from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency fallback
    ChatOpenAI = None  # type: ignore[assignment]
    OpenAIEmbeddings = None  # type: ignore[assignment]

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:  # pragma: no cover - optional dependency fallback
    HuggingFaceEmbeddings = None  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
INDEX_DIR = STORAGE_DIR / "faiss_index"
METADATA_FILE = STORAGE_DIR / "documents.json"
CACHE_MAX_ITEMS = 256
DEFAULT_K = 3
MAX_K = 10
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

logger = logging.getLogger("rag_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def capped_k(value: Optional[int]) -> int:
    if value is None:
        return DEFAULT_K
    return max(1, min(MAX_K, value))


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)
    stream: bool = False


class UploadResponseItem(BaseModel):
    filename: str
    stored_filename: str
    file_type: str
    size_bytes: int
    uploaded_at: str
    chunks: int


class QueryResponse(BaseModel):
    result: str
    source_documents: list[dict[str, Any]]


@dataclass
class AppState:
    embeddings: Any | None = None
    vector_store: FAISS | None = None
    memory: ConversationBufferMemory = field(
        default_factory=lambda: ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
    )
    documents: list[dict[str, Any]] = field(default_factory=list)
    query_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


state = AppState()


class GroundedLLM(LLM):
    model_name: str = "grounded-extractive-llm"

    @property
    def _llm_type(self) -> str:
        return self.model_name

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _extract_context_and_question(self, prompt: str) -> tuple[str, str]:
        context = ""
        question = ""
        context_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)
        if context_match:
            context = context_match.group(1).strip()
        question_match = re.search(r"Question:\s*(.*?)\n\nAnswer", prompt, re.DOTALL)
        if question_match:
            question = question_match.group(1).strip()
        return context, question

    def _answer_from_prompt(self, prompt: str) -> str:
        context, question = self._extract_context_and_question(prompt)
        if not context or not question:
            return "I don't know based on the provided documents."

        question_terms = {
            term
            for term in re.findall(r"[A-Za-z0-9]+", question.lower())
            if len(term) > 2
        }
        if not question_terms:
            return "I don't know based on the provided documents."

        segments = [segment.strip() for segment in re.split(r"\n+|(?<=[.!?])\s+", context) if segment.strip()]
        scored_segments: list[tuple[int, str]] = []
        for segment in segments:
            segment_terms = {
                term
                for term in re.findall(r"[A-Za-z0-9]+", segment.lower())
                if len(term) > 2
            }
            overlap = len(question_terms & segment_terms)
            if overlap > 0:
                scored_segments.append((overlap, segment))

        if not scored_segments:
            return "I don't know based on the provided documents."

        scored_segments.sort(key=lambda item: item[0], reverse=True)
        answer = " ".join(segment for _, segment in scored_segments[:3])
        return answer[:800] if answer else "I don't know based on the provided documents."

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        answer = self._answer_from_prompt(prompt)
        if stop:
            for token in stop:
                answer = answer.split(token, 1)[0]
        return answer

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ):
        answer = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
        for token in answer.split():
            chunk = f"{token} "
            if run_manager is not None:
                try:
                    run_manager.on_llm_new_token(chunk)
                except Exception:
                    pass
            yield GenerationChunk(text=chunk)

app = FastAPI(
    title="Logger RAG API",
    version="1.0.0",
    description="Production-ready FastAPI RAG service backed by FAISS.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a retrieval-only assistant. Answer strictly from the provided context. "
        "If the answer is not explicitly supported by the context, reply with exactly: "
        '"I don\'t know based on the provided documents."\n\n'
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer concisely and only use facts from the context."
    ),
)


async def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not METADATA_FILE.exists():
        METADATA_FILE.write_text("[]", encoding="utf-8")


def load_document_metadata() -> list[dict[str, Any]]:
    if not METADATA_FILE.exists():
        return []
    try:
        payload = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    except json.JSONDecodeError:
        logger.warning("Document metadata file is corrupt; starting fresh.")
    return []


def save_document_metadata(documents: list[dict[str, Any]]) -> None:
    METADATA_FILE.write_text(
        json.dumps(documents, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def build_embeddings() -> Any:
    if os.getenv("OPENAI_API_KEY"):
        if OpenAIEmbeddings is None:
            raise RuntimeError("OpenAI embeddings are unavailable. Install langchain-openai.")
        logger.info("Using OpenAI embeddings.")
        return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

    if HuggingFaceEmbeddings is None:
        raise RuntimeError(
            "No embeddings backend is available. Install langchain-openai or langchain-community with sentence-transformers."
        )

    logger.info("Using HuggingFace embeddings: %s", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


def build_llm(streaming: bool = False, callbacks: Optional[list[Any]] = None) -> Any:
    if ChatOpenAI is None:
        logger.info("Using grounded local LLM fallback.")
        return GroundedLLM()
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("OPENAI_API_KEY is not set; using grounded local LLM fallback.")
        return GroundedLLM()
    return ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        temperature=0,
        streaming=streaming,
        callbacks=callbacks or [],
    )


def load_vector_store() -> FAISS | None:
    if not INDEX_DIR.exists():
        return None
    index_files = list(INDEX_DIR.glob("*"))
    if not index_files:
        return None
    assert state.embeddings is not None
    try:
        return FAISS.load_local(
            str(INDEX_DIR),
            state.embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        logger.warning("Failed to load FAISS index: %s", exc)
        return None


def save_vector_store(vector_store: FAISS) -> None:
    vector_store.save_local(str(INDEX_DIR))


def normalize_cache_key(question: str, k: int) -> str:
    normalized = normalize_text(question)
    memory_signature = hashlib.sha256(
        json.dumps(state.memory.chat_memory.messages, default=str, ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    payload = f"question={normalized}|k={k}|memory={memory_signature}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trim_cache() -> None:
    if len(state.query_cache) <= CACHE_MAX_ITEMS:
        return
    for key in list(state.query_cache.keys())[: len(state.query_cache) - CACHE_MAX_ITEMS]:
        state.query_cache.pop(key, None)


def load_documents_from_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    if suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()
    raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf and .txt are allowed.")


def chunk_documents(documents: list[Document], source_name: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["source_file"] = source_name
    return chunks


def serialize_source_documents(documents: list[Document]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for document in documents:
        serialized.append(
            {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
        )
    return serialized


def sanitize_filename(filename: str) -> str:
    return Path(filename).name


def validate_upload(upload: UploadFile) -> str:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    filename = sanitize_filename(upload.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only .pdf and .txt are allowed.",
        )
    return filename


def get_retriever(vector_store: FAISS, k: int):
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": DEFAULT_SIMILARITY_THRESHOLD},
    )


def retrieve_relevant_documents(question: str, k: int) -> list[Document]:
    if state.vector_store is None:
        return []
    scored_documents = state.vector_store.similarity_search_with_relevance_scores(
        question,
        k=capped_k(k),
    )
    return [document for document, score in scored_documents if score >= DEFAULT_SIMILARITY_THRESHOLD]


def get_qa_chain(retriever: Any, streaming: bool = False, callbacks: Optional[list[Any]] = None) -> ConversationalRetrievalChain:
    llm = build_llm(streaming=streaming, callbacks=callbacks)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT_TEMPLATE},
        verbose=False,
    )


def filter_source_documents(documents: list[Document], max_docs: int) -> list[Document]:
    if not documents:
        return []
    return documents[:max_docs]


def answer_from_cache(cache_key: str) -> Optional[dict[str, Any]]:
    return state.query_cache.get(cache_key)


def store_in_cache(cache_key: str, payload: dict[str, Any]) -> None:
    state.query_cache[cache_key] = payload
    trim_cache()


async def build_or_merge_vector_store(new_chunks: list[Document]) -> None:
    if not new_chunks:
        raise HTTPException(status_code=400, detail="No extractable text was found in the uploaded file.")
    assert state.embeddings is not None
    logger.info("Building FAISS store for %d chunks.", len(new_chunks))
    new_store = await asyncio.to_thread(FAISS.from_documents, new_chunks, state.embeddings)
    if state.vector_store is None:
        state.vector_store = new_store
    else:
        state.vector_store.merge_from(new_store)
    await asyncio.to_thread(save_vector_store, state.vector_store)


async def process_upload(upload: UploadFile) -> UploadResponseItem:
    filename = validate_upload(upload)
    file_id = uuid.uuid4().hex
    stored_filename = f"{file_id}_{filename}"
    stored_path = UPLOADS_DIR / stored_filename
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"Uploaded file {filename} is empty.")
    await asyncio.to_thread(stored_path.write_bytes, raw)

    try:
        loaded_documents = await asyncio.to_thread(load_documents_from_file, stored_path)
        if not loaded_documents:
            raise HTTPException(status_code=400, detail=f"No text could be extracted from {filename}.")
        chunks = await asyncio.to_thread(chunk_documents, loaded_documents, filename)
        await build_or_merge_vector_store(chunks)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to process upload %s", filename)
        raise HTTPException(status_code=500, detail=f"Failed to process {filename}: {exc}") from exc

    metadata = {
        "filename": filename,
        "stored_filename": stored_filename,
        "file_type": Path(filename).suffix.lower().lstrip("."),
        "size_bytes": len(raw),
        "uploaded_at": utc_now(),
        "chunks": len(chunks),
    }
    state.documents.append(metadata)
    save_document_metadata(state.documents)
    logger.info("Uploaded and indexed %s (%d chunks).", filename, len(chunks))
    return UploadResponseItem(**metadata)


async def run_query(question: str, k: int, stream: bool = False, callbacks: Optional[list[Any]] = None) -> dict[str, Any]:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if state.vector_store is None:
        return {"result": "I don't know based on the provided documents.", "source_documents": []}

    k = capped_k(k)
    cache_key = normalize_cache_key(cleaned_question, k)
    cached = answer_from_cache(cache_key)
    if cached is not None:
        logger.info("Cache hit for question=%s k=%s", cleaned_question, k)
        return cached

    relevant_documents = await asyncio.to_thread(retrieve_relevant_documents, cleaned_question, k)
    if not relevant_documents:
        payload = {"result": "I don't know based on the provided documents.", "source_documents": []}
        store_in_cache(cache_key, payload)
        return payload

    retriever = get_retriever(state.vector_store, k)
    chain = get_qa_chain(retriever=retriever, streaming=stream, callbacks=callbacks)

    try:
        result = await chain.ainvoke({"question": cleaned_question})
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    answer = str(result.get("answer", "")).strip()
    source_documents = filter_source_documents(result.get("source_documents", []), k)

    if not source_documents:
        answer = "I don't know based on the provided documents."
    elif not answer:
        answer = "I don't know based on the provided documents."

    payload = {
        "result": answer,
        "source_documents": serialize_source_documents(source_documents),
    }
    store_in_cache(cache_key, payload)
    return payload


async def stream_query(question: str, k: int) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        async with state.lock:
            cleaned_question = question.strip()
            cache_key = normalize_cache_key(cleaned_question, capped_k(k))
            cached = answer_from_cache(cache_key)
            if cached is not None:
                yield f"data: {json.dumps({'event': 'final', **cached}, ensure_ascii=True)}\n\n"
                return

            if state.vector_store is None:
                payload = {
                    "result": "I don't know based on the provided documents.",
                    "source_documents": [],
                }
                store_in_cache(cache_key, payload)
                yield f"data: {json.dumps({'event': 'final', **payload}, ensure_ascii=True)}\n\n"
                return

            relevant_documents = await asyncio.to_thread(retrieve_relevant_documents, cleaned_question, k)
            if not relevant_documents:
                payload = {
                    "result": "I don't know based on the provided documents.",
                    "source_documents": [],
                }
                store_in_cache(cache_key, payload)
                yield f"data: {json.dumps({'event': 'final', **payload}, ensure_ascii=True)}\n\n"
                return

            callback_handler = AsyncIteratorCallbackHandler()
            retriever = get_retriever(state.vector_store, capped_k(k))
            chain = get_qa_chain(retriever=retriever, streaming=True, callbacks=[callback_handler])

            async def task_runner() -> dict[str, Any]:
                return await chain.ainvoke({"question": cleaned_question})

            task = asyncio.create_task(task_runner())

            try:
                async for token in callback_handler.aiter():
                    yield f"data: {json.dumps({'event': 'token', 'token': token}, ensure_ascii=True)}\n\n"
                result = await task
                answer = str(result.get("answer", "")).strip() or "I don't know based on the provided documents."
                source_documents = filter_source_documents(result.get("source_documents", []), capped_k(k))
                if not source_documents:
                    answer = "I don't know based on the provided documents."
                payload = {
                    "result": answer,
                    "source_documents": serialize_source_documents(source_documents),
                }
                store_in_cache(cache_key, payload)
                yield f"data: {json.dumps({'event': 'final', **payload}, ensure_ascii=True)}\n\n"
            except Exception as exc:
                if not task.done():
                    task.cancel()
                logger.exception("Streaming query failed")
                yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)}, ensure_ascii=True)}\n\n"
            finally:
                callback_handler.done.set()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.on_event("startup")
async def startup_event() -> None:
    await ensure_storage()
    state.embeddings = build_embeddings()
    state.documents = load_document_metadata()
    state.vector_store = await asyncio.to_thread(load_vector_store)
    logger.info("Startup complete. Documents=%d, index_loaded=%s", len(state.documents), state.vector_store is not None)


@app.post("/upload", response_model=list[UploadResponseItem])
async def upload_documents(files: list[UploadFile] = File(...)) -> list[UploadResponseItem]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

    async with state.lock:
        results: list[UploadResponseItem] = []
        for upload in files:
            results.append(await process_upload(upload))
        return results


@app.post("/query")
async def query_documents(request: QueryRequest):
    if state.vector_store is None:
        payload = {
            "result": "I don't know based on the provided documents.",
            "source_documents": [],
        }
        if request.stream:
            async def empty_stream() -> AsyncIterator[str]:
                yield f"data: {json.dumps({'event': 'final', **payload}, ensure_ascii=True)}\n\n"

            return StreamingResponse(empty_stream(), media_type="text/event-stream")
        return JSONResponse(content=payload)

    async with state.lock:
        if request.stream:
            return await stream_query(request.question, request.k)
        payload = await run_query(request.question, request.k, stream=False)
        return JSONResponse(content=payload)


@app.get("/documents")
async def list_documents() -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(state.documents, key=lambda item: item.get("uploaded_at", ""), reverse=True)
    return {"documents": ordered}


@app.delete("/history")
async def clear_history() -> dict[str, str]:
    async with state.lock:
        state.memory.clear()
        state.query_cache.clear()
    logger.info("Conversation history cleared.")
    return {"detail": "Conversation history cleared."}


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "documents": len(state.documents),
        "index_loaded": state.vector_store is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
