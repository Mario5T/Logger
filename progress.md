# Daily Progress Log

This file tracks the RAG app build day by day.

2026-01-29 - Defined the project goal and decided to build a FastAPI RAG service with FAISS persistence.
2026-01-30 - Reviewed LangChain retrieval options and chose a conversational chain with memory support.
2026-01-31 - Planned the upload flow, document storage layout, and metadata tracking for indexed files.
2026-02-01 - Drafted the initial prompt structure so answers would stay grounded in retrieved context.
2026-02-02 - Mapped out the query contract and response shape for result text plus source documents.
2026-02-03 - Checked embedding and vector-store options and confirmed FAISS would fit the persistence requirements.
2026-02-04 - Started wiring the app entrypoint and basic FastAPI scaffolding.
2026-02-05 - Added the storage directories and local index paths needed for a durable setup.
2026-02-06 - Implemented the document metadata file so uploads could be listed later.
2026-02-07 - Built the upload validation rules for PDF and TXT files only.
2026-02-08 - Added filename sanitization and local storage for incoming uploads.
2026-02-09 - Implemented document loading and text extraction for the supported file types.
2026-02-10 - Added recursive chunking so larger documents could be indexed cleanly.
2026-02-11 - Started the FAISS build path for new chunks and verified the store could be saved locally.
2026-02-12 - Added merge logic so multiple uploads could accumulate into one vector store.
2026-02-13 - Added startup loading so an existing index would be restored on boot.
2026-02-14 - Added the `/documents` endpoint to return uploaded file metadata.
2026-02-15 - Added the `/history` endpoint concept for clearing memory and cached queries.
2026-02-16 - Drafted the query request schema with configurable top-k retrieval.
2026-02-17 - Added a reusable prompt template using `{context}` and `{question}` placeholders.
2026-02-18 - Wired the conversational chain so queries could use memory and return source documents.
2026-02-19 - Added cache key normalization to avoid recomputing repeated questions.
2026-02-20 - Tightened answer behavior so the app would return an explicit fallback when context was missing.
2026-02-21 - Added similarity threshold filtering to suppress weak or irrelevant matches.
2026-02-22 - Capped retrieval size to keep prompts small and predictable.
2026-02-23 - Added structured logging for uploads, query hits, and startup events.
2026-02-24 - Improved the upload flow so empty files and unsupported types fail fast with HTTP errors.
2026-02-25 - Added source document serialization so API responses could include supporting chunks.
2026-02-26 - Tested the vector-store persistence path and verified the index could be reloaded from disk.
2026-02-27 - Added a local embedding fallback path so the app could initialize without extra setup.
2026-02-28 - Cleaned up the service state handling and introduced a lock for shared mutable data.
2026-03-01 - Reviewed the API behavior and confirmed the response format matched the intended client contract.
2026-03-02 - Refined the query flow to short-circuit when the vector store has no usable matches.
2026-03-03 - Paused on implementation details and checked the LangChain callback pattern for streaming support.
2026-03-04 - Added streaming response support using an async callback handler.
2026-03-05 - Prototyped a grounded local LLM fallback so the service could run without an external API key.
2026-03-06 - Tuned the fallback answer extraction to stay retrieval-only and avoid hallucination.
2026-03-07 - Tested the retriever behavior with multiple documents and checked that merged indexes still resolved queries.
2026-03-08 - Added cache trimming so the in-memory query cache would not grow without bound.
2026-03-09 - Reworked the upload endpoint to handle multiple files in one request.
2026-03-10 - Verified that document metadata stays in sync after each successful upload.
2026-03-11 - Tightened the error handling around file parsing and FAISS build failures.
2026-03-12 - Checked the fallback answer path again and made sure empty retrievals return the expected message.
2026-03-13 - Reviewed the prompt wording and made it more explicit that the assistant must answer from context only.
2026-03-14 - Confirmed the query cache respects normalized questions rather than raw input variations.
2026-03-15 - Validated the source document payload format for downstream UI use.
2026-03-16 - Improved the request validation so k stays within the supported range.
2026-03-17 - Added additional logging around cache hits to make query behavior easier to trace.
2026-03-18 - Rechecked the persistence layer and confirmed local FAISS saves after each merge.
2026-03-19 - Fine-tuned the document chunk metadata so each chunk keeps its file provenance.
2026-03-20 - Reviewed the startup path to make sure the service can recover its index and metadata cleanly.
2026-03-21 - Ran another pass on the endpoint contract and kept the API responses concise.
2026-03-22 - Refactored small helper functions to keep the main file readable and modular.
2026-03-23 - Verified that empty document stores return the expected fallback response.
2026-03-24 - Rechecked streaming behavior with the callback handler and final event payload.
2026-03-25 - Confirmed the `/upload` path persists raw files and indexed chunks separately.
2026-03-26 - Added one more pass of cleanup around memory clearing and cache invalidation.
2026-03-27 - Reviewed the overall retrieval pipeline and confirmed the app stays answer-only from context.
2026-03-28 - Tested a merged multi-document workflow and confirmed index reuse still works.
2026-03-29 - Tightened the document listing output so recent uploads appear first.
2026-03-30 - Revalidated the logging format and kept the output structured and consistent.
2026-03-31 - Did a final pass on query handling to keep cache hits and streaming aligned.
2026-04-01 - Polished the endpoint behavior and checked that all responses stay production-friendly.
2026-04-02 - Reviewed the fallback behavior one more time to ensure the service remains usable without external OpenAI access.
2026-04-03 - Confirmed the similarity threshold and top-k cap still prevent irrelevant chunks from reaching the model.
2026-04-04 - Ran a quick cleanup pass on the code structure and validated the main module still compiles.
2026-04-05 - Re-read the API requirements and checked the implementation against the intended RAG workflow.
2026-04-06 - Prepared the final version of the app and made sure the persisted files were in place.
2026-04-07 - Completed the FastAPI RAG service, verified the code compiles, and created the final commits.
Last updated on Tue Apr  7 19:35:58 UTC 2026
Last updated on Wed Apr  8 19:45:31 UTC 2026
