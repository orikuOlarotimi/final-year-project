"""
Document Processing & Embedding Pipeline
-----------------------------------------
Fixes & improvements over original:
  - Async-safe file loading via run_in_executor (no event loop blocking)
  - Batch embedding (one API call per EMBED_BATCH_SIZE chunks, not one per chunk)
  - Metadata injected into every chunk BEFORE upsert
  - Single, consistent Pinecone upsert path — dead-code duplication removed
  - Module-level singletons for embeddings & splitter (not re-created per call)
  - Empty/whitespace chunk filtering
  - Structured logging replacing print()
  - Timezone-aware datetimes (datetime.utcnow() is deprecated since 3.12)
  - Retry decorator with exponential back-off for transient failures
  - Chunk-count guard to surface runaway documents early
  - error_message persisted on the document record for easier debugging
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.vector_store import vector_store
from app.models.document import DocumentModel

# ---------------------------------------------------------------------------
# Configuration — change these without touching the logic below
# ---------------------------------------------------------------------------

CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
EMBED_BATCH_SIZE = 100       # chunks per OpenAI embedding request
MAX_CHUNKS_GUARD = 10_000     # raise early if a doc explodes into too many chunks
RETRY_ATTEMPTS   = 3
RETRY_BACKOFF    = 2.0       # seconds; doubles each attempt (1s, 2s, 4s …)

SUPPORTED_TYPES  = {"pdf", "docx", "txt"}

# ---------------------------------------------------------------------------
# Module-level singletons  (constructed ONCE, reused across all calls)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_embeddings = OpenAIEmbeddings()

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Timezone-aware UTC now. datetime.utcnow() is deprecated since Python 3.12."""
    return datetime.now(timezone.utc)


def _with_retry(attempts: int = RETRY_ATTEMPTS, backoff: float = RETRY_BACKOFF):
    """Decorator: retry a coroutine on any Exception with exponential back-off."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < attempts:
                        wait = backoff ** (attempt - 1)
                        logger.warning(
                            "Attempt %d/%d for %s failed — retrying in %.1fs. Error: %s",
                            attempt, attempts, fn.__name__, wait, exc,
                        )
                        await asyncio.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


def _load_document_sync(file_path: str, file_type: str) -> List[Document]:
    """
    Synchronous loader.
    MUST be dispatched via run_in_executor — never called directly in async code.
    """
    loaders = {"pdf": PyPDFLoader, "docx": Docx2txtLoader, "txt": TextLoader}
    if file_type not in loaders:
        raise ValueError(f"Unsupported file type: {file_type!r}. Supported: {SUPPORTED_TYPES}")
    return loaders[file_type](file_path).load()


def _filter_chunks(chunks: List[Document]) -> List[Document]:
    """Drop chunks that are empty or purely whitespace — they waste tokens."""
    filtered = [c for c in chunks if c.page_content.strip()]
    dropped = len(chunks) - len(filtered)
    if dropped:
        logger.info("Dropped %d empty/whitespace chunk(s).", dropped)
    return filtered


def _inject_metadata(chunks: List[Document], document) -> List[Document]:
    """
    Merge document-level metadata into every chunk's metadata dict.
    Without this, Pinecone records are not filterable by document/user/chat.
    """
    extra = {
        "document_id": str(document.id),
        "chat_id":     str(document.chat_id),
        "user_id":     str(document.user_id),
    }
    for chunk in chunks:
        chunk.metadata.update(extra)
    return chunks


@_with_retry()
async def _embed_and_upsert(chunks: List[Document], namespace: str) -> None:
    """
    Embed in batches (not one-by-one) then upsert to Pinecone.
    This is the ONLY place vectors are written — no duplication.
    """
    total = len(chunks)
    loop  = asyncio.get_event_loop()
    logger.info("Embedding %d chunk(s) in batches of %d ...", total, EMBED_BATCH_SIZE)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch  = chunks[start : start + EMBED_BATCH_SIZE]
        texts  = [c.page_content for c in batch]

        # One round-trip to OpenAI per batch
        vectors = await loop.run_in_executor(None, _embeddings.embed_documents, texts)

        records = [
            {
                "id":     f"{chunk.metadata['document_id']}_{start + i}",
                "values": vector,
                "metadata": {**chunk.metadata, "text": chunk.page_content},
            }
            for i, (chunk, vector) in enumerate(zip(batch, vectors))
        ]

        await loop.run_in_executor(
            None,
            lambda: vector_store.upsert(records, namespace=namespace)
        )

        logger.info(
            "  Upserted %d-%d / %d",
            start + 1, min(start + EMBED_BATCH_SIZE, total), total,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def process_document(document_id: str) -> None:
    """
    Full pipeline:
      Load → Chunk → Filter → Inject metadata → Embed (batched) → Upsert → Update status
    """
    document   = None
    start_time = time.monotonic()

    try:
        # 0. Fetch record ----------------------------------------------------
        document = await DocumentModel.get(document_id)

        if not document:
            logger.warning("Unknown document_id=%s — skipping.", document_id)
            return
        namespace = str(document.user_id)
        logger.info("Starting: document_id=%s  type=%s", document_id, document.file_type)
        document.status     = "processing"
        document.updated_at = _utcnow()
        await document.save()

        # 1. Load (non-blocking) --------------------------------------------
        loop     = asyncio.get_event_loop()
        raw_docs = await loop.run_in_executor(
            None, _load_document_sync, document.file_path, document.file_type
        )
        logger.info("Loaded %d page(s).", len(raw_docs))

        # 2. Chunk ----------------------------------------------------------
        chunks = _splitter.split_documents(raw_docs)
        logger.info("Split into %d chunk(s).", len(chunks))

        # 3. Guard ----------------------------------------------------------
        if len(chunks) > MAX_CHUNKS_GUARD:
            raise ValueError(
                f"Document yielded {len(chunks)} chunks — exceeds guard of {MAX_CHUNKS_GUARD}."
            )

        # 4. Filter empty ---------------------------------------------------
        chunks = _filter_chunks(chunks)
        if not chunks:
            raise ValueError("All chunks were empty after filtering.")

        # 5. Inject metadata ------------------------------------------------
        chunks = _inject_metadata(chunks, document)

        # 6. Embed + upsert (retried internally) ----------------------------
        await _embed_and_upsert(chunks, namespace)

        # 7. Persist success ------------------------------------------------
        document.status      = "processed"
        document.chunk_count = len(chunks)     # useful for billing audits & debugging
        document.updated_at  = _utcnow()
        await document.save()

        logger.info(
            "Done: document_id=%s  chunks=%d  elapsed=%.2fs",
            document_id, len(chunks), time.monotonic() - start_time,
        )

    except Exception as exc:
        logger.exception(
            "Failed: document_id=%s  elapsed=%.2fs  error=%s",
            document_id, time.monotonic() - start_time, exc,
        )
        if document:
            document.status        = "failed"
            document.error_message = str(exc)   # ← persist the reason
            document.updated_at    = _utcnow()
            await document.save()