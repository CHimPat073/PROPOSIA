"""One-shot ingestion: load -> chunk -> embed -> store in ChromaDB.

Run from project root:
    python -m backend.app.ingestion.ingest_to_chroma
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from backend.app.ingestion.chunker import split_documents
from backend.app.ingestion.embedder import (
    initialize_embedding_model,
    embed_chunks,
)
from backend.app.ingestion.loader import load_documents
from backend.app.database.vector_store import ProposalVectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def assign_chunk_indices(chunks: list[Document]) -> list[Document]:
    """Add a `chunk_index` field to each chunk's metadata.

    The vector store uses this to build deterministic IDs. We assign
    the index per source file so re-runs stay stable even if a file
    is added later (existing chunks keep their original index).
    """
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        idx = counters.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        counters[source] = idx + 1
    return chunks


def run_ingestion() -> ProposalVectorStore:
    documents = load_documents(base_dir=Path("knowledge_base"))
    chunks = split_documents(documents)
    chunks = assign_chunk_indices(chunks)

    model = initialize_embedding_model()
    embedded_chunks = embed_chunks(
        chunks,
        model=model,
        batch_size=32,
        show_progress_bar=True,
    )

    store = ProposalVectorStore()
    inserted = store.add_chunks(embedded_chunks)

    LOGGER.info(
        "Ingestion complete | documents=%d | chunks=%d | new_records=%d | total_records=%d",
        len(documents),
        len(chunks),
        inserted,
        store.count(),
    )
    return store


if __name__ == "__main__":
    run_ingestion()
