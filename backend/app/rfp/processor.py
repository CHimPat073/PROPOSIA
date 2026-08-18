"""RFP processor — orchestrates the RFP pipeline.

This is the only file in the `rfp/` package that combines steps.
It reuses the existing chunker and embedder so we have a single
source of truth for chunk size, overlap, and the embedding model.

Pipeline:
    PDF -> load_rfp_pdf -> parse_rfp -> split_documents -> embed_chunks

Returns EmbeddedChunk records that the caller (later, the Retriever)
can use as queries against the company-knowledge ChromaDB.

Important: this function does NOT touch ChromaDB. RFP embeddings
are returned in-memory only.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer  # noqa: E402

from backend.app.ingestion.chunker import split_documents  # noqa: E402
from backend.app.ingestion.embedder import (  # noqa: E402
    EmbeddedChunk,
    embed_chunks,
    initialize_embedding_model,
)
from backend.app.rfp.loader import load_rfp_pdf  # noqa: E402
from backend.app.rfp.parser import parse_rfp  # noqa: E402

LOGGER = logging.getLogger(__name__)

EXPECTED_DIMENSION = 384


class ProcessedRFP(TypedDict):
    """Bundle returned by process_rfp for downstream consumers."""
    source: str
    page_count: int
    parsed_pages: int
    chunks: list[EmbeddedChunk]


def _validate_dimension(embedded_chunks: list[EmbeddedChunk], expected: int) -> None:
    """Guard against an embedding-model swap producing the wrong vector size."""
    if not embedded_chunks:
        return
    actual = len(embedded_chunks[0]["embedding"])
    if actual != expected:
        raise ValueError(
            f"Embedding dimension mismatch: got {actual}, expected {expected}. "
            "Did you change the embedding model?"
        )


def process_rfp(
    file_path: str | Path,
    embedding_model: SentenceTransformer | None = None,
) -> ProcessedRFP:
    """Run the full RFP pipeline and return embedded chunks (in-memory).

    Args:
        file_path: Path to the RFP PDF.
        embedding_model: Optional pre-loaded model (for tests / batch jobs).

    Returns:
        ProcessedRFP dict with source, counts, and embedded chunks.
    """
    documents = load_rfp_pdf(file_path)
    page_count = len(documents)

    parsed_pages = parse_rfp(documents)
    if not parsed_pages:
        raise ValueError("RFP parsing produced no usable pages.")

    chunks = split_documents(parsed_pages)
    if not chunks:
        raise ValueError("Chunking produced zero chunks; check chunk_size / RFP content.")

    model = embedding_model or initialize_embedding_model()

    embedded = embed_chunks(
        chunks=chunks,
        model=model,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    if not embedded:
        raise ValueError("Embedding produced zero vectors; nothing to return.")

    _validate_dimension(embedded, EXPECTED_DIMENSION)

    source_value = str(parsed_pages[0].metadata.get("source", str(file_path)))

    LOGGER.info(
        "Processed RFP '%s' -> pages=%d, chunks=%d",
        source_value,
        page_count,
        len(embedded),
    )

    return ProcessedRFP(
        source=source_value,
        page_count=page_count,
        parsed_pages=len(parsed_pages),
        chunks=embedded,
    )


def summarize_rfp(processed: ProcessedRFP) -> dict[str, Any]:
    """Tiny helper for the test script: returns counts + first chunk summary."""
    first = processed["chunks"][0]
    return {
        "source": processed["source"],
        "page_count": processed["page_count"],
        "parsed_pages": processed["parsed_pages"],
        "chunk_count": len(processed["chunks"]),
        "first_chunk": {
            "source": first["metadata"].get("source"),
            "page": first["metadata"].get("page"),
            "content_preview": first["text"][:300],
            "embedding_dim": len(first["embedding"]),
        },
    }
