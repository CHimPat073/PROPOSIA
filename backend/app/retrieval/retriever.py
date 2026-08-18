"""Retrieval layer for the AI Sales Proposal Generator.

Coordinates one job:
    query text -> embedding model -> vector store -> ranked chunks.

Responsibilities kept narrow:
- Embed the user's query with the SAME model used during ingestion.
- Hand the embedding to ProposalVectorStore (ChromaDB wrapper).
- Return a flat list of result dicts the rest of the pipeline can use.

This module deliberately knows nothing about ChromaDB internals or about
how the embedding model is built. It only orchestrates them.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer  # noqa: E402

from backend.app.database.vector_store import ProposalVectorStore  # noqa: E402
from backend.app.ingestion.embedder import (  # noqa: E402
    MODEL_NAME,
    initialize_embedding_model,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


class Retriever:
    """High-level retrieval interface for the proposal knowledge base.

    Flow:
        user query
            -> embedding model (BAAI/bge-small-en-v1.5, 384 dims)
            -> ProposalVectorStore.similarity_search
            -> flat list of top-k chunks with metadata + distance
    """

    def __init__(
        self,
        vector_store: ProposalVectorStore | None = None,
        embedding_model: SentenceTransformer | None = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.vector_store: ProposalVectorStore = vector_store or ProposalVectorStore()
        self.embedding_model: SentenceTransformer = (
            embedding_model or initialize_embedding_model(model_name)
        )
        LOGGER.info(
            "Retriever ready | model='%s' | collection_records=%d",
            model_name,
            self.vector_store.count(),
        )

    def _embed_query(self, query: str) -> list[float]:
        """Turn the user's text into a 384-dim vector using the same model
        and normalization that were used during ingestion."""
        vector = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """Return the top-k chunks most relevant to the natural-language query.

        Each result dict has keys: id, text, metadata, distance, similarity.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be greater than 0")

        query_embedding = self._embed_query(query)
        results = self.vector_store.search_by_text(
            query_embedding=query_embedding,
            k=k,
        )
        LOGGER.info(
            "Retrieved %d chunks for query (k=%d).",
            len(results),
            k,
        )
        return results

    def retrieve_from_rfp(
        self,
        processed_rfp: dict[str, Any],
        k: int = DEFAULT_TOP_K,
    ) -> list[list[dict[str, Any]]]:
        """Run one similarity search per RFP chunk.

        Args:
            processed_rfp: Output of `backend.app.rfp.processor.process_rfp`.
                Only the `chunks` key is consumed; each item must have an
                `embedding` list (384 dims) produced by the same model.
            k: Top-k company chunks to return per RFP chunk.

        Returns:
            list[list[dict]] — outer length = number of RFP chunks,
            inner length <= k. Each inner dict has id/text/metadata/distance/similarity.
        """
        chunks = processed_rfp.get("chunks") if isinstance(processed_rfp, dict) else None
        if not chunks:
            raise ValueError(
                "processed_rfp must be a dict containing a non-empty 'chunks' list."
            )
        if k <= 0:
            raise ValueError("k must be greater than 0")

        results_per_chunk: list[list[dict[str, Any]]] = []
        for index, chunk in enumerate(chunks):
            embedding = chunk.get("embedding")
            if not embedding:
                LOGGER.warning("Skipping RFP chunk %d: missing embedding.", index)
                continue
            results_per_chunk.append(
                self.vector_store.search_by_text(
                    query_embedding=embedding,
                    k=k,
                )
            )

        LOGGER.info(
            "RFP retrieval done | rfp_chunks=%d | k_per_chunk=%d | result_lists=%d",
            len(chunks),
            k,
            len(results_per_chunk),
        )
        return results_per_chunk
