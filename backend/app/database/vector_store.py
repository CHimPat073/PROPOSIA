"""ChromaDB-backed vector store for the AI Sales Proposal Generator.

Wraps a single ChromaDB collection (`proposal_knowledge`) with a small,
beginner-friendly API:

    store = ProposalVectorStore()
    store.add_chunks(embedded_chunks)
    store.count()
    store.similarity_search(query, k=3)
    store.get_by_id("chunk_xxxx")

The class is intentionally small so the retrieval + LLM stages can
import it later without extra abstraction.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHROMA_PATH = PROJECT_ROOT / "backend" / "database" / "chroma"

COLLECTION_NAME = "proposal_knowledge"
EMBEDDING_DIMENSION = 384

DistanceFunction = "cosine"


def _project_root() -> Path:
    """Return the project root (folder that contains `backend/`)."""
    return PROJECT_ROOT


def build_chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Build a deterministic, collision-resistant ID for one chunk.

    Same source + same index + same content always produces the same ID,
    so re-running ingestion does not create duplicate rows.
    """
    payload = f"{source}::{chunk_index}::{content}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"chunk_{digest}"


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool — coerce safely."""
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


class ProposalVectorStore:
    """Thin wrapper around a single ChromaDB collection."""

    def __init__(
        self,
        persist_directory: Path | str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        distance_function: str = DistanceFunction,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection_name = collection_name
        self.collection: Collection = self._get_or_create_collection(distance_function)

        LOGGER.info(
            "ChromaDB ready at %s | collection='%s' | records=%d",
            self.persist_directory,
            self.collection_name,
            self.collection.count(),
        )

    def _get_or_create_collection(self, distance_function: str) -> Collection:
        """Get the collection if it exists, else create it with the right space."""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": distance_function},
            )

    def add_chunks(
        self,
        embedded_chunks: list[dict[str, Any]],
        batch_size: int = 256,
    ) -> int:
        """Insert embedded chunks into ChromaDB. Skips duplicates by deterministic ID.

        Each item in `embedded_chunks` must be an `EmbeddedChunk` produced
        by `embedder.embed_chunks` (keys: chunk_id, text, metadata, embedding).
        """
        if not embedded_chunks:
            LOGGER.warning("add_chunks called with no chunks — nothing to insert.")
            return 0

        inserted = 0
        buffer_ids: list[str] = []
        buffer_docs: list[str] = []
        buffer_embeddings: list[list[float]] = []
        buffer_metas: list[dict[str, Any]] = []

        def _flush() -> None:
            nonlocal inserted
            if not buffer_ids:
                return
            existing = set(self.collection.get(ids=buffer_ids).get("ids", []))
            new_ids: list[str] = []
            new_docs: list[str] = []
            new_embeddings: list[list[float]] = []
            new_metas: list[dict[str, Any]] = []
            for i, d, e, m in zip(buffer_ids, buffer_docs, buffer_embeddings, buffer_metas, strict=True):
                if i in existing:
                    continue
                new_ids.append(i)
                new_docs.append(d)
                new_embeddings.append(e)
                new_metas.append(m)
            if new_ids:
                self.collection.add(
                    ids=new_ids,
                    documents=new_docs,
                    embeddings=new_embeddings,
                    metadatas=new_metas,
                )
                inserted += len(new_ids)
            buffer_ids.clear()
            buffer_docs.clear()
            buffer_embeddings.clear()
            buffer_metas.clear()

        for item in embedded_chunks:
            text = item["text"]
            metadata = dict(item.get("metadata", {}))
            chunk_index = int(metadata.get("chunk_index", item.get("chunk_id", 0)))
            source = str(metadata.get("source", "unknown"))

            metadata["chunk_index"] = chunk_index
            sanitized_meta = _sanitize_metadata(metadata)

            vector = item["embedding"]
            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding dimension {len(vector)} != expected {EMBEDDING_DIMENSION}. "
                    "Did you use a different embedding model?"
                )

            chunk_id = build_chunk_id(source=source, chunk_index=chunk_index, content=text)
            buffer_ids.append(chunk_id)
            buffer_docs.append(text)
            buffer_embeddings.append(vector)
            buffer_metas.append(sanitized_meta)

            if len(buffer_ids) >= batch_size:
                _flush()

        _flush()
        LOGGER.info("Inserted %d new chunks (skipped duplicates).", inserted)
        return inserted

    def count(self) -> int:
        """Return how many records are currently in the collection."""
        return self.collection.count()

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 3,
    ) -> dict[str, Any]:
        """Search by a pre-computed query embedding."""
        if len(query_embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} != "
                f"expected {EMBEDDING_DIMENSION}."
            )
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

    def search_by_text(
        self,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """High-level helper: search by text using a pre-built query embedding.

        Returns a flat list of result dicts (easier to print than Chroma's
        nested lists):
            [{"id": ..., "text": ..., "metadata": ..., "distance": ...}, ...]
        """
        raw = self.similarity_search(query_embedding=query_embedding, k=k)

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[dict[str, Any]] = []
        for i, doc, meta, dist in zip(ids, documents, metadatas, distances, strict=True):
            similarity = 1.0 - float(dist)
            results.append(
                {
                    "id": i,
                    "text": doc,
                    "metadata": meta or {},
                    "distance": float(dist),
                    "similarity": similarity,
                }
            )
        return results

    def get_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Fetch a single record by its deterministic ID."""
        raw = self.collection.get(ids=[chunk_id])
        ids = raw.get("ids", [])
        if not ids:
            return None
        return {
            "id": ids[0],
            "text": (raw.get("documents") or [""])[0],
            "metadata": (raw.get("metadatas") or [{}])[0],
        }
