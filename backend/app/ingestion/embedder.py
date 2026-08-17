from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddedChunk(TypedDict):
	chunk_id: int
	text: str
	metadata: dict[str, Any]
	embedding: list[float]


def initialize_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
	"""Initialize and return the embedding model.

	Kept separate from embedding so callers can reuse the model instance.
	"""
	try:
		return SentenceTransformer(model_name)
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(f"Failed to initialize embedding model '{model_name}': {exc}") from exc


def embed_chunks(
	chunks: list[Document],
	model: SentenceTransformer,
	batch_size: int = 32,
	show_progress_bar: bool = True,
	normalize_embeddings: bool = True,
) -> list[EmbeddedChunk]:
	"""Embed LangChain chunk documents and preserve metadata.

	Returns a list of records ready for future vector DB insertion.
	"""
	if not chunks:
		return []

	if batch_size <= 0:
		raise ValueError("batch_size must be greater than 0")

	texts = [chunk.page_content for chunk in chunks]

	try:
		vectors = model.encode(
			texts,
			batch_size=batch_size,
			show_progress_bar=show_progress_bar,
			normalize_embeddings=normalize_embeddings,
			convert_to_numpy=True,
		)
	except Exception as exc:
		raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

	embedded_items: list[EmbeddedChunk] = []
	for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=False)):
		embedded_items.append(
			EmbeddedChunk(
				chunk_id=idx,
				text=chunk.page_content,
				metadata=dict(chunk.metadata),
				embedding=vector.tolist(),
			)
		)

	return embedded_items