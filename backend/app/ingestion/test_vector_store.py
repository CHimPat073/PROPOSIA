"""End-to-end smoke test: load -> chunk -> embed -> store -> query.

Run from project root:
    python -m backend.app.ingestion.test_vector_store
"""

from __future__ import annotations

import logging
from pathlib import Path
try:
    from backend.app.ingestion.chunker import split_documents
    from backend.app.ingestion.embedder import (
        initialize_embedding_model,
        embed_chunks,
    )
    from backend.app.ingestion.ingest_to_chroma import assign_chunk_indices
    from backend.app.ingestion.loader import load_documents
    from backend.app.database.vector_store import ProposalVectorStore
except ModuleNotFoundError:  # pragma: no cover
    from chunker import split_documents
    from embedder import initialize_embedding_model, embed_chunks
    from ingest_to_chroma import assign_chunk_indices
    from loader import load_documents
    from database.vector_store import ProposalVectorStore    

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger("test_vector_store")


def main() -> None:
    documents = load_documents(base_dir=Path("knowledge_base"))
    chunks = split_documents(documents)
    chunks = assign_chunk_indices(chunks)

    model = initialize_embedding_model()
    embedded_chunks = embed_chunks(
        chunks,
        model=model,
        batch_size=32,
        show_progress_bar=False,
    )

    store = ProposalVectorStore()
    inserted = store.add_chunks(embedded_chunks)

    dimension = len(embedded_chunks[0]["embedding"]) if embedded_chunks else 0

    print("\n=== PIPELINE SUMMARY ===")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {len(embedded_chunks)}")
    print(f"Vector dimension: {dimension}")
    print(f"Chroma records: {store.count()}")
    print(f"Newly inserted this run: {inserted}")

    query_text = "Does the company have experience with data engineering and cloud solutions?"
    query_embedding = model.encode(
        query_text, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    results = store.search_by_text(
        query_text=query_text,
        query_embedding=query_embedding,
        k=3,
    )

    print(f"\n=== SEMANTIC SEARCH: {query_text!r} ===")
    for i, result in enumerate(results, start=1):
        print(f"\n-------------------------")
        print(f"RESULT {i}")
        print(f"-------------------------")
        print(f"Source: {result['metadata'].get('source')}")
        print(f"Category: {result['metadata'].get('category', 'n/a')}")
        print(f"Document type: {result['metadata'].get('document_type', 'n/a')}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Similarity: {result['similarity']:.4f}")
        print("Content:")
        print(result["text"])


if __name__ == "__main__":
    main()
