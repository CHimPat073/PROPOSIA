from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from backend.app.ingestion.chunker import split_documents
    from backend.app.ingestion.embedder import embed_chunks, initialize_embedding_model
    from backend.app.ingestion.loader import load_documents
except ModuleNotFoundError:  # pragma: no cover
    from chunker import split_documents
    from embedder import embed_chunks, initialize_embedding_model
    from loader import load_documents


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def run_embedding_pipeline_test() -> None:
    docs = load_documents(base_dir=Path("knowledge_base"))
    chunks = split_documents(docs)

    model = initialize_embedding_model()
    embedded_chunks = embed_chunks(chunks, model=model, batch_size=32, show_progress_bar=True)

    print("\n=== EMBEDDING PIPELINE TEST ===")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings: {len(embedded_chunks)}")

    if embedded_chunks:
        first_embedding = embedded_chunks[0]["embedding"]
        print(f"Embedding dimension: {len(first_embedding)}")
        print(f"First vector values (first 8): {first_embedding[:8]}")
        print(f"Metadata of first chunk: {embedded_chunks[0]['metadata']}")
    else:
        print("No chunks were produced, so no embeddings were generated.")


def run_semantic_similarity_test() -> None:
    model = initialize_embedding_model()

    sentences = [
        "We built a cloud-based inventory tracking platform.",
        "Our team developed an online stock management system.",
        "A volcano erupted near the island during monsoon season.",
    ]

    vectors = model.encode(
        sentences,
        batch_size=3,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    related_similarity = cosine_similarity(vectors[0].tolist(), vectors[1].tolist())
    unrelated_similarity = cosine_similarity(vectors[0].tolist(), vectors[2].tolist())

    print("\n=== SEMANTIC SIMILARITY TEST ===")
    print(f"Related sentence similarity: {related_similarity:.4f}")
    print(f"Unrelated sentence similarity: {unrelated_similarity:.4f}")


def main() -> None:
    run_embedding_pipeline_test()
    run_semantic_similarity_test()


if __name__ == "__main__":
    main()
