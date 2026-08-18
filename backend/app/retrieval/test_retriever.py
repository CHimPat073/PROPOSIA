"""Quick smoke test for the Retriever.

Run from the project root:
    python -m backend.app.retrieval.test_retriever
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.retrieval.retriever import Retriever  # noqa: E402


def _print_section(title: str, width: int = 30) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def _print_result(rank: int, result: dict) -> None:
    metadata = result.get("metadata", {}) or {}
    print(f"\nSource:   {metadata.get('source', 'unknown')}")
    print(f"Category: {metadata.get('category', 'unknown')}")
    print(f"Distance: {result.get('distance', 0.0):.4f}")
    print(f"Content:\n{result.get('text', '')}")


def main() -> None:
    query = "We need a cloud-based inventory management system with real-time analytics."

    retriever = Retriever()

    _print_section("QUERY")
    print(query)

    results = retriever.retrieve(query, k=5)

    if not results:
        print("\nNo results found. Did you run the ingestion pipeline first?")
        return

    for i, result in enumerate(results, start=1):
        _print_section(f"RESULT {i}")
        _print_result(i, result)


if __name__ == "__main__":
    main()
