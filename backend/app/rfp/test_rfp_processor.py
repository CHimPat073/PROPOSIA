"""Smoke test for the RFP processing pipeline.

Run from the project root:

    python -m backend.app.rfp.test_rfp_processor data/rfps/sample_rfp.pdf

If no argument is given, it falls back to data/rfps/sample_rfp.pdf.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.rfp.processor import process_rfp  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

DEFAULT_RFP_PATH = PROJECT_ROOT / "data" / "rfps" / "sample_rfp.pdf"


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _resolve_rfp_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return DEFAULT_RFP_PATH


def main() -> None:
    rfp_path = _resolve_rfp_path()

    _banner("RFP PROCESSING TEST")
    print(f"\nRFP:\n{rfp_path}")

    processed = process_rfp(rfp_path)
    chunks = processed["chunks"]
    first = chunks[0]

    print(f"\nPages loaded:\n{processed['page_count']}")
    print(f"\nParsed documents:\n{processed['parsed_pages']}")
    print(f"\nRFP chunks:\n{len(chunks)}")
    print(f"\nEmbedding dimension:\n{len(first['embedding'])}")

    _banner("FIRST RFP CHUNK")
    print(f"\nSource:\n{first['metadata'].get('source')}")
    print(f"\nPage:\n{first['metadata'].get('page')}")
    print(f"\nContent:\n{first['text']}")
    print(f"\nEmbedding dimensions:\n{len(first['embedding'])}")


if __name__ == "__main__":
    main()
