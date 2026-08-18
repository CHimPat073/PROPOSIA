"""End-to-end demo: PDF RFP / text RFP -> retrieval -> proposal -> chatbot.

Run from project root:

    python -m backend.tests.test_full_pipeline

Modes:
    Default   -> runs both PDF and text RFP paths.
    pdf-only  -> python -m backend.tests.test_full_pipeline pdf
    text-only -> python -m backend.tests.test_full_pipeline text
    chat-only -> python -m backend.tests.test_full_pipeline chat
                 (assumes a proposal already printed; otherwise re-runs text)

Requirements:
    - .env at project root with GROQ_API_KEY=...
    - GROQ_MODEL optionally set
    - knowledge_base/ populated and ingested into ChromaDB
    - data/rfps/sample_rfp.pdf for the PDF path (or pass --pdf <path>)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr so Windows consoles can print non-ASCII characters
# (em-dashes, smart quotes, etc.) emitted by the LLM.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.llm.chat import run_chat_loop  # noqa: E402
from backend.app.llm.client import GroqClient  # noqa: E402
from backend.app.llm.generator import (  # noqa: E402
    generate_proposal_from_rfp,
    generate_proposal_from_text,
)
from backend.app.retrieval.retriever import Retriever  # noqa: E402
from backend.app.rfp.processor import process_rfp  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

DEFAULT_PDF = PROJECT_ROOT / "data" / "rfps" / "sample_rfp.pdf"
SAMPLE_TEXT_RFP = (
    "We need a cloud-based inventory management system with real-time "
    "analytics, predictive demand forecasting, role-based access control, "
    "and integration with our existing SAP ERP. The platform must support "
    "500 concurrent users, 99.9% uptime, and SOC 2 compliance."
)


def _print_proposal(label: str, result: dict) -> None:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(f"\nContext chunks used: {len(result.get('context', []))}")
    print("\n--- RFP INPUT (first 400 chars) ---")
    print(result.get("rfp_text", "")[:400])
    print("\n--- GENERATED PROPOSAL ---")
    print(result.get("proposal", ""))


def _run_pdf(retriever: Retriever, client: GroqClient, pdf_path: Path) -> dict:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Sample RFP PDF not found at {pdf_path}. "
            "Place one there or pass --pdf <path>."
        )
    processed = process_rfp(pdf_path)
    result = generate_proposal_from_rfp(processed, retriever, client)
    _print_proposal(f"PROPOSAL FROM PDF: {pdf_path.name}", result)
    return result


def _run_text(retriever: Retriever, client: GroqClient, text: str) -> dict:
    result = generate_proposal_from_text(text, retriever, client)
    _print_proposal("PROPOSAL FROM TEXT", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end RFP pipeline demo.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "pdf", "text", "chat"],
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("Initializing Retriever + Groq client...")
    retriever = Retriever()
    client = GroqClient()

    last_result: dict | None = None

    if args.mode in {"all", "pdf"}:
        last_result = _run_pdf(retriever, client, args.pdf)

    if args.mode in {"all", "text"}:
        last_result = _run_text(retriever, client, SAMPLE_TEXT_RFP)

    if args.mode == "chat":
        last_result = _run_text(retriever, client, SAMPLE_TEXT_RFP)

    if args.mode == "chat" and last_result is not None:
        run_chat_loop(last_result, client)


if __name__ == "__main__":
    main()
