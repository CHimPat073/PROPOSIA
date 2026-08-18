"""RFP PDF loader.

Single responsibility: a PDF file path -> list[langchain_core.Document].

We deliberately do NOT:
- chunk
- embed
- touch ChromaDB

If you need those, look at `processor.py`.

The loader is strict: missing files, wrong extensions, empty PDFs
and unreadable pages all raise clear errors.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import PyPDFLoader  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

LOGGER = logging.getLogger(__name__)

PDF_EXTENSION = ".pdf"


def _validate_pdf_path(file_path: Path) -> Path:
    """Resolve and validate the input path. Raises ValueError with a clear message."""
    resolved = file_path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"PDF not found: {resolved}")

    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")

    if resolved.suffix.lower() != PDF_EXTENSION:
        raise ValueError(
            f"Expected a {PDF_EXTENSION} file, got '{resolved.suffix}'."
        )

    return resolved


def load_rfp_pdf(file_path: str | Path) -> list[Document]:
    """Load an RFP PDF into LangChain Documents (one per page).

    Args:
        file_path: Path to the RFP PDF.

    Returns:
        list[Document] with metadata carrying at least `source` and `page`.
    """
    resolved = _validate_pdf_path(Path(file_path))

    LOGGER.info("Loading RFP PDF: %s", resolved)

    try:
        loader = PyPDFLoader(str(resolved))
        documents = loader.load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read PDF '{resolved}'. The file may be corrupt or encrypted."
        ) from exc

    if not documents:
        raise ValueError(
            f"PDF '{resolved}' contained no pages (empty or unreadable)."
        )

    LOGGER.info("Loaded %d page(s) from %s", len(documents), resolved.name)
    return documents
