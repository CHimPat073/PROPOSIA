"""RFP text cleaner.

Single responsibility: list[Document] -> cleaned list[Document].

We do NOT use an LLM here. The goal is deterministic, fast, predictable
normalization so downstream chunking is stable.

Steps performed per page:
1. Drop pages whose text is empty after stripping.
2. Normalize line endings -> '\\n'.
3. Collapse runs of blank lines to a single blank line.
4. Collapse runs of horizontal whitespace inside a line.
5. Strip leading/trailing whitespace per line and per page.
6. Tag every document with document_type='rfp' (without losing original metadata).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.documents import Document

LOGGER = logging.getLogger(__name__)

_RFP_DOC_TYPE = "rfp"

_MULTI_NEWLINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r"[^\S\n]+")
_TRAILING_WS_PER_LINE = re.compile(r"[ \t]+$", flags=re.MULTILINE)
_LEADING_WS_PER_LINE = re.compile(r"^[ \t]+", flags=re.MULTILINE)


def _clean_text(raw: str) -> str:
    """Apply deterministic text normalization to one page's content."""
    if not raw:
        return ""

    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_NEWLINES.sub("\n\n", text)
    text = _MULTI_SPACES.sub(" ", text)
    text = _LEADING_WS_PER_LINE.sub("", text)
    text = _TRAILING_WS_PER_LINE.sub("", text)
    return text.strip()


def parse_rfp(documents: list[Document]) -> list[Document]:
    """Clean an RFP's page-level Documents and tag them as RFP documents.

    Args:
        documents: Output of `load_rfp_pdf`.

    Returns:
        Cleaned list[Document]; empty pages are dropped. Each kept document
        carries its original metadata plus `document_type='rfp'`.
    """
    if not documents:
        raise ValueError("No documents to parse. Did the loader return an empty list?")

    cleaned: list[Document] = []
    for index, doc in enumerate(documents):
        original_text = doc.page_content or ""
        normalized = _clean_text(original_text)

        if not normalized:
            LOGGER.warning("Dropping empty RFP page at index %d.", index)
            continue

        merged_metadata: dict[str, Any] = dict(doc.metadata or {})
        merged_metadata.setdefault("source", "unknown_rfp")
        merged_metadata["document_type"] = _RFP_DOC_TYPE

        cleaned.append(
            Document(page_content=normalized, metadata=merged_metadata)
        )

    if not cleaned:
        raise ValueError(
            "All RFP pages were empty after cleaning. Nothing to process."
        )

    LOGGER.info("Parsed %d RFP page(s) after cleaning.", len(cleaned))
    return cleaned
