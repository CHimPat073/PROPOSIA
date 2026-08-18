"""Proposal generator.

End-to-end flow:
    RFP (text or PDF) -> embedded chunks -> KB matches -> dedupe -> prompt -> Groq -> proposal

Two public functions:
    generate_proposal_from_rfp(processed_rfp, retriever, client)
        Use when you already ran `process_rfp` (e.g. PDF RFP).

    generate_proposal_from_text(rfp_text, retriever, client)
        Use for a plain-text RFP (chatbot-style input).

Both return a dict:
    {
        "rfp_text":   "<the RFP the model saw>",
        "proposal":   "<the generated proposal>",
        "context":    [deduped KB chunks used as grounding],
    }
"""

from __future__ import annotations

import logging
from typing import Any

from backend.app.ingestion.embedder import EmbeddedChunk
from backend.app.llm.client import GroqClient
from backend.app.llm.prompt import build_proposal_prompt
from backend.app.retrieval.retriever import Retriever

LOGGER = logging.getLogger(__name__)

# Conservative guard so we don't blow past context windows.
# Chars per token ~ 3.5 for English, so 6000 chars ≈ ~1700 tokens of context,
# leaving headroom for the system prompt, RFP text, and the model's response.
MAX_CONTEXT_CHARS = 6_000


def _rfp_text_from_processed(processed_rfp: dict[str, Any]) -> str:
    chunks: list[EmbeddedChunk] = processed_rfp.get("chunks", [])
    if not chunks:
        raise ValueError("processed_rfp has no chunks.")
    return "\n\n".join(chunk["text"] for chunk in chunks)


def _collect_context(
    retriever: Retriever,
    source: dict[str, Any] | str,
    k: int,
) -> list[dict[str, Any]]:
    """Run retrieval, flatten, dedupe by chunk id, sort by distance, truncate."""
    if isinstance(source, dict):
        per_chunk = retriever.retrieve_from_rfp(source, k=k)
    else:
        per_chunk = [retriever.retrieve(source, k=k)]

    flat: dict[str, dict[str, Any]] = {}
    for chunk_results in per_chunk:
        for item in chunk_results:
            cid = item.get("id")
            if not cid:
                continue
            if cid not in flat or item.get("distance", 1.0) < flat[cid].get("distance", 1.0):
                flat[cid] = item

    ranked = sorted(flat.values(), key=lambda x: x.get("distance", 1.0))

    truncated: list[dict[str, Any]] = []
    total_chars = 0
    for item in ranked:
        text_len = len(item.get("text") or "")
        if total_chars + text_len > MAX_CONTEXT_CHARS:
            break
        truncated.append(item)
        total_chars += text_len

    LOGGER.info(
        "Context built | total_candidates=%d | kept=%d | total_chars=%d",
        len(flat),
        len(truncated),
        total_chars,
    )
    return truncated


def generate_proposal_from_text(
    rfp_text: str,
    retriever: Retriever,
    client: GroqClient,
    k: int = 5,
) -> dict[str, Any]:
    """Generate a proposal directly from a plain-text RFP."""
    if not rfp_text or not rfp_text.strip():
        raise ValueError("rfp_text must be a non-empty string.")

    context = _collect_context(retriever, rfp_text, k=k)
    messages = build_proposal_prompt(rfp_text=rfp_text, context_chunks=context)
    proposal = client.chat(messages)

    return {
        "rfp_text": rfp_text,
        "proposal": proposal,
        "context": context,
    }


def generate_proposal_from_rfp(
    processed_rfp: dict[str, Any],
    retriever: Retriever,
    client: GroqClient,
    k: int = 5,
) -> dict[str, Any]:
    """Generate a proposal from a PDF RFP already run through process_rfp()."""
    rfp_text = _rfp_text_from_processed(processed_rfp)
    return generate_proposal_from_text(
        rfp_text=rfp_text,
        retriever=retriever,
        client=client,
        k=k,
    )
