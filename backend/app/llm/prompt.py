"""Prompt templates for proposal generation and follow-up chat.

Why a dedicated module:
- Prompts are content. The generator is plumbing. Keeping them apart makes
  both readable and lets us iterate on wording without touching the
  generator's logic.

Two prompts are defined:

1. `build_proposal_prompt`  -> system + user messages for first proposal.
2. `build_chat_prompt`      -> system + user messages for follow-up Q&A.

The system prompt explicitly tells the model to ONLY use the provided
company knowledge and to admit when something is missing. This is the
"grounding" contract that makes proposals trustworthy.
"""

from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


SYSTEM_PROPOSAL = (
    "You are an AI sales proposal writer for a technology company.\n"
    "You will be given:\n"
    "  1. The client's Request for Proposal (RFP) text.\n"
    "  2. Relevant excerpts from the company's knowledge base.\n\n"
    "Rules you MUST follow:\n"
    " - Use ONLY information present in the knowledge base excerpts.\n"
    " - If a requirement cannot be satisfied by what is in the knowledge "
    "base, say so explicitly under a 'Gaps & Assumptions' heading.\n"
    " - Cite the source filename in brackets after each claim, e.g. "
    "[services/cloud.md].\n"
    " - Structure the proposal with clear headings: Executive Summary, "
    "Proposed Solution, Why Us (relevant case studies), Timeline & "
    "Pricing (if available), Gaps & Assumptions.\n"
    " - Be concise. Use bullet points. Avoid marketing fluff."
)


SYSTEM_CHAT = (
    "You are an AI sales proposal assistant.\n"
    "You have already produced a proposal grounded in the company's "
    "knowledge base.\n"
    "The user is now asking follow-up questions about that proposal.\n"
    "Rules:\n"
    " - Answer using ONLY the provided proposal text and knowledge base "
    "excerpts.\n"
    " - If the answer is not in those sources, say 'I don't have that "
    "information in the company knowledge base.'\n"
    " - Keep answers short and specific."
)


def build_proposal_prompt(
    rfp_text: str,
    context_chunks: list[dict],
) -> list[ChatMessage]:
    """Build the messages list for the initial proposal generation call."""
    context_block = _format_context(context_chunks)

    user_content = (
        f"=== CLIENT RFP ===\n{rfp_text}\n\n"
        f"=== COMPANY KNOWLEDGE BASE EXCERPTS ===\n{context_block}\n\n"
        "Write the proposal now."
    )

    return [
        ChatMessage(role="system", content=SYSTEM_PROPOSAL),
        ChatMessage(role="user", content=user_content),
    ]


def build_chat_prompt(
    proposal_text: str,
    context_chunks: list[dict],
    history: list[ChatMessage],
    user_question: str,
) -> list[ChatMessage]:
    """Build the messages list for a follow-up Q&A turn.

    `history` already contains prior user/assistant turns (excluding the
    current question, which is appended last).
    """
    context_block = _format_context(context_chunks)

    user_content = (
        f"=== CURRENT PROPOSAL ===\n{proposal_text}\n\n"
        f"=== COMPANY KNOWLEDGE BASE EXCERPTS ===\n{context_block}\n\n"
        f"=== USER QUESTION ===\n{user_question}"
    )

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=SYSTEM_CHAT),
        *history,
        ChatMessage(role="user", content=user_content),
    ]
    return messages


def _format_context(context_chunks: list[dict]) -> str:
    """Render the deduped KB chunks into a single readable block.

    Each chunk becomes:
        [source: <source>]
        <text>

    Empty/whitespace chunks are skipped silently.
    """
    if not context_chunks:
        return "(no relevant company knowledge found)"

    blocks: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        source = chunk.get("metadata", {}).get("source", "unknown")
        distance = chunk.get("distance")
        distance_note = f" (distance={distance:.4f})" if distance is not None else ""
        blocks.append(f"[{i}] source: {source}{distance_note}\n{text}")
    return "\n\n".join(blocks) if blocks else "(no relevant company knowledge found)"
