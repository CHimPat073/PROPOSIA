"""Interactive chatbot loop on top of a generated proposal.

After `generate_proposal_from_*` produces a proposal, the user can ask
follow-up questions like:
    - "make it shorter"
    - "emphasize security"
    - "what case studies did you cite?"
    - "add a 12-month timeline"

Each turn reuses:
    - the original proposal text (so the model has full continuity)
    - the original retrieved KB context (grounding stays stable)
    - the running chat history (so the model remembers earlier turns)

Type 'exit' / 'quit' to leave the loop.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.app.llm.client import GroqClient
from backend.app.llm.prompt import ChatMessage, build_chat_prompt

LOGGER = logging.getLogger(__name__)


def run_chat_loop(
    proposal_result: dict[str, Any],
    client: GroqClient,
    input_fn=input,
    output_fn=print,
) -> None:
    """Run an interactive REPL-style chat over a generated proposal.

    Args:
        proposal_result: Dict returned by generate_proposal_from_text/rfp.
        client: Configured GroqClient.
        input_fn/output_fn: Injectable for testing (default = console).
    """
    proposal_text = proposal_result.get("proposal", "")
    context_chunks = proposal_result.get("context", [])

    if not proposal_text:
        raise ValueError("proposal_result has no 'proposal' text.")

    history: list[ChatMessage] = []

    output_fn("=" * 60)
    output_fn("PROPOSAL CHATBOT — type 'exit' to quit")
    output_fn("=" * 60)
    output_fn(proposal_text)
    output_fn("=" * 60)

    while True:
        try:
            user_question = input_fn("\nYou: ").strip()
        except EOFError:
            output_fn("\n[input ended — exiting chat]")
            break

        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit", "q"}:
            output_fn("Goodbye.")
            break

        messages = build_chat_prompt(
            proposal_text=proposal_text,
            context_chunks=context_chunks,
            history=history,
            user_question=user_question,
        )

        try:
            answer = client.chat(messages)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Chat turn failed.")
            output_fn(f"\n[error] {exc}")
            continue

        history.append(ChatMessage(role="user", content=user_question))
        history.append(ChatMessage(role="assistant", content=answer))

        output_fn(f"\nAssistant:\n{answer}")
