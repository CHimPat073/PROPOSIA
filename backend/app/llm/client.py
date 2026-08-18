"""Groq LLM client wrapper.

Single responsibility: load .env, build a ChatGroq instance, and expose
one `chat(messages) -> str` method.

Configuration:
- Only GROQ_API_KEY is required (from .env or environment).
- The model is auto-discovered at startup by querying Groq's /models
  endpoint, so the code never breaks when a specific model is deprecated
  or unavailable on your account tier.

Why this exists as its own module:
- The generator should not know which provider we're calling.
- Swapping Groq for OpenAI/HuggingFace/Ollama later is a one-file change.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# Preferred models in order — first one your account has access to wins.
# Add more here if Groq releases new ones you want to try first.
MODEL_PREFERENCE_ORDER: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# Substrings that mark a model as NOT chat-completion. Any model whose
# id contains one of these (case-insensitive) is filtered out.
_NON_CHAT_MODEL_HINTS: tuple[str, ...] = (
    "whisper",
    "tts",
    "vision",
    "guard",
    "embed",
    "moderation",
    "orpheus",
    "playai",
    "distil-whisper",
)

# Chat-completion models we will consider as a last-resort fallback if
# none of MODEL_PREFERENCE_ORDER is available. Anything starting with one
# of these prefixes is treated as a valid chat model.
_CHAT_MODEL_PREFIXES: tuple[str, ...] = (
    "llama",
    "llama3",
    "mixtral",
    "gemma",
)

DEFAULT_TEMPERATURE = 0.2


def _list_available_models(api_key: str) -> list[str]:
    """Return chat-capable model IDs accessible to this API key."""
    try:
        response = requests.get(
            GROQ_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to query Groq /models endpoint: {exc}") from exc

    data = response.json()
    raw = [item.get("id", "") for item in data.get("data", []) if item.get("id")]

    # Filter out non-chat models (whisper, tts, vision, embeddings, etc.)
    chat_models = [
        mid for mid in raw
        if not any(hint in mid.lower() for hint in _NON_CHAT_MODEL_HINTS)
    ]
    return chat_models


def _pick_model(api_key: str) -> str:
    """Pick the first chat model in MODEL_PREFERENCE_ORDER that's available."""
    available = _list_available_models(api_key)
    LOGGER.info("Groq chat-capable models available: %s", available)

    # 1) Try the explicit preference list first.
    for candidate in MODEL_PREFERENCE_ORDER:
        if candidate in available:
            return candidate

    # 2) Fall back to any model whose id starts with a known chat prefix.
    for mid in available:
        if mid.lower().startswith(_CHAT_MODEL_PREFIXES):
            return mid

    # 3) Last resort: first available chat model (shouldn't happen given filter).
    if available:
        return available[0]

    raise RuntimeError(
        "No chat-capable Groq models are accessible with this API key. "
        "Check your account at https://console.groq.com/"
    )


class GroqClient:
    """Thin wrapper around ChatGroq. Reads only GROQ_API_KEY."""

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE) -> None:
        # Load .env from project root if present; safe no-op otherwise.
        load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to a .env file at the project root "
                "or export it in your shell."
            )

        model_name = _pick_model(api_key)

        self.client = ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
        )
        LOGGER.info("Groq client ready | model='%s'", model_name)

    def chat(self, messages: list[dict]) -> str:
        """Send a list of {role, content} messages; return the assistant text.

        If the request is rejected as too large (413), the system + context
        portion of the message list is trimmed and the call retried once.
        """
        if not messages:
            raise ValueError("messages must be a non-empty list")

        try:
            response = self.client.invoke(messages)
        except Exception as exc:
            if _is_request_too_large(exc) and len(messages) > 1:
                LOGGER.warning(
                    "Request too large; retrying with trimmed context."
                )
                trimmed = _shrink_last_user_message(messages, keep_fraction=0.5)
                response = self.client.invoke(trimmed)
            else:
                raise

        text = getattr(response, "content", "") or ""
        if not text.strip():
            raise RuntimeError("Groq returned an empty response.")
        return text


def _is_request_too_large(exc: Exception) -> bool:
    """Best-effort detection of 413 / 'too large' errors."""
    msg = str(exc).lower()
    return "413" in msg or "too large" in msg or "request_entity_too_large" in msg


def _shrink_last_user_message(
    messages: list[dict], keep_fraction: float = 0.5
) -> list[dict]:
    """Return a copy of `messages` where the last user message is shortened."""
    if not messages or messages[-1].get("role") != "user":
        return messages

    content = messages[-1].get("content", "")
    if not isinstance(content, str):
        return messages

    cutoff = max(1, int(len(content) * keep_fraction))
    shrunk = messages[:-1] + [
        {**messages[-1], "content": content[:cutoff] + "\n\n[context trimmed for size]"}
    ]
    return shrunk
