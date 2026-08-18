"""AI Sales Proposal Generator — Streamlit UI.

Run from project root:
    streamlit run app.py

Features:
- Tab 1 (RFP Input): upload a PDF or paste text, generate a proposal.
- Tab 2 (Chatbot): ask follow-up questions about the generated proposal.

State management:
- All long-lived objects (Retriever, GroqClient) are cached via
  st.cache_resource so the embedding model loads only once.
- The proposal result and chat history live in st.session_state so they
  survive Streamlit's script reruns.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Force UTF-8 so LLM output (em-dashes, smart quotes) renders correctly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from backend.app.llm.chat import run_chat_loop  # noqa: E402
from backend.app.llm.client import GroqClient  # noqa: E402
from backend.app.llm.generator import (  # noqa: E402
    generate_proposal_from_rfp,
    generate_proposal_from_text,
)
from backend.app.retrieval.retriever import Retriever  # noqa: E402
from backend.app.rfp.processor import process_rfp  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

st.set_page_config(
    page_title="AI Sales Proposal Generator",
    page_icon=":memo:",
    layout="wide",
)


# ---------- Cached heavy resources (load once per session) ----------------- #

@st.cache_resource(show_spinner="Loading embedding model + ChromaDB...")
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource(show_spinner="Connecting to Groq...")
def get_groq_client() -> GroqClient:
    return GroqClient()


# ---------- Session-state defaults ----------------------------------------- #

DEFAULTS = {
    "proposal_result": None,   # dict returned by generate_proposal_from_*
    "chat_history": [],        # list[ChatMessage] for the chatbot loop
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- Sidebar (controls that affect the whole app) ------------------ #

with st.sidebar:
    st.header("Settings")
    k_value = st.slider(
        "KB chunks per RFP chunk (k)",
        min_value=2,
        max_value=10,
        value=5,
        help="Higher k = more context, but slower and may exceed model limits.",
    )
    if st.button("Reset session", type="secondary"):
        st.session_state["proposal_result"] = None
        st.session_state["chat_history"] = []
        st.rerun()


# ---------- Tabs ---------------------------------------------------------- #

tab_proposal, tab_chat = st.tabs(["RFP Input", "Chatbot"])


# ============ TAB 1: RFP INPUT ============================================ #

with tab_proposal:
    st.title("AI Sales Proposal Generator")
    st.caption("Upload an RFP PDF or paste text. The system retrieves company KB context and drafts a grounded proposal.")

    input_mode = st.radio(
        "Input mode",
        options=["Upload PDF", "Paste text"],
        horizontal=True,
    )

    rfp_text: str | None = None
    pdf_path: Path | None = None

    if input_mode == "Upload PDF":
        uploaded = st.file_uploader("RFP PDF", type=["pdf"])
        if uploaded is not None:
            # Streamlit gives us a BytesIO; persist to a temp file the
            # existing pipeline (PyPDFLoader) can read.
            tmp_dir = Path("data") / "rfps" / "_uploads"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = tmp_dir / uploaded.name
            pdf_path.write_bytes(uploaded.read())
    else:
        rfp_text = st.text_area(
            "Paste RFP text",
            height=240,
            placeholder="We need a cloud-based inventory management system...",
        )

    col_gen, col_clear = st.columns([1, 1])
    generate_clicked = col_gen.button("Generate proposal", type="primary")
    clear_clicked = col_clear.button("Clear")

    if clear_clicked:
        st.session_state["proposal_result"] = None
        st.session_state["chat_history"] = []
        st.rerun()

    if generate_clicked:
        if input_mode == "Upload PDF" and pdf_path is None:
            st.error("Please upload a PDF first.")
        elif input_mode == "Paste text" and not (rfp_text and rfp_text.strip()):
            st.error("Please paste some RFP text first.")
        else:
            try:
                with st.spinner("Processing RFP and contacting Groq..."):
                    retriever = get_retriever()
                    client = get_groq_client()

                    if input_mode == "Upload PDF":
                        processed = process_rfp(pdf_path)  # type: ignore[arg-type]
                        result = generate_proposal_from_rfp(
                            processed, retriever, client, k=k_value
                        )
                    else:
                        result = generate_proposal_from_text(
                            rfp_text, retriever, client, k=k_value  # type: ignore[arg-type]
                        )

                st.session_state["proposal_result"] = result
                st.session_state["chat_history"] = []
                st.success("Proposal generated.")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    # ----- Display result ------------------------------------------------ #
    result = st.session_state["proposal_result"]
    if result is not None:
        st.divider()
        left, right = st.columns([2, 1])

        with left:
            st.subheader("Generated proposal")
            st.markdown(result["proposal"])

        with right:
            st.subheader("Retrieved KB context")
            ctx = result.get("context", [])
            if not ctx:
                st.info("No KB chunks were used.")
            else:
                st.caption(f"{len(ctx)} unique chunks used (sorted by relevance)")
                for i, chunk in enumerate(ctx, start=1):
                    source = chunk.get("metadata", {}).get("source", "unknown")
                    distance = chunk.get("distance")
                    dist_str = f"{distance:.4f}" if distance is not None else "—"
                    with st.expander(f"[{i}] {source}  (d={dist_str})"):
                        st.write(chunk.get("text", ""))


def _run_one_chat_turn(
    proposal_result: dict,
    client: GroqClient,
    history: list[dict],
    user_question: str,
) -> str:
    """Send one follow-up message and return the assistant's reply.

    Mirrors the prompt-building logic in `llm.chat.run_chat_loop` but
    returns the answer instead of printing it, which is what Streamlit
    needs (it renders the response itself).
    """
    from backend.app.llm.prompt import build_chat_prompt

    messages = build_chat_prompt(
        proposal_text=proposal_result.get("proposal", ""),
        context_chunks=proposal_result.get("context", []),
        history=history,
        user_question=user_question,
    )
    return client.chat(messages)

# ============ TAB 2: CHATBOT ============================================== #

with tab_chat:
    st.title("Proposal chatbot")
    st.caption("Ask follow-up questions about the generated proposal. Answers are grounded in the same KB context.")

    result = st.session_state["proposal_result"]
    if result is None:
        st.info("Generate a proposal in the 'RFP Input' tab first.")
    else:
        st.subheader("Current proposal")
        with st.expander("Show full proposal", expanded=False):
            st.markdown(result["proposal"])

        st.divider()
        st.subheader("Conversation")

        # Render existing history ------------------------------------------------
        for msg in st.session_state["chat_history"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            with st.chat_message(role):
                st.markdown(content)

        # Chat input -------------------------------------------------------------
        user_question = st.chat_input("Ask a question about the proposal...")
        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = get_groq_client()
                        # Reuse the chatbot's loop logic for consistency.
                        # We pass input_fn=output_fn=no-ops so it doesn't
                        # try to print; we capture the last assistant message.
                        answer = _run_one_chat_turn(
                            proposal_result=result,
                            client=client,
                            history=st.session_state["chat_history"],
                            user_question=user_question,
                        )
                        st.markdown(answer)
                        st.session_state["chat_history"].append(
                            {"role": "user", "content": user_question}
                        )
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": answer}
                        )
                    except Exception as exc:
                        st.error(f"Chat failed: {exc}")


# ---------- One-shot chat turn (Streamlit-friendly) ----------------------- #

def _run_one_chat_turn(
    proposal_result: dict,
    client: GroqClient,
    history: list[dict],
    user_question: str,
) -> str:
    """Send one follow-up message and return the assistant's reply.

    Mirrors the prompt-building logic in `llm.chat.run_chat_loop` but
    returns the answer instead of printing it, which is what Streamlit
    needs (it renders the response itself).
    """
    from backend.app.llm.prompt import build_chat_prompt

    messages = build_chat_prompt(
        proposal_text=proposal_result.get("proposal", ""),
        context_chunks=proposal_result.get("context", []),
        history=history,
        user_question=user_question,
    )
    return client.chat(messages)