"""Movie RAG Chatbot — Streamlit UI."""

import chromadb
import streamlit as st
from dotenv import load_dotenv

from src.common.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from src.llm.client import stream_response
from src.llm.prompt import build_messages
from src.memory.window import add_turn, clear_history, get_history
from src.retrieval.hybrid import hybrid_search

load_dotenv()  # Load .env for local development; Streamlit Cloud uses st.secrets

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie RAG Chatbot",
    page_icon="🎬",
    layout="wide",
)


# ── Helpers ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _check_index() -> int:
    """Return number of chunks in Chroma collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(COLLECTION_NAME)
    return col.count()


def _answer(query: str) -> tuple[str, list[dict]]:
    """Run hybrid search + streaming LLM. Returns (full_answer, chunks)."""
    chunks = hybrid_search(query, top_k=TOP_K)
    history = get_history(st.session_state)
    messages = build_messages(query, chunks, history)

    response_placeholder = st.empty()
    full_text = ""
    for token in stream_response(messages):
        full_text += token
        response_placeholder.markdown(full_text + "▌")
    response_placeholder.markdown(full_text)
    return full_text, chunks


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Movie RAG")
    st.caption("Powered by Wikipedia + Chroma + OpenAI")

    chunk_count = _check_index()
    st.metric("Indexed chunks", chunk_count)
    st.metric("Movies", "60+")
    st.metric("Top-K retrieved", TOP_K)

    if st.button("Clear conversation", use_container_width=True):
        clear_history(st.session_state)
        st.rerun()

    st.divider()
    st.markdown(
        "**Example questions:**\n"
        "- Who directed *Inception*?\n"
        "- What is the plot of *Parasite*?\n"
        "- Compare *The Godfather* and *Goodfellas*.\n"
        "- Which films won the Academy Award for Best Picture?\n"
        "- Tell me more about that movie."
    )

    if chunk_count == 0:
        st.error(
            "Index is empty. Run `make build-index` locally then commit `chroma_db/`."
        )


# ── Main chat area ─────────────────────────────────────────────────────────
st.title("🎬 Movie Knowledge Chatbot")
st.caption(
    "Ask anything about 60+ classic and modern films. "
    "All answers include source citations."
)

# Render conversation history
for msg in get_history(st.session_state):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask about any movie..."):
    if chunk_count == 0:
        st.error("Index is empty. Please build the index first.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and stream answer
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            chunks = hybrid_search(query, top_k=TOP_K)
        history = get_history(st.session_state)
        messages = build_messages(query, chunks, history)

        response_placeholder = st.empty()
        full_text = ""
        for token in stream_response(messages):
            full_text += token
            response_placeholder.markdown(full_text + "▌")
        response_placeholder.markdown(full_text)

        # Show citations
        if chunks:
            with st.expander(f"Sources ({len(chunks)} chunks retrieved)", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    title = chunk["metadata"].get("movie_title", "Unknown")
                    url = chunk["metadata"].get("url", "")
                    snippet = chunk["text"][:300].replace("\n", " ")
                    st.markdown(
                        f"**[{i}] {title}**  \n"
                        f"_{snippet}..._  \n"
                        f"[Wikipedia]({url})"
                    )
                    if i < len(chunks):
                        st.divider()

    # Persist turn to memory
    add_turn(st.session_state, query, full_text)
