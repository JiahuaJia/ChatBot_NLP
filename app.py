"""Movie RAG Chatbot — Streamlit UI."""

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.common.config import CHAT_MODEL, CHROMA_DIR, COLLECTION_NAME, TOP_K
from src.ingest.on_demand import fetch_and_index
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


def _extract_movie_title(query: str) -> str | None:
    """
    Ask GPT to extract a specific movie title from the query.
    Returns the title string, or None if no specific movie is mentioned.
    """
    import os
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    oai = OpenAI(api_key=key)
    resp = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the specific movie title from this query. "
                    "Reply with ONLY the movie title, nothing else. "
                    "If no specific movie title is mentioned, reply with 'none'.\n\n"
                    f"Query: {query}"
                ),
            }
        ],
        temperature=0,
        max_tokens=20,
    )
    result = resp.choices[0].message.content.strip()
    return None if result.lower() == "none" else result


def _is_low_confidence(chunks: list[dict]) -> bool:
    """Return True if retrieval results are sparse or weak."""
    if len(chunks) < 3:
        return True
    if chunks[0].get("rrf_score", 1.0) < 0.02:
        return True
    return False


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Movie RAG")
    st.caption("Powered by Wikipedia + Chroma + OpenAI")

    chunk_count = _check_index()
    st.metric("Indexed chunks", chunk_count)
    st.metric("Base movies", "60")
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
        "- What movies do you recommend?\n"
        "- Tell me about *Dune* (2021)."
    )

    if chunk_count == 0:
        st.error(
            "Index is empty. Run `make build-index` locally then commit `chroma_db/`."
        )


# ── Main chat area ─────────────────────────────────────────────────────────
st.title("🎬 Movie Knowledge Chatbot")
st.caption(
    "Ask anything about 60+ classic and modern films — or any other movie. "
    "Unknown films are fetched automatically. All answers include source citations."
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
        # ── Step 1: Initial retrieval ──────────────────────────────────────
        with st.spinner("Searching knowledge base..."):
            chunks = hybrid_search(query, top_k=TOP_K)

        # ── Step 2: On-demand fetch if confidence is low ───────────────────
        if _is_low_confidence(chunks):
            with st.spinner("Identifying movie..."):
                movie_title = _extract_movie_title(query)

            if movie_title:
                try:
                    with st.spinner(f"Fetching *{movie_title}* from Wikipedia..."):
                        resolved, n_chunks = fetch_and_index(movie_title)
                    st.info(
                        f"📚 Added **{resolved}** ({n_chunks} chunks) to knowledge base."
                    )
                    # Re-run retrieval with the newly indexed movie
                    with st.spinner("Re-searching..."):
                        chunks = hybrid_search(query, top_k=TOP_K)
                except Exception:
                    pass  # Graceful fallback: use original (possibly empty) chunks

        # ── Step 3: Build prompt & stream answer ───────────────────────────
        history = get_history(st.session_state)
        messages = build_messages(query, chunks, history)

        response_placeholder = st.empty()
        full_text = ""
        for token in stream_response(messages):
            full_text += token
            response_placeholder.markdown(full_text + "▌")
        response_placeholder.markdown(full_text)

        # ── Step 4: Show citations ─────────────────────────────────────────
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
