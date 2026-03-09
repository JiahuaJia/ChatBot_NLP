"""Semantic retrieval via Chroma."""

import os

import chromadb
from openai import OpenAI

from src.common.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    RETRIEVAL_CANDIDATES,
)


def _get_openai() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY") or _streamlit_key()
    return OpenAI(api_key=key)


def _streamlit_key() -> str:
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        raise RuntimeError("OPENAI_API_KEY not set")


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def embed_query(query: str) -> list[float]:
    oai = _get_openai()
    resp = oai.embeddings.create(model=EMBED_MODEL, input=[query])
    return resp.data[0].embedding


def semantic_search(query: str, n: int = RETRIEVAL_CANDIDATES) -> list[dict]:
    """Return top-n results from Chroma with id, text, metadata, distance."""
    collection = _get_collection()
    embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist, cid in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        hits.append({"id": cid, "text": doc, "metadata": meta, "score": 1 - dist})
    return hits
