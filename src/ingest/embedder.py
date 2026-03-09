"""Embed chunks and upsert into Chroma (idempotent)."""

import os

import chromadb
from openai import OpenAI

from src.common.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
)
from src.ingest.chunker import Chunk


def _get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY") or _streamlit_key()
    return OpenAI(api_key=key)


def _streamlit_key() -> str:
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        raise RuntimeError("OPENAI_API_KEY not found in environment or Streamlit secrets")


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    oai = _get_client()
    response = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def upsert_chunks(chunks: list[Chunk], batch_size: int = 100) -> int:
    """Upsert chunks into Chroma. Returns number of chunks upserted."""
    collection = get_collection()
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = embed_texts(texts)
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "movie_title": c.movie_title,
                    "url": c.url,
                    "chunk_index": c.chunk_index,
                    "doc_id": c.doc_id,
                    "token_count": c.token_count,
                }
                for c in batch
            ],
        )
        total += len(batch)
        print(f"  upserted {total}/{len(chunks)} chunks")
    return total
