"""BM25 keyword retrieval built on top of the Chroma corpus."""

import re
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from src.common.config import CHROMA_DIR, COLLECTION_NAME, RETRIEVAL_CANDIDATES


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[list[str], list[str], list[dict]]:
    """Load all documents from Chroma once and cache."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    total = collection.count()
    if total == 0:
        return [], [], []
    result = collection.get(include=["documents", "metadatas"])
    ids = result["ids"]
    docs = result["documents"]
    metas = result["metadatas"]
    return ids, docs, metas


def clear_cache() -> None:
    """Clear the corpus cache so newly added movies are included in BM25."""
    _load_corpus.cache_clear()


def bm25_search(query: str, n: int = RETRIEVAL_CANDIDATES) -> list[dict]:
    """Return top-n BM25 results with id, text, metadata, score."""
    ids, docs, metas = _load_corpus()
    if not docs:
        return []

    tokenized_corpus = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [
        {
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "score": float(scores[i]),
        }
        for i in top_indices
        if scores[i] > 0
    ]
