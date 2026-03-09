"""Hybrid retrieval: BM25 + semantic search fused via Reciprocal Rank Fusion."""

from src.common.config import RETRIEVAL_CANDIDATES, RRF_K, TOP_K
from src.retrieval.bm25 import bm25_search
from src.retrieval.semantic import semantic_search


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)


def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve top_k chunks using Reciprocal Rank Fusion of BM25 and semantic results.
    Returns list of dicts: {id, text, metadata, rrf_score}.
    """
    bm25_hits = bm25_search(query, n=RETRIEVAL_CANDIDATES)
    semantic_hits = semantic_search(query, n=RETRIEVAL_CANDIDATES)

    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(bm25_hits):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)
        docs[cid] = hit

    for rank, hit in enumerate(semantic_hits):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)
        if cid not in docs:
            docs[cid] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**docs[cid], "rrf_score": score}
        for cid, score in ranked
    ]
