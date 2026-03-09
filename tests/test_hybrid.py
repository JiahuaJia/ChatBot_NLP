"""Unit tests for RRF fusion logic in src/retrieval/hybrid.py."""

from src.retrieval.hybrid import _rrf_score


def test_rrf_score_decreases_with_rank():
    scores = [_rrf_score(r) for r in range(10)]
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1], "RRF score should decrease as rank increases"


def test_rrf_score_always_positive():
    for rank in range(100):
        assert _rrf_score(rank) > 0


def test_rrf_score_rank_0_highest():
    assert _rrf_score(0) > _rrf_score(1)
    assert _rrf_score(1) > _rrf_score(10)
