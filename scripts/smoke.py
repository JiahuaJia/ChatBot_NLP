#!/usr/bin/env python3
"""Smoke test: run 3 queries and verify citations + top-K results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import TOP_K
from src.llm.client import complete
from src.llm.prompt import build_messages
from src.retrieval.hybrid import hybrid_search

TEST_QUERIES = [
    "Who directed Inception?",
    "What is the plot of Parasite?",
    "Compare The Godfather and Goodfellas.",
]

FAIL = False


def check(condition: bool, msg: str) -> None:
    global FAIL
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        FAIL = True


print("=" * 60)
print("Smoke Test")
print("=" * 60)

for query in TEST_QUERIES:
    print(f"\nQuery: {query!r}")
    chunks = hybrid_search(query, top_k=TOP_K)

    check(len(chunks) == TOP_K, f"top-K={TOP_K} chunks returned (got {len(chunks)})")
    check(
        all("movie_title" in c["metadata"] for c in chunks),
        "all chunks have movie_title metadata",
    )

    messages = build_messages(query, chunks, history=[])
    answer = complete(messages)

    check(bool(answer), "non-empty answer")
    check("[Source:" in answer, "answer contains [Source: ...] citation")
    print(f"  Answer preview: {answer[:120].replace(chr(10), ' ')}...")

print("\n" + "=" * 60)
if FAIL:
    print("SMOKE TEST FAILED")
    sys.exit(1)
else:
    print("SMOKE TEST PASSED")
