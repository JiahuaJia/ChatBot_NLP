"""On-demand Wikipedia fetch → chunk → embed pipeline."""

import hashlib
import time

import wikipedia

from src.ingest.chunker import chunk_document
from src.ingest.embedder import upsert_chunks
from src.retrieval import bm25 as bm25_module


def fetch_and_index(title: str) -> tuple[str, int]:
    """
    Fetch a movie page from Wikipedia, chunk and embed it into Chroma.

    Args:
        title: Movie title to search for (passed directly to Wikipedia).

    Returns:
        (resolved_title, chunk_count) on success.

    Raises:
        ValueError: If Wikipedia cannot find a relevant page.
        wikipedia.exceptions.PageError: If the exact page is not found.
    """
    try:
        page = wikipedia.page(title, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        # Pick the first option that looks like a film
        film_option = next(
            (o for o in e.options if "film" in o.lower() or "movie" in o.lower()),
            e.options[0],
        )
        page = wikipedia.page(film_option, auto_suggest=False)

    content = page.content
    doc = {
        "title": page.title,
        "url": page.url,
        "content": content,
        "doc_id": hashlib.sha256(content.encode()).hexdigest()[:16],
    }

    chunks = chunk_document(doc)
    if not chunks:
        raise ValueError(f"No chunks produced for '{page.title}'")

    upsert_chunks(chunks)

    # Clear BM25 corpus cache so new movie appears in keyword search
    bm25_module.clear_cache()

    time.sleep(0.3)  # be polite to Wikipedia
    return page.title, len(chunks)
