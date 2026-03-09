"""Unit tests for src/ingest/chunker.py."""

import hashlib

from src.ingest.chunker import _make_chunk_id, chunk_document

SAMPLE_DOC = {
    "title": "Test Movie",
    "url": "https://en.wikipedia.org/wiki/Test_Movie",
    "content": "\n\n".join([f"Paragraph {i}. " + "Word " * 80 for i in range(10)]),
    "doc_id": "abc123",
}


def test_chunk_returns_list():
    chunks = chunk_document(SAMPLE_DOC)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_ids_are_unique():
    chunks = chunk_document(SAMPLE_DOC)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunk_ids_are_deterministic():
    chunks1 = chunk_document(SAMPLE_DOC)
    chunks2 = chunk_document(SAMPLE_DOC)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_chunk_metadata():
    chunks = chunk_document(SAMPLE_DOC)
    for c in chunks:
        assert c.movie_title == "Test Movie"
        assert c.url == SAMPLE_DOC["url"]
        assert c.doc_id == "abc123"
        assert isinstance(c.chunk_index, int)
        assert c.token_count > 0


def test_chunk_id_format():
    cid = _make_chunk_id("doc123", 0)
    expected = hashlib.sha256("doc123_0".encode()).hexdigest()[:16]
    assert cid == expected


def test_empty_doc():
    doc = {**SAMPLE_DOC, "content": ""}
    chunks = chunk_document(doc)
    assert chunks == []


def test_short_doc_single_chunk():
    doc = {**SAMPLE_DOC, "content": "A short movie description."}
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "A short movie description."
