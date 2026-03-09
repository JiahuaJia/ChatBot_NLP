"""Paragraph-based chunking with token-aware merging."""

import hashlib
from dataclasses import dataclass

import tiktoken

from src.common.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    movie_title: str
    url: str
    chunk_index: int
    text: str
    token_count: int


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{doc_id}_{chunk_index}".encode()).hexdigest()[:16]


def chunk_document(doc: dict) -> list[Chunk]:
    """Split a movie document into paragraph-merged chunks."""
    content: str = doc["content"]
    doc_id: str = doc["doc_id"]
    title: str = doc["title"]
    url: str = doc["url"]

    raw_paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    buffer = ""
    buffer_tokens = 0
    chunk_index = 0

    for para in raw_paragraphs:
        para_tokens = _count_tokens(para)

        # If a single paragraph exceeds chunk size, split it by sentences
        if para_tokens > CHUNK_SIZE_TOKENS:
            sentences = para.replace(". ", ".\n").split("\n")
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_tokens = _count_tokens(sent)
                if buffer_tokens + sent_tokens > CHUNK_SIZE_TOKENS and buffer:
                    chunks.append(_make_chunk(buffer, chunk_index, doc_id, title, url))
                    # Overlap: keep last CHUNK_OVERLAP_TOKENS worth of buffer
                    buffer = _trim_to_tokens(buffer, CHUNK_OVERLAP_TOKENS)
                    buffer_tokens = _count_tokens(buffer)
                    chunk_index += 1
                buffer = (buffer + " " + sent).strip()
                buffer_tokens += sent_tokens
        else:
            if buffer_tokens + para_tokens > CHUNK_SIZE_TOKENS and buffer:
                chunks.append(_make_chunk(buffer, chunk_index, doc_id, title, url))
                buffer = _trim_to_tokens(buffer, CHUNK_OVERLAP_TOKENS)
                buffer_tokens = _count_tokens(buffer)
                chunk_index += 1
            buffer = (buffer + "\n\n" + para).strip()
            buffer_tokens += para_tokens

    if buffer.strip():
        chunks.append(_make_chunk(buffer, chunk_index, doc_id, title, url))

    # Attach chunk_ids
    for i, c in enumerate(chunks):
        c.chunk_id = _make_chunk_id(doc_id, i)
        c.chunk_index = i

    return chunks


def _make_chunk(text: str, index: int, doc_id: str, title: str, url: str) -> Chunk:
    return Chunk(
        chunk_id=_make_chunk_id(doc_id, index),
        doc_id=doc_id,
        movie_title=title,
        url=url,
        chunk_index=index,
        text=text,
        token_count=_count_tokens(text),
    )


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[-max_tokens:])
