"""OpenAI streaming client."""

import os
from collections.abc import Generator

from openai import OpenAI

from src.common.config import CHAT_MODEL


def _get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY") or _streamlit_key()
    return OpenAI(api_key=key)


def _streamlit_key() -> str:
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        raise RuntimeError("OPENAI_API_KEY not set")


def stream_response(messages: list[dict]) -> Generator[str, None, None]:
    """Stream tokens from OpenAI chat completions."""
    oai = _get_client()
    stream = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        temperature=0.2,
        max_tokens=1024,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def complete(messages: list[dict]) -> str:
    """Non-streaming completion (used in smoke tests)."""
    oai = _get_client()
    resp = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content
