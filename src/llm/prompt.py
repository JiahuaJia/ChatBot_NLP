"""Prompt templates for the movie RAG chatbot."""

SYSTEM_PROMPT = """\
You are a knowledgeable movie expert assistant. Answer questions exclusively based on \
the retrieved context passages below.

Rules:
1. Every factual claim (dates, names, box office figures, awards) MUST include a citation \
in the format [Source: Movie Title].
2. If multiple movies are relevant, cite each one separately.
3. If the context does not contain enough information to answer, say: \
"Based on available sources, I cannot confirm this."
4. Never fabricate information not present in the context.
5. Keep answers concise and well-structured.
"""


def build_messages(
    query: str,
    context_chunks: list[dict],
    history: list[dict],
) -> list[dict]:
    """
    Build the OpenAI messages list.

    Args:
        query: Current user question.
        context_chunks: Retrieved chunks from hybrid_search().
        history: List of {role, content} dicts (already trimmed to window).

    Returns:
        List of messages for chat completions.
    """
    context_text = _format_context(context_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}",
        }
    )
    return messages


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk["metadata"].get("movie_title", "Unknown")
        text = chunk["text"]
        parts.append(f"[{i}] Movie: {title}\n{text}")
    return "\n\n---\n\n".join(parts)
