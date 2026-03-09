"""Sliding-window conversation memory backed by Streamlit session_state."""

from src.common.config import MAX_HISTORY_TURNS


def get_history(session_state) -> list[dict]:
    """Return the current conversation history (list of {role, content})."""
    if "history" not in session_state:
        session_state["history"] = []
    return session_state["history"]


def add_turn(session_state, user_msg: str, assistant_msg: str) -> None:
    """Append a user+assistant turn and trim to MAX_HISTORY_TURNS."""
    history = get_history(session_state)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})

    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        session_state["history"] = history[-max_messages:]


def clear_history(session_state) -> None:
    session_state["history"] = []
