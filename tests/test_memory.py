"""Unit tests for src/memory/window.py."""

from src.common.config import MAX_HISTORY_TURNS
from src.memory.window import add_turn, clear_history, get_history


class _FakeState(dict):
    """Minimal dict-like object to simulate st.session_state."""
    pass


def test_get_history_initializes_empty():
    state = _FakeState()
    history = get_history(state)
    assert history == []


def test_add_turn_appends_two_messages():
    state = _FakeState()
    add_turn(state, "Hello", "Hi there!")
    history = get_history(state)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


def test_sliding_window_trims_oldest():
    state = _FakeState()
    for i in range(MAX_HISTORY_TURNS + 2):
        add_turn(state, f"Q{i}", f"A{i}")
    history = get_history(state)
    assert len(history) == MAX_HISTORY_TURNS * 2
    # Oldest messages should be gone
    assert history[0]["content"] == f"Q{2}"


def test_clear_history():
    state = _FakeState()
    add_turn(state, "Q", "A")
    clear_history(state)
    assert get_history(state) == []


def test_multiple_turns_order_preserved():
    state = _FakeState()
    add_turn(state, "First", "Response1")
    add_turn(state, "Second", "Response2")
    history = get_history(state)
    assert history[0]["content"] == "First"
    assert history[2]["content"] == "Second"
