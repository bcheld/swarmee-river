from __future__ import annotations

from swarmee_river.context.prompt_cache import PromptCacheState


def test_peek_reminder_does_not_consume_pending_chunks() -> None:
    state = PromptCacheState()
    state.queue_one_off("Context A")
    state.queue_one_off("Context B")

    first = state.peek_reminder()
    second = state.peek_reminder()
    popped = state.pop_reminder()

    assert first == second
    assert popped == first
    assert state.peek_reminder() == ""
