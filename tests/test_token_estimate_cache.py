"""Incremental token-estimation cache tests (doc 12 F2)."""

from __future__ import annotations

import pytest

import swarmee_river.context.budgeted_summarizing_conversation_manager as bscm
from swarmee_river.context.budgeted_summarizing_conversation_manager import (
    estimate_tokens,
    invalidate_message_chars_cache,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    invalidate_message_chars_cache()
    yield
    invalidate_message_chars_cache()


def _message(text: str) -> dict:
    return {"role": "user", "content": [{"text": text}]}


def _full_recompute(messages: list[dict]) -> int:
    return sum(len(bscm._extract_message_text(m)) for m in messages)


def test_cached_estimate_matches_full_recompute() -> None:
    messages = [_message(f"message number {i} " * 10) for i in range(50)]
    first = estimate_tokens(system_prompt="sys", messages=messages, chars_per_token=4)
    second = estimate_tokens(system_prompt="sys", messages=messages, chars_per_token=4)
    assert first == second
    expected_chars = len("sys") + _full_recompute(messages)
    assert first == -(-expected_chars // 4)  # ceil


def test_appends_are_incremental_not_full_walks(monkeypatch) -> None:
    messages = [_message(f"m{i}") for i in range(500)]
    estimate_tokens(system_prompt=None, messages=messages, chars_per_token=4)

    calls = []
    real_extract = bscm._extract_message_text
    monkeypatch.setattr(bscm, "_extract_message_text", lambda m: calls.append(1) or real_extract(m))

    messages.append(_message("new tail"))
    estimate_tokens(system_prompt=None, messages=messages, chars_per_token=4)
    # Only the new message and the (potentially streaming) last message are
    # re-extracted — not the 500-message prefix.
    assert len(calls) <= 3, f"expected incremental walk, extracted {len(calls)} messages"


def test_streaming_growth_of_last_message_is_seen() -> None:
    messages = [_message("a"), _message("b")]
    before = estimate_tokens(system_prompt=None, messages=messages, chars_per_token=1)
    messages[-1]["content"].append({"text": "x" * 100})
    after = estimate_tokens(system_prompt=None, messages=messages, chars_per_token=1)
    assert after > before + 90


def test_invalidation_after_in_place_compaction() -> None:
    messages = [_message("x" * 1000) for _ in range(5)]
    messages.append(_message("tail"))
    before = estimate_tokens(system_prompt=None, messages=messages, chars_per_token=1)

    # Simulate tool-result style in-place shrink of a prefix message.
    messages[0]["content"] = [{"text": "compacted"}]
    stale = estimate_tokens(system_prompt=None, messages=messages, chars_per_token=1)
    assert stale == before  # cache cannot see it...

    invalidate_message_chars_cache(messages)
    fresh = estimate_tokens(system_prompt=None, messages=messages, chars_per_token=1)
    assert fresh < before - 900  # ...but explicit invalidation does


def test_trim_triggers_full_recompute() -> None:
    messages = [_message("y" * 100) for _ in range(10)]
    before = estimate_tokens(system_prompt=None, messages=messages, chars_per_token=1)
    del messages[:5]
    after = estimate_tokens(system_prompt=None, messages=messages, chars_per_token=1)
    assert after < before
    assert after == estimate_tokens(system_prompt=None, messages=list(messages), chars_per_token=1)
