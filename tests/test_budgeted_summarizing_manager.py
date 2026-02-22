from __future__ import annotations

import pytest

from swarmee_river.context.budgeted_summarizing_conversation_manager import (
    BudgetedSummarizingConversationManager,
)


def test_reduce_context_ignores_insufficient_messages(monkeypatch):
    manager = BudgetedSummarizingConversationManager()

    def _raise_insufficient(*_args, **_kwargs):
        raise RuntimeError("Cannot summarize: insufficient messages for summarization")

    monkeypatch.setattr(
        "swarmee_river.context.budgeted_summarizing_conversation_manager.SummarizingConversationManager.reduce_context",
        _raise_insufficient,
    )

    manager.reduce_context(agent=object())


def test_reduce_context_propagates_other_errors(monkeypatch):
    manager = BudgetedSummarizingConversationManager()

    def _raise_other(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "swarmee_river.context.budgeted_summarizing_conversation_manager.SummarizingConversationManager.reduce_context",
        _raise_other,
    )

    class _Agent:
        messages = [{"role": "user", "content": [{"text": "a"}]}, {"role": "assistant", "content": [{"text": "b"}]}]

    with pytest.raises(RuntimeError, match="boom"):
        manager.reduce_context(agent=_Agent())


def test_reduce_context_skips_when_messages_too_small(monkeypatch):
    manager = BudgetedSummarizingConversationManager()

    called = {"value": False}

    def _should_not_be_called(*_args, **_kwargs):
        called["value"] = True

    monkeypatch.setattr(
        "swarmee_river.context.budgeted_summarizing_conversation_manager.SummarizingConversationManager.reduce_context",
        _should_not_be_called,
    )

    class _Agent:
        messages = []

    manager.reduce_context(agent=_Agent())
    assert called["value"] is False
