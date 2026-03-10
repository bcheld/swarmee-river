from __future__ import annotations

import pytest

from swarmee_river.context.budgeted_summarizing_conversation_manager import (
    BudgetedSummarizingConversationManager,
    estimate_tokens,
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


def test_apply_management_skips_auto_compaction_when_manual(monkeypatch):
    manager = BudgetedSummarizingConversationManager(max_prompt_tokens=1, compaction_mode="manual")

    called = {"value": False}

    def _should_not_be_called(*_args, **_kwargs):
        called["value"] = True

    monkeypatch.setattr(manager, "reduce_context", _should_not_be_called)

    class _Agent:
        messages = [{"role": "user", "content": [{"text": "hello world"}]}]

    manager.apply_management(agent=_Agent())
    assert called["value"] is False


def test_estimate_tokens_counts_tool_and_reasoning_content() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "plan the tool usage"}}},
                {"toolUse": {"toolUseId": "tool-1", "name": "file_search", "input": {"query": "TODO"}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "tool-1", "status": "success", "content": [{"text": "hit"}]}},
            ],
        },
    ]

    tokens = estimate_tokens(system_prompt="system", messages=messages, chars_per_token=4)

    assert tokens > 0


def test_compact_to_budget_trims_old_messages_when_summarization_insufficient(monkeypatch) -> None:
    manager = BudgetedSummarizingConversationManager(
        max_prompt_tokens=5,
        preserve_recent_messages=2,
        compaction_mode="auto",
    )

    class _Agent:
        messages = [
            {"role": "user", "content": [{"text": "first older message"}]},
            {"role": "assistant", "content": [{"text": "second older message"}]},
            {"role": "user", "content": [{"text": "keep recent one"}]},
            {"role": "assistant", "content": [{"text": "keep recent two"}]},
        ]

    monkeypatch.setattr(manager, "reduce_context", lambda *_args, **_kwargs: None)

    result = manager.compact_to_budget(agent=_Agent())

    assert result["trimmed_messages"] > 0
    assert len(_Agent.messages) == 2


def test_compact_to_budget_reports_over_budget_when_recent_context_alone_is_too_large(monkeypatch) -> None:
    manager = BudgetedSummarizingConversationManager(
        max_prompt_tokens=1,
        preserve_recent_messages=2,
        compaction_mode="auto",
    )

    class _Agent:
        messages = [
            {"role": "user", "content": [{"text": "keep me"}]},
            {"role": "assistant", "content": [{"text": "also keep me"}]},
        ]

    monkeypatch.setattr(manager, "reduce_context", lambda *_args, **_kwargs: None)

    result = manager.compact_to_budget(agent=_Agent())

    assert result["within_budget"] is False
    assert result["trimmed_messages"] == 0
