from __future__ import annotations

from types import SimpleNamespace

import pytest

from swarmee_river.context.budgeted_summarizing_conversation_manager import (
    BudgetedSummarizingConversationManager,
    estimate_tokens,
    estimate_tokens_for_agent,
)
from swarmee_river.utils.fork_utils import SharedPrefixTextForkResult


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


def test_estimate_tokens_counts_tool_schema_overhead() -> None:
    tokens = estimate_tokens(system_prompt=None, messages=[], chars_per_token=4, tool_schema_chars=400)
    assert tokens == 100


def test_estimate_tokens_for_agent_includes_tool_schema_overhead() -> None:
    class _Agent:
        system_prompt = "system"
        messages = [{"role": "user", "content": [{"text": "hello"}]}]
        tools = [
            {
                "name": "shell",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    }
                },
            }
        ]

    baseline = estimate_tokens(system_prompt="system", messages=_Agent.messages, chars_per_token=4)
    estimated = estimate_tokens_for_agent(_Agent(), chars_per_token=4)

    assert estimated > baseline


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

    monkeypatch.setattr(manager, "_compaction_trigger_tokens", lambda: 5)
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

    monkeypatch.setattr(manager, "_compaction_trigger_tokens", lambda: 1)
    monkeypatch.setattr(manager, "reduce_context", lambda *_args, **_kwargs: None)

    result = manager.compact_to_budget(agent=_Agent())

    assert result["within_budget"] is False
    assert result["trimmed_messages"] == 0


def test_compact_to_budget_triggers_before_hard_limit(monkeypatch) -> None:
    manager = BudgetedSummarizingConversationManager(max_prompt_tokens=100000, compaction_mode="auto")

    class _Agent:
        messages = [{"role": "user", "content": [{"text": "hello"}]}]

    estimates = iter([95000, 95000, 85000])
    reduce_calls: list[bool] = []
    monkeypatch.setattr(manager, "estimate_tokens_for_agent", lambda _agent: next(estimates))
    monkeypatch.setattr(manager, "reduce_context", lambda *_args, **_kwargs: reduce_calls.append(True))

    result = manager.compact_to_budget(agent=_Agent())

    assert reduce_calls == [True]
    assert result["compaction_headroom_tokens"] == 10000
    assert result["within_compaction_target"] is True


def test_compact_to_budget_falls_back_to_trim_when_fork_summary_is_unusable(monkeypatch) -> None:
    manager = BudgetedSummarizingConversationManager(
        max_prompt_tokens=100000,
        preserve_recent_messages=2,
        compaction_mode="auto",
    )

    class _Agent:
        messages = [
            {"role": "user", "content": [{"text": "older one"}]},
            {"role": "assistant", "content": [{"text": "older two"}]},
            {"role": "user", "content": [{"text": "recent one"}]},
            {"role": "assistant", "content": [{"text": "recent two"}]},
        ]

    monkeypatch.setattr(
        "swarmee_river.context.budgeted_summarizing_conversation_manager.run_shared_prefix_text_fork",
        lambda *_args, **_kwargs: SharedPrefixTextForkResult(
            text="",
            stop_reason="tool_use",
            message={"role": "assistant", "content": [{"toolUse": {"name": "shell", "toolUseId": "x"}}]},
            used_tool=True,
            diagnostics={"fork_kind": "compaction"},
        ),
    )
    monkeypatch.setattr(
        manager,
        "estimate_tokens_for_agent",
        lambda agent: 95000 if len(agent.messages) > 2 else 50000,
    )

    result = manager.compact_to_budget(agent=_Agent())

    assert result["trimmed_messages"] > 0
    assert "Compaction summary unavailable" in str(result["warning"])
    assert result["fork_kind"] == "compaction"


def _tool_exchange(
    tool_use_id: str,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    text: str,
) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": tool_use_id,
                        "name": tool_name,
                        "input": tool_input,
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": [{"text": text}],
                    }
                }
            ],
        },
    ]


def test_cache_safe_compacts_older_file_reads_before_budget_pass() -> None:
    manager = BudgetedSummarizingConversationManager(strategy="cache_safe", compaction_mode="auto")

    class _Agent:
        messages = (
            _tool_exchange(
                "t1",
                tool_name="file_read",
                tool_input={"path": "a.py", "start_line": 1, "max_lines": 20},
                text="A" * 200,
            )
            + _tool_exchange(
                "t2",
                tool_name="file_read",
                tool_input={"path": "b.py", "start_line": 1, "max_lines": 20},
                text="B" * 200,
            )
            + _tool_exchange(
                "t3",
                tool_name="file_read",
                tool_input={"path": "c.py", "start_line": 1, "max_lines": 20},
                text="C" * 200,
            )
            + _tool_exchange(
                "t4",
                tool_name="file_search",
                tool_input={"query": "needle"},
                text="D" * 200,
            )
        )

    result = manager.compact_to_budget(agent=_Agent())

    texts = [
        item["toolResult"]["content"][0]["text"]
        for message in _Agent.messages
        for item in message.get("content", [])
        if "toolResult" in item
    ]
    assert result["compacted_read_results"] == 2
    assert texts[0].startswith("[cache-compacted]")
    assert texts[1].startswith("[cache-compacted]")
    assert texts[2] == "C" * 200
    assert texts[3] == "D" * 200


def test_long_running_collapses_duplicate_reads_even_within_keep_window() -> None:
    manager = BudgetedSummarizingConversationManager(strategy="long_running", compaction_mode="auto")

    class _Agent:
        messages = (
            _tool_exchange(
                "t1",
                tool_name="file_read",
                tool_input={"path": "dup.py", "start_line": 1, "max_lines": 20},
                text="same excerpt",
            )
            + _tool_exchange(
                "t2",
                tool_name="file_read",
                tool_input={"path": "dup.py", "start_line": 1, "max_lines": 20},
                text="same excerpt",
            )
            + _tool_exchange(
                "t3",
                tool_name="file_read",
                tool_input={"path": "dup.py", "start_line": 1, "max_lines": 20},
                text="same excerpt",
            )
        )

    result = manager.compact_to_budget(agent=_Agent())

    texts = [
        item["toolResult"]["content"][0]["text"]
        for message in _Agent.messages
        for item in message.get("content", [])
        if "toolResult" in item
    ]
    assert result["compacted_read_results"] == 2
    assert texts[0].startswith("[cache-compacted]")
    assert texts[1].startswith("[cache-compacted]")
    assert texts[2] == "same excerpt"


def test_custom_compactable_tools_are_supported() -> None:
    manager = BudgetedSummarizingConversationManager(
        strategy="cache_safe",
        compaction_mode="auto",
        compactable_tool_names=["custom_fetch"],
    )

    class _Agent:
        messages = (
            _tool_exchange(
                "t1",
                tool_name="custom_fetch",
                tool_input={"url": "https://example.com/a"},
                text="A" * 200,
            )
            + _tool_exchange(
                "t2",
                tool_name="custom_fetch",
                tool_input={"url": "https://example.com/b"},
                text="B" * 200,
            )
            + _tool_exchange(
                "t3",
                tool_name="custom_fetch",
                tool_input={"url": "https://example.com/c"},
                text="C" * 200,
            )
        )

    result = manager.compact_to_budget(agent=_Agent())

    texts = [
        item["toolResult"]["content"][0]["text"]
        for message in _Agent.messages
        for item in message.get("content", [])
        if "toolResult" in item
    ]
    assert result["compacted_read_results"] == 1
    assert texts[0].startswith("[cache-compacted]")
    assert texts[-1] == "C" * 200


def test_generate_summary_uses_shared_prefix_compaction_fork(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "swarmee_river.context.budgeted_summarizing_conversation_manager.run_shared_prefix_text_fork",
        lambda agent, *, kind, prompt_text, extra_fields=None: (
            captured.update(
                {
                    "agent": agent,
                    "kind": kind,
                    "prompt_text": prompt_text,
                    "extra_fields": extra_fields,
                }
            )
            or SharedPrefixTextForkResult(
                text="Assistant: skip\nDurable summary\n[tool result] shell",
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Durable summary"}]},
                used_tool=False,
                diagnostics={"fork_kind": "compaction", "fork_prefix_hash": "abc123"},
            )
        ),
    )

    manager = BudgetedSummarizingConversationManager()
    agent = SimpleNamespace(
        messages=[
            {"role": "user", "content": [{"text": "older one"}]},
            {"role": "assistant", "content": [{"text": "older two"}]},
            {"role": "user", "content": [{"text": "recent one"}]},
        ],
        _swarmee_compaction_extra_fields={"compaction_headroom_tokens": 8192},
    )
    messages = agent.messages[:2]

    summary = manager._generate_summary(messages, agent=agent)

    assert captured["agent"] is agent
    assert captured["kind"] == "compaction"
    assert "oldest 2 messages" in str(captured["prompt_text"]).lower()
    assert captured["extra_fields"] == {"compaction_headroom_tokens": 8192}
    assert summary == {"role": "user", "content": [{"text": "Durable summary"}]}
    assert manager._last_compaction_fork_diagnostics == {
        "fork_kind": "compaction",
        "fork_prefix_hash": "abc123",
    }


def test_balanced_strategy_leaves_read_results_uncompacted() -> None:
    manager = BudgetedSummarizingConversationManager(strategy="balanced", compaction_mode="auto")

    class _Agent:
        messages = _tool_exchange(
            "t1",
            tool_name="file_read",
            tool_input={"path": "a.py", "start_line": 1, "max_lines": 20},
            text="raw excerpt",
        ) + _tool_exchange(
            "t2",
            tool_name="file_search",
            tool_input={"query": "needle"},
            text="search excerpt",
        )

    result = manager.compact_to_budget(agent=_Agent())

    texts = [
        item["toolResult"]["content"][0]["text"]
        for message in _Agent.messages
        for item in message.get("content", [])
        if "toolResult" in item
    ]
    assert result["compacted_read_results"] == 0
    assert texts == ["raw excerpt", "search excerpt"]
