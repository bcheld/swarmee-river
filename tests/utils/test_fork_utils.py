from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
from typing import Any

from swarmee_river.context.prompt_cache import PromptCacheState
from swarmee_river.utils import fork_utils


class _FakeToolRegistry:
    def __init__(self) -> None:
        self.registry = OrderedDict(
            [
                ("alpha", {"tool_name": "alpha"}),
                ("beta", {"tool_name": "beta"}),
            ]
        )
        self.dynamic_tools: dict[str, Any] = {}

    def get_all_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {"name": "alpha", "inputSchema": {"json": {"type": "object", "properties": {}}}},
            {"name": "beta", "inputSchema": {"json": {"type": "object", "properties": {}}}},
        ]

    def get_all_tools_config(self) -> dict[str, dict[str, Any]]:
        return {
            "alpha": {"name": "alpha", "inputSchema": {"json": {"type": "object", "properties": {}}}},
            "beta": {"name": "beta", "inputSchema": {"json": {"type": "object", "properties": {}}}},
        }


def _parent_agent() -> SimpleNamespace:
    prompt_cache = PromptCacheState()
    prompt_cache.queue_one_off("Reminder block")
    return SimpleNamespace(
        model=SimpleNamespace(stream=lambda *_args, **_kwargs: object()),
        system_prompt="base system prompt",
        _system_prompt_content=None,
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        tool_registry=_FakeToolRegistry(),
        _swarmee_prompt_cache=prompt_cache,
        _swarmee_current_invocation_state={"swarmee": {"provider": "bedrock", "tier": "deep"}},
    )


def test_run_shared_prefix_text_fork_preserves_parent_prefix(monkeypatch) -> None:
    parent = _parent_agent()
    model_calls: list[dict[str, Any]] = []
    pending_before = parent._swarmee_prompt_cache.peek_reminder()

    def _fake_stream(
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        system_prompt_content: list[dict[str, Any]] | None = None,
        invocation_state: dict[str, Any] | None = None,
    ) -> object:
        model_calls.append(
            {
                "messages": messages,
                "tool_specs": tool_specs,
                "system_prompt": system_prompt,
                "system_prompt_content": system_prompt_content,
                "invocation_state": invocation_state,
            }
        )
        return object()

    async def _fake_process_stream(_stream: object):
        yield {
            "stop": (
                "end_turn",
                {"role": "assistant", "content": [{"text": "Fork result"}]},
                None,
                None,
            )
        }

    parent.model.stream = _fake_stream
    monkeypatch.setattr(fork_utils, "process_stream", _fake_process_stream)

    result = fork_utils.run_shared_prefix_text_fork(parent, kind="compaction", prompt_text="Compact now")

    assert result.text == "Fork result"
    assert model_calls
    call = model_calls[0]
    assert call["messages"][:-1] == parent.messages
    assert "Reminder block" in call["messages"][-1]["content"][0]["text"]
    assert "Compact now" in call["messages"][-1]["content"][0]["text"]
    assert [item["name"] for item in call["tool_specs"]] == ["alpha", "beta"]
    assert call["system_prompt"] == "base system prompt"
    assert call["invocation_state"]["swarmee"]["fork_kind"] == "compaction"
    assert result.diagnostics["fork_used_pending_reminder"] is True
    assert parent._swarmee_prompt_cache.peek_reminder() == pending_before
    assert parent._swarmee_prompt_cache.pop_reminder() == pending_before


def test_create_shared_prefix_child_agent_seeds_instruction_and_tool_order(monkeypatch) -> None:
    parent = _parent_agent()
    created: dict[str, Any] = {}

    class _FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            created.update(kwargs)
            self.hooks = SimpleNamespace(add_hook=lambda _hook: None)

    monkeypatch.setattr(fork_utils, "Agent", _FakeAgent)

    _child, snapshot = fork_utils.create_shared_prefix_child_agent(
        parent_agent=parent,
        kind="strand",
        seed_instruction="Do strand work",
        tool_allowlist=["beta"],
        callback_handler=None,
    )

    assert snapshot.parent_message_count == 1
    assert created["messages"][0] == parent.messages[0]
    assert "Reminder block" in created["messages"][1]["content"][0]["text"]
    assert "Do strand work" in created["messages"][1]["content"][0]["text"]
    assert [tool["tool_name"] for tool in created["tools"]] == ["alpha", "beta"]
    assert len(created["hooks"]) == 1
    assert isinstance(created["hooks"][0], fork_utils.StaticToolAllowlistHooks)
