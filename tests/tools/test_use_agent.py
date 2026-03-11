from __future__ import annotations

from typing import Any

import tools.use_agent as use_agent_module
from swarmee_river.utils.fork_utils import SharedPrefixTextForkResult


def _text(result: dict) -> str:
    return (result.get("content") or [{"text": ""}])[0].get("text", "")


def test_use_agent_requires_prompt() -> None:
    result = use_agent_module.use_agent(prompt=" ", agent=object())
    assert result.get("status") == "error"
    assert "prompt is required" in _text(result).lower()


def test_use_agent_requires_agent_model() -> None:
    class Parent:
        model = None

    result = use_agent_module.use_agent(prompt="hi", agent=Parent())
    assert result.get("status") == "error"
    assert "requires an agent context" in _text(result).lower()


def test_use_agent_invokes_shared_prefix_text_fork(monkeypatch) -> None:
    class Parent:
        model = object()

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        use_agent_module,
        "run_shared_prefix_text_fork",
        lambda agent, *, kind, prompt_text, extra_fields=None: (
            calls.append({"agent": agent, "kind": kind, "prompt_text": prompt_text, "extra_fields": extra_fields}) or
            SharedPrefixTextForkResult(
                text="echo:ping",
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "echo:ping"}]},
                used_tool=False,
                diagnostics={},
            )
        ),
    )

    result = use_agent_module.use_agent(prompt="ping", system_prompt="sys", agent=Parent())
    assert result.get("status") == "success"
    assert _text(result) == "echo:ping"
    assert calls and calls[0]["kind"] == "use_agent"
    assert "sys" in calls[0]["prompt_text"]


def test_use_agent_rejects_tool_use_in_text_only_fork(monkeypatch) -> None:
    class Parent:
        model = object()

    monkeypatch.setattr(
        use_agent_module,
        "run_shared_prefix_text_fork",
        lambda *_args, **_kwargs: SharedPrefixTextForkResult(
            text="",
            stop_reason="tool_use",
            message={"role": "assistant", "content": [{"toolUse": {"name": "shell", "toolUseId": "t1"}}]},
            used_tool=True,
            diagnostics={},
        ),
    )

    result = use_agent_module.use_agent(prompt="ping", agent=Parent())
    assert result.get("status") == "error"
    assert "rejected a tool call" in _text(result).lower()
