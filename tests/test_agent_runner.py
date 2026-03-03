from __future__ import annotations

import asyncio
import threading
from typing import Any

from swarmee_river import agent_runner


class _SleepAgent:
    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    async def invoke_async(self, _query: str, **_kwargs: Any) -> str:
        await asyncio.sleep(self._delay_s)
        return "ok"


def test_invoke_agent_emits_stall_warning_for_bedrock(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_BEDROCK_STALL_WARN_SEC", "0.04")
    monkeypatch.setenv("SWARMEE_BEDROCK_STALL_DIAG_DUMP", "false")

    callback_calls: list[dict[str, Any]] = []

    def _callback_handler(**kwargs: Any) -> None:
        callback_calls.append(dict(kwargs))

    result = agent_runner.invoke_agent(
        _SleepAgent(delay_s=0.15),
        "hello",
        callback_handler=_callback_handler,
        interrupt_event=threading.Event(),
        invocation_state={"swarmee": {"provider": "bedrock"}},
    )

    assert result == "ok"
    warning_calls = [item for item in callback_calls if isinstance(item.get("warning_text"), str)]
    assert warning_calls, "expected at least one stall warning callback"
    assert "stalled" in warning_calls[0]["warning_text"].lower()


def test_invoke_agent_stall_warning_is_throttled(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_BEDROCK_STALL_WARN_SEC", "0.05")
    monkeypatch.setenv("SWARMEE_BEDROCK_STALL_DIAG_DUMP", "false")

    callback_calls: list[dict[str, Any]] = []

    def _callback_handler(**kwargs: Any) -> None:
        callback_calls.append(dict(kwargs))

    result = agent_runner.invoke_agent(
        _SleepAgent(delay_s=0.18),
        "hello",
        callback_handler=_callback_handler,
        interrupt_event=threading.Event(),
        invocation_state={"swarmee": {"provider": "bedrock"}},
    )

    assert result == "ok"
    warning_calls = [item for item in callback_calls if isinstance(item.get("warning_text"), str)]
    assert len(warning_calls) >= 2


def test_windows_loop_policy_auto_does_not_override(monkeypatch) -> None:
    set_calls: list[object] = []

    monkeypatch.setattr(agent_runner.os, "name", "nt", raising=False)
    monkeypatch.setenv("SWARMEE_WINDOWS_EVENT_LOOP_POLICY", "auto")
    monkeypatch.setattr(agent_runner.asyncio, "set_event_loop_policy", lambda policy: set_calls.append(policy))

    with agent_runner._windows_loop_policy_context():
        pass

    assert set_calls == []


def test_windows_loop_policy_selector_applies_and_restores(monkeypatch) -> None:
    previous_policy = object()
    set_calls: list[object] = []

    class _SelectorPolicy:
        pass

    monkeypatch.setattr(agent_runner.os, "name", "nt", raising=False)
    monkeypatch.setenv("SWARMEE_WINDOWS_EVENT_LOOP_POLICY", "selector")
    monkeypatch.setattr(agent_runner.asyncio, "WindowsSelectorEventLoopPolicy", _SelectorPolicy, raising=False)
    monkeypatch.setattr(agent_runner.asyncio, "get_event_loop_policy", lambda: previous_policy)
    monkeypatch.setattr(agent_runner.asyncio, "set_event_loop_policy", lambda policy: set_calls.append(policy))

    with agent_runner._windows_loop_policy_context():
        pass

    assert len(set_calls) == 2
    assert isinstance(set_calls[0], _SelectorPolicy)
    assert set_calls[1] is previous_policy


def test_windows_loop_policy_proactor_applies_and_restores(monkeypatch) -> None:
    previous_policy = object()
    set_calls: list[object] = []

    class _ProactorPolicy:
        pass

    monkeypatch.setattr(agent_runner.os, "name", "nt", raising=False)
    monkeypatch.setenv("SWARMEE_WINDOWS_EVENT_LOOP_POLICY", "proactor")
    monkeypatch.setattr(agent_runner.asyncio, "WindowsProactorEventLoopPolicy", _ProactorPolicy, raising=False)
    monkeypatch.setattr(agent_runner.asyncio, "get_event_loop_policy", lambda: previous_policy)
    monkeypatch.setattr(agent_runner.asyncio, "set_event_loop_policy", lambda policy: set_calls.append(policy))

    with agent_runner._windows_loop_policy_context():
        pass

    assert len(set_calls) == 2
    assert isinstance(set_calls[0], _ProactorPolicy)
    assert set_calls[1] is previous_policy
