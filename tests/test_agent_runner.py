from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from typing import Any

import pytest

from swarmee_river import agent_runner
from swarmee_river.interrupts import AgentInterruptedError


class _SleepAgent:
    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    async def invoke_async(self, _query: str, **_kwargs: Any) -> str:
        await asyncio.sleep(self._delay_s)
        return "ok"


class _ThreadRecordingAgent:
    def __init__(self, ctx_var: contextvars.ContextVar[str] | None = None) -> None:
        self.ctx_var = ctx_var
        self.invoke_thread_id: int | None = None

    async def invoke_async(self, _query: str, **_kwargs: Any) -> str:
        self.invoke_thread_id = threading.get_ident()
        if self.ctx_var is not None:
            return self.ctx_var.get()
        return "ok"


class _NeverFinishesAgent:
    async def invoke_async(self, _query: str, **_kwargs: Any) -> str:
        while True:
            await asyncio.sleep(0.05)


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


def test_resolve_agent_invoke_mode_defaults_to_isolated(monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_AGENT_INVOKE_MODE", raising=False)
    assert agent_runner._resolve_agent_invoke_mode() == "isolated"


def test_resolve_agent_invoke_mode_invalid_falls_back_to_isolated(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AGENT_INVOKE_MODE", "invalid")
    assert agent_runner._resolve_agent_invoke_mode() == "isolated"


def test_invoke_agent_isolated_mode_runs_in_different_thread(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AGENT_INVOKE_MODE", "isolated")
    agent = _ThreadRecordingAgent()
    caller_tid = threading.get_ident()

    result = agent_runner.invoke_agent(
        agent,
        "hello",
        callback_handler=lambda **_kwargs: None,
        interrupt_event=threading.Event(),
        invocation_state={"swarmee": {"provider": "openai"}},
    )

    assert result == "ok"
    assert agent.invoke_thread_id is not None
    assert agent.invoke_thread_id != caller_tid


def test_invoke_agent_direct_mode_runs_in_caller_thread(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AGENT_INVOKE_MODE", "direct")
    agent = _ThreadRecordingAgent()
    caller_tid = threading.get_ident()

    result = agent_runner.invoke_agent(
        agent,
        "hello",
        callback_handler=lambda **_kwargs: None,
        interrupt_event=threading.Event(),
        invocation_state={"swarmee": {"provider": "openai"}},
    )

    assert result == "ok"
    assert agent.invoke_thread_id == caller_tid


def test_invoke_agent_isolated_mode_propagates_contextvars(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AGENT_INVOKE_MODE", "isolated")
    ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("invoke_test_ctx")
    token = ctx_var.set("ctx-value")
    try:
        agent = _ThreadRecordingAgent(ctx_var=ctx_var)
        result = agent_runner.invoke_agent(
            agent,
            "hello",
            callback_handler=lambda **_kwargs: None,
            interrupt_event=threading.Event(),
            invocation_state={"swarmee": {"provider": "openai"}},
        )
    finally:
        ctx_var.reset(token)

    assert result == "ctx-value"


def test_invoke_agent_interrupt_sets_force_stop_and_raises(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AGENT_INVOKE_MODE", "isolated")
    callback_calls: list[dict[str, Any]] = []
    interrupt_event = threading.Event()

    def _callback_handler(**kwargs: Any) -> None:
        callback_calls.append(dict(kwargs))

    def _trigger_interrupt() -> None:
        time.sleep(0.08)
        interrupt_event.set()

    interrupter = threading.Thread(target=_trigger_interrupt, daemon=True)
    interrupter.start()

    with pytest.raises(AgentInterruptedError):
        agent_runner.invoke_agent(
            _NeverFinishesAgent(),
            "hello",
            callback_handler=_callback_handler,
            interrupt_event=interrupt_event,
            invocation_state={"swarmee": {"provider": "openai"}},
        )

    interrupter.join(timeout=0.2)
    assert any(call.get("force_stop") is True for call in callback_calls)


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


def test_windows_loop_policy_selector_applies_and_restores_through_invoke_agent(monkeypatch) -> None:
    previous_policy = object()
    set_calls: list[object] = []

    class _SelectorPolicy:
        pass

    monkeypatch.setattr(agent_runner.os, "name", "nt", raising=False)
    monkeypatch.setenv("SWARMEE_WINDOWS_EVENT_LOOP_POLICY", "selector")
    monkeypatch.setenv("SWARMEE_AGENT_INVOKE_MODE", "isolated")
    monkeypatch.setattr(agent_runner.asyncio, "WindowsSelectorEventLoopPolicy", _SelectorPolicy, raising=False)
    monkeypatch.setattr(agent_runner.asyncio, "get_event_loop_policy", lambda: previous_policy)
    monkeypatch.setattr(agent_runner.asyncio, "set_event_loop_policy", lambda policy: set_calls.append(policy))

    result = agent_runner.invoke_agent(
        _SleepAgent(delay_s=0),
        "hello",
        callback_handler=lambda **_kwargs: None,
        interrupt_event=threading.Event(),
        invocation_state={"swarmee": {"provider": "openai"}},
    )

    assert result == "ok"
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


def test_windows_loop_policy_proactor_applies_and_restores_through_invoke_agent(monkeypatch) -> None:
    previous_policy = object()
    set_calls: list[object] = []

    class _ProactorPolicy:
        pass

    monkeypatch.setattr(agent_runner.os, "name", "nt", raising=False)
    monkeypatch.setenv("SWARMEE_WINDOWS_EVENT_LOOP_POLICY", "proactor")
    monkeypatch.setenv("SWARMEE_AGENT_INVOKE_MODE", "isolated")
    monkeypatch.setattr(agent_runner.asyncio, "WindowsProactorEventLoopPolicy", _ProactorPolicy, raising=False)
    monkeypatch.setattr(agent_runner.asyncio, "get_event_loop_policy", lambda: previous_policy)
    monkeypatch.setattr(agent_runner.asyncio, "set_event_loop_policy", lambda policy: set_calls.append(policy))

    result = agent_runner.invoke_agent(
        _SleepAgent(delay_s=0),
        "hello",
        callback_handler=lambda **_kwargs: None,
        interrupt_event=threading.Event(),
        invocation_state={"swarmee": {"provider": "openai"}},
    )

    assert result == "ok"
    assert len(set_calls) == 2
    assert isinstance(set_calls[0], _ProactorPolicy)
    assert set_calls[1] is previous_policy
