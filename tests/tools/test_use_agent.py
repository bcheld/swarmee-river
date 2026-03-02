from __future__ import annotations

from typing import Any

import tools.use_agent as use_agent_module


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


def test_use_agent_invokes_toolless_sub_agent(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class FakeSubAgent:
        async def invoke_async(self, prompt: str) -> Any:  # noqa: D401
            return {"message": [{"text": f"echo:{prompt}"}]}

    def fake_create_sub_agent(*, parent_agent: Any, system_prompt: str) -> Any:
        seen["parent_agent"] = parent_agent
        seen["system_prompt"] = system_prompt
        return FakeSubAgent()

    class Parent:
        model = object()

    monkeypatch.setattr(use_agent_module, "create_sub_agent", fake_create_sub_agent)

    result = use_agent_module.use_agent(prompt="ping", system_prompt="sys", agent=Parent())
    assert result.get("status") == "success"
    assert _text(result) == "echo:ping"
    assert seen.get("system_prompt") == "sys"
    assert seen.get("parent_agent") is not None
