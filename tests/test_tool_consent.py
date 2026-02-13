from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.hooks.tool_consent import ToolConsentHooks
from swarmee_river.settings import SafetyConfig, ToolRule


def test_tool_consent_non_interactive_fails_closed():
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="shell", default="ask", remember=True)])
    hook = ToolConsentHooks(safety, interactive=False, auto_approve=False, prompt=lambda _text: "y")

    event = SimpleNamespace(tool_use={"name": "shell"}, invocation_state={}, cancel_tool=False)
    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "non-interactive" in str(event.cancel_tool)


def test_tool_consent_remembers_denial_for_session():
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="shell", default="ask", remember=True)])
    calls: list[str] = []

    def prompt(_text: str) -> str:
        calls.append("asked")
        return "v"

    hook = ToolConsentHooks(safety, interactive=True, auto_approve=False, prompt=prompt)

    event1 = SimpleNamespace(tool_use={"name": "shell"}, invocation_state={}, cancel_tool=False)
    hook.before_tool_call(event1)
    assert event1.cancel_tool

    event2 = SimpleNamespace(tool_use={"name": "shell"}, invocation_state={}, cancel_tool=False)
    hook.before_tool_call(event2)
    assert event2.cancel_tool

    # Only prompted once; second time used remembered decision.
    assert len(calls) == 1


def test_plan_approval_counts_as_consent():
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="shell", default="ask", remember=True)])
    hook = ToolConsentHooks(safety, interactive=True, auto_approve=False, prompt=lambda _text: "n")

    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"mode": "execute", "enforce_plan": True, "allowed_tools": ["shell"]}},
        cancel_tool=False,
    )
    hook.before_tool_call(event)
    assert event.cancel_tool is False

