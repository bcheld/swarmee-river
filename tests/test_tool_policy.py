from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.hooks.tool_policy import ToolPolicyHooks


def test_plan_mode_blocks_non_allowlisted_tools() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"mode": "plan"}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "blocked in plan mode" in str(event.cancel_tool)


def test_plan_mode_allows_structured_output_tool_from_invocation_state() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "WorkPlan"},
        invocation_state={"swarmee": {"mode": "plan", "plan_allowed_tools": ["WorkPlan"]}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool is False
