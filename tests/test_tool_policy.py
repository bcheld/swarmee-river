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


def test_windows_powershell_blocks_posix_shell_commands() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "shell", "input": {"command": "grep -R TODO src"}},
        invocation_state={
            "swarmee": {"mode": "execute", "runtime_environment": {"os": "windows", "shell_family": "powershell"}}
        },
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "POSIX-specific" in str(event.cancel_tool)


def test_windows_powershell_allows_explicit_bash_shell_commands() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "shell", "input": {"command": "bash -lc 'grep -R TODO src'"}},
        invocation_state={
            "swarmee": {"mode": "execute", "runtime_environment": {"os": "windows", "shell_family": "powershell"}}
        },
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool is False


def test_execute_mode_blocks_workplan_tool() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "WorkPlan"},
        invocation_state={"swarmee": {"mode": "execute"}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "only allowed in plan mode" in str(event.cancel_tool)


def test_execute_mode_enforce_plan_blocks_when_allowlist_is_empty() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"mode": "execute", "enforce_plan": True, "allowed_tools": []}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "not in approved plan" in str(event.cancel_tool)
