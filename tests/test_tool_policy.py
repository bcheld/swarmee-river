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


def test_plan_mode_allows_repo_inspection_tools() -> None:
    hook = ToolPolicyHooks()
    for name in ["file_read", "file_list", "file_search", "read", "grep", "list", "glob", "project_context", "todoread"]:
        event = SimpleNamespace(
            tool_use={"name": name},
            invocation_state={"swarmee": {"mode": "plan"}},
            cancel_tool=False,
        )
        hook.before_tool_call(event)
        assert event.cancel_tool is False


def test_plan_mode_blocks_risky_opencode_aliases() -> None:
    hook = ToolPolicyHooks()
    for name in ["bash", "patch", "write", "edit"]:
        event = SimpleNamespace(
            tool_use={"name": name},
            invocation_state={"swarmee": {"mode": "plan"}},
            cancel_tool=False,
        )
        hook.before_tool_call(event)
        assert event.cancel_tool
        assert "blocked in plan mode" in str(event.cancel_tool)


def test_plan_mode_blocks_todowrite() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "todowrite"},
        invocation_state={"swarmee": {"mode": "plan"}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "blocked in plan mode" in str(event.cancel_tool)


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


def test_execute_mode_blocks_repeated_project_context_loop() -> None:
    hook = ToolPolicyHooks()
    invocation_state = {"swarmee": {"mode": "execute"}}
    last_event = None

    for _ in range(7):
        event = SimpleNamespace(
            tool_use={"name": "project_context", "input": {"action": "summary"}},
            invocation_state=invocation_state,
            cancel_tool=False,
        )
        hook.before_tool_call(event)
        last_event = event

    assert last_event is not None
    assert last_event.cancel_tool
    assert "Repeated project_context loop detected" in str(last_event.cancel_tool)


def test_execute_mode_blocks_project_context_after_total_cap_even_when_varied() -> None:
    hook = ToolPolicyHooks()
    invocation_state = {"swarmee": {"mode": "execute"}}
    last_event = None

    for i in range(7):
        event = SimpleNamespace(
            tool_use={"name": "project_context", "input": {"action": "search", "query": f"q{i}"}},
            invocation_state=invocation_state,
            cancel_tool=False,
        )
        hook.before_tool_call(event)
        last_event = event

    assert last_event is not None
    assert last_event.cancel_tool
    assert "Repeated project_context loop detected" in str(last_event.cancel_tool)


def test_execute_mode_blocks_shell_file_inspection_commands() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "shell", "input": {"command": "sed -n '1,240p' src/swarmee_river/settings.py"}},
        invocation_state={"swarmee": {"mode": "execute"}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "list/glob/file_list/file_search/file_read" in str(event.cancel_tool)


def test_execute_mode_blocks_shell_file_inspection_commands_via_bash_alias() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "bash", "input": {"command": "sed -n '1,240p' src/swarmee_river/settings.py"}},
        invocation_state={"swarmee": {"mode": "execute"}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "list/glob/file_list/file_search/file_read" in str(event.cancel_tool)


def test_execute_mode_allows_shell_non_file_inspection_commands() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "shell", "input": {"command": "pytest -q"}},
        invocation_state={"swarmee": {"mode": "execute"}},
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


def test_execute_mode_enforce_plan_allows_alias_when_underlying_is_approved() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "patch"},
        invocation_state={"swarmee": {"mode": "execute", "enforce_plan": True, "allowed_tools": ["patch_apply"]}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool is False
