from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.hooks.tool_policy import ToolPolicyHooks
from swarmee_river.permissions import evaluate_permission_action
from swarmee_river.settings import PermissionRule, SafetyConfig, ToolRule


def test_evaluate_permission_action_defaults_unchanged_without_permission_rules() -> None:
    safety = SafetyConfig(tool_consent="ask", tool_rules=[], permission_rules=[])

    action_shell, remember_shell = evaluate_permission_action(
        safety=safety,
        tool_name="shell",
        tool_use={"name": "shell", "input": {"command": "echo hi"}},
    )
    assert action_shell == "ask"
    assert remember_shell is True

    action_read, remember_read = evaluate_permission_action(
        safety=safety,
        tool_name="file_read",
        tool_use={"name": "file_read", "input": {"path": "README.md"}},
    )
    assert action_read == "allow"
    assert remember_read is True

    safety_with_rule = SafetyConfig(
        tool_consent="ask",
        tool_rules=[ToolRule(tool="file_read", default="deny", remember=False)],
        permission_rules=[],
    )
    action_rule, remember_rule = evaluate_permission_action(
        safety=safety_with_rule,
        tool_name="file_read",
        tool_use={"name": "file_read", "input": {"path": "README.md"}},
    )
    assert action_rule == "deny"
    assert remember_rule is False


def test_permission_rule_can_force_deny_for_low_risk_tool() -> None:
    safety = SafetyConfig(
        tool_consent="ask",
        tool_rules=[ToolRule(tool="file_read", default="allow", remember=True)],
        permission_rules=[PermissionRule(tool="file_read", action="deny", remember=False, when={})],
    )

    action, remember = evaluate_permission_action(
        safety=safety,
        tool_name="file_read",
        tool_use={"name": "file_read", "input": {"path": "README.md"}},
    )

    assert action == "deny"
    assert remember is False


def test_permission_rule_command_regex_matches_shell_commands() -> None:
    safety = SafetyConfig(
        tool_consent="ask",
        tool_rules=[],
        permission_rules=[
            PermissionRule(
                tool="shell",
                action="deny",
                remember=True,
                when={"command_regex": r"^rm\s+-rf\b"},
            )
        ],
    )

    action, remember = evaluate_permission_action(
        safety=safety,
        tool_name="bash",
        tool_use={"name": "bash", "input": {"command": "rm -rf /tmp/cache"}},
    )

    assert action == "deny"
    assert remember is True


def test_permission_rule_host_glob_and_method_match_http_request() -> None:
    safety = SafetyConfig(
        tool_consent="ask",
        tool_rules=[],
        permission_rules=[
            PermissionRule(
                tool="http_request",
                action="deny",
                remember=True,
                when={"host_glob": "*.internal.example.com", "method": "POST"},
            )
        ],
    )

    action, remember = evaluate_permission_action(
        safety=safety,
        tool_name="http_request",
        tool_use={
            "name": "http_request",
            "input": {"url": "https://api.internal.example.com/v1/jobs", "method": "POST"},
        },
    )

    assert action == "deny"
    assert remember is True


def test_permission_rules_deny_is_hard_enforced_by_tool_policy_hooks(monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_ENABLE_TOOLS", raising=False)
    monkeypatch.delenv("SWARMEE_DISABLE_TOOLS", raising=False)
    safety = SafetyConfig(
        tool_consent="ask",
        tool_rules=[],
        permission_rules=[PermissionRule(tool="file_read", action="deny", remember=True, when={})],
    )
    hook = ToolPolicyHooks(safety=safety)
    event = SimpleNamespace(
        tool_use={"name": "file_read", "input": {"path": "README.md"}},
        invocation_state={"swarmee": {"mode": "execute"}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool == "Tool 'file_read' blocked by permission_rules safety policy."


def test_permission_rules_deny_enforced_even_when_bypass_tool_consent_enabled(monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_ENABLE_TOOLS", raising=False)
    monkeypatch.delenv("SWARMEE_DISABLE_TOOLS", raising=False)
    monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
    safety = SafetyConfig(
        tool_consent="ask",
        tool_rules=[],
        permission_rules=[PermissionRule(tool="file_read", action="deny", remember=True, when={})],
    )
    hook = ToolPolicyHooks(safety=safety)
    event = SimpleNamespace(
        tool_use={"name": "file_read", "input": {"path": "README.md"}},
        invocation_state={"swarmee": {"mode": "execute"}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool == "Tool 'file_read' blocked by permission_rules safety policy."


def test_tool_rules_deny_is_not_hard_enforced_by_tool_policy_hooks(monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_ENABLE_TOOLS", raising=False)
    monkeypatch.delenv("SWARMEE_DISABLE_TOOLS", raising=False)
    safety = SafetyConfig(
        tool_consent="ask",
        tool_rules=[ToolRule(tool="file_read", default="deny", remember=True)],
        permission_rules=[],
    )
    hook = ToolPolicyHooks(safety=safety)
    event = SimpleNamespace(
        tool_use={"name": "file_read", "input": {"path": "README.md"}},
        invocation_state={"swarmee": {"mode": "execute"}},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool is False
