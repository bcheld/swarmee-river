from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from swarmee_river.hooks.tool_consent import ToolConsentHooks
from swarmee_river.settings import SafetyConfig, ToolRule


def test_tool_consent_non_interactive_fails_closed():
    safety = SafetyConfig(tool_consent="ask", tool_rules=[])
    hook = ToolConsentHooks(safety, interactive=False, auto_approve=False, prompt=lambda _text, _payload=None: "y")

    event = SimpleNamespace(tool_use={"name": "shell"}, invocation_state={}, cancel_tool=False)
    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "non-interactive" in str(event.cancel_tool)


def test_tool_consent_remembers_denial_for_session():
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="shell", default="ask", remember=True)])
    calls: list[str] = []

    def prompt(_text: str, _payload=None) -> str:
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
    hook = ToolConsentHooks(safety, interactive=True, auto_approve=False, prompt=lambda _text, _payload=None: "n")

    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"mode": "execute", "enforce_plan": True, "allowed_tools": ["shell"]}},
        cancel_tool=False,
    )
    hook.before_tool_call(event)
    assert event.cancel_tool is False


def test_tool_consent_prompt_includes_shell_command_context():
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="shell", default="ask", remember=True)])
    prompts: list[str] = []

    def prompt(text: str, _payload=None) -> str:
        prompts.append(text)
        return "y"

    hook = ToolConsentHooks(safety, interactive=True, auto_approve=False, prompt=prompt)

    event = SimpleNamespace(
        tool_use={"name": "shell", "input": {"command": "ls -al", "cwd": "/tmp/work"}},
        invocation_state={},
        cancel_tool=False,
    )
    hook.before_tool_call(event)

    assert event.cancel_tool is False
    assert prompts
    assert "Command: ls -al" in prompts[0]
    assert "CWD: /tmp/work" in prompts[0]


def test_tool_consent_prompt_includes_editor_diff_preview(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="editor", default="ask", remember=True)])
    prompts: list[str] = []
    payloads: list[dict[str, object] | None] = []

    def prompt(text: str, payload=None) -> str:
        prompts.append(text)
        payloads.append(payload)
        return "y"

    hook = ToolConsentHooks(safety, interactive=True, auto_approve=False, prompt=prompt)
    event = SimpleNamespace(
        tool_use={
            "name": "editor",
            "input": {
                "command": "replace",
                "path": "notes.txt",
                "old_str": "hello",
                "new_str": "goodbye",
                "cwd": str(tmp_path),
            },
        },
        invocation_state={},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool is False
    assert prompts
    assert "Changed paths: notes.txt" in prompts[0]
    assert "--- a/notes.txt" in prompts[0]
    assert "+++ b/notes.txt" in prompts[0]
    assert payloads and payloads[0] is not None
    assert payloads[0]["changed_paths"] == ["notes.txt"]


def test_tool_consent_prompt_includes_non_text_change_summary(tmp_path: Path) -> None:
    target = tmp_path / "image.bin"
    target.write_bytes(b"\x00\x01before")

    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="editor", default="ask", remember=True)])
    prompts: list[str] = []
    payloads: list[dict[str, object] | None] = []

    def prompt(text: str, payload=None) -> str:
        prompts.append(text)
        payloads.append(payload)
        return "y"

    hook = ToolConsentHooks(safety, interactive=True, auto_approve=False, prompt=prompt)
    event = SimpleNamespace(
        tool_use={
            "name": "editor",
            "input": {"command": "write", "path": "image.bin", "file_text": "after\n", "cwd": str(tmp_path)},
        },
        invocation_state={},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool is False
    assert "Non-text changes:" in prompts[0]
    assert "image.bin: binary -> text" in prompts[0]
    assert payloads and payloads[0] is not None
    assert payloads[0]["non_text_change_summary"]


def test_tool_consent_blocks_unpreviewable_patch_apply() -> None:
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="patch_apply", default="ask", remember=True)])
    calls: list[str] = []

    def prompt(_text: str, _payload=None) -> str:
        calls.append("asked")
        return "y"

    hook = ToolConsentHooks(safety, interactive=True, auto_approve=False, prompt=prompt)
    event = SimpleNamespace(
        tool_use={"name": "patch_apply", "input": {"patch": "not a unified diff"}},
        invocation_state={},
        cancel_tool=False,
    )

    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "trustworthy pre-approval diff preview" in str(event.cancel_tool)
    assert calls == []


def test_tool_consent_invocation_state_auto_approve_overrides_non_interactive_for_shell():
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="shell", default="ask", remember=True)])
    hook = ToolConsentHooks(safety, interactive=False, auto_approve=False, prompt=lambda _text, _payload=None: "n")

    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"auto_approve": True}},
        cancel_tool=False,
    )
    hook.before_tool_call(event)

    assert event.cancel_tool is False


def test_tool_consent_invocation_state_auto_approve_overrides_non_interactive_for_patch() -> None:
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="patch_apply", default="ask", remember=True)])
    hook = ToolConsentHooks(safety, interactive=False, auto_approve=False, prompt=lambda _text, _payload=None: "n")

    event = SimpleNamespace(
        tool_use={"name": "patch_apply", "input": {"patch": "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n"}},
        invocation_state={"swarmee": {"auto_approve": True}},
        cancel_tool=False,
    )
    hook.before_tool_call(event)

    assert event.cancel_tool is False


def test_tool_consent_invocation_state_can_disable_session_auto_approve():
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="shell", default="ask", remember=True)])
    hook = ToolConsentHooks(safety, interactive=False, auto_approve=True, prompt=lambda _text, _payload=None: "y")

    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"auto_approve": False}},
        cancel_tool=False,
    )
    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "non-interactive" in str(event.cancel_tool)


def test_tool_consent_session_override_allow_short_circuits_prompt() -> None:
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="shell", default="ask", remember=True)])
    calls: list[str] = []

    def _prompt(_text: str, _payload=None) -> str:
        calls.append("asked")
        return "n"

    hook = ToolConsentHooks(safety, interactive=True, auto_approve=False, prompt=_prompt)
    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"session_safety_overrides": {"tool_consent": "allow"}}},
        cancel_tool=False,
    )
    hook.before_tool_call(event)

    assert event.cancel_tool is False
    assert calls == []


def test_tool_consent_session_override_deny_blocks_tool() -> None:
    safety = SafetyConfig(tool_consent="ask", tool_rules=[ToolRule(tool="shell", default="allow", remember=True)])
    hook = ToolConsentHooks(safety, interactive=True, auto_approve=True, prompt=lambda _text, _payload=None: "y")
    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"session_safety_overrides": {"tool_consent": "deny"}}},
        cancel_tool=False,
    )
    hook.before_tool_call(event)

    assert event.cancel_tool
    assert "session safety override" in str(event.cancel_tool)
