#!/usr/bin/env python3
"""
Tests for TUI subprocess helpers.
"""

from __future__ import annotations

import sys

from swarmee_river.tui import app as tui_app


def test_build_swarmee_cmd_run_mode():
    prompt = "show git status"
    command = tui_app.build_swarmee_cmd(prompt, auto_approve=True)
    assert command == [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--yes", prompt]


def test_build_swarmee_cmd_plan_mode():
    prompt = "show git status"
    command = tui_app.build_swarmee_cmd(prompt, auto_approve=False)
    assert command == [sys.executable, "-u", "-m", "swarmee_river.swarmee", prompt]


def test_build_plan_mode_prompt_wraps_user_request():
    wrapped = tui_app.build_plan_mode_prompt("  fix the failing tests  ")
    assert wrapped.startswith("Create a concrete work plan")
    assert wrapped.endswith("Request:\nfix the failing tests")


def test_model_select_options_only_includes_configured_provider_tiers(monkeypatch):
    class _Tier:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

    class _Provider:
        def __init__(self, tiers: dict[str, _Tier]) -> None:
            self.tiers = tiers

    class _Models:
        def __init__(self) -> None:
            self.provider = "openai"
            self.default_tier = "balanced"
            self.providers = {
                "openai": _Provider({"balanced": _Tier("gpt-5-mini")}),
                "bedrock": _Provider({"balanced": _Tier("anthropic.claude-sonnet")}),
            }
            self.tiers = {"balanced": _Tier("global-balanced"), "fast": _Tier("global-fast")}

    class _Settings:
        def __init__(self) -> None:
            self.models = _Models()

    import swarmee_river.settings as settings_module
    import swarmee_river.utils.provider_utils as provider_utils_module

    monkeypatch.setattr(settings_module, "load_settings", lambda: _Settings())
    monkeypatch.setattr(
        provider_utils_module,
        "resolve_model_provider",
        lambda **_kwargs: ("openai", None),
    )

    options, selected = tui_app.model_select_options()
    values = [value for _label, value in options]

    assert "__auto__" in values
    assert "openai|balanced" in values
    assert "openai|fast" not in values
    assert "bedrock|balanced" not in values
    assert selected == "__auto__"


def test_looks_like_plan_output():
    text = "Some output\nProposed plan:\n- Step 1\n- Step 2\nPlan generated. Re-run with --yes to execute."
    assert tui_app.looks_like_plan_output(text) is True
    assert tui_app.looks_like_plan_output("normal run output") is False


def test_extract_plan_section_returns_only_plan_block():
    output = (
        "preface\n"
        "Proposed plan:\n"
        "- Summary: Fix issue\n"
        "- Steps:\n"
        "  1. Reproduce\n"
        "\n"
        "Plan generated. Re-run with --yes (or set SWARMEE_AUTO_APPROVE=true) to execute.\n"
        "postface\n"
    )
    extracted = tui_app.extract_plan_section(output)
    assert extracted == "Proposed plan:\n- Summary: Fix issue\n- Steps:\n  1. Reproduce"


def test_extract_plan_section_returns_none_when_missing_marker():
    assert tui_app.extract_plan_section("plain execution output") is None


def test_extract_plan_section_from_output_ignores_jsonl_event_lines():
    output = '{"event":"plan","rendered":"Proposed plan:\\n- Summary: Json only"}\n'
    assert tui_app.extract_plan_section_from_output(output) is None


def test_extract_plan_section_from_output_extracts_plain_plan_lines():
    output = (
        '{"event":"thinking","text":"planning"}\n'
        "Proposed plan:\n"
        "- Summary: Fix issue\n"
        "- Steps:\n"
        "  1. Reproduce\n"
        "Plan generated. Re-run with --yes (or set SWARMEE_AUTO_APPROVE=true) to execute.\n"
    )
    extracted = tui_app.extract_plan_section_from_output(output)
    assert extracted == "Proposed plan:\n- Summary: Fix issue\n- Steps:\n  1. Reproduce"


def test_is_multiline_newline_key_detects_shift_enter():
    class _Event:
        key = "shift+enter"
        character = None
        aliases: list[str] = []

    assert tui_app.is_multiline_newline_key(_Event()) is True


def test_is_multiline_newline_key_detects_newline_alias():
    class _Event:
        key = "ctrl+j"
        character = "\n"
        aliases = ["ctrl+j", "newline"]

    assert tui_app.is_multiline_newline_key(_Event()) is True


def test_is_multiline_newline_key_rejects_plain_enter():
    class _Event:
        key = "enter"
        character = "\r"
        aliases = ["enter", "ctrl+m"]

    assert tui_app.is_multiline_newline_key(_Event()) is False


def test_is_multiline_newline_key_detects_shift_ctrl_m_variant():
    class _Event:
        key = "shift+ctrl+m"
        name = "shift_ctrl_m"
        character = "\r"
        aliases = ["shift+ctrl+m"]

    assert tui_app.is_multiline_newline_key(_Event()) is True


def test_is_multiline_newline_key_detects_shift_enter_alias():
    class _Event:
        key = "enter"
        name = "enter"
        character = "\r"
        aliases = ["enter", "ctrl+m", "shift+enter"]

    assert tui_app.is_multiline_newline_key(_Event()) is True


def test_spawn_swarmee_configures_subprocess_run_mode(monkeypatch):
    captured: dict[str, object] = {}
    fake_proc = object()

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    proc = tui_app.spawn_swarmee("hello from tui", auto_approve=True)

    assert proc is fake_proc
    command = captured["command"]
    kwargs = captured["kwargs"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1:4] == ["-u", "-m", "swarmee_river.swarmee"]
    assert "--yes" in command
    assert "hello from tui" in command
    assert kwargs["stdin"] is tui_app.subprocess.PIPE
    assert kwargs["stdout"] is tui_app.subprocess.PIPE
    assert kwargs["stderr"] is tui_app.subprocess.STDOUT
    assert kwargs["text"] is True
    assert kwargs["errors"] == "replace"
    assert kwargs["bufsize"] == 1
    env = kwargs["env"]
    assert isinstance(env, dict)
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["SWARMEE_SPINNERS"] == "0"
    assert "PYTHONWARNINGS" in env
    assert "Http_requestTool" in str(env["PYTHONWARNINGS"])


def test_spawn_swarmee_configures_subprocess_plan_mode(monkeypatch):
    captured: dict[str, object] = {}
    fake_proc = object()

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    proc = tui_app.spawn_swarmee("hello from tui", auto_approve=False)

    assert proc is fake_proc
    command = captured["command"]
    kwargs = captured["kwargs"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1:4] == ["-u", "-m", "swarmee_river.swarmee"]
    assert "--yes" not in command
    assert "hello from tui" in command
    assert kwargs["stdin"] is tui_app.subprocess.PIPE
    assert kwargs["stderr"] is tui_app.subprocess.STDOUT
    assert kwargs["errors"] == "replace"
    env = kwargs["env"]
    assert isinstance(env, dict)
    assert env["SWARMEE_SPINNERS"] == "0"
    assert "PYTHONWARNINGS" in env
    assert "Http_requestTool" in str(env["PYTHONWARNINGS"])


def test_spawn_swarmee_sets_session_id_env_when_provided(monkeypatch):
    captured: dict[str, object] = {}
    fake_proc = object()

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    proc = tui_app.spawn_swarmee("hello", auto_approve=False, session_id="abc123")

    assert proc is fake_proc
    env = captured["kwargs"]["env"]
    assert isinstance(env, dict)
    assert env["SWARMEE_SESSION_ID"] == "abc123"


def test_spawn_swarmee_applies_env_overrides(monkeypatch):
    captured: dict[str, object] = {}
    fake_proc = object()

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    proc = tui_app.spawn_swarmee(
        "hello",
        auto_approve=False,
        env_overrides={"SWARMEE_MODEL_PROVIDER": "openai", "SWARMEE_MODEL_TIER": "fast"},
    )

    assert proc is fake_proc
    env = captured["kwargs"]["env"]
    assert isinstance(env, dict)
    assert env["SWARMEE_MODEL_PROVIDER"] == "openai"
    assert env["SWARMEE_MODEL_TIER"] == "fast"


def test_write_to_proc_writes_newline_and_flushes():
    class FakeStdin:
        def __init__(self) -> None:
            self.payload = ""
            self.flush_calls = 0

        def write(self, text: str) -> None:
            self.payload += text

        def flush(self) -> None:
            self.flush_calls += 1

    class FakeProc:
        def __init__(self) -> None:
            self.stdin = FakeStdin()

    proc = FakeProc()
    assert tui_app.write_to_proc(proc, "y") is True
    assert proc.stdin.payload == "y\n"
    assert proc.stdin.flush_calls == 1


def test_write_to_proc_returns_false_when_stdin_missing():
    class FakeProc:
        stdin = None

    assert tui_app.write_to_proc(FakeProc(), "n") is False


def test_detect_consent_prompt_matches_cli_prompt():
    assert tui_app.detect_consent_prompt("~ consent> ") is not None
    assert tui_app.detect_consent_prompt("Allow tool 'shell'? [y/n]") is not None
    assert tui_app.detect_consent_prompt("normal output line") is None


def test_parse_output_line_extracts_artifact_from_truncation_line():
    line = "[tool result truncated: kept 2000 chars of 9000; full output saved to .swarmee/artifacts/abc123.txt]"
    event = tui_app.parse_output_line(line)
    assert event is not None
    assert event.kind == "artifact"
    assert event.meta == {"path": ".swarmee/artifacts/abc123.txt"}


def test_parse_output_line_extracts_patch_path():
    event = tui_app.parse_output_line("patch: /tmp/swarmee/patches/plan.patch")
    assert event is not None
    assert event.kind == "artifact"
    assert event.meta == {"path": "/tmp/swarmee/patches/plan.patch"}


def test_parse_output_line_returns_none_for_unmatched_line():
    assert tui_app.parse_output_line("regular assistant output") is None


def test_parse_output_line_classifies_provider_fallback_as_noise():
    event = tui_app.parse_output_line(
        "[provider] No AWS credentials detected for Bedrock; falling back to OpenAI because OPENAI_API_KEY is set."
    )
    assert event is not None
    assert event.kind == "noise"


def test_parse_output_line_classifies_homebrew_ps_warning():
    event = tui_app.parse_output_line(
        "/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted"
    )
    assert event is not None
    assert event.kind == "warning"


def test_sanitize_output_text_strips_ansi_and_carriage_returns():
    text = "hello\r\n\x1b[31mred\x1b[0m"
    assert tui_app.sanitize_output_text(text) == "hello\nred"


def test_update_consent_capture_collects_recent_lines():
    consent_active = False
    consent_buffer: list[str] = []
    lines = [
        "some normal line",
        "Allow tool 'shell'? [y/n/a/v]",
        "context line 1",
        "~ consent> ",
        "context line 2",
    ]
    for line in lines:
        consent_active, consent_buffer = tui_app.update_consent_capture(
            consent_active,
            consent_buffer,
            line,
            max_lines=3,
        )

    assert consent_active is True
    assert consent_buffer == ["context line 1", "~ consent> ", "context line 2"]


def test_add_recent_artifacts_dedupes_and_caps():
    existing = ["a.txt", "b.txt", "c.txt"]
    updated = tui_app.add_recent_artifacts(existing, ["b.txt", "d.txt", "e.txt"], max_items=4)
    assert updated == ["c.txt", "b.txt", "d.txt", "e.txt"]


# ---------------------------------------------------------------------------
# parse_tui_event tests
# ---------------------------------------------------------------------------


def test_parse_tui_event_valid_json():
    result = tui_app.parse_tui_event('{"event":"text_delta","data":"hello"}')
    assert result == {"event": "text_delta", "data": "hello"}


def test_parse_tui_event_non_json_returns_none():
    assert tui_app.parse_tui_event("plain text output") is None
    assert tui_app.parse_tui_event("") is None
    assert tui_app.parse_tui_event("Error: something failed") is None


def test_parse_tui_event_malformed_json_returns_none():
    assert tui_app.parse_tui_event('{"event": "text_delta"') is None
    assert tui_app.parse_tui_event("{bad json}") is None


def test_parse_tui_event_rejects_non_object_json():
    assert tui_app.parse_tui_event("[]") is None
    assert tui_app.parse_tui_event('"event"') is None


def test_parse_tui_event_strips_whitespace():
    result = tui_app.parse_tui_event('  {"event":"thinking","text":"hmm"}  ')
    assert result == {"event": "thinking", "text": "hmm"}


def test_spawn_swarmee_sets_tui_events_env(monkeypatch):
    """spawn_swarmee should set SWARMEE_TUI_EVENTS=1 in the subprocess env."""
    captured_env = {}

    def fake_popen(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))

        class FakeProc:
            pid = 12345
            stdin = None
            stdout = None
            stderr = None

            def poll(self):
                return 0

        return FakeProc()

    monkeypatch.setattr(tui_app.subprocess, "Popen", fake_popen)
    tui_app.spawn_swarmee("test prompt", auto_approve=True, session_id="abc123")
    assert captured_env.get("SWARMEE_TUI_EVENTS") == "1"
    assert captured_env.get("SWARMEE_SPINNERS") == "0"


def test_format_tool_input_shell():
    from swarmee_river.tui.widgets import _format_tool_input

    result = _format_tool_input("shell", {"command": "git status", "cwd": "/tmp"})
    assert "Command: git status" in result
    assert "CWD: /tmp" in result


def test_format_tool_input_generic():
    from swarmee_river.tui.widgets import _format_tool_input

    result = _format_tool_input("custom_tool", {"key": "value"})
    assert '"key"' in result
    assert '"value"' in result


def test_plan_card_renders_steps():
    from swarmee_river.tui.widgets import PlanCard

    card = PlanCard(plan_json={
        "summary": "Fix login",
        "steps": ["Read auth module", "Add validation", "Update tests"],
    })
    rendered = card._render_from_status()
    assert "Fix login" in rendered
    assert "1." in rendered
    assert "Read auth module" in rendered
    assert "/approve" in rendered


def test_plan_card_mark_step_complete():
    from swarmee_river.tui.widgets import PlanCard

    card = PlanCard(plan_json={
        "summary": "Test",
        "steps": ["Step A", "Step B"],
    })
    assert card._step_status == [False, False]
    card.mark_step_complete(0)
    assert card._step_status == [True, False]


def test_command_palette_filter():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/pl")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/plan"


def test_command_palette_filter_no_match():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/zzz")
    assert len(palette._filtered) == 0
    assert palette.get_selected() is None


def test_command_palette_move_selection():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/co")
    assert len(palette._filtered) == 6  # /copy, /copy plan, /copy issues, /copy last, /copy all, /consent
    palette.move_selection(1)
    assert palette._selected_index == 1


def test_assistant_message_accumulates_deltas():
    from swarmee_river.tui.widgets import AssistantMessage

    msg = AssistantMessage()
    msg._buffer.append("Hello ")
    msg._buffer.append("**world**")
    assert msg.full_text == "Hello **world**"
    assert msg.finalize() == "Hello **world**"


def test_status_bar_refresh_display():
    from swarmee_river.tui.widgets import StatusBar

    bar = StatusBar()
    # Access internal content set by update()
    def _get_text(widget):
        return widget._Static__content  # type: ignore[attr-defined]

    assert "idle" in _get_text(bar)

    bar.set_state("running")
    bar.set_model("Model: openai/balanced")
    bar.set_tool_count(3)
    bar.set_elapsed(12.4)
    text = _get_text(bar)
    assert "running" in text
    assert "Tools: 3" in text
    assert "12.4s" in text


def test_status_bar_counts():
    from swarmee_river.tui.widgets import StatusBar

    bar = StatusBar()
    bar.set_counts(warnings=2, errors=1)
    text = bar._Static__content  # type: ignore[attr-defined]
    assert "warn=2" in text
    assert "err=1" in text


def test_command_palette_includes_copy_last():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/copy l")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/copy last"


def test_command_palette_includes_open_and_search():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/op")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/open"

    palette.filter("/se")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/search"


def test_session_save_load(tmp_path, monkeypatch):
    """Session save/load round-trips prompt history and settings."""
    import swarmee_river.tui.app as app_mod

    monkeypatch.setattr(app_mod, "sessions_dir", lambda: tmp_path)

    session_file = tmp_path / "tui_session.json"
    import json

    data = {
        "prompt_history": ["hello", "world"],
        "last_prompt": "world",
        "plan_text": "my plan",
        "artifacts": ["/tmp/a.txt"],
        "model_provider_override": "openai",
        "model_tier_override": "fast",
        "default_auto_approve": True,
        "split_ratio": 3,
    }
    session_file.write_text(json.dumps(data))

    loaded = json.loads(session_file.read_text())
    assert loaded["prompt_history"] == ["hello", "world"]
    assert loaded["split_ratio"] == 3
    assert loaded["model_provider_override"] == "openai"


def test_stop_process_escalates(monkeypatch):
    class FakeProc:
        def __init__(self) -> None:
            self.send_signal_calls: list[int] = []
            self.terminate_called = False
            self.kill_called = False
            self.wait_calls = 0

        def poll(self):
            return None

        def send_signal(self, sig: int) -> None:
            self.send_signal_calls.append(sig)

        def terminate(self) -> None:
            self.terminate_called = True

        def kill(self) -> None:
            self.kill_called = True

        def wait(self, timeout: float):
            self.wait_calls += 1
            raise tui_app.subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

    proc = FakeProc()
    monkeypatch.setattr(tui_app.os, "name", "posix")

    tui_app.stop_process(proc, timeout_s=0.01)

    if hasattr(tui_app.signal, "SIGINT"):
        assert proc.send_signal_calls == [tui_app.signal.SIGINT]
    else:
        assert proc.send_signal_calls == []
    assert proc.terminate_called is True
    assert proc.kill_called is True
