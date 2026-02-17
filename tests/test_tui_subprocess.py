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
