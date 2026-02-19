from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarmee_river.hooks.jsonl_logger import JSONLLoggerHooks


def test_jsonl_logger_redacts_known_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWARMEE_LOG_EVENTS", "true")
    monkeypatch.setenv("SWARMEE_LOG_REDACT", "true")
    monkeypatch.setenv("SWARMEE_LOG_DIR", str(tmp_path / "logs"))

    secret = "sk-test-secret-12345678901234567890"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    hook = JSONLLoggerHooks()
    hook._log("test", {"value": f"prefix {secret} suffix"})  # noqa: SLF001

    raw = hook._log_path.read_text(encoding="utf-8")  # noqa: SLF001
    assert secret not in raw
    data = json.loads(raw.splitlines()[-1])
    assert "<redacted>" in str(data)


def test_jsonl_logger_emits_tui_events_for_tool_and_final_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWARMEE_LOG_EVENTS", "true")
    monkeypatch.setenv("SWARMEE_LOG_REDACT", "false")
    monkeypatch.setenv("SWARMEE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SWARMEE_TUI_EVENTS", "true")

    hook = JSONLLoggerHooks()
    invocation_state = {"swarmee": {"invocation_id": "inv-1", "tool_t0": {}, "model_t0": {}}}

    before_tool = SimpleNamespace(
        invocation_state=invocation_state,
        tool_use={"toolUseId": "tool-1", "name": "shell", "input": {"command": "echo hi"}},
    )
    after_tool = SimpleNamespace(
        invocation_state=invocation_state,
        tool_use={"toolUseId": "tool-1", "name": "shell"},
        result={"status": "success", "toolUseId": "tool-1"},
    )
    after_invocation = SimpleNamespace(invocation_state=invocation_state, result="final answer")

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        hook.before_tool_call(before_tool)
        hook.after_tool_call(after_tool)
        hook.after_invocation(after_invocation)
    finally:
        sys.stdout = old_stdout

    events = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    assert any(event.get("event") == "tool_start" and event.get("tool_use_id") == "tool-1" for event in events)
    assert any(event.get("event") == "tool_result" and event.get("tool_use_id") == "tool-1" for event in events)
    assert any(event.get("event") == "final_result" and event.get("text") == "final answer" for event in events)


def test_jsonl_logger_emits_thinking_event_before_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWARMEE_LOG_EVENTS", "true")
    monkeypatch.setenv("SWARMEE_LOG_REDACT", "false")
    monkeypatch.setenv("SWARMEE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SWARMEE_TUI_EVENTS", "true")

    hook = JSONLLoggerHooks()
    invocation_state = {"swarmee": {"invocation_id": "inv-1", "tool_t0": {}, "model_t0": {}}}
    before_model = SimpleNamespace(invocation_state=invocation_state, messages=[])

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        hook.before_model_call(before_model)
    finally:
        sys.stdout = old_stdout

    events = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    assert any(event.get("event") == "thinking" for event in events)
