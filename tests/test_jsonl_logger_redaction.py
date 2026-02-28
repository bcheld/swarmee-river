from __future__ import annotations

import json
from pathlib import Path

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


def test_before_model_call_logs_context_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWARMEE_LOG_EVENTS", "true")
    monkeypatch.setenv("SWARMEE_LOG_REDACT", "false")
    monkeypatch.setenv("SWARMEE_LOG_DIR", str(tmp_path / "logs"))

    hook = JSONLLoggerHooks()

    class FakeEvent:
        def __init__(self) -> None:
            self.invocation_state: dict = {
                "swarmee": {
                    "invocation_id": "inv-1",
                    "model_t0": {},
                    "system_prompt_chars": 5000,
                    "tool_count": 20,
                    "tool_schema_chars": 15000,
                    "model_id": "gpt-4o",
                },
            }
            self.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "Thanks"},
            ]

    hook.before_model_call(FakeEvent())

    raw = hook._log_path.read_text(encoding="utf-8")  # noqa: SLF001
    data = json.loads(raw.splitlines()[-1])
    assert data["event"] == "before_model_call"
    assert data["messages"] == 4
    assert data["system_prompt_chars"] == 5000
    assert data["tool_count"] == 20
    assert data["tool_schema_chars"] == 15000
    assert data["model_id"] == "gpt-4o"
    assert data["message_breakdown"] == {"system": 1, "user": 2, "assistant": 1}


def test_after_model_call_logs_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWARMEE_LOG_EVENTS", "true")
    monkeypatch.setenv("SWARMEE_LOG_REDACT", "false")
    monkeypatch.setenv("SWARMEE_LOG_DIR", str(tmp_path / "logs"))

    hook = JSONLLoggerHooks()

    class FakeEvent:
        def __init__(self) -> None:
            self.invocation_state: dict = {
                "swarmee": {
                    "invocation_id": "inv-1",
                    "model_t0": {},
                    "model_id": "anthropic.claude-3-5-sonnet",
                },
                "swarmee_model_call_id": "call-1",
            }
            self.usage = {"input_tokens": 1000, "output_tokens": 300}
            self.response = {"role": "assistant", "content": "done"}

    hook.after_model_call(FakeEvent())

    raw = hook._log_path.read_text(encoding="utf-8")  # noqa: SLF001
    data = json.loads(raw.splitlines()[-1])
    assert data["event"] == "after_model_call"
    assert data["model_id"] == "anthropic.claude-3-5-sonnet"
    assert data["usage"]["input_tokens"] == 1000
