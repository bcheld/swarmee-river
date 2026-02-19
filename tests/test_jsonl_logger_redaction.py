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
