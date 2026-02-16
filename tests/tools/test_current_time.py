from __future__ import annotations

from tools.current_time import current_time


def _text(result: dict) -> str:
    return (result.get("content") or [{"text": ""}])[0].get("text", "")


def test_current_time_returns_iso_timestamp() -> None:
    result = current_time()
    assert result.get("status") == "success"
    text = _text(result)
    assert "T" in text
    assert text.endswith("+00:00") or text.endswith("Z")

