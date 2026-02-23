from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return an RFC3339-style UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_error_event(code: str, message: str, *, detail: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"event": "error", "code": code, "message": message}
    if isinstance(detail, str) and detail.strip():
        payload["detail"] = detail.strip()
    return payload


def encode_jsonl(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"


def parse_jsonl_command(raw_line: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Parse a single JSONL command line.

    Returns:
        (command, error_event). Exactly one of these is non-None for non-empty input.
    """
    text = raw_line.strip()
    if not text:
        return None, None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, make_error_event("invalid_json", "Invalid JSON command", detail=str(exc))

    if not isinstance(parsed, dict):
        return None, make_error_event("invalid_payload", "Command payload must be a JSON object")

    return parsed, None
