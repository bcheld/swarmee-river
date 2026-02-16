from __future__ import annotations

import os
import re
from typing import Any, Optional

from strands import tool

_SENSITIVE_KEY_RE = re.compile(
    r"(?:"
    r"secret|token|password|passwd|private|credential|session|cookie|"
    r"access[_-]?key|api[_-]?key"
    r")",
    re.IGNORECASE,
)


def _is_sensitive_key(key: str) -> bool:
    return bool(_SENSITIVE_KEY_RE.search(key or ""))


def _format_value(key: str, value: str, *, redact: bool) -> str:
    if redact and _is_sensitive_key(key):
        return "<redacted>"
    return value


@tool
def environment(
    *,
    action: Optional[str] = None,
    command: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[str] = None,
    keys: Optional[list[str]] = None,
    prefix: Optional[str] = None,
    reveal: bool = False,
    redact: bool = True,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Cross-platform fallback for `strands_tools.environment`.

    Supported actions (aliases accepted):
    - list: list env var names (optionally `reveal=True` to include values)
    - get: get a value (redacted by default for sensitive keys)
    - set: set a value
    - unset: remove a key
    - export/dump: render `export KEY=value` lines (values redacted by default)
    """
    op = (action or command or "list").strip().lower()

    def _truncate(text: str) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n… (truncated to {max_chars} chars) …"

    if op in {"list", "ls"}:
        selected = sorted(os.environ.keys())
        if prefix:
            selected = [k for k in selected if k.startswith(prefix)]
        if keys:
            want = {str(k) for k in keys if str(k).strip()}
            selected = [k for k in selected if k in want]

        if not reveal:
            text = "\n".join(selected) if selected else "(no environment variables)"
            return {"status": "success", "content": [{"text": _truncate(text)}]}

        lines = [f"{k}={_format_value(k, os.environ.get(k, ''), redact=redact)}" for k in selected]
        text = "\n".join(lines) if lines else "(no environment variables)"
        return {"status": "success", "content": [{"text": _truncate(text)}]}

    if op in {"export", "dump"}:
        selected = sorted(os.environ.keys())
        if prefix:
            selected = [k for k in selected if k.startswith(prefix)]
        if keys:
            want = {str(k) for k in keys if str(k).strip()}
            selected = [k for k in selected if k in want]
        lines = [f"export {k}={_format_value(k, os.environ.get(k, ''), redact=redact)}" for k in selected]
        text = "\n".join(lines) if lines else "(no environment variables)"
        return {"status": "success", "content": [{"text": _truncate(text)}]}

    if op == "get":
        k = (key or "").strip()
        if not k:
            return {"status": "error", "content": [{"text": "key is required for get"}]}
        if k not in os.environ:
            return {"status": "success", "content": [{"text": "(not set)"}]}
        v = os.environ.get(k, "")
        return {"status": "success", "content": [{"text": _truncate(_format_value(k, v, redact=redact))}]}

    if op == "set":
        k = (key or "").strip()
        if not k:
            return {"status": "error", "content": [{"text": "key is required for set"}]}
        if value is None:
            return {"status": "error", "content": [{"text": "value is required for set"}]}
        if "\x00" in k or "\n" in k or "\r" in k:
            return {"status": "error", "content": [{"text": "invalid key"}]}
        os.environ[k] = str(value)
        shown = _format_value(k, os.environ.get(k, ""), redact=redact)
        return {"status": "success", "content": [{"text": f"Set {k}={shown}"}]}

    if op in {"unset", "del", "delete", "remove"}:
        k = (key or "").strip()
        if not k:
            return {"status": "error", "content": [{"text": "key is required for unset"}]}
        existed = k in os.environ
        os.environ.pop(k, None)
        return {
            "status": "success",
            "content": [{"text": f"Unset {k}" if existed else f"{k} was not set"}],
        }

    return {
        "status": "error",
        "content": [
            {
                "text": "Unsupported action. Use one of: list, get, set, unset, export/dump "
                "(action=... or command=...)."
            }
        ],
    }

