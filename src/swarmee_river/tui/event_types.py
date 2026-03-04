"""Parsing/types helpers for TUI daemon output events."""

from __future__ import annotations

import json as _json
import os
import re
from dataclasses import dataclass
from typing import Any

from swarmee_river.diagnostics import session_issues_path
from swarmee_river.tui.text_sanitize import _extract_paths_from_text, sanitize_output_text

_TRUNCATED_ARTIFACT_RE = re.compile(r"full output saved to (?P<path>[^\]]+)")
_PROVIDER_NOISE_PREFIX = "[provider]"
_PROVIDER_FALLBACK_PHRASE = "falling back to"
_TRACEBACK_FILE_LINE_RE = re.compile(r'^File ".*", line \d+')
_BEDROCK_CHUNK_KEYS = {
    "messageStart",
    "messageStop",
    "contentBlockStart",
    "contentBlockDelta",
    "contentBlockStop",
    "metadata",
}
_NON_TEXT_EVENT_TOKENS = {
    "after_invocation",
    "after_model_call",
    "after_tool_call",
    "before_model_call",
    "complete",
    "delta",
    "llm_start",
    "message_complete",
    "message_delta",
    "model_start",
    "output_text_complete",
    "output_text_delta",
    "text_complete",
    "text_delta",
    "thinking",
}


@dataclass(frozen=True)
class ParsedEvent:
    kind: str
    text: str
    meta: dict[str, str] | None = None


def parse_output_line(line: str) -> ParsedEvent | None:
    """Best-effort parser for notable subprocess output events."""
    text = line.rstrip("\n")
    stripped = text.strip()
    lower = stripped.lower()

    if stripped.startswith(_PROVIDER_NOISE_PREFIX) and _PROVIDER_FALLBACK_PHRASE in lower:
        return ParsedEvent(kind="noise", text=text)

    if "~ consent>" in lower:
        return ParsedEvent(kind="consent_prompt", text=text)

    if stripped.startswith("Proposed plan:"):
        return ParsedEvent(kind="plan_header", text=text)

    if "[tool result truncated:" in lower:
        match = _TRUNCATED_ARTIFACT_RE.search(text)
        if match:
            path = match.group("path").strip()
            return ParsedEvent(kind="artifact", text=text, meta={"path": path})
        return ParsedEvent(kind="tool_truncated", text=text)

    if lower.startswith("patch:"):
        path = stripped.split(":", 1)[1].strip()
        if path:
            return ParsedEvent(kind="artifact", text=text, meta={"path": path})
        return ParsedEvent(kind="patch", text=text)

    if lower.startswith("backups:"):
        rest = stripped.split(":", 1)[1].strip()
        paths = _extract_paths_from_text(rest)
        if paths:
            return ParsedEvent(kind="artifact", text=text, meta={"paths": ",".join(paths)})
        return ParsedEvent(kind="backups", text=text)

    if stripped.startswith("Error:") or stripped.startswith("ERROR:"):
        return ParsedEvent(kind="error", text=text)

    if stripped.startswith("Traceback (most recent call last):"):
        details_path = session_issues_path(os.getenv("SWARMEE_SESSION_ID") or None)
        return ParsedEvent(
            kind="warning",
            text=f"WARN: daemon emitted a Python traceback (details in {details_path}).",
        )

    if _TRACEBACK_FILE_LINE_RE.match(stripped):
        return ParsedEvent(kind="noise", text=text)

    if "_watchdog_fsevents.py" in lower or "_fsevents.add_watch" in lower:
        return ParsedEvent(kind="noise", text=text)

    if "cannot start fsevents stream" in lower:
        return ParsedEvent(
            kind="warning",
            text="WARN: filesystem watcher fsevents backend unavailable; continuing without it.",
        )

    if "operation not permitted" in lower and "/bin/ps" in lower:
        return ParsedEvent(kind="warning", text=text)

    if "warning" in lower or "deprecationwarning" in lower or "runtimewarning" in lower or "userwarning" in lower:
        return ParsedEvent(kind="warning", text=text)

    if lower.startswith("[tool ") and any(token in lower for token in {" start", " started", " running"}):
        return ParsedEvent(kind="tool_start", text=text)

    if lower.startswith("[tool ") and any(token in lower for token in {" done", " end", " finished", " stopped"}):
        return ParsedEvent(kind="tool_stop", text=text)

    return None


def parse_tui_event(line: str) -> dict[str, Any] | None:
    """Parse a JSONL event line emitted by TuiCallbackHandler. Returns None for non-JSON lines."""
    stripped = sanitize_output_text(line).strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = _json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, _json.JSONDecodeError):
        return None


def extract_tui_text_chunk(event: dict[str, Any]) -> str:
    """Extract a text chunk from a structured TUI event payload."""
    extracted = _extract_text_value(event)
    if isinstance(extracted, str):
        return extracted
    return ""


def _extract_text_from_delta_payload(payload: dict[str, Any]) -> str:
    for key in ("text", "data", "output_text", "outputText", "textDelta"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_text_from_bedrock_payload(payload: Any) -> str:
    if isinstance(payload, list):
        for item in payload:
            extracted = _extract_text_from_bedrock_payload(item)
            if extracted:
                return extracted
        return ""
    if not isinstance(payload, dict):
        return ""

    for key in ("contentBlockDelta", "content_block_delta"):
        block = payload.get(key)
        if not isinstance(block, dict):
            continue
        delta_payload = block.get("delta")
        if isinstance(delta_payload, dict):
            extracted = _extract_text_from_delta_payload(delta_payload)
            if extracted:
                return extracted
        if isinstance(delta_payload, str) and delta_payload:
            return delta_payload
        extracted = _extract_text_from_delta_payload(block)
        if extracted:
            return extracted

    for key in ("delta", "text_delta", "content_delta"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            extracted = _extract_text_from_delta_payload(nested)
            if extracted:
                return extracted
    return ""


def _extract_text_value(payload: Any, *, depth: int = 5) -> str:
    if depth < 0:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        for item in payload:
            extracted = _extract_text_value(item, depth=depth - 1)
            if extracted:
                return extracted
        return ""
    if not isinstance(payload, dict):
        return ""

    bedrock = _extract_text_from_bedrock_payload(payload)
    if bedrock:
        return bedrock

    for key in ("data", "text", "delta", "content", "output_text", "outputText", "textDelta"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value

    for key in ("delta", "text_delta", "content_delta"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            extracted = _extract_text_from_delta_payload(nested)
            if extracted:
                return extracted

    nested_priority: list[Any] = []
    for marker in _BEDROCK_CHUNK_KEYS:
        if marker in payload:
            nested_priority.append(payload.get(marker))
    for key in ("message", "event", "chunk", "content", "response", "payload"):
        if key in payload:
            value = payload.get(key)
            if key in {"event", "type", "kind"} and isinstance(value, str):
                if value.strip().lower() in _NON_TEXT_EVENT_TOKENS:
                    continue
            nested_priority.append(value)

    for nested in nested_priority:
        extracted = _extract_text_value(nested, depth=depth - 1)
        if extracted:
            return extracted

    for key, nested in payload.items():
        if key in {"event", "type", "kind"} and isinstance(nested, str):
            if nested.strip().lower() in _NON_TEXT_EVENT_TOKENS:
                continue
        extracted = _extract_text_value(nested, depth=depth - 1)
        if extracted:
            return extracted
    return ""


def detect_consent_prompt(line: str) -> str | None:
    """Detect consent-related subprocess output lines."""
    normalized = line.strip().lower()
    if "~ consent>" in normalized:
        return "prompt"
    if "allow tool '" in normalized:
        return "header"
    return None


def update_consent_capture(
    consent_active: bool,
    consent_buffer: list[str],
    line: str,
    *,
    max_lines: int = 20,
) -> tuple[bool, list[str]]:
    """Update consent capture state from a single output line."""
    kind = detect_consent_prompt(line)
    if kind is None and not consent_active:
        return consent_active, consent_buffer

    updated = list(consent_buffer)
    updated.append(line.rstrip("\n"))
    if len(updated) > max_lines:
        updated = updated[-max_lines:]
    return True, updated


__all__ = [
    "ParsedEvent",
    "detect_consent_prompt",
    "extract_tui_text_chunk",
    "parse_output_line",
    "parse_tui_event",
    "update_consent_capture",
]
