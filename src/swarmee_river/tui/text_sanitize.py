"""Pure output sanitization and plan extraction helpers for the TUI."""

from __future__ import annotations

import json as _json
import re
from typing import Any

_PATH_TOKEN_RE = re.compile(r"[A-Za-z]:\\[^\s,;]+|/(?:[^\s,;]+)|\./[^\s,;]+|\.\./[^\s,;]+")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC_ESCAPE_RE = re.compile(r"\x1b\][^\x1b\x07]*(?:\x07|\x1b\\)")


def _extract_paths_from_text(text: str) -> list[str]:
    return [match.strip() for match in _PATH_TOKEN_RE.findall(text) if match.strip()]


def sanitize_output_text(text: str) -> str:
    """Remove common control sequences that render poorly in a TUI transcript."""
    cleaned = text.replace("\r", "")
    cleaned = _OSC_ESCAPE_RE.sub("", cleaned)
    cleaned = _ANSI_ESCAPE_RE.sub("", cleaned)
    # Remove any stray ESC bytes left by malformed or partial sequences.
    return cleaned.replace("\x1b", "")


def extract_plan_section(output: str) -> str | None:
    """Extract the one-shot plan block beginning at 'Proposed plan:' if present."""
    marker = "Proposed plan:"
    marker_index = output.find(marker)
    if marker_index < 0:
        return None

    trailing_hint_prefix = "Plan generated. Re-run with --yes"
    candidate = output[marker_index:]
    extracted_lines: list[str] = []
    for line in candidate.splitlines():
        if not extracted_lines:
            local_index = line.find(marker)
            if local_index >= 0:
                line = line[local_index:]
        if line.strip().startswith(trailing_hint_prefix):
            break
        extracted_lines.append(line.rstrip())

    while extracted_lines and not extracted_lines[-1].strip():
        extracted_lines.pop()

    if not extracted_lines:
        return None

    extracted = "\n".join(extracted_lines).strip()
    return extracted or None


def looks_like_plan_output(text: str) -> bool:
    """Detect whether one-shot output likely contains a generated plan."""
    return extract_plan_section(text) is not None


def render_tui_hint_after_plan() -> str:
    """Hint shown when a plan-only run is detected."""
    return "Plan detected. Use Run > Plan controls to review, refine, or continue."


def _parse_tui_event_line(line: str) -> dict[str, Any] | None:
    stripped = sanitize_output_text(line).strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = _json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, _json.JSONDecodeError):
        return None


def extract_plan_section_from_output(output: str) -> str | None:
    """Extract plan text from mixed output by ignoring structured JSONL event lines."""
    plain_lines = [line for line in output.splitlines() if _parse_tui_event_line(line) is None]
    if not plain_lines:
        return None
    return extract_plan_section("\n".join(plain_lines))


__all__ = [
    "_extract_paths_from_text",
    "extract_plan_section",
    "extract_plan_section_from_output",
    "looks_like_plan_output",
    "render_tui_hint_after_plan",
    "sanitize_output_text",
]
