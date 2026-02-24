"""Pure session-sidebar domain helpers for the TUI."""

from __future__ import annotations

import json as _json
from typing import Any


def build_session_issue_sidebar_items(issues: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build SidebarList payloads from structured session issues."""

    def _truncate_text(value: str, *, max_chars: int = 88) -> str:
        text = value.strip().replace("\n", " ")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"

    items: list[dict[str, str]] = []
    for issue in issues:
        issue_id = str(issue.get("id", "")).strip()
        if not issue_id:
            continue
        severity = str(issue.get("severity", "warning")).strip().lower()
        if severity not in {"warning", "error"}:
            severity = "warning"
        title = str(issue.get("title", "")).strip() or "Issue"
        created_at = str(issue.get("created_at", "")).strip()
        text = str(issue.get("text", "")).strip()
        subtitle_parts = []
        if created_at:
            subtitle_parts.append(created_at)
        if text:
            subtitle_parts.append(_truncate_text(text, max_chars=88))
        subtitle = " | ".join(subtitle_parts)
        items.append(
            {
                "id": issue_id,
                "title": title,
                "subtitle": subtitle,
                "state": "error" if severity == "error" else "warning",
            }
        )
    return items


def render_session_issue_detail_text(issue: dict[str, Any] | None) -> str:
    """Render a detail panel body for a selected session issue."""
    if not isinstance(issue, dict):
        return "(no issue selected)"
    lines = [
        f"Severity: {str(issue.get('severity', 'warning')).strip() or 'warning'}",
        f"Title: {str(issue.get('title', 'Issue')).strip() or 'Issue'}",
        f"When: {str(issue.get('created_at', '')).strip() or '(unknown)'}",
        "",
        str(issue.get("text", "")).strip() or "(no details)",
    ]
    tool_use_id = str(issue.get("tool_use_id", "")).strip()
    if tool_use_id:
        lines.append("")
        lines.append(f"Tool Use ID: {tool_use_id}")
    tool_name = str(issue.get("tool_name", "")).strip()
    if tool_name:
        lines.append(f"Tool: {tool_name}")
    next_tier = str(issue.get("next_tier", "")).strip()
    if next_tier:
        lines.append(f"Suggested tier: {next_tier}")
    return "\n".join(lines)


def session_issue_actions(issue: dict[str, Any] | None) -> list[dict[str, str]]:
    """Return available action buttons for a selected session issue."""
    if not isinstance(issue, dict):
        return []
    category = str(issue.get("category", "")).strip().lower()
    tool_use_id = str(issue.get("tool_use_id", "")).strip()
    actions: list[dict[str, str]] = []
    if category == "tool_failure" and tool_use_id:
        actions.append({"id": "session_issue_retry_tool", "label": "Retry", "variant": "default"})
        actions.append({"id": "session_issue_skip_tool", "label": "Skip", "variant": "default"})
        actions.append({"id": "session_issue_escalate_tier", "label": "Escalate", "variant": "default"})
        actions.append({"id": "session_issue_interrupt", "label": "Interrupt", "variant": "default"})
    return actions


def normalize_session_view_mode(mode: str | None) -> str:
    """Normalize session panel mode for Timeline/Issues toggle."""
    normalized = str(mode or "").strip().lower()
    if normalized == "issues":
        return "issues"
    return "timeline"


def classify_session_timeline_event_kind(event: dict[str, Any] | None) -> str:
    """Classify timeline event kind used for badges/icons."""
    if not isinstance(event, dict):
        return "event"
    name = str(event.get("event", "")).strip().lower()
    has_error = bool(str(event.get("error", "")).strip())
    if name == "after_tool_call":
        if has_error or event.get("success") is False:
            return "error"
        return "tool"
    if name == "after_model_call":
        return "model"
    if name == "after_invocation":
        return "invocation"
    if has_error:
        return "error"
    return "event"


def summarize_session_timeline_event(event: dict[str, Any] | None) -> str:
    """Render compact one-line timeline summary."""
    if not isinstance(event, dict):
        return "event"
    kind = classify_session_timeline_event_kind(event)
    duration = event.get("duration_s")
    duration_text = ""
    if isinstance(duration, (int, float)):
        duration_text = f" ({float(duration):.1f}s)"
    if kind in {"tool", "error"}:
        tool = str(event.get("tool", "")).strip() or "unknown"
        label = f"tool: {tool}{duration_text}"
        if kind == "error":
            return f"{label} error"
        return label
    if kind == "model":
        return f"model call{duration_text}"
    if kind == "invocation":
        return f"invocation{duration_text}"
    return (str(event.get("event", "")).strip() or "event") + duration_text


def build_session_timeline_sidebar_items(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build SidebarList payload for session timeline events."""
    icon_map = {
        "tool": "⚙",
        "model": "◉",
        "invocation": "▶",
        "error": "✖",
        "event": "•",
    }
    state_map = {
        "tool": "default",
        "model": "default",
        "invocation": "active",
        "error": "error",
        "event": "default",
    }
    items: list[dict[str, str]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", "")).strip() or f"timeline-{index + 1}"
        kind = classify_session_timeline_event_kind(event)
        summary = summarize_session_timeline_event(event)
        ts = str(event.get("ts", "")).strip()
        label = str(event.get("event", "")).strip().lower()
        subtitle = ts if ts else label
        if ts and label:
            subtitle = f"{ts} | {label}"
        items.append(
            {
                "id": event_id,
                "title": f"{icon_map.get(kind, '•')} {summary}",
                "subtitle": subtitle,
                "state": state_map.get(kind, "default"),
            }
        )
    return items


def render_session_timeline_detail_text(event: dict[str, Any] | None) -> str:
    """Render detail body for selected timeline event."""
    if not isinstance(event, dict):
        return "(no timeline event selected)"
    payload = dict(event)
    payload.pop("id", None)
    summary = summarize_session_timeline_event(event)
    try:
        rendered = _json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = str(payload)
    return f"Summary: {summary}\n\nPayload:\n{rendered}"


def session_timeline_actions(event: dict[str, Any] | None) -> list[dict[str, str]]:
    """Actions available for selected timeline event."""
    if not isinstance(event, dict):
        return []
    return [
        {"id": "session_timeline_copy_json", "label": "Copy JSON", "variant": "default"},
        {"id": "session_timeline_copy_summary", "label": "Copy summary", "variant": "default"},
    ]


__all__ = [
    "build_session_issue_sidebar_items",
    "build_session_timeline_sidebar_items",
    "classify_session_timeline_event_kind",
    "normalize_session_view_mode",
    "render_session_issue_detail_text",
    "render_session_timeline_detail_text",
    "session_issue_actions",
    "session_timeline_actions",
    "summarize_session_timeline_event",
]
