"""Pure session-sidebar domain helpers for the TUI."""

from __future__ import annotations

import json as _json
import time as _time
from datetime import datetime, timezone
from typing import Any


def _relative_time(ts_str: str) -> str:
    """Convert an ISO timestamp string to a relative time like '2s ago'."""
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Timestamps from jsonl_logger use time.localtime() — treat as local time.
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        delta = _time.time() - dt.timestamp()
        if delta < 0:
            return "just now"
        if delta < 60:
            return f"{int(delta)}s ago"
        if delta < 3600:
            return f"{int(delta / 60)}m ago"
        if delta < 86400:
            return f"{int(delta / 3600)}h ago"
        return f"{int(delta / 86400)}d ago"
    except Exception:
        return ts_str


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
            subtitle_parts.append(_relative_time(created_at))
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
        f"When: {_relative_time(str(issue.get('created_at', '')))}",
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
    """Normalize session panel mode for Timeline/Artifacts toggle."""
    normalized = str(mode or "").strip().lower()
    if normalized in {"artifacts", "issues"}:
        return normalized
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


# ---------------------------------------------------------------------------
# Model call detail helpers
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4


def _fmt_tokens(count: int) -> str:
    """Format a token count as e.g. '3.2k' or '142'."""
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def _extract_cached_tokens(usage: dict[str, Any]) -> int:
    """Extract cached input tokens across provider formats."""
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, (int, float)) and cached:
            return int(cached)
    for key in ("cache_read_input_tokens", "cacheReadInputTokens"):
        val = usage.get(key)
        if isinstance(val, (int, float)) and val:
            return int(val)
    return 0


def _est_tokens(chars: int) -> str:
    """Rough token estimate from character count."""
    tokens = chars // _CHARS_PER_TOKEN
    return f"~{_fmt_tokens(tokens)} tokens"


def _render_model_call_detail(event: dict[str, Any]) -> str:
    """Render a purpose-built detail view for LLM model call events."""
    lines: list[str] = []
    summary = summarize_session_timeline_event(event)
    lines.append(f"Summary: {summary}")

    # --- Token Usage ---
    usage = event.get("usage")
    if isinstance(usage, dict):
        inp = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("inputTokens") or 0
        out = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("outputTokens") or 0
        cached = _extract_cached_tokens(usage)
        total = inp + out
        lines.append("")
        lines.append("── Token Usage ──")
        lines.append(f"  Input tokens:    {inp:>8,}")
        lines.append(f"  Output tokens:   {out:>8,}")
        lines.append(f"  Total tokens:    {total:>8,}")
        if cached:
            pct = (cached / inp * 100) if inp else 0
            lines.append(f"  Cached input:    {cached:>8,}  ({pct:.0f}% cache hit)")
        else:
            lines.append(f"  Cached input:    {0:>8,}  (no cache hit)")
    else:
        lines.append("")
        lines.append("── Token Usage ──")
        lines.append("  (no usage data available)")

    # --- Context Composition ---
    sys_chars = event.get("system_prompt_chars")
    tool_count = event.get("tool_count")
    tool_chars = event.get("tool_schema_chars")
    msg_count = event.get("messages")
    breakdown = event.get("message_breakdown")
    has_composition = any(v is not None for v in (sys_chars, tool_count, tool_chars, msg_count, breakdown))

    if has_composition:
        lines.append("")
        lines.append("── Context Composition ──")
        if isinstance(sys_chars, (int, float)):
            lines.append(f"  System prompt:   {int(sys_chars):>8,} chars  ({_est_tokens(int(sys_chars))})")
        if isinstance(tool_chars, (int, float)):
            suffix = f"  ({tool_count} tools)" if isinstance(tool_count, int) else ""
            lines.append(f"  Tool schemas:    {int(tool_chars):>8,} chars  ({_est_tokens(int(tool_chars))}){suffix}")
        elif isinstance(tool_count, int):
            lines.append(f"  Tool count:      {tool_count:>8,}")
        if isinstance(msg_count, int):
            lines.append(f"  Messages:        {msg_count:>8,}")
        if isinstance(breakdown, dict) and breakdown:
            parts = [f"{role}={count}" for role, count in sorted(breakdown.items())]
            lines.append(f"  Breakdown:       {', '.join(parts)}")

    # --- Metadata ---
    model_id = event.get("model_id")
    duration = event.get("duration_s")
    model_call_id = event.get("model_call_id")
    if any(v is not None for v in (model_id, duration, model_call_id)):
        lines.append("")
        lines.append("── Metadata ──")
        if model_id:
            lines.append(f"  Model:           {model_id}")
        if isinstance(duration, (int, float)):
            lines.append(f"  Duration:        {float(duration):.2f}s")
        if model_call_id:
            lines.append(f"  Call ID:         {model_call_id}")

    return "\n".join(lines)


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
        label = f"{tool}{duration_text}"
        if kind == "error":
            return f"{label} (error)"
        return label
    if kind == "model":
        usage = event.get("usage")
        if isinstance(usage, dict):
            inp = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("inputTokens") or 0
            out = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("outputTokens") or 0
            parts = [f"{_fmt_tokens(inp)} in", f"{_fmt_tokens(out)} out"]
            cached = _extract_cached_tokens(usage)
            if cached:
                parts.append(f"{_fmt_tokens(cached)} cached")
            return f"LLM call{duration_text} — {' / '.join(parts)}"
        return f"LLM call{duration_text}"
    if kind == "invocation":
        return f"invocation{duration_text}"
    return (str(event.get("event", "")).strip() or "event") + duration_text


def build_session_timeline_sidebar_items(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build SidebarList payload for session timeline events."""
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
        kind = classify_session_timeline_event_kind(event)
        event_id = str(event.get("id", "")).strip() or f"timeline-{index + 1}"
        summary = summarize_session_timeline_event(event)
        ts = str(event.get("ts", "")).strip()
        subtitle = _relative_time(ts) if ts else ""
        items.append(
            {
                "id": event_id,
                "title": summary,
                "subtitle": subtitle,
                "state": state_map.get(kind, "default"),
            }
        )
    return items


def render_session_timeline_detail_text(event: dict[str, Any] | None) -> str:
    """Render detail body for selected timeline event."""
    if not isinstance(event, dict):
        return "(no timeline event selected)"
    kind = classify_session_timeline_event_kind(event)
    if kind == "model":
        return _render_model_call_detail(event)
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
