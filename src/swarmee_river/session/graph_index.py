from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from swarmee_river.state_paths import logs_dir, sessions_dir

_GRAPH_INDEX_SCHEMA = "session_graph_index"
_GRAPH_INDEX_SCHEMA_VERSION = 1
_GRAPH_INDEX_FILENAME = "graph_index.json"
_MESSAGES_LOG_FILENAME = "messages.jsonl"


def _iso_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _session_dir(session_id: str, *, cwd: Path | None = None) -> Path:
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    return sessions_dir(cwd=cwd) / sid


def _messages_log_path(session_id: str, *, cwd: Path | None = None) -> Path:
    return _session_dir(session_id, cwd=cwd) / _MESSAGES_LOG_FILENAME


def _graph_index_path(session_id: str, *, cwd: Path | None = None) -> Path:
    return _session_dir(session_id, cwd=cwd) / _GRAPH_INDEX_FILENAME


def _safe_json_loads(line: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(line)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_replay_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _extract_replay_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    if isinstance(content, dict):
        if content.get("toolUse") or content.get("toolResult"):
            return ""
        for key in ("text", "data", "delta", "output_text", "outputText", "textDelta"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
        nested = content.get("content")
        if nested is not None:
            nested_text = _extract_replay_text(nested)
            if nested_text:
                return nested_text
    return ""


def _extract_text_from_message(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    extracted = _extract_replay_text(message.get("content"))
    if extracted:
        return extracted
    for key in ("text", "data", "output_text"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _read_latest_messages_snapshot(path: Path) -> list[Any]:
    if not path.exists():
        return []

    latest: list[Any] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                parsed = _safe_json_loads(line)
                if parsed is None:
                    continue
                messages = parsed.get("messages")
                if isinstance(messages, list):
                    latest = messages
    except OSError:
        return []

    return latest


def _discover_session_log_paths(session_id: str, *, cwd: Path | None = None) -> list[Path]:
    try:
        matches = list(logs_dir(cwd=cwd).glob(f"*_{session_id}.jsonl"))
    except OSError:
        return []
    if not matches:
        return []

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    matches.sort(key=_mtime)
    return matches


def _build_turns(messages: list[Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current_turn: dict[str, Any] | None = None

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        text = _extract_text_from_message(message).strip()

        if role == "user":
            # Tool-only user entries appear in replay streams and should not create turns.
            if not text:
                continue
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = {
                "turn_id": len(turns) + 1,
                "user_text": text,
                "assistant_text": "",
                "user_message_index": index,
                "assistant_message_indexes": [],
            }
            continue

        if role != "assistant" or current_turn is None or not text:
            continue

        assistant_text = str(current_turn.get("assistant_text", ""))
        current_turn["assistant_text"] = f"{assistant_text}\n\n{text}" if assistant_text else text
        assistant_indexes = current_turn.get("assistant_message_indexes")
        if isinstance(assistant_indexes, list):
            assistant_indexes.append(index)

    if current_turn is not None:
        turns.append(current_turn)
    return turns


def _coerce_duration(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return None


def _detect_tool_outcome(result_field: Any) -> tuple[bool | None, str | None]:
    payload: Any | None = None
    if isinstance(result_field, dict):
        payload = result_field
    elif isinstance(result_field, str):
        raw = result_field.strip()
        if not raw:
            return None, None
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, None
    else:
        return None, None

    if not isinstance(payload, dict):
        return None, None

    error_value = payload.get("error")
    if error_value not in (None, "", False):
        return False, str(error_value)

    for key in ("ok", "success"):
        value = payload.get(key)
        if isinstance(value, bool):
            return value, None if value else f"{key}=false"

    is_error = payload.get("isError")
    if isinstance(is_error, bool):
        return (not is_error), "isError=true" if is_error else None

    status = payload.get("status")
    if isinstance(status, str):
        status_lower = status.strip().lower()
        if status_lower and any(token in status_lower for token in ("error", "fail", "exception")):
            return False, status
        if status_lower in {"ok", "success", "succeeded", "completed"}:
            return True, None

    return None, None


def _build_events_and_tools(log_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    active_paths = [p for p in log_paths if p.exists()]
    if not active_paths:
        return [], {}, 0

    events: list[dict[str, Any]] = []
    tool_counts: dict[str, int] = {}
    errors = 0
    notable = {"after_tool_call", "after_model_call", "after_invocation", "before_model_call"}

    # Temporary storage for before_model_call entries, keyed by model_call_id,
    # so we can merge them into the corresponding after_model_call event.
    pending_before: dict[str, dict[str, Any]] = {}

    for log_path in active_paths:
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parsed = _safe_json_loads(line)
                    if parsed is None:
                        continue
                    event_name = str(parsed.get("event", "")).strip()
                    if event_name not in notable:
                        continue

                    entry: dict[str, Any] = {"event": event_name}
                    ts = parsed.get("ts")
                    if isinstance(ts, str) and ts.strip():
                        entry["ts"] = ts.strip()
                    invocation_id = parsed.get("invocation_id")
                    if isinstance(invocation_id, str) and invocation_id.strip():
                        entry["invocation_id"] = invocation_id.strip()
                    duration = _coerce_duration(parsed.get("duration_s"))
                    if duration is not None:
                        entry["duration_s"] = duration

                    if event_name == "after_tool_call":
                        tool = str(parsed.get("tool", "")).strip()
                        if tool:
                            entry["tool"] = tool
                            tool_counts[tool] = tool_counts.get(tool, 0) + 1
                        tool_use_id = str(parsed.get("toolUseId", "")).strip()
                        if tool_use_id:
                            entry["tool_use_id"] = tool_use_id

                        success, error = _detect_tool_outcome(parsed.get("result"))
                        if success is not None:
                            entry["success"] = success
                        if isinstance(error, str) and error.strip():
                            entry["error"] = error.strip()
                            errors += 1
                        elif success is False:
                            errors += 1

                    elif event_name == "before_model_call":
                        # Stash context composition metrics; merge into after_model_call later.
                        call_id = str(parsed.get("model_call_id", "")).strip()
                        before_data: dict[str, Any] = {}
                        for key in (
                            "messages",
                            "system_prompt_chars",
                            "tool_count",
                            "tool_schema_chars",
                            "model_id",
                            "message_breakdown",
                        ):
                            val = parsed.get(key)
                            if val is not None:
                                before_data[key] = val
                        if call_id:
                            pending_before[call_id] = before_data
                        # Don't emit before_model_call as a separate timeline event.
                        continue

                    elif event_name == "after_model_call":
                        call_id = str(parsed.get("model_call_id", "")).strip()
                        if call_id:
                            entry["model_call_id"] = call_id
                        # Merge context composition from the matching before_model_call.
                        before = pending_before.pop(call_id, None) if call_id else None
                        if isinstance(before, dict):
                            entry.update(before)
                        # Preserve usage and model_id from the after event itself.
                        usage = parsed.get("usage")
                        if isinstance(usage, dict):
                            entry["usage"] = usage
                        model_id = parsed.get("model_id")
                        if isinstance(model_id, str) and model_id.strip():
                            entry["model_id"] = model_id.strip()

                    events.append(entry)
        except OSError:
            continue

    return events, dict(sorted(tool_counts.items())), errors


def build_session_graph_index(session_id: str, *, cwd: Path | None = None) -> dict[str, Any]:
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")

    messages_log_path = _messages_log_path(sid, cwd=cwd)
    log_paths = _discover_session_log_paths(sid, cwd=cwd)

    messages = _read_latest_messages_snapshot(messages_log_path)
    turns = _build_turns(messages)
    events, tool_counts, error_count = _build_events_and_tools(log_paths)

    return {
        "schema": _GRAPH_INDEX_SCHEMA,
        "schema_version": _GRAPH_INDEX_SCHEMA_VERSION,
        "session_id": sid,
        "generated_at": _iso_ts(),
        "sources": {
            "messages_log": str(messages_log_path) if messages_log_path.exists() else None,
            "logs_files": [str(p) for p in log_paths if p.exists()],
        },
        "stats": {
            "turns": len(turns),
            "tools": sum(tool_counts.values()),
            "errors": error_count,
        },
        "tools": {
            "counts": tool_counts,
        },
        "turns": turns,
        "events": events,
    }


def load_session_graph_index(session_id: str) -> dict[str, Any] | None:
    path = _graph_index_path(session_id)
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def write_session_graph_index(session_id: str, index: dict[str, Any]) -> Path:
    path = _graph_index_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(index) if isinstance(index, dict) else {}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
