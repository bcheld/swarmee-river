from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from strands.hooks import HookRegistry, HookProvider
from strands.hooks.events import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)

from swarmee_river.hooks._compat import event_messages, model_response_payload, register_hook_callback


def _truthy_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _iso_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _truncate(value: str, limit: int) -> str:
    if limit <= 0:
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + f"...[truncated {len(value) - limit} chars]"


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


class JSONLLoggerHooks(HookProvider):
    """
    Lightweight JSONL logging of model/tool/invocation events for performance monitoring and replay.

    Writes one JSON object per line to a session log file in `.swarmee/logs/`.
    Optionally uploads the log to S3 if `SWARMEE_LOG_S3_BUCKET` is set.
    """

    def __init__(self) -> None:
        self.enabled = _truthy_env("SWARMEE_LOG_EVENTS", True)
        self.log_dir = Path(
            os.getenv(
                "SWARMEE_LOG_DIR",
                str(Path.cwd() / ".swarmee" / "logs"),
            )
        )
        self.session_id = os.getenv("SWARMEE_SESSION_ID", uuid.uuid4().hex)
        self.max_field_chars = int(os.getenv("SWARMEE_LOG_MAX_FIELD_CHARS", "8000"))

        self.s3_bucket = os.getenv("SWARMEE_LOG_S3_BUCKET")
        self.s3_prefix = os.getenv("SWARMEE_LOG_S3_PREFIX", "swarmee/logs").strip("/")

        self._lock = threading.Lock()
        self._log_path = self.log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{self.session_id}.jsonl"

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        register_hook_callback(registry, BeforeInvocationEvent, self.before_invocation)
        register_hook_callback(registry, AfterInvocationEvent, self.after_invocation)
        register_hook_callback(registry, BeforeToolCallEvent, self.before_tool_call)
        register_hook_callback(registry, AfterToolCallEvent, self.after_tool_call)
        register_hook_callback(registry, BeforeModelCallEvent, self.before_model_call)
        register_hook_callback(registry, AfterModelCallEvent, self.after_model_call)

    def _write_append(self, line: str) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = dict(payload)
        payload.update({"ts": _iso_ts(), "event": event, "session_id": self.session_id})
        self._write_append(json.dumps(payload, ensure_ascii=False))

    def before_invocation(self, event: BeforeInvocationEvent) -> None:
        inv_id = uuid.uuid4().hex
        event.invocation_state.setdefault("swarmee", {})
        event.invocation_state["swarmee"].update(
            {
                "invocation_id": inv_id,
                "t0": time.time(),
                "tool_t0": {},
                "model_t0": {},
            }
        )

        messages = event_messages(event)
        input_summary = {
            "input_items": len(messages) if isinstance(messages, list) else None,
        }
        self._log("before_invocation", {"invocation_id": inv_id, **input_summary})

    def after_invocation(self, event: AfterInvocationEvent) -> None:
        sw_state = event.invocation_state.get("swarmee", {}) if isinstance(event.invocation_state, dict) else {}
        inv_id = sw_state.get("invocation_id")
        t0 = sw_state.get("t0")
        duration_s = round(time.time() - t0, 3) if isinstance(t0, (int, float)) else None

        result_repr = _truncate(str(event.result), self.max_field_chars) if event.result is not None else None
        self._log("after_invocation", {"invocation_id": inv_id, "duration_s": duration_s, "result": result_repr})

        if self.s3_bucket:
            threading.Thread(target=self._upload_to_s3, daemon=True).start()

    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        sw = event.invocation_state.get("swarmee", {})
        inv_id = sw.get("invocation_id")
        tool_use = event.tool_use or {}
        tool_use_id = tool_use.get("toolUseId")
        tool_name = tool_use.get("name")
        sw.get("tool_t0", {})[tool_use_id] = time.time()

        tool_input = tool_use.get("input")
        tool_input_repr = _truncate(_safe_json(tool_input), self.max_field_chars) if tool_input else None
        self._log(
            "before_tool_call",
            {"invocation_id": inv_id, "toolUseId": tool_use_id, "tool": tool_name, "input": tool_input_repr},
        )

    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        sw = event.invocation_state.get("swarmee", {})
        inv_id = sw.get("invocation_id")
        tool_use = event.tool_use or {}
        tool_use_id = tool_use.get("toolUseId")
        tool_name = tool_use.get("name")
        t0 = sw.get("tool_t0", {}).pop(tool_use_id, None)
        duration_s = round(time.time() - t0, 3) if isinstance(t0, (int, float)) else None

        result_repr = _truncate(_safe_json(event.result), self.max_field_chars) if event.result else None
        self._log(
            "after_tool_call",
            {
                "invocation_id": inv_id,
                "toolUseId": tool_use_id,
                "tool": tool_name,
                "duration_s": duration_s,
                "result": result_repr,
            },
        )

    def before_model_call(self, event: BeforeModelCallEvent) -> None:
        sw = event.invocation_state.get("swarmee", {})
        inv_id = sw.get("invocation_id")
        call_id = uuid.uuid4().hex
        sw.get("model_t0", {})[call_id] = time.time()

        messages = event_messages(event)
        msg_count = len(messages) if isinstance(messages, list) else None
        self._log("before_model_call", {"invocation_id": inv_id, "model_call_id": call_id, "messages": msg_count})
        event.invocation_state["swarmee_model_call_id"] = call_id

    def after_model_call(self, event: AfterModelCallEvent) -> None:
        sw = event.invocation_state.get("swarmee", {})
        inv_id = sw.get("invocation_id")
        call_id = event.invocation_state.pop("swarmee_model_call_id", None)
        t0 = sw.get("model_t0", {}).pop(call_id, None)
        duration_s = round(time.time() - t0, 3) if isinstance(t0, (int, float)) else None

        response_payload = model_response_payload(event)
        resp_summary = _truncate(str(response_payload), self.max_field_chars)
        self._log(
            "after_model_call",
            {
                "invocation_id": inv_id,
                "model_call_id": call_id,
                "duration_s": duration_s,
                "response": resp_summary,
            },
        )

    def _upload_to_s3(self) -> None:
        if not self.s3_bucket:
            return
        try:
            import boto3
        except Exception:
            return

        key = f"{self.s3_prefix}/{self._log_path.name}"
        try:
            s3 = boto3.client("s3")
            body = self._log_path.read_bytes()
            s3.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=body,
                ContentType="application/x-ndjson",
            )
        except Exception:
            return
