"""Compatibility tests for callback-style extra event fields in TuiCallbackHandler."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable

from swarmee_river.handlers.callback_handler import TuiCallbackHandler


def _capture_events(handler: TuiCallbackHandler, fn: Callable[[], None]) -> list[dict]:
    buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = buffer
    try:
        fn()
    finally:
        sys.stdout = original_stdout
    lines = [line for line in buffer.getvalue().strip().splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_token_field_emits_text_delta() -> None:
    handler = TuiCallbackHandler()
    events = _capture_events(handler, lambda: handler.callback_handler(token="hello"))
    assert events == [{"event": "text_delta", "data": "hello"}]


def test_callback_style_tool_events_emit_start_progress_result() -> None:
    handler = TuiCallbackHandler()

    def run() -> None:
        handler.callback_handler(tool_start={"tool_use_id": "tool-1", "tool": "shell", "input": {"command": "echo hi"}})
        handler.callback_handler(tool_progress={"tool_use_id": "tool-1", "stream": "stdout", "content": "hi\n"})
        handler.callback_handler(tool_end={"tool_use_id": "tool-1", "status": "success"})

    events = _capture_events(handler, run)
    event_types = [event["event"] for event in events]
    assert "tool_start" in event_types
    assert "tool_progress" in event_types
    assert "tool_result" in event_types
    assert any(event.get("event") == "tool_progress" and event.get("content") == "hi\n" for event in events)


def test_tool_progress_emits_tool_start_when_history_missing() -> None:
    handler = TuiCallbackHandler()

    events = _capture_events(
        handler,
        lambda: handler.callback_handler(tool_progress={"tool_use_id": "tool-2", "content": "chunk"}),
    )
    assert events[0]["event"] == "tool_start"
    assert events[0]["tool_use_id"] == "tool-2"
    assert events[1]["event"] == "tool_progress"
    assert events[1]["tool_use_id"] == "tool-2"
    assert events[1]["content"] == "chunk"
