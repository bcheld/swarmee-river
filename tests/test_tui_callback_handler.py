"""Tests for TuiCallbackHandler JSONL event emission."""

from __future__ import annotations

import json
import io
import sys
from threading import Event

from swarmee_river.handlers.callback_handler import TuiCallbackHandler


def _capture_events(handler: TuiCallbackHandler, fn) -> list[dict]:
    """Run fn, capture stdout, return parsed JSONL events."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn()
    finally:
        sys.stdout = old
    lines = [l for l in buf.getvalue().strip().split("\n") if l.strip()]
    return [json.loads(l) for l in lines]


def test_text_delta_emitted():
    h = TuiCallbackHandler()
    events = _capture_events(h, lambda: h.callback_handler(data="hello"))
    assert len(events) == 1
    assert events[0] == {"event": "text_delta", "data": "hello"}


def test_text_complete_emitted():
    h = TuiCallbackHandler()
    events = _capture_events(h, lambda: h.callback_handler(data="done", complete=True))
    assert len(events) == 2
    assert events[0]["event"] == "text_delta"
    assert events[1]["event"] == "text_complete"


def test_thinking_emitted():
    h = TuiCallbackHandler()
    events = _capture_events(h, lambda: h.callback_handler(reasoningText="pondering..."))
    assert len(events) == 1
    assert events[0] == {"event": "thinking", "text": "pondering..."}


def test_tool_start_and_result():
    h = TuiCallbackHandler()

    def run():
        # Simulate tool input streaming
        h.callback_handler(current_tool_use={"toolUseId": "t1", "name": "shell", "input": {"command": "ls"}})
        # Simulate tool result
        h.callback_handler(message={
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "status": "success"}}],
        })

    events = _capture_events(h, run)
    types = [e["event"] for e in events]
    assert "tool_start" in types
    assert "tool_result" in types
    result_event = [e for e in events if e["event"] == "tool_result"][0]
    assert result_event["tool"] == "shell"
    assert result_event["status"] == "success"
    assert "duration_s" in result_event


def test_force_stop_no_output():
    h = TuiCallbackHandler()
    events = _capture_events(h, lambda: h.callback_handler(force_stop=True))
    assert events == []


def test_interrupt_event_no_output():
    h = TuiCallbackHandler()
    ev = Event()
    ev.set()
    h.interrupt_event = ev
    events = _capture_events(h, lambda: h.callback_handler(data="should not appear"))
    assert events == []


def test_throttle_warning():
    h = TuiCallbackHandler()
    events = _capture_events(h, lambda: h.callback_handler(event_loop_throttled_delay=30))
    assert len(events) == 1
    assert events[0]["event"] == "warning"
    assert "30" in events[0]["text"]


def test_tool_input_emitted_from_assistant_message():
    """tool_input event is emitted when assistant message contains toolUse with dict input."""
    h = TuiCallbackHandler()

    def run():
        # First trigger tool_start via current_tool_use
        h.callback_handler(current_tool_use={"toolUseId": "t1", "name": "shell", "input": {"command": "ls"}})
        # Then simulate assistant message with finalized toolUse
        h.callback_handler(message={
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "shell", "input": {"command": "ls -la", "cwd": "/tmp"}}}],
        })

    events = _capture_events(h, run)
    tool_input_events = [e for e in events if e["event"] == "tool_input"]
    assert len(tool_input_events) == 1
    assert tool_input_events[0]["tool_use_id"] == "t1"
    assert tool_input_events[0]["input"] == {"command": "ls -la", "cwd": "/tmp"}


def test_tool_input_not_emitted_for_non_dict_input():
    """tool_input event should not be emitted when toolUse input is not a dict."""
    h = TuiCallbackHandler()

    def run():
        h.callback_handler(message={
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t2", "name": "shell", "input": "partial string"}}],
        })

    events = _capture_events(h, run)
    tool_input_events = [e for e in events if e["event"] == "tool_input"]
    assert len(tool_input_events) == 0
