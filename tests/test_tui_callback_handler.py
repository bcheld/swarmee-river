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


def test_tool_start_and_result_for_empty_input_tool():
    h = TuiCallbackHandler()

    def run():
        h.callback_handler(current_tool_use={"toolUseId": "t-empty", "name": "noop", "input": {}})
        h.callback_handler(message={
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t-empty", "status": "success"}}],
        })

    events = _capture_events(h, run)
    assert any(event.get("event") == "tool_start" and event.get("tool_use_id") == "t-empty" for event in events)
    assert any(event.get("event") == "tool_result" and event.get("tool_use_id") == "t-empty" for event in events)


def test_tool_result_emitted_even_without_tool_history():
    h = TuiCallbackHandler()
    events = _capture_events(
        h,
        lambda: h.callback_handler(message={
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "unknown", "status": "error"}}],
        }),
    )
    assert events == [
        {
            "event": "tool_result",
            "tool_use_id": "unknown",
            "tool": "unknown",
            "status": "error",
            "duration_s": 0.0,
        }
    ]


def test_tool_result_emitted_from_result_payload():
    h = TuiCallbackHandler()

    def run():
        h.callback_handler(current_tool_use={"toolUseId": "t1", "name": "shell", "input": {"command": "ls"}})
        h.callback_handler(result={"toolUseId": "t1", "status": "success"})

    events = _capture_events(h, run)
    assert any(event.get("event") == "tool_start" and event.get("tool_use_id") == "t1" for event in events)
    assert any(event.get("event") == "tool_result" and event.get("tool_use_id") == "t1" for event in events)


def test_tool_progress_content_rate_limited_and_flushed_before_result(monkeypatch):
    h = TuiCallbackHandler()
    now = {"value": 100.0}

    def _mono() -> float:
        return now["value"]

    def _wall() -> float:
        return 1000.0 + now["value"]

    monkeypatch.setattr("swarmee_river.handlers.callback_handler.time.monotonic", _mono)
    monkeypatch.setattr("swarmee_river.handlers.callback_handler.time.time", _wall)

    def run():
        h.callback_handler(current_tool_use={"toolUseId": "t1", "name": "shell", "input": {"command": "ls"}})
        h.callback_handler(current_tool_use={"toolUseId": "t1", "stdout": "first\n"})
        now["value"] = 100.05
        h.callback_handler(current_tool_use={"toolUseId": "t1", "stdout": "second\n"})
        now["value"] = 100.06
        h.callback_handler(message={"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "status": "success"}}]})

    events = _capture_events(h, run)
    progress_events = [event for event in events if event.get("event") == "tool_progress" and event.get("content")]
    assert [event.get("content") for event in progress_events] == ["first\n", "second\n"]
    result_index = [idx for idx, event in enumerate(events) if event.get("event") == "tool_result"][0]
    second_progress_index = [idx for idx, event in enumerate(events) if event.get("event") == "tool_progress"][-1]
    assert second_progress_index < result_index


def test_tool_progress_heartbeat_emitted_for_long_running_tool(monkeypatch):
    h = TuiCallbackHandler()
    now = {"value": 5.0}

    def _mono() -> float:
        return now["value"]

    def _wall() -> float:
        return 1000.0 + now["value"]

    monkeypatch.setattr("swarmee_river.handlers.callback_handler.time.monotonic", _mono)
    monkeypatch.setattr("swarmee_river.handlers.callback_handler.time.time", _wall)

    def run():
        h.callback_handler(current_tool_use={"toolUseId": "t-heartbeat", "name": "shell", "input": {"command": "sleep 5"}})
        now["value"] = 7.3
        h.callback_handler(current_tool_use={"toolUseId": "t-heartbeat", "input": {"command": "sleep 5"}})

    events = _capture_events(h, run)
    heartbeat_events = [
        event
        for event in events
        if event.get("event") == "tool_progress" and "content" not in event and event.get("tool_use_id") == "t-heartbeat"
    ]
    assert len(heartbeat_events) == 1
    assert float(heartbeat_events[0].get("elapsed_s", 0.0)) >= 2.0


def test_tool_output_preview_capped_to_4kb(monkeypatch):
    h = TuiCallbackHandler()
    now = {"value": 1.0}

    monkeypatch.setattr("swarmee_river.handlers.callback_handler.time.monotonic", lambda: now["value"])
    monkeypatch.setattr("swarmee_river.handlers.callback_handler.time.time", lambda: 1000.0 + now["value"])

    large_chunk = "x" * 6000

    def run():
        h.callback_handler(current_tool_use={"toolUseId": "t-cap", "name": "shell", "input": {"command": "echo"}})
        h.callback_handler(current_tool_use={"toolUseId": "t-cap", "stdout": large_chunk})

    _capture_events(h, run)
    info = h.tool_histories.get("t-cap")
    assert info is not None
    assert len(str(info.get("output_preview", ""))) <= 4096


def test_result_string_emits_text_when_no_stream_deltas():
    h = TuiCallbackHandler()
    events = _capture_events(h, lambda: h.callback_handler(result="final response"))
    assert events == [
        {"event": "text_delta", "data": "final response"},
        {"event": "text_complete"},
    ]


def test_result_fallback_does_not_duplicate_streamed_text():
    h = TuiCallbackHandler()

    def run():
        h.callback_handler(data="hello")
        h.callback_handler(complete=True)
        h.callback_handler(result="hello")

    events = _capture_events(h, run)
    assert events == [
        {"event": "text_delta", "data": "hello"},
        {"event": "text_complete"},
    ]


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


def test_assistant_message_text_emitted_when_data_is_empty():
    h = TuiCallbackHandler()
    events = _capture_events(
        h,
        lambda: h.callback_handler(
            message={"role": "assistant", "content": [{"text": "hello from message"}]},
        ),
    )
    assert events == [{"event": "text_delta", "data": "hello from message"}]


def test_assistant_message_cumulative_text_is_delta_deduped():
    h = TuiCallbackHandler()

    def run():
        h.callback_handler(message={"role": "assistant", "content": [{"text": "hel"}]})
        h.callback_handler(message={"role": "assistant", "content": [{"text": "hello"}]})
        h.callback_handler(message={"role": "assistant", "content": [{"text": "hello"}]})

    events = _capture_events(h, run)
    assert events == [
        {"event": "text_delta", "data": "hel"},
        {"event": "text_delta", "data": "lo"},
    ]


def test_extra_event_fields_text_emits_text_delta():
    h = TuiCallbackHandler()
    events = _capture_events(h, lambda: h.callback_handler(delta="hello from extra"))
    assert events == [{"event": "text_delta", "data": "hello from extra"}]


def test_init_event_loop_resets_text_fallback_state():
    h = TuiCallbackHandler()

    def run():
        h.callback_handler(data="first")
        h.callback_handler(init_event_loop=True)
        h.callback_handler(result="second")

    events = _capture_events(h, run)
    assert events == [
        {"event": "text_delta", "data": "first"},
        {"event": "text_delta", "data": "second"},
        {"event": "text_complete"},
    ]


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


def test_plan_step_updates_emitted_from_text_markers():
    h = TuiCallbackHandler()

    def run():
        invocation_state = {"swarmee": {"plan_step_count": 2}}
        h.callback_handler(data="Starting step 1: Inspect logs\n", invocation_state=invocation_state)
        h.callback_handler(data="Completed step 1.\n", invocation_state=invocation_state)
        h.callback_handler(
            data="Starting step 2: Apply fix\nCompleted step 2.\n",
            invocation_state=invocation_state,
        )

    events = _capture_events(h, run)
    plan_events = [event for event in events if event.get("event") in {"plan_step_update", "plan_complete"}]
    assert plan_events == [
        {"event": "plan_step_update", "step_index": 0, "status": "in_progress", "note": "Inspect logs"},
        {"event": "plan_step_update", "step_index": 0, "status": "completed"},
        {"event": "plan_step_update", "step_index": 1, "status": "in_progress", "note": "Apply fix"},
        {"event": "plan_step_update", "step_index": 1, "status": "completed"},
        {"event": "plan_complete", "completed_steps": 2, "total_steps": 2},
    ]


def test_plan_marker_buffer_flushes_on_text_complete():
    h = TuiCallbackHandler()
    events = _capture_events(
        h,
        lambda: h.callback_handler(
            data="Starting step 1: Prepare",
            complete=True,
            invocation_state={"swarmee": {"plan_step_count": 1}},
        ),
    )
    assert any(
        event.get("event") == "plan_step_update"
        and event.get("step_index") == 0
        and event.get("status") == "in_progress"
        for event in events
    )
    assert any(event.get("event") == "text_complete" for event in events)


def test_plan_step_updates_emitted_from_plan_progress_tool():
    h = TuiCallbackHandler()

    def run():
        invocation_state = {"swarmee": {"plan_step_count": 1}}
        h.callback_handler(
            current_tool_use={
                "toolUseId": "p1",
                "name": "plan_progress",
                "input": {"step": 1, "status": "in_progress", "note": "Setting up"},
            },
            invocation_state=invocation_state,
        )
        h.callback_handler(
            current_tool_use={
                "toolUseId": "p1",
                "name": "plan_progress",
                "input": {"step_index": 0, "status": "completed"},
            },
            invocation_state=invocation_state,
        )

    events = _capture_events(h, run)
    assert any(
        event.get("event") == "plan_step_update"
        and event.get("step_index") == 0
        and event.get("status") == "in_progress"
        and event.get("note") == "Setting up"
        for event in events
    )
    assert any(
        event.get("event") == "plan_step_update"
        and event.get("step_index") == 0
        and event.get("status") == "completed"
        for event in events
    )
    assert any(event.get("event") == "plan_complete" and event.get("total_steps") == 1 for event in events)
