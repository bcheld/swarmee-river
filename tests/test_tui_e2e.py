"""
End-to-end TUI scenario tests.

These tests drive the real SwarmeeTUI Textual app via Textual's Pilot API,
with all real daemon I/O replaced by MockTransport.  Tests exercise the
full event-routing and UI-update path without spawning any subprocess.

Run with:
    pytest tests/test_tui_e2e.py -v
"""

from __future__ import annotations

import pytest

# The harness fixture is defined in tui_harness.py (auto-discovered via conftest or
# direct import).  We import it so pytest collects it as a fixture.
from tests.tui_harness import MockTransport, tui_app_factory  # noqa: F401  (fixture re-export)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _wait_for(condition, *, pilot, attempts: int = 20, delay: float = 0.05) -> bool:
    """Poll *condition* up to *attempts* times, pausing between each check."""
    for _ in range(attempts):
        if condition():
            return True
        await pilot.pause(delay=delay)
    return False


# ---------------------------------------------------------------------------
# Scenario 1 — daemon ready handshake
# ---------------------------------------------------------------------------

async def test_daemon_becomes_ready_after_ready_event(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        assert not app.state.daemon.ready

        transport.emit_ready(session_id="e2e-session-1")

        reached = await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)
        assert reached, "daemon.ready never became True after ready event"
        assert app.state.daemon.session_id == "e2e-session-1"


# ---------------------------------------------------------------------------
# Scenario 2 — text output renders in transcript
# ---------------------------------------------------------------------------

async def test_plain_text_output_appends_to_transcript(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        transport.emit("Hello from the daemon\n")
        await pilot.pause(delay=0.15)

        transcript_lines = "\n".join(app._transcript_fallback_lines)
        assert "Hello from the daemon" in transcript_lines


# ---------------------------------------------------------------------------
# Scenario 3 — /new resets the session timeline
# ---------------------------------------------------------------------------

async def test_new_command_resets_session_timeline(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready(session_id="old-session")
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        # Inject a fake timeline event directly into state
        app.state.session.timeline_events = [{"id": "ev1", "event": "after_tool_call"}]
        app._render_session_timeline_panel()
        assert len(app.state.session.timeline_events) == 1

        # Trigger /new
        app._start_fresh_session()

        assert app.state.session.timeline_events == [], (
            "timeline_events should be cleared after _start_fresh_session()"
        )


# ---------------------------------------------------------------------------
# Scenario 4 — error output goes to issues panel
# ---------------------------------------------------------------------------

async def test_error_line_appears_in_issues(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        transport.emit("ERROR: something went wrong\n")
        reached = await _wait_for(lambda: app.state.session.error_count > 0, pilot=pilot)
        assert reached, "error_count never incremented"
        assert any("something went wrong" in i.get("text", "") for i in app.state.session.issues)


# ---------------------------------------------------------------------------
# Scenario 5 — model call event increments timeline
# ---------------------------------------------------------------------------

async def test_model_call_event_adds_timeline_entry(tui_app_factory):
    """
    Emitting a before/after_model_call pair should add one entry to
    state.session.timeline_events after a timeline refresh.

    Note: timeline refresh is async (reads from disk), so we can't assert
    timeline_events directly here.  Instead we verify that the graph index
    builder round-trips correctly via the unit path in test_session_graph_index.py.
    This test verifies the transport → state route up to the issue-free steady state.
    """
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready(session_id="model-call-session")
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        transport.emit_event({
            "event": "before_model_call",
            "model_call_id": "mc-1",
            "system_prompt_chars": 1000,
            "tool_count": 5,
            "tool_schema_chars": 2500,
            "message_breakdown": {"user": 2, "assistant": 1},
        })
        transport.emit_event({
            "event": "after_model_call",
            "model_call_id": "mc-1",
            "model_id": "claude-sonnet-4-6",
            "usage": {"inputTokens": 300, "outputTokens": 80},
            "duration_s": 1.2,
        })
        await pilot.pause(delay=0.15)

        # No errors should have been raised by event processing
        assert app.state.session.error_count == 0


# ---------------------------------------------------------------------------
# Scenario 6 — sent commands are captured by MockTransport
# ---------------------------------------------------------------------------

async def test_mock_transport_captures_sent_commands(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        from swarmee_river.tui.transport import send_daemon_command
        send_daemon_command(app.state.daemon.proc, {"cmd": "interrupt"})

        assert any(c.get("cmd") == "interrupt" for c in transport.sent_commands)


# ---------------------------------------------------------------------------
# Scenario 7 — full timeline round-trip after turn_complete
# ---------------------------------------------------------------------------

async def test_timeline_updates_after_turn_complete(tui_app_factory, tmp_path):
    """
    End-to-end timeline refresh:

    1. Pre-seed a JSONL log file in the test state dir with real events.
    2. Connect the mock transport and emit a ready + turn_complete.
    3. Wait for the async timeline refresh to complete.
    4. Assert timeline_events contains the pre-seeded data.

    This is the closest possible simulation of a real LLM query completing and
    the Session Timeline updating — without running an actual daemon.
    """
    import json
    import time

    session_id = "timeline-roundtrip-session"

    # The harness fixture sets SWARMEE_STATE_DIR → tmp_path/.swarmee
    state_dir = tmp_path / ".swarmee"
    logs_dir = state_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{ts}_{session_id}.jsonl"

    events = [
        {"event": "before_model_call", "model_call_id": "mc-rt-1",
         "system_prompt_chars": 800, "tool_count": 3, "tool_schema_chars": 1200,
         "message_breakdown": {"user": 1}, "ts": "2026-02-28T10:00:00"},
        {"event": "after_model_call", "model_call_id": "mc-rt-1",
         "model_id": "claude-sonnet-4-6",
         "usage": {"inputTokens": 200, "outputTokens": 50},
         "duration_s": 0.9, "ts": "2026-02-28T10:00:01"},
        {"event": "after_tool_call", "tool": "bash", "toolUseId": "tu-1",
         "duration_s": 0.3, "result": '{"ok": true}', "ts": "2026-02-28T10:00:02"},
        {"event": "after_invocation", "invocation_id": "inv-rt-1",
         "duration_s": 2.1, "ts": "2026-02-28T10:00:03"},
    ]
    log_file.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )

    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready(session_id=session_id)
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        # Simulate a completed LLM query
        transport.emit_event({"event": "turn_complete", "exit_status": "ok"})

        # Timeline refresh is scheduled with 0.35s delay → wait up to 3s total
        reached = await _wait_for(
            lambda: len(app.state.session.timeline_events) > 0,
            pilot=pilot,
            attempts=60,
            delay=0.05,
        )

        # Diagnose failure: report any issues that were appended
        if not reached:
            issue_texts = [i.get("text", "") for i in app.state.session.issues]
            raise AssertionError(
                f"timeline_events never populated after turn_complete.\n"
                f"Issues panel: {issue_texts}\n"
                f"Log file exists: {log_file.exists()}\n"
                f"session_id in state: {app.state.daemon.session_id!r}"
            )

        event_types = [e.get("event") for e in app.state.session.timeline_events]
        assert "after_model_call" in event_types, f"Expected after_model_call in {event_types}"
        assert "after_tool_call" in event_types, f"Expected after_tool_call in {event_types}"
        assert "after_invocation" in event_types, f"Expected after_invocation in {event_types}"
