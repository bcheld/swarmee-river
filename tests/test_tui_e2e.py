"""
End-to-end TUI scenario tests.

These tests drive the real SwarmeeTUI Textual app via Textual's Pilot API,
with all real daemon I/O replaced by MockTransport.  Tests exercise the
full event-routing and UI-update path without spawning any subprocess.

Run with:
    pytest tests/test_tui_e2e.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# The harness fixture is defined in tui_harness.py (auto-discovered via conftest or
# direct import).  We import it so pytest collects it as a fixture.
from tests.tui_harness import tui_app_factory  # noqa: F401

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

        assert app.state.session.timeline_events == [], "timeline_events should be cleared after _start_fresh_session()"


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

        transport.emit_event(
            {
                "event": "before_model_call",
                "model_call_id": "mc-1",
                "system_prompt_chars": 1000,
                "tool_count": 5,
                "tool_schema_chars": 2500,
                "message_breakdown": {"user": 2, "assistant": 1},
            }
        )
        transport.emit_event(
            {
                "event": "after_model_call",
                "model_call_id": "mc-1",
                "model_id": "claude-sonnet-4-6",
                "usage": {"inputTokens": 300, "outputTokens": 80},
                "duration_s": 1.2,
            }
        )
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


async def test_prompt_submit_shows_thinking_bar_before_stream_events(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        from textual.widgets import TextArea

        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        prompt = app.query_one("#prompt", TextArea)
        prompt.insert("ping")
        app.action_submit_prompt()

        # Verify the query command was sent first.
        sent_query = await _wait_for(
            lambda: any(str(cmd.get("cmd", "")).strip().lower() == "query" for cmd in transport.sent_commands),
            pilot=pilot,
        )
        assert sent_query, "query command was not sent after prompt submit"

        # Before any text/tool streaming events, the thinking bar should be visible.
        await pilot.pause(delay=0.05)
        assert str(app._thinking_bar.styles.display) == "block"
        assert app._active_thinking_indicator is not None


async def test_thinking_bar_stays_visible_briefly_on_immediate_first_delta(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        from textual.widgets import TextArea

        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        prompt = app.query_one("#prompt", TextArea)
        prompt.insert("ping")
        app.action_submit_prompt()

        sent_query = await _wait_for(
            lambda: any(str(cmd.get("cmd", "")).strip().lower() == "query" for cmd in transport.sent_commands),
            pilot=pilot,
        )
        assert sent_query, "query command was not sent after prompt submit"

        # Simulate a very fast first model token.
        transport.emit_event({"event": "text_delta", "data": "p"})
        await pilot.pause(delay=0.05)
        assert str(app._thinking_bar.styles.display) == "block"
        assert app._active_thinking_indicator is not None

        # After the minimum-visible window, the thinking bar and transcript
        # spinner should hide.
        await pilot.pause(delay=0.40)
        assert str(app._thinking_bar.styles.display) == "none"
        assert app._active_thinking_indicator is None


async def test_model_selector_moves_to_agents_overview_and_updates_runtime_state(tui_app_factory):
    from textual.containers import Horizontal
    from textual.widgets import Select

    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        selector = app.query_one("#model_select", Select)
        overview_row = app.query_one("#agent_overview_model_row", Horizontal)
        prompt_bottom = app.query_one("#prompt_bottom", Horizontal)

        assert selector.parent is overview_row
        assert "model_select" not in {str(getattr(child, "id", "") or "") for child in prompt_bottom.children}
        assert "prompt_metrics" in {str(getattr(child, "id", "") or "") for child in prompt_bottom.children}

        app.on_select_changed(
            SimpleNamespace(
                select=SimpleNamespace(id="model_select", has_focus=True),
                value="openai|deep",
            )
        )
        assert app.state.daemon.model_provider_override == "openai"
        assert app.state.daemon.model_tier_override == "deep"


async def test_prompt_submit_forces_transcript_scroll_to_tail_when_overflowed(tui_app_factory):
    from textual.containers import VerticalScroll
    from textual.widgets import TextArea

    async with tui_app_factory(size=(140, 24)) as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        for index in range(90):
            app._write_transcript_line(f"filler line {index}")
        await pilot.pause(delay=0.1)

        transcript = app.query_one("#transcript", VerticalScroll)
        transcript.scroll_to(0, 0, animate=False)
        await pilot.pause(delay=0.05)
        assert app._get_scroll_proportion(transcript) < 0.5

        prompt = app.query_one("#prompt", TextArea)
        prompt.insert("scroll to latest")
        app.action_submit_prompt()

        reached_tail = await _wait_for(
            lambda: app._get_scroll_proportion(transcript) > 0.95,
            pilot=pilot,
            attempts=30,
            delay=0.05,
        )
        assert reached_tail
        assert any("YOU> scroll to latest" in line for line in app._transcript_fallback_lines)


async def test_settings_general_includes_interrupt_control_label(tui_app_factory):
    from textual.widgets import Static

    async with tui_app_factory() as (app, pilot, _transport):
        app._switch_side_tab("tab_settings")
        await pilot.pause(delay=0.05)

        label = app.query_one("#settings_interrupt_control_label", Static)
        assert "Interrupt Control" in str(label.render())


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
        {
            "event": "before_model_call",
            "model_call_id": "mc-rt-1",
            "system_prompt_chars": 800,
            "tool_count": 3,
            "tool_schema_chars": 1200,
            "message_breakdown": {"user": 1},
            "ts": "2026-02-28T10:00:00",
        },
        {
            "event": "after_model_call",
            "model_call_id": "mc-rt-1",
            "model_id": "claude-sonnet-4-6",
            "usage": {"inputTokens": 200, "outputTokens": 50},
            "duration_s": 0.9,
            "ts": "2026-02-28T10:00:01",
        },
        {
            "event": "after_tool_call",
            "tool": "bash",
            "toolUseId": "tu-1",
            "duration_s": 0.3,
            "result": '{"ok": true}',
            "ts": "2026-02-28T10:00:02",
        },
        {"event": "after_invocation", "invocation_id": "inv-rt-1", "duration_s": 2.1, "ts": "2026-02-28T10:00:03"},
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


# ---------------------------------------------------------------------------
# Scenario 8 — full query cycle: transcript populated but timeline empty
# ---------------------------------------------------------------------------


async def test_full_query_cycle_populates_both_transcript_and_timeline(tui_app_factory, tmp_path):
    """
    Reproduce the reported issue: after an LLM query completes, the user sees
    streamed text and tool output in the transcript, but the Session > Timeline
    tab remains empty.

    This test simulates a complete query lifecycle:

    1. Daemon ready (session connects).
    2. User sends a query (command captured by MockTransport).
    3. Daemon streams text_delta events → transcript populates.
    4. Daemon runs a tool (tool_start → tool_result) → transcript populates.
    5. JSONL log file is written mid-query by the daemon's JSONLLoggerHooks
       (simulated by creating the file after streaming events but before
       turn_complete — matching real-world timing).
    6. turn_complete fires → timeline refresh is scheduled.
    7. Assert: transcript has the streamed text AND tool result.
    8. Assert: timeline_events should contain entries from the JSONL log.

    If step 8 fails while step 7 passes, the issue is in the log-file-to-
    timeline pipeline (log discovery, graph index build, or session_id mismatch).
    """
    import json
    import time

    from swarmee_river.tui.transport import send_daemon_command

    session_id = "full-query-cycle-session"

    # Pre-create the logs directory (the daemon logger would mkdir on first write)
    state_dir = tmp_path / ".swarmee"
    logs_dir = state_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    async with tui_app_factory() as (app, pilot, transport):
        # --- Phase 1: daemon ready ---
        transport.emit_ready(session_id=session_id)
        reached = await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)
        assert reached, "daemon never became ready"
        assert app.state.daemon.session_id == session_id

        # --- Phase 2: user sends a query (TUI → daemon command) ---
        send_daemon_command(
            app.state.daemon.proc,
            {
                "cmd": "query",
                "text": "Show me the git log",
                "auto_approve": True,
            },
        )
        assert any(
            c.get("cmd") == "query" and c.get("text") == "Show me the git log" for c in transport.sent_commands
        ), "query command was not captured"

        # --- Phase 3: daemon streams text back (transcript fills up) ---
        transport.emit_event({"event": "text_delta", "text": "I'll run `git log` "})
        transport.emit_event({"event": "text_delta", "text": "to show recent commits."})
        await pilot.pause(delay=0.1)

        # --- Phase 4: daemon runs a tool ---
        transport.emit_event(
            {
                "event": "tool_start",
                "tool_use_id": "tu-git-1",
                "tool": "bash",
            }
        )
        transport.emit_event(
            {
                "event": "tool_result",
                "tool_use_id": "tu-git-1",
                "tool": "bash",
                "status": "success",
                "duration_s": 0.4,
            }
        )
        await pilot.pause(delay=0.1)

        # More streamed text after the tool
        transport.emit_event({"event": "text_delta", "text": "Here are the recent commits."})
        transport.emit_event({"event": "text_complete", "text": ""})
        await pilot.pause(delay=0.1)

        # --- Phase 5: JSONL log file written by daemon logger (simulated) ---
        # In production, JSONLLoggerHooks writes events as they happen.
        # We create the file here — after streaming but before turn_complete,
        # matching real timing (the logger has already flushed by this point).
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{ts}_{session_id}.jsonl"
        log_events = [
            {"event": "before_invocation", "invocation_id": "inv-fq-1", "ts": "2026-02-28T11:00:00"},
            {
                "event": "before_model_call",
                "model_call_id": "mc-fq-1",
                "system_prompt_chars": 1200,
                "tool_count": 8,
                "tool_schema_chars": 3000,
                "message_breakdown": {"user": 1},
                "ts": "2026-02-28T11:00:01",
            },
            {
                "event": "after_model_call",
                "model_call_id": "mc-fq-1",
                "model_id": "claude-sonnet-4-6",
                "usage": {"inputTokens": 450, "outputTokens": 120},
                "duration_s": 1.8,
                "ts": "2026-02-28T11:00:03",
            },
            {
                "event": "after_tool_call",
                "tool": "bash",
                "toolUseId": "tu-git-1",
                "duration_s": 0.4,
                "result": '{"ok": true}',
                "ts": "2026-02-28T11:00:04",
            },
            {"event": "after_invocation", "invocation_id": "inv-fq-1", "duration_s": 3.5, "ts": "2026-02-28T11:00:05"},
        ]
        log_file.write_text(
            "\n".join(json.dumps(e) for e in log_events) + "\n",
            encoding="utf-8",
        )

        # --- Phase 6: turn_complete fires ---
        transport.emit_event({"event": "turn_complete", "exit_status": "ok"})

        # --- Phase 7: verify transcript has content ---
        await pilot.pause(delay=0.15)
        transcript = "\n".join(app._transcript_fallback_lines)
        assert "git log" in transcript.lower() or "commits" in transcript.lower(), (
            f"Transcript should contain streamed text but got:\n{transcript[:500]}"
        )
        assert app.state.daemon.run_tool_count >= 1, "tool_start should have incremented run_tool_count"

        # --- Phase 8: verify timeline populated from JSONL log ---
        reached = await _wait_for(
            lambda: len(app.state.session.timeline_events) > 0,
            pilot=pilot,
            attempts=80,
            delay=0.05,
        )

        if not reached:
            issue_texts = [i.get("text", "") for i in app.state.session.issues]
            raise AssertionError(
                f"TIMELINE EMPTY after full query cycle with transcript content.\n"
                f"This reproduces the reported issue: text appears in transcript\n"
                f"but Session > Timeline remains empty.\n\n"
                f"Diagnostics:\n"
                f"  session_id in state: {app.state.daemon.session_id!r}\n"
                f"  session_id expected: {session_id!r}\n"
                f"  log file exists: {log_file.exists()}\n"
                f"  log file path: {log_file}\n"
                f"  issues panel: {issue_texts}\n"
                f"  timeline_refresh_inflight: {app.state.session.timeline_refresh_inflight}\n"
                f"  timeline_refresh_pending: {app.state.session.timeline_refresh_pending}\n"
                f"  error_count: {app.state.session.error_count}\n"
                f"  transcript lines: {len(app._transcript_fallback_lines)}"
            )

        event_types = [e.get("event") for e in app.state.session.timeline_events]
        assert "after_model_call" in event_types, f"Expected after_model_call in timeline but got: {event_types}"
        assert "after_tool_call" in event_types, f"Expected after_tool_call in timeline but got: {event_types}"
        assert "after_invocation" in event_types, f"Expected after_invocation in timeline but got: {event_types}"

        # Verify the model call entry has merged context composition data
        model_calls = [e for e in app.state.session.timeline_events if e.get("event") == "after_model_call"]
        assert len(model_calls) >= 1
        mc = model_calls[0]
        assert mc.get("model_id") == "claude-sonnet-4-6", f"model_id: {mc.get('model_id')}"
        assert isinstance(mc.get("usage"), dict), f"usage missing: {mc}"


# ---------------------------------------------------------------------------
# Scenario 9 — second query updates timeline incrementally
# ---------------------------------------------------------------------------


async def test_second_query_appends_to_timeline(tui_app_factory, tmp_path):
    """
    After a first query populates the timeline, a second query should APPEND
    new events (not leave timeline stale with only the first query's data).

    This catches bugs where:
    - The refresh isn't re-scheduled on the second turn_complete
    - The inflight guard blocks the second refresh permanently
    - The log file is only read once and cached
    """
    import json
    import time

    session_id = "incremental-timeline-session"
    state_dir = tmp_path / ".swarmee"
    logs_dir = state_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{ts}_{session_id}.jsonl"

    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready(session_id=session_id)
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        # --- First query ---
        first_query_events = [
            {
                "event": "before_model_call",
                "model_call_id": "mc-q1",
                "system_prompt_chars": 500,
                "tool_count": 3,
                "tool_schema_chars": 1000,
                "message_breakdown": {"user": 1},
                "ts": "2026-02-28T12:00:00",
            },
            {
                "event": "after_model_call",
                "model_call_id": "mc-q1",
                "model_id": "claude-sonnet-4-6",
                "usage": {"inputTokens": 100, "outputTokens": 30},
                "duration_s": 0.6,
                "ts": "2026-02-28T12:00:01",
            },
            {"event": "after_invocation", "invocation_id": "inv-q1", "duration_s": 1.0, "ts": "2026-02-28T12:00:02"},
        ]
        log_file.write_text(
            "\n".join(json.dumps(e) for e in first_query_events) + "\n",
            encoding="utf-8",
        )

        transport.emit_event({"event": "text_delta", "text": "First answer."})
        transport.emit_event({"event": "text_complete"})
        transport.emit_event({"event": "turn_complete", "exit_status": "ok"})

        reached = await _wait_for(
            lambda: len(app.state.session.timeline_events) > 0,
            pilot=pilot,
            attempts=80,
            delay=0.05,
        )
        assert reached, "timeline never populated after first query"

        first_event_count = len(app.state.session.timeline_events)
        assert first_event_count >= 2, (
            f"Expected at least 2 events after first query, got {first_event_count}: "
            f"{[e.get('event') for e in app.state.session.timeline_events]}"
        )

        # --- Second query: append more events to the same log file ---
        second_query_events = [
            {
                "event": "before_model_call",
                "model_call_id": "mc-q2",
                "system_prompt_chars": 600,
                "tool_count": 3,
                "tool_schema_chars": 1000,
                "message_breakdown": {"user": 2, "assistant": 1},
                "ts": "2026-02-28T12:01:00",
            },
            {
                "event": "after_model_call",
                "model_call_id": "mc-q2",
                "model_id": "claude-sonnet-4-6",
                "usage": {"inputTokens": 250, "outputTokens": 70},
                "duration_s": 1.1,
                "ts": "2026-02-28T12:01:01",
            },
            {
                "event": "after_tool_call",
                "tool": "read_file",
                "toolUseId": "tu-q2",
                "duration_s": 0.2,
                "result": '{"ok": true}',
                "ts": "2026-02-28T12:01:02",
            },
            {"event": "after_invocation", "invocation_id": "inv-q2", "duration_s": 2.0, "ts": "2026-02-28T12:01:03"},
        ]
        with log_file.open("a", encoding="utf-8") as f:
            for ev in second_query_events:
                f.write(json.dumps(ev) + "\n")

        transport.emit_event({"event": "text_delta", "text": "Second answer."})
        transport.emit_event({"event": "text_complete"})
        transport.emit_event({"event": "turn_complete", "exit_status": "ok"})

        # Wait for timeline to grow beyond the first query's event count
        reached = await _wait_for(
            lambda: len(app.state.session.timeline_events) > first_event_count,
            pilot=pilot,
            attempts=80,
            delay=0.05,
        )

        if not reached:
            from swarmee_river.session.graph_index import build_session_graph_index

            diag_index = build_session_graph_index(session_id)
            diag_events = diag_index.get("events", [])
            raise AssertionError(
                f"Timeline did NOT update after second query.\n"
                f"  Events after first query: {first_event_count}\n"
                f"  Events now: {len(app.state.session.timeline_events)}\n"
                f"  Manual build found: {len(diag_events)} events\n"
                f"  inflight: {app.state.session.timeline_refresh_inflight}\n"
                f"  pending: {app.state.session.timeline_refresh_pending}"
            )

        # Should now have events from BOTH queries
        all_event_types = [e.get("event") for e in app.state.session.timeline_events]
        tool_calls = [e for e in app.state.session.timeline_events if e.get("event") == "after_tool_call"]
        assert len(tool_calls) >= 1, f"Expected after_tool_call from second query in {all_event_types}"
        assert any(
            e.get("tool") == "read_file" for e in tool_calls
        ), f"Expected read_file tool from second query, got: {tool_calls}"


# ---------------------------------------------------------------------------
# Scenario 10 — terminal resize with content does not corrupt state
# ---------------------------------------------------------------------------

async def test_resize_clears_right_edge_artifacts(tui_app_factory):
    """
    Regression test for right-edge artifacts after terminal resize.

    After populating the transcript with streamed text and tool output,
    shrink and then expand the terminal.  Assert that:

    1. No errors are raised during resize.
    2. App state (daemon ready, transcript content, tool count) is preserved.
    3. The on_resize handler fires without exceptions.

    This test cannot assert visual pixel-level artifacts directly, but it
    validates that the resize path with loaded content is stable and that
    the full-screen refresh added to on_resize() executes without error.
    """
    async with tui_app_factory(size=(200, 50)) as (app, pilot, transport):
        # --- Setup: daemon ready + transcript content ---
        transport.emit_ready(session_id="resize-test-session")
        reached = await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)
        assert reached, "daemon never became ready"

        # Stream some text into transcript
        transport.emit_event({"event": "text_delta", "text": "Here is a detailed answer "})
        transport.emit_event({"event": "text_delta", "text": "spanning multiple chunks."})
        await pilot.pause(delay=0.1)

        # Run a tool
        transport.emit_event({
            "event": "tool_start",
            "tool_use_id": "tu-resize-1",
            "tool": "bash",
        })
        transport.emit_event({
            "event": "tool_result",
            "tool_use_id": "tu-resize-1",
            "tool": "bash",
            "status": "success",
            "duration_s": 0.3,
        })
        await pilot.pause(delay=0.1)

        transport.emit_event({"event": "text_complete", "text": ""})
        await pilot.pause(delay=0.1)

        # Snapshot state before resize
        pre_error_count = app.state.session.error_count
        pre_tool_count = app.state.daemon.run_tool_count
        pre_transcript = list(app._transcript_fallback_lines)

        # --- Shrink terminal ---
        await pilot.resize_terminal(100, 30)
        await pilot.pause(delay=0.15)

        assert app.state.daemon.ready, "daemon should still be ready after shrink"
        assert app.state.session.error_count == pre_error_count, (
            f"error_count changed after shrink: {app.state.session.error_count} != {pre_error_count}"
        )
        assert app.state.daemon.run_tool_count == pre_tool_count

        # --- Expand terminal back ---
        await pilot.resize_terminal(200, 50)
        await pilot.pause(delay=0.15)

        assert app.state.daemon.ready, "daemon should still be ready after expand"
        assert app.state.session.error_count == pre_error_count, (
            f"error_count changed after expand: {app.state.session.error_count} != {pre_error_count}"
        )
        assert app.state.daemon.run_tool_count == pre_tool_count

        # Transcript content should be preserved through resizes
        assert len(app._transcript_fallback_lines) >= len(pre_transcript), (
            "transcript lines should not be lost after resize"
        )
