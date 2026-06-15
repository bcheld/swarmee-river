"""Plan workflow lifecycle tests (doc 14 F2/F3/F4).

Covers: stale step-update events must not corrupt the displayed plan, and
approving a plan must not lose it when the dispatch to the daemon fails.
"""

from __future__ import annotations

import pytest

from tests.tui_harness import tui_app_factory  # noqa: F401  (fixture)


async def _wait_for(condition, *, pilot, attempts: int = 40, delay: float = 0.05) -> bool:
    for _ in range(attempts):
        if condition():
            return True
        await pilot.pause(delay=delay)
    return condition()


def _plan_event(run_id: str) -> dict:
    return {
        "event": "plan",
        "plan_run_id": run_id,
        "rendered": "Proposed plan",
        "plan_json": {
            "summary": "Test plan",
            "steps": [
                {"description": "step one"},
                {"description": "step two"},
            ],
        },
    }


@pytest.mark.asyncio
async def test_stale_step_update_without_run_id_is_discarded(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        transport.emit_event(_plan_event("plan-new"))
        await _wait_for(lambda: app.state.plan.plan_run_id == "plan-new", pilot=pilot)
        assert app.state.plan.current_step_statuses == ["pending", "pending"]

        # A stale event from a previous run carries no plan_run_id.
        transport.emit_event({"event": "plan_step_update", "step_index": 0, "status": "completed"})
        # And one from a different run.
        transport.emit_event(
            {"event": "plan_step_update", "plan_run_id": "plan-old", "step_index": 1, "status": "completed"}
        )
        await pilot.pause(delay=0.2)
        assert app.state.plan.current_step_statuses == ["pending", "pending"]

        # A matching event still lands.
        transport.emit_event(
            {"event": "plan_step_update", "plan_run_id": "plan-new", "step_index": 0, "status": "completed"}
        )
        reached = await _wait_for(
            lambda: app.state.plan.current_step_statuses[0] == "completed", pilot=pilot
        )
        assert reached


@pytest.mark.asyncio
async def test_stale_plan_complete_is_discarded(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        transport.emit_event(_plan_event("plan-new"))
        await _wait_for(lambda: app.state.plan.plan_run_id == "plan-new", pilot=pilot)

        transport.emit_event({"event": "plan_complete"})
        await pilot.pause(delay=0.2)
        assert app.state.plan.current_step_statuses == ["pending", "pending"]
        assert not app.state.plan.completion_announced


def _pending_plan_payload() -> dict:
    return {
        "kind": "pending_work_plan",
        "plan_run_id": "plan-keep",
        "original_request": "do the thing",
        "current_plan": {
            "kind": "work_plan",
            "summary": "Do the thing",
            "steps": [{"description": "only step"}],
        },
    }


@pytest.mark.asyncio
async def test_approve_keeps_pending_plan_when_dispatch_fails(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        app._set_pending_plan_record(_pending_plan_payload())
        assert app._pending_plan_record() is not None

        # Daemon goes away before the user approves.
        app.state.daemon.ready = False
        app._dispatch_plan_action("approve")
        await pilot.pause(delay=0.1)

        assert app._pending_plan_record() is not None, "failed dispatch must keep the plan"
        assert not app.state.daemon.query_active

        # Daemon back: approval now succeeds and the record is consumed.
        app.state.daemon.ready = True
        app._dispatch_plan_action("approve")
        await pilot.pause(delay=0.1)
        assert app._pending_plan_record() is None
        sent = [cmd for cmd in transport.sent_commands if cmd.get("cmd") == "query"]
        assert sent and sent[-1].get("approved_plan", {}).get("plan_run_id") == "plan-keep"


@pytest.mark.asyncio
async def test_replan_with_no_prompt_gives_recovery_guidance(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        app._last_prompt = None
        app._dispatch_plan_action("replan")
        await pilot.pause(delay=0.1)

        transcript = "\n".join(app._transcript_fallback_lines)
        assert "no saved request to replan" in transcript
        assert "prompt" in transcript  # actionable guidance, not a dead-end


@pytest.mark.asyncio
async def test_planning_phase_follows_lifecycle(tui_app_factory):
    from swarmee_river.tui.state import PlanningPhase

    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)
        assert app.state.plan.phase == PlanningPhase.IDLE

        # Plan run dispatched -> GENERATING
        app._start_run("plan something", auto_approve=False, mode="plan")
        assert app.state.plan.phase == PlanningPhase.GENERATING

        # Plan arrives -> REVIEWING
        transport.emit_event(_plan_event("plan-xyz"))
        reached = await _wait_for(
            lambda: app.state.plan.phase == PlanningPhase.REVIEWING, pilot=pilot
        )
        assert reached

        # Turn completes while reviewing: phase must stay REVIEWING.
        transport.emit_event({"event": "turn_complete", "exit_status": "ok"})
        await pilot.pause(delay=0.2)
        assert app.state.plan.phase == PlanningPhase.REVIEWING

        # Clearing the plan -> IDLE
        app._dispatch_plan_action("clearplan")
        await pilot.pause(delay=0.1)
        assert app.state.plan.phase == PlanningPhase.IDLE


@pytest.mark.asyncio
async def test_approve_transitions_to_executing_then_idle(tui_app_factory):
    from swarmee_river.tui.state import PlanningPhase

    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        transport.emit_event(_plan_event("plan-exec"))
        await _wait_for(lambda: app.state.plan.phase == PlanningPhase.REVIEWING, pilot=pilot)

        app._set_pending_plan_record(_pending_plan_payload())
        app._dispatch_plan_action("approve")
        await pilot.pause(delay=0.1)
        assert app.state.plan.phase == PlanningPhase.EXECUTING

        transport.emit_event({"event": "turn_complete", "exit_status": "ok"})
        reached = await _wait_for(lambda: app.state.plan.phase == PlanningPhase.IDLE, pilot=pilot)
        assert reached
