# 14 — Planning Feature: Workflow Reliability Overhaul

**Date:** 2026-06-10
**Status:** In progress — see implementation status below
**Theme:** User complaints about the plan/approve/execute workflow (the most-reported area).

---

**Implementation status (2026-06-12):**
- DONE — F1: explicit PlanningPhase with validated transition table; visibility branches on phase (7fe1114).
- DONE — F2: strict run-id matching for plan_step_update/plan_complete (995ad19).
- DONE — F3/F4: approve keeps the plan until dispatch is acknowledged; replan fallback + recovery guidance (995ad19).
- DONE — F8: plan UI failures now logged via ui_guard (d8f6046).
- REMAINING — F5 layout bounds hardening (pilot test exists), F6 async preflight validation, F7 interaction papercuts.

## Problem Statement

The planning workflow's state is spread across many loosely coupled variables
(`plan.pending_record`, `plan.plan_json`, `plan.received_structured_plan`,
`plan.plan_run_id`, `plan.current_step_statuses`, `daemon.query_active`,
`_last_run_auto_approve`) with no explicit lifecycle. Recent history shows the
symptom pattern: three of the last seven commits are planning hotfixes
("Fix plan UI Continue button not rendering", "Strengthen planning context
handoff", "Fix planning pane controls"). Each fix patched one manifestation;
this plan addresses the underlying causes.

---

## Findings

### F1 (High) — No explicit planning state machine

Phase is *inferred* from combinations of flags scattered across
`tui/event_router.py:671-785`, `tui/mixins/plan.py:257-349`, and
`tui/mixins/session.py:50-92`. Example fragility: whether a pending record is
kept depends on `app._last_run_auto_approve`, which is only set in
`_start_run` and goes stale across interrupts/daemon restarts.
`received_structured_plan` is set on any plan event and never reset until the
next one, so stale plan UI can show between runs.

**Fix:** Introduce a `PlanningPhase` enum
(`IDLE → GENERATING → REVIEWING → EXECUTING`, plus `REPLAN_PENDING`) on
`PlanState`, with a single `transition()` helper that validates moves and logs
invalid ones. Event handlers and visibility logic branch on the phase, not on
flag combinations. This is the keystone fix — F2, F3, and F6 become much
smaller once phase is explicit.

### F2 (High) — Stale `plan_step_update` events corrupt step display

The guard at `tui/event_router.py:728-732`:

```python
if event_plan_run_id is not None and event_plan_run_id != current_plan_run_id:
    return True
```

accepts any step update whose event carries **no** `plan_run_id` — regardless
of which plan is on screen. Plain (non-structured) plans have `plan_run_id =
None`, so after a quick replan, queued updates from the *previous* run land on
the *new* plan's step rows: wrong ☐/▶/☑ markers and premature completion
animation. This matches "jumbled plan steps" reports.

**Fix:** Require both IDs to be present and equal; drop updates when the
current plan has no run id. Assign a run id to plain plans too (generate one
TUI-side on receipt) so every plan is addressable.

### F3 (High) — Approve clears the pending plan before the daemon confirms

`_dispatch_plan_action("approve")` (`tui/mixins/plan.py:393-408`) calls
`_clear_pending_plan_record()` at line 400, *before* `_start_run` dispatches
the execute command. If the daemon is down, times out, or rejects the run, the
approved plan — including its `planning_context` (findings digest, tool
summaries) — is already gone from UI state. The user cannot retry approval
and must re-plan from scratch.

**Fix:** Move the pending record into an `executing_plan` slot on approval and
clear it only on run-start acknowledgment (or completion); on dispatch
failure, restore it to pending so the buttons come back.

### F4 (Medium) — Replan dead-end: "no previous prompt to replan"

`tui/mixins/plan.py:410-415` aborts replan when both
`pending.original_request` and `_last_prompt` are empty (possible after
session restore or plain plans). The message gives no recovery path; the only
escape is restarting.

**Fix:** Fall back to deriving a prompt from the plan summary, and when truly
nothing exists, keep the user in plan view with an actionable message ("type
refinement in the prompt and press Continue"). Persist `original_request` with
the session so restores don't lose it.

### F5 (Medium) — Continue-button layout remains one CSS tweak from regressing

Commit d15583f fixed the Continue button being pushed off-screen, but the
structure that caused it persists: the actions row sits after an unbounded
`#engage_plan_items` container (`tui/views/engage.py:43-51`,
CSS in `tui/app.py:~1175`). Any reintroduction of `height: 1fr` (or a very
large plan in a small terminal) hides the buttons again.

**Fix:** Bound the scrollable plan content (`#engage_plan_scroll` takes
`1fr`; items/questions get `max-height` + internal scroll) and dock the
actions row so it cannot be displaced. Add a pilot test at 80×24 asserting the
Continue/Approve buttons are within the visible region with a 50-step plan.

### F6 (Medium) — Execute preflight blocks the daemon thread

`_execute_with_plan` (`swarmee.py:~2384-2387`) synchronously runs
`validate_plan_preconditions(... capture_shared_prefix_fork(agent, ...))`,
hashing the full conversation history before execution starts. On long
sessions this reads as a multi-second hang right after the user presses
Approve — the worst possible moment for a stall. The planning-context
preamble injected into the prompt is also unbounded in size.

**Fix:** Run the snapshot/validation off-thread with a timeout and emit a
"validating plan preconditions…" status event so the UI shows progress; cap
the planning-context preamble (truncate findings digest, top-N tool
summaries).

### F7 (Low–Medium) — Plan widget interaction papercuts

- Checkbox → comment-input toggle (`tui/app.py:~2814-2828`,
  `tui/widgets.py:~1970-1978`) queries children that may not be mounted yet;
  failures are suppressed, so the first click sometimes does nothing.
- `PlanQuestionRow._sync_height` (`tui/widgets.py:~2034-2079`) re-runs
  `textwrap.wrap` over the full answer on every keystroke — visible typing
  lag with several questions open.

**Fix:** Defer the toggle via `call_after_refresh`; debounce height sync
(~150ms) and cache the wrap count per (text, width).

### F8 (Cross-cutting) — Failures in plan UI are invisible

Plan code leans on `contextlib.suppress(Exception)` for widget queries (e.g.
`tui/mixins/plan.py:14-71` and throughout). The Continue-button bug class —
a selector typo silently swallowed — is enabled by this pattern. Covered in
depth by doc 17; called out here because planning is where it has bitten
users hardest.

---

## Proposed Plan

| Phase | Work | Findings |
|-------|------|----------|
| 1 | `PlanningPhase` state machine + phase-driven visibility | F1 |
| 2 | Strict run-id matching; run ids for plain plans | F2 |
| 3 | Approve/ replan lifecycle: keep record until ack; replan fallbacks | F3, F4 |
| 4 | Layout bounds + docked actions row + small-terminal pilot test | F5 |
| 5 | Async preflight validation + bounded planning context | F6 |
| 6 | Interaction papercuts (deferred toggle, debounced height) | F7 |

## Test Strategy

- Unit: state-machine transition table (every event in every phase); step
  update routing with mismatched/missing run ids; approve-failure restores
  pending record.
- Pilot tests: plan review at 80×24 with large plans; rapid replan followed by
  stale step events; approve with daemon stopped → error + buttons return.
