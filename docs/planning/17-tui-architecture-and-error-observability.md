# 17 — TUI Architecture & Error Observability

**Date:** 2026-06-10
**Status:** Proposed
**Theme:** Structural causes behind the recurring breakage: god-class composition, blanket exception suppression, and test gaps. This is the "stop the bleeding for good" plan that makes docs 11-16 stick.

---

## Problem Statement

The complaint areas (planning, settings, menus, indicators) keep regressing
because of three structural properties:

1. **God-class mixin composition.** `SwarmeeTUI` (`tui/app.py:~1034`) inherits
   from 12+ mixins (`TranscriptMixin`, `PlanMixin`, `SettingsMixin`,
   `AgentStudioMixin`, …) totaling ~6,000 lines plus app.py's 3,812 and
   widgets.py's 5,560. Mixins freely share `self.state`, private attributes,
   and each other's methods; any change risks cross-mixin breakage via MRO and
   implicit contracts.
2. **Blanket exception suppression.** There are **253** occurrences of
   `contextlib.suppress(Exception)` across `src/` — 92 in `tui/widgets.py`
   alone, 26 in `tui/event_router.py`, 25 in `swarmee.py`. Widget-selector
   typos, unmounted-widget races, and logic errors all fail silently. The
   shipped "Continue button not rendering" bug (fixed in d15583f) was exactly
   this class: a wrong selector swallowed by suppression, discovered by users
   rather than by a log line or test.
3. **Test gaps on the paths that break.** E2E tests exist, but there is no
   coverage for widget visibility under constrained sizes, streaming-path
   cost, dispatch-queue behavior, or settings round-trips.

---

## Findings & Direction

### F1 (High) — Replace silent suppression with a logged-failure helper

A mechanical but high-value migration: introduce one helper, e.g.

```python
def ui_guard(app, label: str):
    """Like contextlib.suppress, but logs and counts the failure."""
```

that logs at WARNING with the label, increments a per-label counter, and (for
repeated failures) surfaces a one-time diagnostic line in the session issues
panel. Migrate call sites in priority order: `tui/mixins/plan.py`,
`tui/event_router.py`, `tui/mixins/settings.py`, then `tui/widgets.py`.
Pure `query_one` lookups should mostly become non-suppressed — a missing
selector is a bug, not an expected condition.

**Payoff:** every future "feature silently does nothing" bug produces a log
trail; many latent bugs surface immediately during the migration itself.

### F2 (Medium-High) — Incremental decomposition of the mixin god-class

A big-bang rewrite is not warranted; an incremental extraction is:

1. Define the seams: `OutputManager` (transcript writing + batching, absorbs
   doc 15 work), `MetricsSink` (doc 12's last-write-wins slots), and the
   planning state machine object (doc 14 F1). Each is an owned attribute, not
   a mixin.
2. New code goes into components; mixin methods become thin delegating shims.
3. One mixin at a time gets retired when its surface is fully delegated.

Rule to adopt now: **no new methods on mixins; no mixin reaching into another
mixin's privates.** Enforce with a lightweight lint (grep-based CI check is
fine to start).

### F3 (Medium) — widgets.py split

`tui/widgets.py` (5,560 lines, 92 suppressions) mixes streaming message
widgets, plan rows, command palette, action sheet, metrics bar, and helpers.
Split by feature (`widgets/streaming.py`, `widgets/plan.py`,
`widgets/palette.py`, `widgets/metrics.py`) with a re-exporting
`widgets/__init__.py` so imports don't break. Mostly mechanical; do it early
so docs 12-15's changes land in the new layout instead of deepening the
monolith.

### F4 (Medium) — Targeted regression test harness

The recurring bug classes each get a cheap, durable guard:

| Bug class | Test |
|-----------|------|
| Controls pushed off-screen / hidden | Pilot tests at 80×24 and 60×20 asserting named controls are within the visible region (plan buttons, action sheet, footer priority chips) |
| Silent selector breakage | After F1, a test asserting zero `ui_guard` failures during a scripted happy-path session |
| Streaming cost regressions | Budget tests from doc 15 F4 |
| Settings round-trip loss | Property-style test: load → mutate one field → save → reload preserves all other fields incl. unknown keys |
| Dropped state events | Flood test from doc 12 |

### F5 (Low) — Diagnostics surface for users

Once F1 exists, add a `/diagnostics` view summarizing: suppressed-failure
counts by label, dropped-dispatch counts, last event-loop stall duration.
Turns "all sorts of issues" reports into actionable tickets ("diagnostics
shows 40 failures in plan.visibility").

---

## Proposed Plan

| Phase | Work | Findings |
|-------|------|----------|
| 1 | `ui_guard` helper + migrate plan/event_router/settings call sites | F1 |
| 2 | widgets.py split by feature | F3 |
| 3 | Pilot-test harness for visibility + settings round-trip tests | F4 |
| 4 | Component extraction (OutputManager, MetricsSink, plan state machine) | F2 |
| 5 | `/diagnostics` view | F5 |

## Sequencing With the Other Plans

Recommended overall order across docs 11-17, optimizing for user-visible
relief first:

1. **15-P1** markdown coalescing (biggest perceived-perf win, small diff)
2. **13-P1** footer binding dedupe (one-day fix for the literal complaint)
3. **11-P1** SettingsStore (kills the data-loss class)
4. **12-P1/P2** indicator blocking + staleness
5. **17-P1** ui_guard migration (surfaces remaining latent bugs)
6. **14-P1/P2/P3** planning state machine + lifecycle
7. Remaining phases per-doc, with **17-P3 tests** landing alongside each.
