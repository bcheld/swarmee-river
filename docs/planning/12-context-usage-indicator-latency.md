# 12 — Context/Usage Indicators: Latency and Staleness

**Date:** 2026-06-10
**Status:** Proposed
**Theme:** User complaints that token/context/cost indicators lag behind activity or show stale values.

---

## Problem Statement

The context-fill and usage/cost indicators are fed by `usage`/`context` events
from the daemon, routed through `tui/event_router.py` into widgets in
`tui/widgets.py`. The update path does blocking work on the UI thread,
recomputes token estimates over the full conversation repeatedly, and shares a
lossy dispatch queue with transcript rendering — so indicators fall behind
during exactly the moments users watch them (active streaming).

---

## Findings

### F1 (Critical) — `load_settings()` disk read on every usage event

`_compute_usage_cost_fallback()` (`tui/event_router.py:94-122`) calls
`load_settings()` (line 102) — a full read + JSON parse of
`.swarmee/settings.json` — synchronously on the UI thread, every time a usage
event arrives without a precomputed `cost_usd`. During a multi-call turn this
fires repeatedly and stalls the event loop.

**Fix:** Cache the parsed settings (the `SettingsStore` proposed in doc 11
covers this); pass pricing config into the router once. Also make
`TuiMetricsHooks` (`hooks/tui_metrics.py:~252`) always attach `cost_usd` so the
fallback is genuinely exceptional.

### F2 (Critical) — Full-history token re-estimation, repeatedly per turn

`estimate_tokens()` (`context/budgeted_summarizing_conversation_manager.py:239-255`)
walks every message and runs `_extract_message_text()` (lines 195-237: recursive
extraction + JSON-encoding of tool inputs/reasoning) with no memoization —
O(total conversation size) each call. It is invoked multiple times per turn:

- `swarmee.py:~2006` before each invocation,
- `swarmee.py:~2060` after compaction,
- again inside `_emit_tui_context_event_if_enabled()`,
- `tui/event_router.py:~374` when the TUI handles a `context` event.

`estimate_tool_schema_chars()` (`swarmee.py:~811`) similarly re-serializes
every tool schema on each call even though tools don't change mid-turn.

**Impact:** Context indicator update cost grows linearly with conversation
length; long sessions see indicators progressively lag.

**Fix:**
1. Memoize per-message extracted-text length (keyed by message identity);
   estimate becomes "cached sum + new messages only".
2. Compute the estimate once per turn, attach it to the emitted event, and have
   the TUI consume the value instead of recomputing.
3. Cache tool-schema size at agent build time; invalidate on tool set change.

### F3 (High) — Usage/context events dropped or queued behind transcript rendering

All daemon events — transcript lines, tool output, usage, context — go through
one sequential path: `mixins/output.py:~493-505` → event router, dispatched
across threads via `TranscriptMixin` (`mixins/transcript.py:13-98`). Two
consequences:

- During heavy streaming/tool output, usage events wait behind rendering work,
  so the indicator visibly trails the transcript.
- The backlog (`_THREAD_DISPATCH_QUEUE_MAX = 256`, `transcript.py:91-98`)
  drops the **oldest** entries when full. A dropped usage/context event means
  the indicator silently sticks at a stale value until the next event happens
  to arrive — matching the "indicator frozen mid-turn" reports.

**Fix:** Metrics events are tiny and *last-write-wins* — they should never
queue. Maintain a dedicated latest-metrics slot (single mutable cell per
indicator, overwritten by each new event) drained by a 100ms UI timer. Never
drop them; dropping transcript lines under pressure is acceptable, dropping
state is not.

### F4 (Medium) — Indicator animation timer thrash

`ContextMetricsBar` (`tui/widgets.py:~2233-2255`) animates at a 50ms interval
and restarts the animation when a new value lands. With rapid usage events the
animation restarts continuously: constant 20 FPS re-render work and a bar that
appears jittery or perpetually "catching up".

**Fix:** Debounce incoming values (animate toward the latest target rather than
restarting), drop the tick rate to 100ms, and stop the timer when the target is
reached. Pairs naturally with the F3 latest-value slot.

### F5 (Low) — Cost/cached-token values read via mutable shared state

The usage handler (`tui/event_router.py:377-403`) writes
`app.state.daemon.last_usage` / `last_provider_cached_input_tokens` and then
reads them back to compute display values. Interleaved events can briefly
display mismatched token/cost pairs.

**Fix:** Compute display values purely from the event payload and pass them to
`set_provider_usage()` directly; treat state as a record of what was displayed,
not an input.

---

## Proposed Plan

| Phase | Work | Findings |
|-------|------|----------|
| 1 | Cache settings/pricing for the cost path; ensure daemon always sends `cost_usd` | F1 |
| 2 | Latest-value metrics slot + 100ms drain timer; exempt metrics from drop logic | F3, F5 |
| 3 | Incremental token estimation with per-message memoization; once-per-turn emission | F2 |
| 4 | Animation debounce/retarget | F4 |

Phases 1–2 are small and remove the *blocking* and *staleness* causes; phase 3
removes the *growth-with-session-length* cause.

## Test Strategy

- Unit: token-estimate memoization correctness (same totals as full recompute);
  metrics slot last-write-wins under interleaving.
- Perf regression test: estimate cost on a 500-message synthetic history must
  be O(new messages) on the second call.
- TUI pilot test: flood 1,000 transcript lines + 50 usage events; assert final
  indicator equals the last usage event (no drops, no staleness).
