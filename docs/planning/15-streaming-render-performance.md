# 15 — Streaming & Transcript Rendering Performance

**Date:** 2026-06-10
**Status:** Proposed
**Theme:** TUI lag/unresponsiveness while the agent is streaming output — the cross-cutting cause behind several complaint categories ("all sorts of issues").

---

## Problem Statement

During streaming, the TUI does quadratic rendering work and floods the Textual
event loop with per-line cross-thread dispatches. This is the substrate on
which other symptoms grow: indicator lag (doc 12) and sluggish plan widgets
(doc 14) are partly downstream of a saturated UI thread.

---

## Findings

### F1 (Critical) — Full markdown re-parse on every streamed token

Three widgets re-parse the *entire accumulated message* through Rich's
`Markdown` on each delta:

- `AssistantMessage.append_delta` (`tui/widgets.py:1321-1324`):
  ```python
  self._buffer.append(text)
  full = "".join(self._buffer)
  self.update(RichMarkdown(full))
  ```
- `AssistantStreamBlock.append_delta` (`tui/widgets.py:~1408-1414`) — same,
  plus a redundant double body refresh.
- `ReasoningBlock.append_delta` (`tui/widgets.py:~1490-1495`) — same.

Rich markdown parsing is O(message length), called once per delta → O(n²)
total per message. A 10k-character response parsed hundreds of times pegs the
CPU and makes the whole app (keystrokes, indicators, plan buttons) feel frozen
during generation.

**Fix:** Time-based coalescing: accumulate deltas and re-render at most every
~150ms (single timer per streaming widget), render final markdown once on
`finalize()`. Optionally render plain `RichText` during streaming and switch
to markdown on finalize — eliminates parsing from the hot path entirely.

### F2 (High) — Per-line `call_from_thread` storm from the daemon reader

The daemon output reader (`tui/mixins/daemon.py:~143-156`) dispatches **every
line** to the UI thread via `_call_from_thread_safe`, and the dispatch backlog
(`tui/mixins/transcript.py:13-98`) holds max 256 entries, dropping the oldest
on overflow. Under verbose tool output:

- the Textual loop saturates with queued callbacks (keypresses starve), and
- events are silently dropped — including state-bearing ones (see doc 12 F3),
  surfacing as a one-line warning at most every 5 seconds.

**Fix:** Batch in the reader thread: buffer lines and flush to the UI as one
callback per ~50ms tick or 100 lines, whichever first. Apply backpressure
(brief reader sleep) instead of dropping when the UI is behind. Route
state-bearing events (usage/context/plan-step) through last-write-wins slots
so they are exempt from both queuing and dropping.

### F3 (Medium) — Historical transcript re-parses markdown on re-render

`render_assistant_message()` (`tui/widgets.py:~82`) builds a fresh
`RichMarkdown` for completed messages too; tab switches and scrollback over a
long session re-pay full parse cost for every visible message.

**Fix:** Cache the rendered renderable per message (keyed by content hash);
invalidate never (messages are immutable once finalized).

### F4 (Medium) — No performance regression tests on the streaming path

`tests/` covers callback-handler behavior and E2E flows, but nothing asserts
streaming cost envelopes, so O(n²) regressions (F1) and dispatch floods (F2)
ship invisibly.

**Fix:** Add micro-benchmarks as tests with generous thresholds:

- `append_delta` × 200 on a growing 10k-char message completes < 0.5s;
- 500 rapid cross-thread dispatches: zero state-bearing events dropped;
- a "render budget" counter test: ≤ ~10 markdown parses per streamed message
  (validates coalescing stays in place).

---

## Proposed Plan

| Phase | Work | Findings |
|-------|------|----------|
| 1 | Coalesced streaming renders (150ms timer, finalize-once markdown) | F1 |
| 2 | Reader-side line batching + backpressure; exempt state events from drops | F2 |
| 3 | Renderable cache for finalized messages | F3 |
| 4 | Streaming perf regression tests | F4 |

Phase 1 is the single highest-impact change in this whole optimization effort:
it removes the dominant CPU cost from the moment users most notice lag.

## Dependencies / Interactions

- Doc 12 (indicators) phase 2 shares the F2 last-write-wins mechanism — build
  once, use for both.
- Doc 14 (planning) F7 typing lag is partially masked by F1; fix both.
