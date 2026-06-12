# 16 — Startup Latency: Lazy Imports and Event-Driven Readiness

**Date:** 2026-06-10
**Status:** In progress — see implementation status below
**Theme:** Slow CLI invocation and slow TUI first paint.

---

**Implementation status (2026-06-12):**
- DONE — F1 (package half): PEP 562 lazy loading for swarmee_river/__init__.py and models providers; `import swarmee_river.settings` 1.27s -> 0.07s (102d7f5).
- REMAINING — F1 (CLI half): swarmee.py still imports strands at module scope (Agent + MaxTokensReachedException in 5 except clauses); fast `--help` needs the arg parser extracted to a light module and the console script repointed. F2 backoff polling, F3 paint-first startup.

## Problem Statement

Every `swarmee` invocation — including `--help` — pays for importing the full
tool and hook surface, and the TUI additionally waits on polling loops before
first paint. Users experience this as "the app is slow" before they've even
typed anything.

---

## Findings

### F1 (High) — Eager import of all hooks and tools at module load

`swarmee.py:50-228` imports every hook implementation
(`FileDiffReviewHooks`, `JSONLLoggerHooks`, `MaxTokensRetryHooks`,
`TuiMetricsHooks`, `SessionS3Hooks`, …) at module level, and `tools.py:10-45`
imports 20+ tool modules (`athena_query`, `editor`, `git`, `swarm`,
`agent_graph`, …) eagerly. Several of these pull heavy transitive
dependencies (boto3, provider SDKs). The cost lands on every CLI entrypoint.

**Fix:**
1. Move tool imports inside `get_tools()` (and import per-tool lazily within
   it) so only commands that build an agent pay the cost.
2. Lazy-import hooks at agent assembly, not module load.
3. Make `--help`/argument errors return before importing the agent stack
   (parse args in a thin entry module).
4. Add a CI check: `python -X importtime -c "import swarmee_river.cli"` with a
   budget, so heavy imports can't creep back in.

### F2 (Medium) — Broker readiness discovered by fixed-interval polling

`ensure_runtime_broker()` (`runtime_service/client.py:~220-230`) polls
reachability every 100ms up to a 6s deadline; shutdown has a similar loop
(`client.py:~262-268`). `transport.py:137-142` busy-waits in 10ms steps for
socket close. There is no readiness signal and no feedback in the UI while
waiting — slow broker start looks like a hang.

**Fix:** Start with a 10ms interval and back off exponentially (cap ~200ms);
show a "starting runtime…" status during the wait. Longer-term, have the
broker signal readiness (write discovery file last / accept a hello on the
socket) so the client can block on connect instead of polling.

### F3 (Medium) — First paint waits on the runtime instead of rendering immediately

The TUI completes broker discovery/hello before showing the interface, so all
of F1+F2 latency is serial with first paint.

**Fix:** Paint the shell immediately with a "connecting to runtime" state in
the status bar; do broker startup in a worker and enable the prompt when the
hello completes. This converts startup time from "blank terminal" to
"responsive app, briefly disabled input" — a large perceived-latency win even
before the actual costs shrink.

---

## Proposed Plan

| Phase | Work | Findings |
|-------|------|----------|
| 1 | Lazy tool/hook imports; thin CLI entry; importtime budget in CI | F1 |
| 2 | Paint-first TUI startup with async broker connect | F3 |
| 3 | Backoff polling + readiness signaling in the broker protocol | F2 |

## Measurement

Before starting, capture baselines so improvement is demonstrable:

- `hyperfine 'swarmee --help'` (target: <150ms from ~2-3s today),
- `python -X importtime -m swarmee_river.cli 2>importtime.log` top offenders,
- time-to-first-paint of `swarmee tui` (manual or pilot-test timestamp).
