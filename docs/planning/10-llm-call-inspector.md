# 10 — LLM Call Inspector: Implementation Report

**Date:** 2025-02-28
**Status:** Feature implemented; known follow-up bugs documented

---

## Objective

Enable users to explore and understand what is being passed to LLM endpoints
via the **Run › Session** tab in the TUI.

> User story: *"How do I know if I am sending the appropriate context to the LLM?"*

---

## What Was Implemented

### Phase 1 — Context Metadata Injection (`swarmee.py`)

`run_agent()` now injects context metadata into `invocation_state["swarmee"]`
before each agent invocation:

| Key | Value |
|-----|-------|
| `system_prompt_chars` | Length of the agent's system prompt |
| `tool_count` | Number of tools registered on the agent |
| `tool_schema_chars` | Total serialised size of all tool schemas |

### Phase 2 — JSONL Logger Enrichment (`hooks/jsonl_logger.py`)

- **`before_model_call`** payload now includes a `message_breakdown` (per-role
  message counts), `system_prompt_chars`, `tool_count`, `tool_schema_chars`,
  and `total_input_chars`.
- **`after_model_call`** payload now includes `model_id`.

### Phase 3 — Graph Index Merge (`session/graph_index.py`)

- `before_model_call` added to the notable-event set.
- During index construction, each `before_model_call` is held in a
  `pending_before` dict keyed by `(turn, cycle)`. When the matching
  `after_model_call` arrives it is merged — the `after` event inherits
  `message_breakdown`, `system_prompt_chars`, `tool_count`,
  `tool_schema_chars`, and `total_input_chars` from the `before` event.

### Phase 4 — Timeline Visibility (`tui/sidebar_session.py`)

Model-call events now render in the session timeline with a one-line summary:

```
⚡ Model call  ─  1.2k in / 340 out
```

The token counts are estimated at ~4 chars per token when exact token fields
are absent.

### Phase 5 — Detail Renderer (`tui/sidebar_session.py`)

Selecting a model-call event shows a detail panel with three sections:

1. **Token Usage** — input / output / cached-read tokens (or char-based
   estimates).
2. **Context Composition** — system prompt chars, tool schemas chars, tool
   count, and per-role message breakdown.
3. **Metadata** — model ID, latency, turn, cycle.

### Phase 6 — Tests

| Test file | New tests |
|-----------|-----------|
| `tests/test_jsonl_logger_redaction.py` | 2 (enrichment assertions) |
| `tests/test_session_graph_index.py` | `before_model_call` merging + enriched fields |
| `tests/test_tui_subprocess.py` | 3 tests + `SWARMEE_LOG_EVENTS` assertion |

All 556 tests pass, 1 skipped.

---

## Bug Fix: `SWARMEE_LOG_EVENTS` Disabled in Daemon

### Root Cause

`.swarmee/settings.json` contained `"SWARMEE_LOG_EVENTS": "false"`. This value
was propagated to the daemon subprocess environment via
`_apply_project_settings_env_overrides()`, which entirely disabled JSONL
logging.  Without JSONL events there is nothing for `build_session_graph_index`
to read, so the timeline stayed empty (or showed cached stale data).

### Fix

Force `SWARMEE_LOG_EVENTS=1` in both daemon-spawn paths:

- `src/swarmee_river/tui/transport.py` — `_build_swarmee_subprocess_env()`
- `src/swarmee_river/runtime_service/server.py` — `_start_session_process()`

---

## Known Remaining Issues

### 1. Single-Log-File Discovery

`_discover_session_log_path()` in `session/graph_index.py` returns **only the
single most-recently-modified** log file matching `*_{session_id}.jsonl`. When
a session spans multiple daemon restarts (each creating a new log file) the
older events are invisible to the graph index builder.

**Fix needed:** Return all matching log files sorted by timestamp; update
`_build_events_and_tools()` to iterate over them.

### 2. `/new` Does Not Reset Timeline

`_start_fresh_session()` in `tui/mixins/session.py` clears the restore-state
but never calls `_reset_session_timeline_panel()`. After `/new`, the timeline
still displays events from the previous session.

**Fix needed:** Add `self._reset_session_timeline_panel()` to
`_start_fresh_session()`.

### 3. Silent Exception Fallback in Timeline Refresh

`_refresh_session_timeline_async()` catches **all** exceptions from
`build_session_graph_index` and silently falls back to the cached
`graph_index.json`.  Build failures (e.g. missing log file, corrupt JSON) are
invisible to the user.

**Fix needed:** Log the exception to the issues panel or transcript.

---

## Files Changed

| File | Summary |
|------|---------|
| `src/swarmee_river/swarmee.py` | Context metadata injection |
| `src/swarmee_river/hooks/jsonl_logger.py` | Enriched before/after model call payloads |
| `src/swarmee_river/session/graph_index.py` | before_model_call merge logic |
| `src/swarmee_river/tui/sidebar_session.py` | Timeline rendering + detail view |
| `src/swarmee_river/tui/transport.py` | Force `SWARMEE_LOG_EVENTS=1` |
| `src/swarmee_river/runtime_service/server.py` | Force `SWARMEE_LOG_EVENTS=1` |
| `src/swarmee_river/tui/mixins/session.py` | Debug logging removed |
| `tests/test_jsonl_logger_redaction.py` | 2 new tests |
| `tests/test_session_graph_index.py` | before_model_call + enriched assertions |
| `tests/test_tui_subprocess.py` | 3 new tests + SWARMEE_LOG_EVENTS assertion |
