# Implementation Plan: Long-Running Daemon + UX Fixes

## Context

The TUI currently spawns a **new subprocess per prompt**, losing all conversation history. The CLI REPL keeps a single `Agent` instance alive across turns (messages accumulate automatically inside the Strands Agent). We need to bring that same multi-turn capability to the TUI by switching to a long-running daemon subprocess.

Additionally, the user identified several UX issues to address alongside this architectural change.

## User Feedback Summary

| # | Issue | Action |
|---|-------|--------|
| 1 | ThinkingIndicator not appearing; no intermediate output beyond tool notifications | Debug event flow; ensure `thinking` events reach TUI |
| 2 | Can't copy text beyond visible scroll area | Already works via `/copy last`, `/copy all`; no change needed |
| 3 | Collapsible tool details are static, not updated dynamically | Enhance tool_progress events with richer content (stdout chunks, status updates) |
| 5 | Plan workflow too rigid; don't start in plan mode; populate plan dynamically from chat | Remove `build_plan_mode_prompt` wrapping; add approve/replan/clear buttons to sidebar |
| 5b | Upon plan completion, prompt user to clear; record finished plans | Add post-plan-completion flow |
| 7 | "Auto" model label is confusing | Display resolved model name instead of "Auto" |
| 10 | Artifacts need better organization (but no file tree needed) | Defer to later |
| 11 | Stamp model used + timestamp on turns | Add metadata to transcript widgets |

## Architecture: Long-Running Daemon Subprocess

### Current Flow
```
User types prompt → TUI spawns `python -m swarmee_river.swarmee <prompt>` → subprocess exits
User types another prompt → TUI spawns NEW subprocess → no memory of previous turn
```

### New Flow
```
TUI starts → spawns `python -m swarmee_river.swarmee --tui-daemon` (once)
User types prompt → TUI sends JSONL to subprocess stdin: {"cmd":"query","text":"..."}
Subprocess runs query against persistent Agent → streams JSONL events to stdout
User types another prompt → same subprocess, same Agent, full conversation history
```

### Protocol

**TUI → Subprocess (stdin JSONL commands):**
- `{"cmd": "query", "text": "user prompt", "auto_approve": true}`
- `{"cmd": "consent_response", "choice": "y"}`
- `{"cmd": "interrupt"}` — cancel current query
- `{"cmd": "shutdown"}` — graceful exit

**Subprocess → TUI (stdout JSONL events):** — existing events unchanged, plus:
- `{"event": "ready"}` — daemon started, agent initialized
- `{"event": "turn_complete"}` — query finished, ready for next
- `{"event": "model_info", "model": "bedrock/balanced (anthropic.claude-sonnet...)", "tier": "balanced"}`

## Implementation Plan

### Step 1: Add `--tui-daemon` mode to `swarmee.py`

**File: `src/swarmee_river/swarmee.py`**

After the existing `if args.query[0] == "tui":` block (~line 437), add `--tui-daemon` argparse flag. When set:

1. Create agent (same as existing `create_agent()`)
2. Emit `{"event": "ready"}`
3. Enter stdin read loop:
   - Read one JSONL line from stdin
   - Parse command
   - If `cmd == "query"`: call `run_agent(text)`, emit events via existing callback handler, emit `{"event": "turn_complete"}` when done
   - If `cmd == "consent_response"`: unblock the consent prompt (use a `threading.Event` + shared variable, similar to existing `_consent_prompt` TUI path)
   - If `cmd == "interrupt"`: set interrupt_event
   - If `cmd == "shutdown"`: break loop, exit

Key: Reuse existing `run_agent()`, `_consent_prompt()`, `_generate_plan()`, hooks, etc. The daemon is essentially the CLI REPL but with JSONL stdin/stdout instead of interactive terminal I/O.

### Step 2: Update TUI to use persistent subprocess

**File: `src/swarmee_river/tui/app.py`**

- On app mount, spawn the daemon subprocess once (`spawn_swarmee_daemon()`)
- Replace `_start_run()` to send a `{"cmd": "query", ...}` JSONL line to the subprocess stdin instead of spawning a new process
- `_stream_output()` runs continuously on the daemon's stdout (not per-run)
- Handle `turn_complete` event to mark run as finished
- Handle `ready` event on startup
- `_stop_run()` sends `{"cmd": "interrupt"}` instead of killing the process
- `action_quit()` sends `{"cmd": "shutdown"}` then waits for exit

### Step 3: Remove forced plan-mode wrapping

**File: `src/swarmee_river/tui/app.py`**

- Remove `build_plan_mode_prompt()` call in `_start_run()` — all prompts go directly to agent
- The agent decides whether to plan based on the query, not forced wrapping
- `/plan <prompt>` still works by sending `{"cmd": "query", "text": "...", "mode": "plan"}`
- `/run <prompt>` sends with `"auto_approve": true`
- Default mode: just send the prompt as-is, let the agent decide

### Step 4: Model display fix

**File: `src/swarmee_river/tui/app.py`**

- In `model_select_options()`, replace "Auto" label with resolved model name (already computed by `resolve_model_config_summary()`)
- Add `model_info` event to daemon protocol so TUI always shows actual model being used

### Step 5: Plan sidebar buttons + completion flow

**File: `src/swarmee_river/tui/widgets.py` + `app.py`**

- Add clickable buttons (Textual `Button` widgets) for Approve / Replan / Clear below the plan panel
- On plan completion (all steps checked): show "Plan complete. Clear plan?" prompt
- Record completed plans in session data for history

### Step 6: Turn metadata (timestamp + model)

**File: `src/swarmee_river/tui/widgets.py`**

- Add optional `timestamp` and `model` params to `UserMessage` and `AssistantMessage`
- Display as dim right-aligned text: `[dim]claude-sonnet · 2:34 PM[/dim]`
- TUI populates from `model_info` event data

## Files to Modify

| File | Changes |
|------|---------|
| `src/swarmee_river/swarmee.py` | Add `--tui-daemon` flag + daemon loop |
| `src/swarmee_river/tui/app.py` | Persistent subprocess, remove plan-mode wrapping, model display, plan buttons |
| `src/swarmee_river/tui/widgets.py` | Plan buttons, turn metadata, thinking fix |
| `tests/test_tui_subprocess.py` | Tests for daemon protocol |
| `tests/test_tui_callback_handler.py` | Existing tests should still pass |

## Verification

1. Start TUI → daemon subprocess spawns, "ready" event received
2. Send first prompt → get response with thinking indicator + tool calls + text
3. Send second prompt → agent remembers first conversation (multi-turn works)
4. `/plan` sends planning query; approve/replan/clear buttons work
5. Interrupt (Ctrl+C) sends interrupt command, doesn't kill subprocess
6. Model name shown correctly (not "Auto")
7. Timestamps visible on turns
8. `pytest tests/` passes

## Priority Order for Implementation

1. **Step 1 + Step 2** — Daemon mode + TUI integration (critical path, enables multi-turn)
2. **Step 3** — Remove plan-mode wrapping (blocks user feedback)
3. **Step 4** — Model display fix (quick win)
4. **Step 6** — Turn metadata (polish)
5. **Step 5** — Plan buttons + completion flow (nice-to-have, can follow)

## Existing Code to Reuse

- `swarmee.py::create_agent()` — agent factory (lines 749-763)
- `swarmee.py::run_agent()` — query execution with model escalation (lines 813-865)
- `swarmee.py::_consent_prompt()` — consent flow (lines 697-721)
- `swarmee.py::_generate_plan()` — plan generation (lines 891-924)
- `swarmee.py::_render_plan()` — plan rendering (lines 867-889)
- `tui/app.py::_handle_tui_event()` — JSONL event dispatch (lines 1245-1345)
- `tui/app.py::parse_tui_event()` — JSONL parsing utility
- `tui/widgets.py::PlanCard` — existing plan display widget
- `session/store.py` — session persistence for daemon state

## Open Questions for Implementation

1. How to handle consent_response IPC? Current TUI path uses `input()` on parent stdin. In daemon, need a threading.Event + shared variable to unblock `_consent_prompt()` waiting inside `run_agent()`.

2. Should daemon maintain its own SessionStore, or share TUI's? Probably separate—daemon just runs queries, TUI manages UI state.

3. Model escalation—does it work in daemon? Yes, `model_manager` is in the closure, same logic applies.

4. Should we buffer partial tool output and stream via tool_progress events, or leave that for later? Leave for later (Step 3 is "enhance tool_progress" but current implementation is sufficient for MVP).
