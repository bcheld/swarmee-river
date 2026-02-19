# TUI Current State Evaluation

## Overview

The TUI has progressed significantly since the initial JSONL event protocol was introduced. Most of the original 11 UX recommendations are now implemented. This document evaluates the current state against the original recommendations and identifies remaining gaps.

## Original 11 Recommendations: Status

| # | Recommendation | Status | Notes |
|---|---|---|---|
| 1 | **Streaming output & progressive rendering** | ✅ Delivered | JSONL events, Markdown rendering, typed widgets, thinking indicator |
| 2 | **Copy-friendly text selection** | ✅ Delivered | `/copy`, `/copy all`, `/copy last`, `/copy issues` commands |
| 3 | **Dynamic tool progress indicators** | ⚠️ Partial | Static collapsible tool blocks; no real-time stdout streaming |
| 4 | **Inline artifact metadata** | ⚠️ Partial | Artifacts listed; no icons, diff badges, or smart organization |
| 5 | **Dynamic plan generation** | ❌ Missing | Plan workflow still plan-first; doesn't populate incrementally from chat |
| 6 | **Keyboard navigation & discovery** | ✅ Delivered | Command palette with `/` prefix, arrow key nav, search |
| 7 | **Model/tier selector** | ⚠️ Partial | Selector works; "Auto" label confusing; doesn't show resolved model |
| 8 | **Session persistence** | ✅ Delivered | `tui_session.json` saves history, artifacts, plan, model overrides |
| 9 | **Error/warning toast notifications** | ❌ Missing | Errors shown in issues panel, not as toasts |
| 10 | **Token/cost counter** | ❌ Missing | Usage data available but not surfaced in status bar |
| 11 | **Timestamps on turns** | ⚠️ Partial | Timestamps/model not visible on individual turns |

---

## Current Architecture: One-Shot Subprocesses

**Current Flow:**
```
User types prompt → TUI spawns python -m swarmee_river.swarmee <prompt>
Subprocess runs agent(prompt), prints JSONL events
Subprocess exits
→ **All conversation history lost**

User types another prompt → TUI spawns NEW subprocess
New agent created, has no memory of previous turn
→ **Each prompt starts fresh**
```

**Key Problem:** The Strands `Agent` class accumulates messages automatically across calls, but we spawn a new agent each time. The CLI REPL keeps a single agent alive for the entire session—we need to do the same in the TUI.

---

## User Feedback on Current Implementation

Based on recent review, the user identified these specific issues:

### 1. **Thinking Indicator Not Appearing**
- ThinkingIndicator widget defined but not appearing in practice
- Likely: `thinking` events not reaching TUI, or event flow broken
- **Fix:** Debug event emission and TUI event parsing

### 2. **Collapsible Tool Details Are Static**
- Tool blocks show initial input details only
- No updates as tool runs (no streaming stdout, no live progress)
- **Fix:** Enhance `tool_progress` events with richer content (e.g., partial output, live status updates)

### 3. **Plan Workflow Too Rigid**
- TUI forces plan mode: every prompt wrapped by `build_plan_mode_prompt()`
- User prefers: dynamic plan population (chat leads to incremental edits to displayed plan)
- User doesn't want to start in plan mode
- **Fix:** Remove forced wrapping; let agent decide; add approve/replan/clear buttons to sidebar

### 4. **No Buttons for Plan Actions**
- `/approve`, `/replan`, `/clearplan` are command-based (requires typing)
- **Fix:** Add sidebar buttons for these actions

### 5. **Plan Completion Flow Missing**
- Upon final plan step completion, should prompt user to clear plan
- Finished plans should be recorded in conversation history
- **Fix:** Add post-completion UX

### 6. **"Auto" Model Label Confusing**
- Dropdown shows "Auto (bedrock/balanced (claude-sonnet))" but "Auto" is confusing
- **Fix:** Display resolved model name directly, not "Auto"

### 7. **No Timestamps or Model Stamps on Turns**
- Each turn should show timestamp and model used
- **Fix:** Add metadata to UserMessage and AssistantMessage widgets

### 8. **Artifacts Organization Weak**
- Current: flat list of artifact paths
- User prefers: no file tree (VSCode/Sagemaker already have that), but better organization
- **Defer:** Can improve later (not critical path)

---

## Recommended Next Steps

### **High Priority (Blocking multi-turn usage)**

1. **Implement Long-Running Daemon Subprocess**
   - Add `--tui-daemon` mode to `swarmee.py`
   - TUI spawns daemon once on startup
   - TUI sends `{"cmd": "query", "text": "..."}` via stdin instead of spawning new process
   - Daemon emits events continuously to stdout
   - Same Agent instance persists across turns → full conversation history
   - **Impact:** Enables multi-turn conversation, fixing the fundamental architecture issue

### **Medium Priority (Polish & usability)**

2. **Fix Thinking Indicator Event Flow**
   - Debug why `thinking` events not appearing
   - Ensure TUI callback handler emits them
   - Verify TUI event parsing handles them

3. **Remove Forced Plan-Mode Wrapping**
   - Delete `build_plan_mode_prompt()` logic from `_start_run()`
   - Let agent decide whether to plan based on query content
   - `/plan <prompt>` sends to agent as planning mode request
   - `/run <prompt>` sends with auto_approve=true

4. **Add Plan Sidebar Buttons**
   - Textual `Button` widgets in plan panel: Approve / Replan / Clear
   - On completion: show "Plan complete. Clear?" prompt
   - Record completed plans in session

5. **Fix Model Display**
   - Replace "Auto" label with resolved model name
   - Add `model_info` event to daemon protocol

6. **Add Turn Metadata (Timestamp + Model)**
   - Enhance UserMessage/AssistantMessage widgets
   - Display dim right-aligned: `[dim]model · timestamp[/dim]`

### **Low Priority (Nice-to-have)**

7. **Enhance Tool Progress Display**
   - Stream stdout chunks to TUI via tool_progress events
   - Show live activity inside collapsible blocks

8. **Toast Notifications for Errors**
   - Use `self.notify()` on error events
   - Keep issues panel as secondary view

9. **Token/Cost Counter**
   - Surface `usage` data in status bar
   - Show token count, estimated cost

10. **Better Artifact Organization**
    - Smart grouping (by type, by turn, by date)
    - Icon badges for file types

---

## Architecture Decision: Daemon Subprocess

### Why It Matters

The current one-shot subprocess architecture is fundamentally at odds with multi-turn conversation. Every prompt starts fresh because:

1. New `Agent` instance created each time
2. Strands Agent's internal message history is lost
3. No shared context between turns
4. Tools see no conversation history

The Strands `Agent` class **automatically accumulates messages** within a single instance. We just need to keep that instance alive.

### Solution: `--tui-daemon` Mode

**Add to `swarmee.py`:**
- New argparse flag: `--tui-daemon`
- When set, don't create agent and exit
- Instead: create agent, emit `{"event": "ready"}`, enter stdin read loop
- Read JSONL commands: `{"cmd": "query", "text": "..."}` or `{"cmd": "consent_response", "choice": "y"}`
- For each query, call existing `run_agent()` → callbacks emit JSONL events to stdout
- Emit `{"event": "turn_complete"}` when done
- Reuse ALL existing logic: hooks, conversation managers, consent flow, plan generation

**Update TUI (`tui/app.py`):**
- On app mount: spawn daemon once with `--tui-daemon`
- Replace `_start_run()`: send `{"cmd": "query", ...}` to daemon stdin instead of spawning subprocess
- Single long-lived `_stream_output()` thread reads daemon stdout continuously
- Handle `turn_complete` event to mark run as finished
- `_stop_run()` sends `{"cmd": "interrupt"}` instead of killing process
- `action_quit()` sends `{"cmd": "shutdown"}` then waits

**Benefits:**
- Same Agent instance across all turns → full conversation history
- Reuses all existing logic (no duplication)
- Minimal changes to TUI (just IPC protocol)
- Compatible with all existing features (hooks, plan generation, consent, escalation)

---

## Files Affected

| File | Change | Complexity |
|---|---|---|
| `src/swarmee_river/swarmee.py` | Add `--tui-daemon` mode + stdin command loop | Medium |
| `src/swarmee_river/tui/app.py` | Persistent subprocess, JSONL stdin writer, event handling | Medium |
| `src/swarmee_river/tui/widgets.py` | Plan buttons, turn metadata, event flow fixes | Low |
| `tests/test_tui_subprocess.py` | Tests for daemon protocol | Low |

---

## Verification Checklist

After implementation:

- [ ] Start TUI → daemon subprocess spawns, emits `{"event": "ready"}`
- [ ] Send first prompt → response with thinking + tools + text
- [ ] Send second prompt → agent remembers first conversation (multi-turn works)
- [ ] `/plan <prompt>` generates plan; buttons (Approve/Replan/Clear) work
- [ ] Interrupt (Ctrl+C) pauses turn without killing subprocess
- [ ] Model selector shows resolved name (not "Auto")
- [ ] Timestamps visible on turns
- [ ] `pytest tests/` passes
- [ ] Session restore works with new architecture
