# Sidebar Refinement & Control Plane Polish (Complete)

Date: February 25, 2026

## Overview
This work completes the implementation of the 4-tab "Control Plane" architecture (Engage | Agents | Scaffold | Settings) with significant UX polish, dynamic orchestrator context, and session scoping functionality.

## Changes Summary

### 1. Engage Tab → Execution View: Orchestrator Status Display
- **Status line** shows orchestrator configuration dynamically: `"Orchestrator: anthropic/balanced (claude-sonnet-4)"`
- Updates immediately when model is changed via the model selector
- **Empty state** message: `"No active plan. Enter a prompt to get started, or switch to Planning..."`

### 2. Plan Approval Workflow
- **Approve / Replan / Clear buttons** now hidden by default (`display: none` CSS)
- Only visible when `pending_prompt` is set (i.e., agent has generated a plan awaiting user approval)
- Buttons appear inline with the plan text for easy interaction

### 3. Dynamic Sidebar Width Clamping
- **"Start Plan" button** triggers sidebar expansion; now clamps to a minimum **1:1 ratio** (sidebar : transcript)
  - Prevents shrinking past 50% width to allow full plan editing
  - Automatically switches to Planning view
  - Seeds prompt input with `/plan` command
- **"Continue" button** clamps back to default **2:1 ratio** (sidebar : transcript)
  - Restores normal execution-focused layout
  - Switches back to Execution view to show running tasks

### 4. Session Timeline Cleanup & Formatting
- **Relative timestamps** replace raw ISO 8601 datetimes: `"2s ago"`, `"5m ago"`, `"1h ago"`
- **Filtered events**: `after_model_call` events are hidden; only tool calls, errors, and invocations are shown
- **Cleaner event titles**: `"shell (2.1s)"` instead of `"⚙ tool: shell (2.1s)"`
- Improves readability for quick session review

### 5. Agents Tab → "Overview" Sub-View
- **"Profile" renamed to "Overview"** (better mental model for users unfamiliar with agent terminology)
- Header text changed from "Effective Session Profile" to **"Orchestrator Agent"**
- **Help text** added: *"The orchestrator processes your prompts and coordinates tools."*

### 6. Agents Tab → Team Help Text
- Added descriptive help: *"Define multi-agent presets the orchestrator can invoke via the swarm tool."*
- Clarifies the relationship between the orchestrator and sub-agents

### 7. Agents Tab → Tools & Safety Polish
- **Cleaner placeholder text**:
  - `"tool_consent: ask | allow | deny"` (instead of cryptic JSON)
  - `"tool_allowlist: tool1, tool2, ..."` (instead of technical override syntax)
- **Help text**: *"Control which tools the orchestrator can use and set safety policies."*
- Makes safety overrides more accessible to non-technical users

### 8. SidebarDetail Scrolling Improvements
- **Preview content** now wrapped in `VerticalScroll` widget with gray scrollbar theme
- **Container sizing** changed from `height: auto` to `height: 1fr` so it fills available space and scrolls naturally
- Prevents long previews from pushing buttons off-screen

### 9. Scaffold Tab Cleanup
- **Artifact titles simplified**: `"config.yaml"` with subtitle `"file · /path/to/config.yaml"`
- **KB empty state**: `"No knowledge bases connected. Use /kb to connect one."`
- **Session panel scrollbar** colors matched to gray theme for visual consistency

### 10. Settings Tab Functionality
#### Environment Variables (MVP)
- Auto-populates with relevant env vars (`SWARMEE_MODEL_PROVIDER`, API keys, etc.)
- API keys and sensitive values are masked (e.g., `sk-***...` for `OPENAI_API_KEY`)
- **"Set" button** writes changes to `os.environ` and refreshes the app state

#### Directory Scoping (MVP)
- **Current scope path** displayed at the top (e.g., `"Scope: /Users/bcheld/dev/stuff/.swarmee"`)
- **Path input field** allows direct text entry
- **DirectoryTree widget** starts at `$HOME` (not `.`) for better navigation
- **"Set Scope" button**:
  - Creates `.swarmee/` directory in the selected path if it doesn't exist
  - Updates `SWARMEE_STATE_DIR` environment variable
  - Updates the displayed scope path
  - Future session commands respect the new scoping

## Architecture Notes

### View Files Created
- `src/swarmee_river/tui/views/engage.py` — Execution, Planning, and Session views
- `src/swarmee_river/tui/views/agents.py` — Orchestrator, Roster, and Builder views
- `src/swarmee_river/tui/views/scaffold.py` — Context, SOPs, KBs, and Artifacts views
- `src/swarmee_river/tui/views/settings.py` — Environment and Scoping views

### State Management
- Added `engage_view_mode`, `agents_view_mode`, `scaffold_view_mode`, `settings_view_mode` to `AppState`
- Each tab manages its own sub-view state via dedicated setter methods in `app.py`
- Tab button presses route to view switcher logic vs. command actions

### Session Scoping Logic
- Updated `scope_root()` in `src/swarmee_river/state_paths.py` to **prioritize explicit `.swarmee/` directories**
  - Walks up the directory tree looking for `.swarmee/` first
  - Falls back to git repository root, then `~/.swarmee/`
  - Enables per-project or per-task workspace isolation

## Testing
- All 136 TUI tests pass
- No regressions in callback handling, event routing, or session persistence
- DirectoryTree integration successful; no coroutine warnings in core tests

## UX Benefits
1. **Clear orchestrator context** — Users always see what model and tier the orchestrator is using
2. **Guided planning** — Expand/clamping UI makes plan approval workflow obvious
3. **Cleaner history** — Relative timestamps and filtered events reduce cognitive load
4. **Informed configuration** — Help text and improved naming make agent/tool/safety setup self-documenting
5. **Local workspace scoping** — Users can now manage multiple projects without polluting `~/.swarmee/`

## Next Steps (Future)
- Implement interactive plan step editing (checkboxes + comment fields) in Planning view
- Add team preset visual builder (dropdown for models, checkbox grid for tools)
- Enhance KB management UI (browse, upload, delete)
- Add search/filter to Artifact list by title and content
- Implement live cost/token tracking per context item

---

**Commit message:**
```
refactor(tui): implement 4-tab sidebar control plane with orchestrator context, 
session scoping, and UX polish

- Restructure sidebar from 6 tabs to 4: Engage, Agents, Scaffold, Settings
- Add orchestrator status display in Execution view
- Implement dynamic sidebar width clamping (1:1 planning, 2:1 execution)
- Polish session timeline with relative timestamps and filtered events
- Rename Profile to Overview, improve agent terminology and help text
- Add scrolling to SidebarDetail previews; match scrollbar colors to theme
- Implement Settings tab with env var management and directory scoping
- Update scope_root() to prioritize .swarmee/ directories for per-project isolation
- All tests passing; no regressions in callback or event routing
```
