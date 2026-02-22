# High-Impact Agent Prompts

Prompts designed to be sent to coding agents, ordered by expected impact. Each addresses a gap that is architecturally significant, non-obvious, or requires cross-cutting awareness of how the TUI, daemon, hooks, and widget layers interact.

These prompts are informed by a fresh assessment of the codebase against the planning docs and user feedback as of 2026-02-22.

---

## Assessment Summary

| Area | Planning Docs Said | Reality |
|------|--------------------|---------|
| Daemon subprocess | High priority, blocking | Fully implemented and working |
| Multi-turn conversation | Blocked by one-shot arch | Working via daemon |
| Text selectability | `/copy` commands sufficient | Users say transcript text is not selectable; Static widgets confirmed |
| Tool consent UX | Text-based y/n/a/v | Still text-only; users want buttons/checkboxes |
| Plan actions | Need sidebar buttons | Buttons exist in Plan tab; but consent has no equivalent |
| Side panel tabs | 4 tabs: Plan, Help, Issues, Artifacts | All present but shallow; no KB/SOP/context source integration |
| Token/cost counter | Missing | StatusBar now shows tokens, cost, cache stats |
| Turn metadata | Missing | Timestamps + model stamps implemented |

**Key gaps remaining:** transcript selectability, consent UX, side panel depth, and context source integration. The prompts below target these plus deeper architectural improvements that unlock compounding value.

---

## Prompt 1: Replace Static transcript widgets with a selectable RichLog

**Why this is high-impact:** Users praised the formatting but can't select text. The transcript uses `Static` widgets which are not text-selectable in Textual. This is the #1 user complaint. Fixing it also unlocks native search-within-transcript and eliminates the need for the `/copy` command workaround for most use cases.

**Why a junior wouldn't think of it:** The naive fix is to wrap each message in a `TextArea(read_only=True)`, but that creates hundreds of competing TextArea instances with independent scroll, focus, and cursor state - it would be a performance and UX disaster. The correct approach uses Textual's `RichLog` widget which is purpose-built for append-only styled output with text selection support.

```
TASK: Replace the transcript's VerticalScroll + Static widget pattern with a single RichLog widget.

CONTEXT:
- File: src/swarmee_river/tui/app.py
- File: src/swarmee_river/tui/widgets.py
- The transcript area (#transcript) is currently a VerticalScroll container.
- Individual messages (UserMessage, AssistantMessage, SystemMessage, ToolCallBlock, ConsentCard, PlanCard) are mounted as child Static widgets inside it.
- Static widgets do NOT support text selection in Textual. Users cannot select/copy text from the output pane.
- The transcript has a MAX_TRANSCRIPT_WIDGETS limit (500) and a pruning mechanism.

REQUIREMENTS:
1. Replace the VerticalScroll(id="transcript") with a RichLog(id="transcript") from textual.widgets.
2. RichLog supports .write() with Rich renderables (Markdown, Text, Panel, etc.) and has built-in text selection.
3. Refactor _mount_transcript_widget() to append Rich renderables to the RichLog instead of mounting child widgets.
4. For UserMessage: write a Rich Text with "[bold cyan]YOU>[/bold cyan] {text}" markup + optional dim timestamp line.
5. For AssistantMessage: this is trickier because it accumulates deltas. Create a pattern where:
   - On first text_delta, write a placeholder line to the RichLog.
   - Track the current assistant message buffer in an instance variable.
   - On text_complete/finalize, replace the last entry or append the final rendered RichMarkdown.
   - NOTE: RichLog does not support in-place updates of previously written content. The recommended pattern is to use RichLog's `clear()` and re-render approach only if absolutely necessary, OR accept that streaming deltas appear as progressive appends (which is actually fine UX - each delta chunk appears as it arrives). The simplest correct approach: buffer all deltas silently, then write the complete RichMarkdown on text_complete. Show a "thinking..." line during streaming that gets visually superseded.
6. For ToolCallBlock: write a Rich Panel or Text with the tool header. On result, write a follow-up line. Collapsible expand/collapse is not available in RichLog, so show a compact one-liner (e.g., "⚙ shell (2.3s) ✓") with the option to expand details via a /expand command or by clicking.
7. For ConsentCard: write a Rich Panel with border showing the consent prompt and [y]/[n]/[a]/[v] options.
8. For PlanCard: write a Rich Panel with the plan steps.
9. For SystemMessage: write dim Rich Text.
10. Remove the MAX_TRANSCRIPT_WIDGETS pruning logic - RichLog handles its own max_lines parameter.
11. Update _get_all_text() and copy-related methods to use RichLog's exported text.
12. Keep the /copy commands working but note that users can now also natively select text.
13. Update action_copy_selection() to pull selected text from the RichLog.
14. Ensure scroll-to-bottom behavior is preserved (RichLog has auto_scroll parameter).

IMPORTANT CONSTRAINTS:
- Do NOT create separate TextArea widgets per message. That approach does not work at scale.
- Do NOT break the existing JSONL event handling in _handle_tui_event(). The refactoring is in how events are rendered, not how they are received.
- Test that keyboard shortcuts (Esc to interrupt, Ctrl+C to copy, etc.) still work after the change.
- The ToolCallBlock collapsible behavior will be lost. This is acceptable - a compact one-liner is preferable to broken selection.
```

---

## Prompt 2: Consent as an inline interactive widget with clickable buttons

**Why this is high-impact:** Tool approval is "extremely clunky" per user feedback. Currently users must type `y`, `n`, `a`, or `v` (or use `/consent y`). In a GUI TUI, this should be clickable buttons. This is the #2 user complaint and directly affects the moment-to-moment usability of the tool.

**Why a junior wouldn't think of it:** The challenge is that consent is blocking - the daemon's `ToolConsentHooks.before_tool_call()` holds a lock and waits synchronously on `self._prompt()`. Simply adding buttons to the TUI isn't enough; you need the buttons to resolve the same IPC path that the keyboard shortcut `y/n/a/v` uses. The widget must be ephemeral (auto-removed after decision) and must handle the case where the daemon times out or the user interrupts.

```
TASK: Replace the text-based consent prompt with an inline interactive consent widget containing clickable buttons.

CONTEXT:
- File: src/swarmee_river/tui/widgets.py - ConsentCard is currently a Static with markup text showing [y] [n] [a] [v]
- File: src/swarmee_river/tui/app.py - _submit_consent_choice() sends {"cmd": "consent_response", "choice": "..."} to daemon
- File: src/swarmee_river/hooks/tool_consent.py - ToolConsentHooks.before_tool_call() blocks on self._prompt() waiting for response
- The consent_prompt JSONL event carries context (tool name, input details) and options list
- Current keyboard handling: on_key() catches single y/n/a/v keys when _consent_pending is True

REQUIREMENTS:
1. Create a new ConsentPrompt widget (NOT a Static) that contains:
   - A Rich-formatted context display showing the tool name and input summary
   - A horizontal row of Textual Button widgets: "Yes (y)", "No (n)", "Always (a)", "Never (v)"
   - Each button styled distinctly: Yes=success variant, No=error variant, Always=primary, Never=warning
   - Keyboard shortcut hints in the button labels

2. When a consent_prompt event arrives:
   - Mount the ConsentPrompt widget at the bottom of the transcript (or as an overlay anchored above the prompt box)
   - Focus automatically shifts to the first button (Yes) for keyboard accessibility
   - Tab/arrow keys navigate between buttons
   - The existing y/n/a/v single-key shortcuts should ALSO still work (don't remove keyboard support)

3. When a button is pressed:
   - Call _submit_consent_choice() with the corresponding choice letter
   - Immediately remove or collapse the ConsentPrompt widget (replace with a dim one-liner showing the decision: "✓ shell allowed" or "✗ shell denied")
   - Restore focus to the prompt TextArea

4. Handle edge cases:
   - If the daemon sends a new consent_prompt while one is already displayed, replace the existing one
   - If the user interrupts the run (/stop or Esc), dismiss any pending consent widget
   - If the daemon's turn_complete arrives before the user responds, dismiss the consent widget

5. Update the side panel's #consent TextArea:
   - Either remove it entirely (the inline widget replaces it) OR repurpose it as a consent history log showing past decisions

6. Remove the dedicated #consent TextArea from the compose() layout if the inline widget fully replaces it.

7. Preserve the existing on_key() handler for y/n/a/v as a keyboard shortcut path - the buttons are additive, not replacing keyboard support.

STYLING:
- The consent widget should visually stand out from normal transcript content (use a border or distinct background)
- Buttons should be large enough to be easily clickable in a terminal
- The widget should not take more than 3-4 lines of vertical space
```

---

## Prompt 3: Context Sources tab with pluggable data connectors

**Why this is high-impact:** Users want to "connect additional information sources to bring into context." The side panel currently has 4 shallow tabs. Adding a Context Sources tab with the ability to attach URLs, files, KB references, and SOP selections would transform the TUI from a chat interface into an actual workbench. This is the #3 user complaint.

**Why a junior wouldn't think of it:** The naive approach is to add a text box where users paste URLs. The high-impact approach is to build a pluggable connector model that feeds into the daemon's `<system-reminder>` injection pipeline - this means context sources persist across turns and benefit from prompt caching (because they go through `prompt_cache.queue_if_changed()`). The architecture must respect the cache-safety constraints documented in cache_guidance.md.

```
TASK: Add a "Context" tab to the side panel with pluggable context source management, wired into the daemon's system-reminder pipeline.

CONTEXT:
- File: src/swarmee_river/tui/app.py - compose() defines the side panel with TabbedContent containing Plan, Help, Issues, Artifacts tabs
- File: src/swarmee_river/context/prompt_cache.py - queue_if_changed() injects <system-reminder> blocks into user messages
- File: src/swarmee_river/swarmee.py - the daemon reads commands from stdin and runs queries against a persistent Agent
- File: docs/planning/cache_guidance.md - prompt caching rules: stable prefix, dynamic context via system-reminders
- File: src/swarmee_river/packs.py - Pack system provides tools, SOPs, and system prompts

The daemon already supports <system-reminder> injection for preflight context, project map, and active SOPs. We need to extend this to user-specified context sources.

REQUIREMENTS:

### Phase 1: Context tab UI
1. Add a new TabPane("Context", id="tab_context") to the side panel TabbedContent, positioned after Plan and before Help.
2. The Context tab contains:
   - A header label "Active Context Sources"
   - A scrollable list of currently attached context sources, each showing:
     - Source type icon (📄 file, 🌐 URL, 📚 KB, 📋 SOP, 📝 note)
     - Source name/path (truncated if long)
     - A "✕" remove button per source
   - Below the list, an "Add Source" button row with type selectors:
     - "File" - opens a text input for a file path
     - "Note" - opens a text input for free-form context text
     - "SOP" - opens a selection list of available SOPs from sops/ directory
     - "KB" - opens a text input for a knowledge base ID
3. When a source is added or removed, the change is sent to the daemon as a new command:
   - {"cmd": "set_context_sources", "sources": [{"type": "file", "path": "/path/to/file"}, {"type": "note", "text": "..."}, {"type": "sop", "name": "bugfix"}, {"type": "kb", "id": "kb-123"}]}

### Phase 2: Daemon-side context injection
4. In swarmee.py daemon loop, handle the "set_context_sources" command:
   - For "file" sources: read the file content, truncate to a reasonable limit (e.g., 8000 chars), and queue it via prompt_cache.queue_if_changed() with a unique key like "user_context_file_{path}"
   - For "note" sources: queue the raw text via prompt_cache.queue_if_changed()
   - For "sop" sources: resolve the SOP path and load its content, queue via prompt_cache
   - For "kb" sources: store the KB ID so that retrieve() calls include it in the next query
5. Context sources should be injected as <system-reminder> blocks, NOT as changes to the system prompt (to preserve prompt cache).
6. When a source is removed, queue an empty string for its key to clear it from the next turn's reminders.

### Phase 3: Persistence
7. Save the active context sources list in the TUI session file (tui_session.json) so they persist across TUI restarts.
8. On TUI mount, restore context sources from session and send them to the daemon after it's ready.

### Phase 4: Slash command interface
9. Add /context command as an alternative to the tab UI:
   - /context add file <path>
   - /context add note <text>
   - /context add sop <name>
   - /context remove <index>
   - /context list
   - /context clear

IMPORTANT CONSTRAINTS:
- Context source content must go through the <system-reminder> pipeline, NOT modify the system prompt. This is critical for prompt cache stability.
- File content should be read lazily (on first query after adding) and cached. Re-read only if the file's mtime changes.
- Total injected context should be capped (e.g., 32K chars across all sources) with a warning if exceeded.
- The SOP selector should dynamically list available SOPs by scanning sops/ directory and any enabled pack sop directories.
```

---

## Prompt 4: Streaming tool output in collapsible blocks

**Why this is high-impact:** Tool calls (especially `shell`) currently show only a static header that updates to a one-liner result. Users praised the detailed information but tools like shell commands can run for tens of seconds with no visible progress. Streaming stdout from tools into the TUI provides the "am I stuck?" signal that prevents users from interrupting good runs and gives immediate value for long-running operations.

**Why a junior wouldn't think of it:** The data is already flowing through the daemon but being discarded. The callback handler in `callback_handler.py` receives tool stdout but only emits a summary. The fix requires threading stdout chunks through the JSONL event protocol as `tool_progress` events with content, then having the TUI render them progressively. The tricky part is rate-limiting the render updates to avoid flooding the TUI with redraws.

```
TASK: Stream tool stdout/stderr through the daemon's JSONL event protocol and render it progressively in the TUI's tool blocks.

CONTEXT:
- File: src/swarmee_river/handlers/callback_handler.py - receives tool execution callbacks
- File: src/swarmee_river/tui/widgets.py - ToolCallBlock shows tool header + collapsible details
- File: src/swarmee_river/tui/app.py - _handle_tui_event() dispatches JSONL events
- The daemon already emits tool_start and tool_result events
- ToolCallBlock.update_progress(chars) exists but only updates a character count

REQUIREMENTS:

### Daemon side (callback handler):
1. In the callback handler, when a tool produces stdout/stderr output, emit periodic JSONL events:
   {"event": "tool_progress", "tool_use_id": "...", "content": "line of output\n", "stream": "stdout"}
2. Rate-limit emissions to at most one event per 200ms per tool invocation to avoid flooding.
3. Buffer output between emissions and send the accumulated chunk.
4. On tool completion, flush any remaining buffer before emitting tool_result.

### TUI side (widget + event handler):
5. In _handle_tui_event(), when a tool_progress event with "content" field arrives:
   - Find the corresponding ToolCallBlock by tool_use_id
   - Append the content to the block's output buffer
   - Update the block's display
6. In ToolCallBlock, add an output display area:
   - Below the header, show the last N lines (e.g., 8) of accumulated output
   - Style output as dim monospace text
   - Auto-scroll within the block as new output arrives
   - If using the RichLog approach from Prompt 1, render tool output as indented dim text lines appended to the log
7. On tool_result, collapse the live output and show the final summary one-liner.

### Strands tool integration:
8. For the shell tool specifically, capture subprocess stdout/stderr via the existing Strands tool infrastructure. The shell tool in strands-agents-tools pipes output through the tool's return value, but we need streaming DURING execution.
9. If direct stdout capture is not feasible for all tools, at minimum:
   - Emit a heartbeat tool_progress event every 2 seconds for long-running tools: {"event": "tool_progress", "tool_use_id": "...", "elapsed_s": 5.2}
   - This gives the TUI enough to show "⚙ shell running... (5.2s)" as a live counter

CONSTRAINTS:
- Do NOT buffer unlimited output in memory. Cap the retained output per tool at 4KB.
- Rate-limit TUI redraws from tool_progress events to at most 5 per second.
- The tool_progress event schema must be backwards compatible (TUI should gracefully ignore unknown fields).
```

---

## Prompt 5: Session conversation persistence across TUI restarts

**Why this is high-impact:** The TUI saves prompt history, plan text, and preferences across restarts - but NOT the actual conversation messages. When a user restarts the TUI, all conversation context is lost even though it could be restored. The daemon creates a fresh Agent each time. This makes the TUI feel disposable rather than persistent, which undermines trust in longer workflows.

**Why a junior wouldn't think of it:** The obvious approach (save messages to disk, reload on start) misses the prompt caching implications. If you reload 50K tokens of conversation history, you pay full input token cost with zero cache hits on the first turn because the cache is cold. The correct approach is to reload messages into the Agent AND emit them as a synthetic event stream so the TUI rebuilds its visual transcript, while accepting that the first turn after restart will be cache-cold (but subsequent turns will build cache normally).

```
TASK: Persist conversation messages across TUI daemon restarts and restore them on startup.

CONTEXT:
- File: src/swarmee_river/session/store.py - SessionStore already has save_messages() and load_messages() methods
- File: src/swarmee_river/swarmee.py - daemon creates Agent in _build_agent_runtime(), agent.messages is the conversation history
- File: src/swarmee_river/tui/app.py - _spawn_daemon() starts the daemon, _load_session() restores TUI state
- The daemon emits {"event": "ready", "session_id": "..."} on startup

REQUIREMENTS:

### Daemon side:
1. When the daemon receives a "query" command and the turn completes successfully, persist the agent's messages via SessionStore.save_messages(). Do this on a background thread to avoid blocking.
2. Add a new daemon command: {"cmd": "restore_session", "session_id": "..."}
   - Load messages from SessionStore.load_messages(session_id)
   - Set agent.messages = loaded_messages
   - Emit {"event": "session_restored", "turn_count": N, "message_count": M} to confirm
   - Then emit synthetic events to rebuild the TUI transcript:
     {"event": "replay_turn", "role": "user", "text": "...", "timestamp": "..."}
     {"event": "replay_turn", "role": "assistant", "text": "...", "model": "...", "timestamp": "..."}
   - Emit {"event": "replay_complete"} when done
3. After emitting "ready", if the daemon detects a previous session exists for the same working directory, emit {"event": "session_available", "session_id": "...", "turn_count": N} so the TUI can offer to restore.

### TUI side:
4. When the TUI receives "session_available", show a system message: "Previous session found (N turns). Type /restore to resume or /new to start fresh."
5. Add /restore and /new commands:
   - /restore: send {"cmd": "restore_session", "session_id": "..."} to daemon
   - /new: do nothing (start fresh, which is the current default)
6. When replay_turn events arrive, render them in the transcript exactly as if they were live events (UserMessage, AssistantMessage with model/timestamp metadata).
7. When replay_complete arrives, show a system message: "Session restored (N turns)."
8. Save the current session_id in tui_session.json so it's available on next TUI startup.

### Message format:
9. Messages should be saved in the Strands Agent message format (the same dicts that agent.messages contains). Do NOT re-serialize or transform them.
10. Include a version field in the saved session so future format changes can be handled gracefully.

CONSTRAINTS:
- Message persistence should be append-only during a session (write after each turn, don't rewrite the entire file).
- Cap saved message history at 200 messages (roughly 100 turns). Older messages are dropped from the persisted file but remain in the live agent's memory.
- The restore flow should NOT attempt to re-execute any tools or re-invoke any APIs. It's purely loading message history for context.
- Handle gracefully: corrupted session files, version mismatches, missing messages.
```

---

## Prompt 6: Smart plan-to-execution pipeline with inline step tracking

**Why this is high-impact:** The plan workflow exists but the transition from plan to execution is a "mode switch" that loses visual continuity. When a plan is approved and execution begins, users can't see which plan step the agent is currently working on. The plan tab shows the plan, the transcript shows tool calls, but nothing connects them. Making the agent report progress against plan steps creates an "are we there yet?" signal that builds trust and lets users intervene at the right granularity.

**Why a junior wouldn't think of it:** This requires modifying the agent's system prompt during execution to include the plan AND adding a hook that matches tool calls against plan steps. It's a cross-cutting concern spanning the prompt injection layer, the hook layer, and the TUI widget layer. The key insight is that the plan steps can be injected as `<system-reminder>` content (cache-safe) and the agent can be instructed to emit step-transition markers.

```
TASK: Wire plan step tracking through from the agent's execution to the TUI's plan display, so users can see which step is currently executing.

CONTEXT:
- File: src/swarmee_river/planning.py - WorkPlan Pydantic model with steps list
- File: src/swarmee_river/tui/widgets.py - PlanCard has mark_step_complete(step_index) method
- File: src/swarmee_river/tui/app.py - _dispatch_plan_action("approve") starts execution with auto_approve=True
- File: src/swarmee_river/hooks/tool_policy.py - enforces plan-mode tool restrictions
- File: src/swarmee_river/context/prompt_cache.py - queue_if_changed() for system-reminder injection

Currently: when a plan is approved, execution starts but there's no mechanism for the agent to signal which plan step it's working on.

REQUIREMENTS:

### Agent-side step tracking:
1. When a plan is approved and execution begins, inject the plan steps into the agent's context via <system-reminder> with instructions:
   "You are executing the following plan. Before starting each step, emit a brief status message indicating which step you are beginning. Format: 'Starting step N: <description>'. After completing a step, emit: 'Completed step N.'"
2. This injection should use prompt_cache.queue_if_changed("active_plan", ...) so it's cache-safe.

### Daemon-side step detection:
3. In the daemon, monitor the agent's text output for step transition markers.
4. When the agent emits "Starting step N" or "Completed step N", emit a JSONL event:
   {"event": "plan_step_update", "step_index": N-1, "status": "in_progress"|"completed"}
5. Alternatively, add a lightweight tool called "plan_progress" that the agent can call:
   plan_progress(step=2, status="in_progress", note="Setting up database schema")
   This is more reliable than text parsing and gives richer metadata.

### TUI-side step visualization:
6. In _handle_tui_event(), handle "plan_step_update" events:
   - Call PlanCard.mark_step_in_progress(step_index) for "in_progress" (show a ▶ icon)
   - Call PlanCard.mark_step_complete(step_index) for "completed" (show ☑ icon)
7. Add mark_step_in_progress() to PlanCard widget (currently only mark_step_complete exists).
8. In the Plan tab's TextArea, update the step display to show current progress.
9. When all steps are completed, emit a plan_complete event and show the "Plan complete. Clear?" prompt.
10. Update the StatusBar to show "Step 3/7" during plan execution.

### Plan-progress tool approach (recommended over text parsing):
11. Add a "plan_progress" tool to the tool registry that:
    - Accepts: step (int), status ("in_progress" | "completed"), note (optional str)
    - Emits the plan_step_update JSONL event
    - Returns a confirmation message to the agent
    - Is always available during execute mode (add to tool_policy's allowed tools)
12. In the system-reminder injection for the active plan, include: "Use the plan_progress tool to report your progress on each step."

CONSTRAINTS:
- The plan_progress tool should be lightweight (no API calls, no file I/O).
- Plan step indices are 0-based internally but 1-based in agent-facing text.
- If the agent skips a step or does steps out of order, handle gracefully (don't crash).
- The plan injection MUST go through <system-reminder>, NOT modify the system prompt.
```

---

## Prompt 7: Unified error recovery with toast notifications and retry affordances

**Why this is high-impact:** Errors currently appear in the Issues tab (which users may not be looking at) and as text in the transcript. There are no toasts, no retry buttons, and no distinction between transient errors (rate limits, timeouts) and permanent errors (bad credentials, tool not found). A unified error UX with appropriate retry affordances would dramatically reduce the "what went wrong?" debugging cycle.

**Why a junior wouldn't think of it:** Textual has `self.notify()` for toast notifications, but the real value is in classifying errors and offering the right recovery action. Rate limit errors should auto-retry with backoff. Model errors should offer to escalate tier. Tool errors should offer to retry or skip. This requires a classification layer that understands the error taxonomy of AWS Bedrock, OpenAI, and Ollama providers.

```
TASK: Add error classification, toast notifications, and inline retry/skip affordances for errors in the TUI.

CONTEXT:
- File: src/swarmee_river/tui/app.py - errors currently written to transcript and issues panel
- File: src/swarmee_river/session/models.py - maybe_escalate() handles retryable model errors
- File: src/swarmee_river/tui/widgets.py - no error-specific widgets exist
- Textual provides self.notify(message, severity="error"|"warning"|"information") for toast popups

REQUIREMENTS:

### Error classification:
1. Create an error classifier function that categorizes errors into:
   - TRANSIENT: rate limits, throttling, timeouts, 5xx responses -> auto-retry with backoff
   - ESCALATABLE: model capacity errors, context too long -> offer tier escalation
   - TOOL_ERROR: tool execution failures -> offer retry or skip
   - AUTH_ERROR: credential/permission failures -> show fix instructions, don't retry
   - FATAL: unrecoverable errors -> show error, stop run
2. The classifier should handle error patterns from Bedrock (ThrottlingException, ModelNotReadyException), OpenAI (rate_limit_error, context_length_exceeded), and Ollama (connection refused).

### Toast notifications:
3. When an error event arrives in _handle_tui_event():
   - Classify the error
   - Show a Textual toast via self.notify() with appropriate severity
   - Toast message should be concise: "Rate limited - retrying in 5s" or "shell tool failed"
4. Keep the existing issues panel logging for full error details.

### Inline retry affordances:
5. For TOOL_ERROR: after the error in the transcript, show an inline widget with two buttons:
   - "Retry" - resend the same tool call
   - "Skip" - tell the agent to continue without the tool result
   Both should send appropriate commands to the daemon.
6. For ESCALATABLE: show a widget with:
   - "Escalate to [next tier]" - trigger model escalation
   - "Continue" - keep trying with current tier
7. For TRANSIENT: auto-retry with exponential backoff (1s, 2s, 4s, max 30s), show a countdown toast. After 3 retries, fall through to ESCALATABLE.

### Daemon protocol:
8. Enhance error events with classification metadata:
   {"event": "error", "message": "...", "category": "transient|escalatable|tool_error|auth_error|fatal", "retryable": true, "tool_use_id": "...", "retry_after_s": 5}
9. Add daemon commands for error recovery:
   {"cmd": "retry_tool", "tool_use_id": "..."}
   {"cmd": "skip_tool", "tool_use_id": "..."}

CONSTRAINTS:
- Toast notifications should auto-dismiss after 5 seconds for transient errors, persist until dismissed for fatal errors.
- Don't show toasts for warnings (keep those in the issues panel only).
- Auto-retry should respect the daemon's existing escalation logic in maybe_escalate().
- Rate limit retry should NOT create new API calls - it should queue the retry in the daemon's event loop.
```

---

## Prompt 8: Side panel SOP browser with activation toggle

**Why this is high-impact:** SOPs exist in the codebase but are only activatable via `--sop <name>` CLI flag at startup. Users can't browse, preview, or toggle SOPs from within the TUI. Since SOPs directly shape agent behavior (injected into system prompt via `<system-reminder>`), making them accessible from the side panel gives users real-time control over the agent's operational framework.

**Why a junior wouldn't think of it:** SOPs are loaded from multiple sources (local `sops/` dir, pack SOP dirs, `strands-agents-sops` package) with a resolution chain. Toggling an SOP mid-session means injecting/removing it from the `<system-reminder>` pipeline without breaking prompt cache. The SOP text must go through `prompt_cache.queue_if_changed()` with a stable key.

```
TASK: Add an SOP browser to the TUI side panel that lets users browse, preview, and toggle SOPs during a session.

CONTEXT:
- File: src/swarmee_river/sops/ - contains bugfix.sop.md, code-change.sop.md, repo-onboarding.sop.md, security-review.sop.md, swarm.sop.md, tool-migration.sop.md
- File: src/swarmee_river/packs.py - enabled_sop_paths() returns SOP directories from enabled packs
- File: src/swarmee_river/swarmee.py - resolve_effective_sop_paths() resolves SOPs from CLI args + packs + strands-agents-sops
- File: src/swarmee_river/context/prompt_cache.py - queue_if_changed("active_sop", content) injects SOP into system-reminders
- SOPs are markdown files that guide agent behavior patterns

REQUIREMENTS:

### SOP discovery:
1. Create a function that discovers all available SOPs from:
   - Local sops/ directory
   - Enabled pack sop directories
   - The strands-agents-sops package (if installed)
2. Return a list of {name, path, source, first_paragraph_preview} for each SOP.

### Side panel integration:
3. Extend the Help tab OR add a new "SOPs" tab (if adding a tab, place it after Context).
4. Display a list of available SOPs, each with:
   - SOP name
   - Source label (local, pack name, or strands-sops)
   - A toggle switch or checkbox to activate/deactivate
   - The first paragraph of the SOP as a preview
5. When a user toggles an SOP on:
   - Read the SOP markdown content
   - Send to daemon: {"cmd": "set_sop", "name": "bugfix", "content": "..."}
   - Daemon queues it via prompt_cache.queue_if_changed("active_sop", combined_sop_content)
6. When toggled off:
   - Send {"cmd": "set_sop", "name": "bugfix", "content": null}
   - Daemon re-queues with the SOP removed from the combined content
7. Multiple SOPs can be active simultaneously (they get concatenated).

### Slash command:
8. /sop list - show available SOPs
9. /sop activate <name> - activate an SOP
10. /sop deactivate <name> - deactivate an SOP
11. /sop preview <name> - show full SOP content in transcript

CONSTRAINTS:
- SOP content injection MUST use the <system-reminder> pipeline, not modify the system prompt.
- Keep a combined SOP key in prompt_cache rather than one key per SOP (to minimize cache key proliferation).
- SOP toggling should take effect on the NEXT query, not retroactively.
```

---

## Prompt 9: Prompt metrics with token estimation and context budget visualization

**Why this is high-impact:** The StatusBar shows token usage AFTER a turn completes, but users have no visibility into how much of the context window is consumed BEFORE they send a query. When context is nearly full, the next query triggers expensive compaction. Showing a pre-send token estimate and a visual context budget bar lets users make informed decisions about when to compact, start a new session, or trim context sources.

**Why a junior wouldn't think of it:** Token estimation before sending requires counting tokens client-side, which means either bundling a tokenizer or using a heuristic (chars/4). The real insight is that the daemon already tracks `prompt_tokens_est` and `budget_tokens` in context events - these just need to be surfaced prominently rather than buried in the status bar. Adding a visual fill bar transforms an abstract number into an intuitive "fuel gauge."

```
TASK: Add a context budget visualization to the TUI showing pre-send token estimates and a visual fill bar.

CONTEXT:
- File: src/swarmee_river/tui/app.py - StatusBar.set_context() already receives prompt_tokens_est and budget_tokens
- File: src/swarmee_river/tui/widgets.py - StatusBar.refresh_display() shows "ctx 45k/200k" as text
- File: src/swarmee_river/hooks/tui_metrics.py - emits context events with token estimates
- The daemon emits {"event": "context", "prompt_tokens_est": N, "budget_tokens": M} after each turn

REQUIREMENTS:

1. Add a visual context budget bar to the prompt area (below or beside the prompt TextArea):
   - Show a horizontal bar filled proportionally to prompt_tokens_est/budget_tokens
   - Color coding: green (<50%), yellow (50-80%), red (>80%)
   - Text label: "Context: 45k / 200k (23%)"
   - When >80%, add a warning indicator and tooltip: "Context nearly full. Consider /compact or /new."

2. Add a prompt size estimator:
   - When the user types in the prompt TextArea, estimate the token count of their input
   - Use a simple heuristic: len(text) / 4 (rough chars-to-tokens ratio)
   - Show this in the prompt metrics area: "~250 tokens"
   - Update on every keystroke (debounced to 200ms)

3. Add a /compact command:
   - Send {"cmd": "compact"} to the daemon
   - Daemon triggers conversation manager summarization
   - Show progress: "Compacting context..." toast
   - On completion: update the context budget display with new values

4. Update the context budget display in real-time:
   - After each turn, the daemon sends updated context metrics
   - The budget bar should animate smoothly when values change
   - If context exceeds 90%, show a persistent warning in the status bar

5. In the prompt_bottom area (where prompt_metrics Static lives):
   - Replace the empty Static with the budget bar widget
   - Keep the model selector dropdown on the right

CONSTRAINTS:
- Token estimation is approximate - do NOT bundle a full tokenizer library.
- The budget bar should be compact (1 line height, fits in the prompt footer area).
- The /compact command should only work when no query is active.
```

---

## Prompt 10: Keyboard-driven workflow with modal action sheets

**Why this is high-impact:** Power users in terminal applications expect keyboard-driven workflows. The current TUI has good keyboard basics (Enter to submit, Esc to interrupt, y/n/a/v for consent) but lacks modal action sheets that let users quickly perform common operations without typing commands. Think of it like VS Code's Ctrl+Shift+P command palette, but for agent operations.

**Why a junior wouldn't think of it:** The CommandPalette widget already exists for `/` commands, but it's a text filter - not an action sheet. A modal action sheet is contextual: during a consent prompt it shows consent actions, during plan review it shows plan actions, during idle it shows general actions. The key insight is that the appropriate actions change based on TUI state, and the action sheet should reflect this.

```
TASK: Extend the command palette into a context-sensitive action sheet triggered by a keyboard shortcut.

CONTEXT:
- File: src/swarmee_river/tui/widgets.py - CommandPalette widget filters slash commands by prefix
- File: src/swarmee_river/tui/app.py - on_key() handles keyboard shortcuts, command palette shown on "/" key

REQUIREMENTS:

1. Add a Ctrl+K (or Ctrl+Space) keyboard shortcut that opens a context-sensitive action sheet.

2. The action sheet shows different actions based on current TUI state:

   When IDLE (no query active, no consent pending):
   - "New query" (focus prompt)
   - "Plan mode" (prefix prompt with /plan)
   - "Run mode" (prefix prompt with /run)
   - "Restore session" (if previous session available)
   - "Compact context"
   - "Switch model tier" -> sub-menu with tier options

   When RUNNING (query active):
   - "Stop run" (Esc)
   - "View plan progress" (switch to Plan tab)
   - "View issues" (switch to Issues tab)

   When CONSENT PENDING:
   - "Allow (y)"
   - "Deny (n)"
   - "Always allow (a)"
   - "Never allow (v)"

   When PLAN REVIEW (plan generated, awaiting action):
   - "Approve plan"
   - "Replan"
   - "Clear plan"
   - "Edit plan" (future: allow inline plan editing)

3. Each action has a keyboard shortcut hint displayed on the right side.
4. Actions are navigable with arrow keys and selectable with Enter.
5. The action sheet dismisses on Esc or after selecting an action.
6. Actions execute immediately on selection (no confirmation step).

7. The existing "/" command palette should continue to work as-is. The action sheet is an overlay on top, not a replacement.

CONSTRAINTS:
- The action sheet should render as a centered overlay, not inline in the transcript.
- Use Textual's layer system for the overlay.
- Maximum 10 items visible at once; scroll if more.
- Each item: icon + label + shortcut hint, single line.
```

---

## Usage Notes

These prompts are ordered by estimated user-facing impact:

1. **Text selectability** - Fixes the #1 user complaint
2. **Consent buttons** - Fixes the #2 user complaint
3. **Context sources** - Fixes the #3 user complaint
4. **Streaming tool output** - Addresses tool progress transparency gap
5. **Session persistence** - Enables persistent workflows
6. **Plan step tracking** - Connects planning to execution
7. **Error recovery** - Reduces debugging friction
8. **SOP browser** - Makes SOPs accessible without CLI flags
9. **Context budget viz** - Prevents surprise compaction
10. **Action sheets** - Power user acceleration

Prompts 1-3 directly address current user feedback. Prompts 4-6 address architectural gaps that compound over time. Prompts 7-10 are polish that differentiates the product.

Each prompt is designed to be self-contained and can be sent to a coding agent independently. However, Prompt 1 (RichLog) should be done before Prompt 4 (streaming tool output) since the rendering approach changes.
