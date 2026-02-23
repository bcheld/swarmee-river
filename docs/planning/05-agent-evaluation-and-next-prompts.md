# Agent Performance Evaluation & Next Prompts (Batch 3)

## Batch 2 Agent Performance Assessment

| Prompt | Goal | Outcome | Grade |
|--------|------|---------|-------|
| B1 | Fix consent button visibility | Buttons now use `compact=True`, heights set to `auto`, `max-height: 7`, separator added, `_signal_attention()` with bell+toast, `_flash_highlight()` with `-highlight` CSS class, `show_confirmation()` for post-decision feedback. Thorough implementation. | A |
| B2 | TextArea transcript (skipped) | User chose B3 instead. | N/A |
| B3 | Dual-mode transcript toggle | Implemented: `#transcript` (RichLog) + `#transcript_text` (TextArea, hidden by default). Ctrl+T and `/text` toggle. `_sync_transcript_text_widget()` populates TextArea from fallback lines. `_set_transcript_mode()` swaps visibility. | B |
| B4 | Copy cleanup | `_get_transcript_text()` still tries nonexistent RichLog APIs (export_text, get_text, to_text) before falling back to `_transcript_fallback_lines`. Improved but not fully cleaned. `_get_richlog_selection_text()` still probes nonexistent attributes. | C |
| B5 | Consent animation | Merged into B1 — `_signal_attention()` (toast + bell) and `_flash_highlight()` (2s border emphasis) implemented directly in the ConsentPrompt widget. | A |
| B6 | Streaming debounce | `_streaming_buffer`, `_schedule_streaming_flush()`, `_cancel_streaming_flush_timer()`, `_flush_streaming_buffer()` all implemented for text_delta events. Tool progress has separate `_schedule_tool_progress_flush()` with timer. | A |

### What went right
- Consent is now fully functional: visible buttons with compact styling, toast + bell attention, highlight animation, post-choice confirmation
- Streaming debounce is solid: separate buffer/flush mechanisms for both assistant text and tool progress
- The dual-mode toggle works mechanically — Ctrl+T swaps between RichLog and TextArea

### What the user feedback reveals

**1. "Lost visibility into tool call details"**

The old `ToolCallBlock` widget (still in widgets.py, lines 548-621) had a `Collapsible` that showed tool input details — shell commands, file paths, JSON payloads — inside an expandable block. The RichLog migration replaced this with `render_tool_start_line()` which only shows `"⚙ {tool_name} running..."`. Tool input details are captured in the `_tool_blocks` record (`record["input"] = event.get("input", {})`) but never rendered to the transcript. The `/expand <id>` command exists but users would need to know the tool_use_id and type a command — completely non-discoverable.

The previous collapsible blocks showed details like:
- `shell` → `Command: git status` / `CWD: /path`
- `file_read` → `Path: src/foo.py`
- `file_write` → `Path: src/bar.py`

Now users see only `⚙ shell running...` → `⚙ shell (2.3s) ✓`. They lost the "what" entirely.

**2. "Tool indicators should update in-place instead of new lines"**

Currently, each tool lifecycle produces 2-4 separate transcript lines:
```
⚙ shell [tool-1] running...        ← tool_start
  │ output line 1                    ← tool_progress (optional)
  │ output line 2                    ← tool_progress (optional)
⚙ shell (2.3s) ✓ [tool-1]          ← tool_result
```

Users want the start line to UPDATE to the result line, not have both appear. In a RichLog, you can't update previously written content. But you can use a different approach: render tools as short-lived Textual widgets that update in-place, then replace with a final static line.

**3. "Can't see thinking stream / want dynamic thinking like Claude Code"**

The daemon emits `{"event": "thinking", "text": "..."}` when `reasoningText` is present (callback_handler.py line 832-833). The TUI receives it and sets `_current_thinking = True` (line 3740-3741) but does NOTHING with the text content. It doesn't render the thinking text or show any dynamic indicator. The static `"thinking..."` message from `render_thinking_message()` only appears when the first `text_delta` arrives (line 3725-3728), not on the `thinking` event itself.

For models that stream extended thinking (GPT-5.2, Claude with extended thinking), the thinking content is simply discarded. Claude Code shows an animated "Thinking..." indicator with character-count progress. More importantly, models like GPT-5.2 that expose reasoning tokens should have that content shown — it's valuable context for understanding why the model is doing what it does.

**4. "Toggle loses scroll position"**

`_set_transcript_mode("text")` calls `_sync_transcript_text_widget()` which does `text_widget.load_text(text)` and then `_scroll_transcript_text_to_end()`. When switching back to rich mode, `rich_widget.scroll_end()` is called. Both always jump to the bottom. If the user was reading something mid-transcript, they lose their place in both directions.

---

## Prompt Batch 3

### Prompt C1: Inline tool input details on the start line, with in-place status updates

**Why this is high-impact:** Users lost the most valuable information from the old UI — what each tool is actually doing. The `_format_tool_input()` function already exists and produces excellent summaries (`Command: git status`, `Path: src/foo.py`). It just isn't being called in the RichLog rendering path. Additionally, the start→result lifecycle produces visual clutter by writing separate lines. This prompt fixes both.

```
TASK: Show tool input details inline on the tool start line, and update tool lines in-place when results arrive rather than appending new lines.

CONTEXT:
- File: src/swarmee_river/tui/widgets.py — render_tool_start_line() at line 57, _format_tool_input() at line 527
- File: src/swarmee_river/tui/app.py — tool_start event at line 3743, tool_input event at line 3804, tool_result event at line 3810
- The _tool_blocks dict stores tool records with "input", "status", "duration_s" fields
- The transcript is a RichLog — RichLog.write() appends content; it CANNOT update previously written content
- The old ToolCallBlock was a Static/Collapsible widget mounted in a VerticalScroll — it COULD update in place

THE PROBLEMS:
1. Tool start lines show only "⚙ shell running..." with no input details
2. Tool results write a NEW line instead of updating the start line
3. Users see clutter: start line + optional progress lines + result line per tool

APPROACH: Use a mounted Static widget (not RichLog.write) for each active tool call, then finalize it.

Since RichLog can't update in place, but we NEED in-place updates for tools, use a hybrid:
- For regular content (messages, plans, system text): continue using RichLog.write()
- For tool call blocks: mount a lightweight Static widget at the END of the RichLog
  NOTE: RichLog does NOT support mounting child widgets. This means we need a different container strategy.

REVISED APPROACH: Write tool lines to the RichLog but use a line-index tracking scheme to replace them.

Actually, the cleanest solution is:
- Use a Horizontal container at the bottom of the transcript area (above or overlaying the RichLog) as a "tool status strip"
- Active tool calls render as compact updateable Static widgets in this strip
- When a tool completes, the strip entry is replaced with a final summary, and after a short delay, the final summary is also written to the RichLog (for scrollback history)

SIMPLEST CORRECT APPROACH: Just include tool input details on the start line. Accept the two-line pattern (start + result) but make each line information-rich.

REQUIREMENTS:

### Part A: Show tool input details on the start line

1. Create a new render function render_tool_start_line_with_input():
   def render_tool_start_line_with_input(
       tool_name: str,
       *,
       tool_input: dict | None = None,
       tool_use_id: str | None = None,
   ) -> RichText:
       """Render tool start with inline input summary."""
       rendered = RichText()
       rendered.append("⚙ ", style="dim")
       rendered.append(tool_name, style="bold dim")
       if tool_input:
           summary = _format_tool_input_oneliner(tool_name, tool_input)
           if summary:
               rendered.append(f"  {summary}", style="dim")
       rendered.append(" ...", style="dim")
       return rendered

2. Create _format_tool_input_oneliner() — a single-line version of _format_tool_input():
   - shell → "$ git status" (just the command, truncated to 80 chars)
   - file_read/read → "← src/foo.py" (path only)
   - file_write/write/edit → "→ src/bar.py" (path only, arrow indicates write)
   - editor → "✎ src/baz.py" (edit indicator)
   - http_request → "⇄ GET https://..." (method + URL, truncated)
   - Other tools → truncated JSON of first key-value pair, max 60 chars
   - ALWAYS truncate the whole line to max 100 chars

3. In app.py, the tool_start event handler currently writes the start line BEFORE tool_input arrives:
   - tool_start fires first (no input yet)
   - tool_input fires shortly after (has the input dict)

   Problem: by the time tool_input arrives, the start line is already written to RichLog.

   Solution: DELAY writing the start line. When tool_start fires:
   - Create the _tool_blocks record as before
   - Set a short timer (100ms) before writing the start line
   - If tool_input arrives within that 100ms, include the input in the start line
   - If the timer expires without tool_input, write without details
   - This is a common pattern for "coalescing" sequential events

   Implementation:
   - Add _tool_pending_start: dict[str, float] = {} (tool_use_id -> timestamp)
   - On tool_start: record the tool, set timer, don't write yet
   - On tool_input: if tool is pending, cancel timer, write with details
   - Timer callback: if still pending, write without details
   - 100ms is short enough that users won't notice the delay

### Part B: Make tool result lines more informative

4. Update render_tool_result_line() to include the tool input summary:
   - "✓ shell (2.3s) — $ git status" for success
   - "✗ file_write (0.5s) → src/broken.py (error)" for failure
   This way even the result line tells users WHAT was done, not just THAT something succeeded.

### Part C: Suppress the start line for fast tools

5. If a tool completes in < 500ms AND tool_start hasn't been rendered yet (still in the pending timer):
   - Skip writing the start line entirely
   - Write ONLY the result line (which now includes input details from Part B)
   - This eliminates the start+result double-line for fast operations like file_read

### Part D: Update plain text for text mode

6. Ensure the plain_text= kwarg passed to _mount_transcript_widget() includes the input summary:
   - "⚙ shell — $ git status ..."
   - "✓ shell (2.3s) — $ git status"

CONSTRAINTS:
- The 100ms coalescing timer must be a Textual timer (self.set_timer), not threading.
- Don't break tool_progress rendering — progress chunks should still appear between start and result.
- The _tool_blocks record structure must remain compatible with /expand and render_tool_details_panel().
- Tool use IDs should NOT appear in the default display — only show them in /expand output. Users don't need to see "tool-7a3f...".
```

---

### Prompt C2: Streaming thinking indicator with content preview

**Why this is high-impact:** Users specifically called out wanting to see the thinking stream. The daemon already emits `{"event": "thinking", "text": "..."}` with the actual reasoning text, but the TUI discards it. For models that expose extended thinking (GPT-5.2, Claude extended thinking), this is valuable context. Even for models that don't, a dynamic "Thinking..." indicator with a character count (like Claude Code shows) provides better feedback than the current static text.

```
TASK: Render thinking events as a dynamic, updating indicator that shows thinking content or at minimum a live character count.

CONTEXT:
- File: src/swarmee_river/handlers/callback_handler.py — line 832: emits {"event": "thinking", "text": str(reasoningText)}
- File: src/swarmee_river/tui/app.py — line 3740: receives "thinking" event, sets _current_thinking = True but discards the text
- File: src/swarmee_river/tui/widgets.py — render_thinking_message() returns static "thinking..." RichText
- Claude Code shows: "Thinking..." with a pulsing animation and character count that updates as reasoning tokens stream
- GPT-5.2 streams reasoning_content tokens that users want to see

THE PROBLEM:
- thinking events arrive with text content but it's thrown away
- The "thinking..." placeholder only appears on the FIRST text_delta, not on the thinking event itself
- There's no visual feedback during the thinking phase (which can last 10-30+ seconds for complex queries)
- Users with GPT-5.2 get reasoning tokens that would be valuable to display

REQUIREMENTS:

### Part A: Accumulate and display thinking content

1. Add instance variables:
   _thinking_buffer: list[str] = []
   _thinking_char_count: int = 0
   _thinking_display_timer: Timer | None = None

2. When a "thinking" event arrives:
   - Extract the text: thinking_text = str(event.get("text", ""))
   - Append to _thinking_buffer
   - Update _thinking_char_count
   - If this is the FIRST thinking event in a turn (buffer was empty):
     - Write an initial thinking indicator to the transcript
   - Schedule/reschedule a display update timer (200ms debounce)

3. Create render_thinking_indicator() function in widgets.py:
   def render_thinking_indicator(
       *,
       char_count: int = 0,
       preview: str | None = None,
       elapsed_s: float = 0.0,
   ) -> RichText:
       """Dynamic thinking indicator with optional content preview."""
       rendered = RichText()
       # Animated dots (caller passes frame index)
       rendered.append("💭 ", style="dim")
       rendered.append("Thinking", style="bold dim")
       if char_count > 0:
           rendered.append(f" ({char_count} chars)", style="dim")
       if elapsed_s > 0:
           rendered.append(f" {elapsed_s:.0f}s", style="dim")
       if preview:
           # Show last ~60 chars of thinking as a dim preview
           truncated = preview[-60:].replace("\n", " ").strip()
           if truncated:
               rendered.append(f"\n  ╰ {truncated}", style="dim italic")
       return rendered

4. The display update timer callback:
   - Compute elapsed time since first thinking event
   - Get a preview string from the last chunk in _thinking_buffer
   - Write render_thinking_indicator() to the transcript
   - Since RichLog can't update in place, use the SAME approach as tool coalescing:
     DON'T write individual thinking updates to the RichLog. Instead, maintain a SINGLE
     thinking indicator widget that floats at the bottom of the visible area.

### Part B: Floating thinking indicator (not in RichLog)

5. Better approach: Add a ThinkingBar widget (similar to StatusBar) that appears when thinking is active:
   - Docked just above the status bar or at the bottom of the transcript
   - Shows: "💭 Thinking... (1,247 chars · 8s)"
   - If thinking text is available, shows a dim scrolling preview of the last line
   - Disappears when the first text_delta or tool_start event arrives

   This avoids the RichLog update problem entirely — it's a separate widget that updates in place.

6. Create ThinkingBar in widgets.py:
   class ThinkingBar(Static):
       DEFAULT_CSS = """
       ThinkingBar {
           dock: bottom;
           height: auto;
           max-height: 2;
           padding: 0 1;
           background: $surface;
           color: $text-muted;
           display: none;
       }
       """
       def show_thinking(self, *, char_count: int, elapsed_s: float, preview: str = ""):
           # render and update
           self.styles.display = "block"

       def hide_thinking(self):
           self.styles.display = "none"

7. In compose(), yield ThinkingBar(id="thinking_bar") between transcript and status bar.

8. In _handle_tui_event():
   - "thinking" event: update ThinkingBar, accumulate buffer
   - "text_delta" event: hide ThinkingBar, dismiss thinking
   - "tool_start" event: hide ThinkingBar, dismiss thinking
   - "turn_complete" event: hide ThinkingBar, reset thinking state

9. When thinking finishes (first non-thinking event arrives):
   - Write a SINGLE summary line to the RichLog: "💭 Thought for 12s (3,847 chars)"
   - Optionally: if the full thinking text was > 0 chars, make it available via /thinking command
   - Store the full thinking buffer in _last_thinking_text for /thinking to display

### Part C: /thinking command to view full reasoning

10. Add /thinking slash command:
    - Writes the full accumulated thinking text to the transcript
    - Useful for understanding why the model made certain decisions
    - If no thinking text available: "No thinking content from this turn."

11. Add /thinking to CommandPalette.TUI_COMMANDS.

### Part D: Timer-based animation

12. While thinking is active, run a 500ms interval timer that:
    - Cycles through animation frames: "Thinking.", "Thinking..", "Thinking..."
    - Updates the elapsed time counter
    - Updates the ThinkingBar display

13. Clean up the timer when thinking ends.

CONSTRAINTS:
- ThinkingBar must NOT be inside the RichLog — it's a separate dock widget.
- Thinking text may be large (10K+ chars for complex reasoning). Don't render it all — just show char count and a preview.
- The /thinking command output should be truncated to a reasonable length (e.g., last 5000 chars) with a note if truncated.
- Don't show thinking preview if the model doesn't emit reasoning text (many models send empty thinking events). In that case, just show the animated "Thinking..." with elapsed time.
- The thinking indicator must be clearly visually distinct from tool progress indicators.
```

---

### Prompt C3: Preserve scroll position across rich/text mode toggle

**Why this is high-impact:** Users complained that toggling loses their place. The fix is conceptually simple but requires mapping between two different scroll models (RichLog line index vs. TextArea character offset).

```
TASK: Preserve the user's approximate scroll position when toggling between rich and text transcript modes.

CONTEXT:
- File: src/swarmee_river/tui/app.py — _set_transcript_mode() at line 1990
- RichLog uses widget-based scrolling (scroll_offset, virtual_size)
- TextArea uses text-based scrolling (cursor position, scroll_offset)
- _transcript_fallback_lines maps 1:1 with RichLog entries (approximately)
- Currently: switching to text mode always scrolls to end; switching to rich always scrolls to end

THE PROBLEM:
User is reading something mid-transcript, presses Ctrl+T to select text, and gets jumped to the bottom. They lose their place. Same when toggling back.

APPROACH:

1. Before switching modes, capture the current scroll position as a PROPORTIONAL value:
   - For RichLog: proportion = scroll_offset.y / max(1, virtual_size.height - size.height)
   - For TextArea: proportion = scroll_offset.y / max(1, virtual_size.height - size.height)
   (Both Textual widgets expose scroll_offset and virtual_size)

2. After switching modes and populating content:
   - Apply the same proportion to the new widget:
   - target_y = proportion * max(0, virtual_size.height - size.height)
   - widget.scroll_to(0, target_y, animate=False)

3. Update _set_transcript_mode():

   def _set_transcript_mode(self, mode: str, *, notify: bool = True) -> None:
       ...
       if normalized == "text":
           # Capture position from rich widget
           proportion = self._get_scroll_proportion(rich_widget)
           self._sync_transcript_text_widget()
           rich_widget.styles.display = "none"
           text_widget.styles.display = "block"
           # Restore position in text widget (after a short delay for layout)
           self.set_timer(0.05, lambda: self._set_scroll_proportion(text_widget, proportion))
           ...
       else:
           # Capture position from text widget
           proportion = self._get_scroll_proportion(text_widget)
           text_widget.styles.display = "none"
           rich_widget.styles.display = "block"
           self.set_timer(0.05, lambda: self._set_scroll_proportion(rich_widget, proportion))
           ...

4. Helper methods:

   def _get_scroll_proportion(self, widget: Any) -> float:
       """Get 0.0-1.0 proportion of current scroll position."""
       try:
           scroll_y = widget.scroll_offset.y
           max_scroll = widget.virtual_size.height - widget.size.height
           if max_scroll <= 0:
               return 1.0
           return min(1.0, max(0.0, scroll_y / max_scroll))
       except Exception:
           return 1.0  # Default to bottom if anything fails

   def _set_scroll_proportion(self, widget: Any, proportion: float) -> None:
       """Set scroll position from 0.0-1.0 proportion."""
       try:
           max_scroll = widget.virtual_size.height - widget.size.height
           if max_scroll <= 0:
               return
           target = int(proportion * max_scroll)
           widget.scroll_to(0, target, animate=False)
       except Exception:
           pass

5. The 50ms timer delay is needed because Textual doesn't compute virtual_size immediately after display changes. The widget needs one layout pass first.

6. Special case: if the user was at the BOTTOM (proportion > 0.95), stay at the bottom in the new mode. This preserves the "following new content" behavior.

CONSTRAINTS:
- The proportion mapping is approximate — rich and text modes have different line heights. That's fine; "approximately right" is much better than "always bottom."
- Do NOT try to map exact line numbers — the line counts differ between RichLog and TextArea.
- The 50ms delay should be the minimum needed for layout. If it's not enough, try 100ms.
- This must work in both directions: rich→text and text→rich.
```

---

### Prompt C4: Render tool input summary on the result line (quick win, no timer complexity)

**Why this is a good fallback if C1's timer approach is too complex:** If the agent struggles with the coalescing timer in C1, here's a simpler approach that still addresses the core user complaint: just put the tool input summary on the RESULT line. The start line stays as-is ("⚙ shell running...") but the result line becomes "✓ shell (2.3s) — $ git status".

```
TASK: Include a one-line tool input summary on the tool_result line rendered to the transcript.

CONTEXT:
- File: src/swarmee_river/tui/widgets.py — render_tool_result_line() at line 63
- File: src/swarmee_river/tui/app.py — tool_result handler at line 3810, _tool_blocks stores records with "input" field
- The tool_input event populates record["input"] before tool_result fires

REQUIREMENTS:

1. Add a _format_tool_input_oneliner() function to widgets.py:
   def _format_tool_input_oneliner(tool_name: str, tool_input: dict | None) -> str:
       if not tool_input or not isinstance(tool_input, dict):
           return ""
       canonical = tool_name.lower().strip()
       if canonical in ("shell", "bash", "python_repl"):
           cmd = str(tool_input.get("command", "")).strip()
           return f"$ {cmd[:80]}" if cmd else ""
       if canonical in ("file_read", "read"):
           path = str(tool_input.get("path", "")).strip()
           return f"← {path}" if path else ""
       if canonical in ("file_write", "write"):
           path = str(tool_input.get("path", "")).strip()
           return f"→ {path}" if path else ""
       if canonical in ("editor", "edit", "file_edit"):
           path = str(tool_input.get("path", tool_input.get("file_path", ""))).strip()
           return f"✎ {path}" if path else ""
       if canonical == "http_request":
           method = str(tool_input.get("method", "GET")).upper()
           url = str(tool_input.get("url", "")).strip()
           return f"{method} {url[:60]}" if url else ""
       if canonical in ("glob", "file_search", "file_list"):
           pattern = str(tool_input.get("pattern", tool_input.get("query", ""))).strip()
           return pattern[:60] if pattern else ""
       if canonical == "retrieve":
           query = str(tool_input.get("query", "")).strip()
           return f"? {query[:60]}" if query else ""
       # Generic: show first key=value
       for key, value in tool_input.items():
           val_str = str(value).strip()
           if val_str and len(val_str) < 80:
               return f"{key}: {val_str[:60]}"
           break
       return ""

2. Update render_tool_result_line() signature and implementation:
   def render_tool_result_line(
       tool_name: str,
       *,
       status: str,
       duration_s: float,
       tool_use_id: str | None = None,
       tool_input: dict | None = None,   # NEW parameter
   ) -> RichText:
       succeeded = status == "success"
       status_glyph = "✓" if succeeded else "✗"
       status_style = "green" if succeeded else "red"
       rendered = RichText()
       rendered.append(status_glyph, style=f"bold {status_style}")
       rendered.append(f" {tool_name} ({duration_s:.1f}s)")
       input_summary = _format_tool_input_oneliner(tool_name, tool_input)
       if input_summary:
           rendered.append(f" — {input_summary}", style="dim")
       if not succeeded:
           rendered.append(f" ({status})", style="dim red")
       return rendered

3. In app.py tool_result handler, pass the stored input to the render function:
   - record["input"] should be populated by the time tool_result fires
   - Change the render call to:
     render_tool_result_line(
         tool_name,
         status=status,
         duration_s=duration_s,
         tool_use_id=tid,
         tool_input=record.get("input") if record else None,
     )
   - Also update the plain_text to include the summary

4. ALSO update render_tool_start_line() to include input when available:
   def render_tool_start_line(
       tool_name: str,
       *,
       tool_use_id: str | None = None,
       tool_input: dict | None = None,   # NEW parameter
   ) -> RichText:
       rendered = RichText()
       rendered.append("⚙ ", style="dim")
       rendered.append(tool_name, style="dim")
       input_summary = _format_tool_input_oneliner(tool_name, tool_input)
       if input_summary:
           rendered.append(f" — {input_summary}", style="dim")
       rendered.append(" ...", style="dim")
       return rendered

   This won't help for the initial tool_start (input hasn't arrived yet), but it future-proofs the function.

CONSTRAINTS:
- _format_tool_input_oneliner must never return more than ~100 characters
- Don't remove tool_use_id parameter from render_tool_result_line — keep backward compat even if we don't display it by default
- The input summary on the result line should be dim/muted — the status glyph and tool name are the primary info
```

---

## Execution Order

1. **C4** first — it's a quick, no-risk win that immediately restores tool input visibility on result lines. Can be done independently.
2. **C1** second — the full coalescing timer approach for richer tool display. Builds on C4's `_format_tool_input_oneliner`.
3. **C2** third — thinking indicator is the biggest new feature. Independent of tool changes.
4. **C3** fourth — scroll position fix for the mode toggle. Quick and self-contained.

C4 and C3 are fully independent and can be sent in parallel.
C1 depends on C4 (reuses `_format_tool_input_oneliner`).
C2 is independent of all others.
