# Agent Performance Evaluation & Next Prompts (Batch 2)

## Agent Performance Assessment

### What the agents accomplished (from Prompt Batch 1)

| Prompt | Goal | Outcome | Grade |
|--------|------|---------|-------|
| 1. RichLog transcript | Make text selectable | Migrated to RichLog. Render functions created. Old Static widgets preserved as legacy. **But text is still not selectable.** | C |
| 2. Consent buttons | Clickable consent | ConsentPrompt widget built with Button widgets, wired into on_button_pressed handler. **But buttons are not visible to users.** | C |
| 3. Context Sources tab | Pluggable context panel | Fully implemented: Context tab with file/note/SOP/KB sources, context_sources_list VerticalScroll, Add/Remove buttons, /context commands, SOP catalog in separate SOPs tab. | A |
| 4. Streaming tool output | Progressive tool display | render_tool_progress_chunk(), render_tool_heartbeat_line(), _queue_tool_progress_content() all implemented. Tool output streams to transcript. | A |
| 5. Session persistence | Survive TUI restart | Not directly assessed but session_id tracking, /restore, /new commands added to CommandPalette. | B |
| 6. Plan step tracking | Inline step progress | plan_step_update events handled, PlanCard.mark_step_in_progress() added, StatusBar.set_plan_step() added, render_plan_panel_with_status() with ▶/☑/☐ markers. plan_progress tool whitelisted in tool_policy.py. | A |
| 7. Error recovery | Toast + retry/skip | ErrorActionPrompt widget with Retry/Skip/Escalate/Continue buttons. Toast notifications via _notify(). Error category handling in event dispatch. | A |
| 8. SOP browser | Toggle SOPs in panel | Full SOPs tab with catalog, _refresh_sop_catalog(), /sop commands. | A |
| 9. Context budget viz | Token fuel gauge | ContextBudgetBar widget with animated fill bar, color-coded thresholds, tooltip warning, prompt token estimate. Replaces old static prompt_metrics. | A |
| 10. Action sheet | Ctrl+K context menu | ActionSheet widget with keyboard nav, state-aware action lists, Dismissed/ActionSelected messages. | A |

### Key diagnosis of failures

**Prompt 1 failed because of a wrong assumption in the prompt itself.** I stated that RichLog "has built-in text selection support" — this is false. [Textual issue #5333](https://github.com/Textualize/textual/issues/5333) confirms RichLog does NOT support mouse text selection. The agents faithfully migrated to RichLog and added fallback methods (`_get_richlog_selection_text` tries `selected_text`, `selection_text`, `get_selection_text` etc.) but none of those APIs exist on RichLog. The transcript is now a RichLog that can display Rich renderables but users still cannot select text with their mouse/cursor. **This is my fault, not the agent's.**

**Prompt 2 partially failed due to CSS height constraints.** The ConsentPrompt widget is correctly built: 4 Button widgets (Yes/No/Always/Never) in a Horizontal, wired to `_submit_consent_choice()` via `on_button_pressed()`. The problem is visual: the widget has `max-height: 4` with `#consent_context` at `height: 2` and `#consent_actions` at `height: 1`. Textual Buttons with variants (success/error/primary/warning) render with borders and padding that require more than 1 row of height. The `height: 1` on the button row clips the buttons so they're invisible. Additionally, the ConsentPrompt is placed outside the pane area (between StatusBar and prompt_box) at a level where it may not receive enough vertical space.

### Pattern observations

1. **Agents are excellent at widget creation and event wiring** — ConsentPrompt, ErrorActionPrompt, ActionSheet, ContextBudgetBar are all well-structured with proper CSS, compose(), message handling.
2. **Agents struggle with visual layout tuning** — height constraints, overflow behavior, and how Textual renders buttons in tight spaces are hard to get right without visual testing.
3. **Agents implement what they're told verbatim** — when the prompt said "use RichLog which has text selection," they used RichLog. They didn't independently verify the claim. Prompts must be factually correct about library capabilities.
4. **Agents handle cross-cutting concerns well** — the plan_progress tool whitelisting in tool_policy.py, consent button IDs matching on_button_pressed patterns, context budget wiring to status bar — all correctly threaded across files.

---

## Remaining User Issues (Priority Order)

1. **Text selectability in transcript** — Users STILL cannot select text. RichLog doesn't support it. This requires a fundamentally different approach.
2. **Consent buttons invisible** — The widget exists but CSS clips the buttons. Users don't know how to respond.

---

## Prompt Batch 2

### Prompt B1: Fix consent widget visibility — increase height, fix button rendering

**Why this is the highest-priority fix:** Users see the consent prompt appear but cannot see the buttons. This is a pure CSS/layout bug. The widget, event wiring, and IPC are all correct — only the visual rendering is broken.

**Root cause:** `ConsentPrompt #consent_actions` has `height: 1` but Textual Buttons with variant styling (borders, padding) need at least 3 rows. The `max-height: 4` on the parent clips the overflow.

```
TASK: Fix the ConsentPrompt widget so its buttons are visible and clickable.

CONTEXT:
- File: src/swarmee_river/tui/widgets.py — ConsentPrompt class (line 213)
- File: src/swarmee_river/tui/app.py — CSS for #consent_prompt (line 1592), compose() yields it at line 1836, on_button_pressed at line 4856

THE PROBLEM:
Users report that "response options are not visible" in the consent box. The ConsentPrompt widget has four Button widgets (Yes/No/Always/Never) but the CSS constrains them into an invisible space:

1. ConsentPrompt has max-height: 4
2. #consent_context has height: 2  (uses 2 of the 4 rows)
3. #consent_actions has height: 1  (only 1 row left for 4 buttons)
4. Textual Buttons with variant="success"|"error"|"primary"|"warning" render with borders and padding that need AT LEAST 3 rows of height
5. Result: buttons are rendered but clipped — users see the context text but not the response options

FIXES REQUIRED:

1. In widgets.py, update ConsentPrompt DEFAULT_CSS:
   - Remove max-height: 4 (or increase to max-height: 8)
   - Change #consent_context from height: 2 to height: auto (let it size to content)
   - Change #consent_actions from height: 1 to height: 3 (enough for bordered buttons)
   - OR better: change #consent_actions to height: auto so Textual calculates the needed height

2. In app.py CSS for #consent_prompt:
   - Change min-height: 0 to min-height: 5 (guarantee enough room)
   - Ensure height: auto is preserved so it can grow

3. Make the buttons more compact to save space. For each button in compose():
   - Add compact=True parameter (Textual Button supports this)
   - This removes borders and padding, making buttons fit in 1 row
   - Example: Button("Yes (y)", id="consent_choice_y", variant="success", compact=True)

4. ALTERNATIVELY (if compact=True doesn't make them visible enough):
   - Instead of Textual Button widgets, use styled Static widgets that look like buttons
   - Each Static renders as: "[on green] Yes (y) [/]  [on red] No (n) [/]  [on blue] Always (a) [/]  [on yellow] Never (v) [/]"
   - Wire mouse_down or click events to dispatch the choice
   - This is more predictable than Textual Button height behavior

5. Test that after the fix:
   - The consent prompt appears with VISIBLE response options
   - Users can click the buttons with a mouse
   - Users can still press y/n/a/v keys (the on_key handler on ConsentPrompt must still work)
   - Arrow keys navigate between choices
   - After choosing, the widget hides and focus returns to the prompt

6. Also verify the placement: ConsentPrompt is yielded between StatusBar and prompt_box (line 1836 of app.py). This is outside the #panes container. Confirm it renders in a visible area. If it's hidden behind other widgets, move it INSIDE the prompt_box Vertical, above the PromptTextArea, so it appears directly above where the user is typing.

IMPORTANT:
- Do NOT change the button IDs (consent_choice_y, consent_choice_n, consent_choice_a, consent_choice_v) — they are referenced by on_button_pressed().
- Do NOT change the _submit_consent_choice() flow — only fix the visual rendering.
- Do NOT remove the keyboard shortcut handling in ConsentPrompt.on_key().
```

---

### Prompt B2: Make transcript text selectable via a TextArea-backed dual view

**Why this is critical:** This is the #1 user complaint and the previous attempt (RichLog migration) did not fix it because RichLog does not support text selection ([Textual issue #5333](https://github.com/Textualize/textual/issues/5333)). We need a different approach entirely.

**The correct approach:** A read-only TextArea can be selected from. The transcript already maintains a plain-text fallback buffer (`_transcript_fallback_lines`). We can layer a TextArea on top of or alongside the RichLog, letting users toggle between "rich view" (pretty but not selectable) and "text view" (plain but selectable). Or, more practically: replace RichLog with a TextArea that receives ANSI-stripped plain text, since users have told us selectability matters more than formatting.

```
TASK: Make the transcript pane text-selectable by replacing the RichLog with a read-only TextArea, or by adding a selectable text layer alongside it.

CONTEXT:
- File: src/swarmee_river/tui/app.py — transcript is a RichLog(id="transcript") yielded at line 1769
- File: src/swarmee_river/tui/widgets.py — render_*() functions return Rich renderables (RichText, RichMarkdown, RichPanel)
- The app maintains _transcript_fallback_lines: list[str] for plain-text copy support
- _mount_transcript_widget() writes Rich renderables to the RichLog AND records plain_text to the fallback buffer
- RichLog does NOT support mouse text selection (confirmed: https://github.com/Textualize/textual/issues/5333)
- Textual TextArea(read_only=True) DOES support mouse text selection and Ctrl+C copy

THE FUNDAMENTAL PROBLEM:
RichLog renders beautiful styled output but users cannot select any of it. The /copy commands work as a workaround but users expect to click-and-drag to select text, which is standard in any text interface.

APPROACH — Replace RichLog with read-only TextArea:

1. In compose(), replace:
     yield RichLog(id="transcript")
   with:
     yield TextArea(text="", read_only=True, show_line_numbers=False, id="transcript", soft_wrap=True)

2. Update _mount_transcript_widget() to append plain text to the TextArea instead of writing Rich renderables:
   - Instead of transcript.write(renderable), do:
     current = transcript.text
     new_text = plain_text or str(renderable)
     transcript.text = current + new_text + "\n"
   - NOTE: TextArea.text assignment is the correct API. Do NOT use .load_text() or .insert() unless you specifically need cursor positioning.
   - NOTE: For performance, if transcript.text exceeds a threshold (e.g., 500KB), trim from the front.

3. The render_*() functions in widgets.py remain available for the plain-text conversion. Add plain-text equivalents:
   - render_user_message() already takes text + timestamp — just format as "YOU> {text}\n{timestamp}"
   - render_assistant_message() — use the raw markdown text (not Rich-formatted)
   - render_tool_start_line() — "⚙ {tool_name} running..."
   - render_tool_result_line() — "⚙ {tool_name} ({duration_s:.1f}s) ✓" or "✗ ({status})"
   - render_consent_panel() — "Consent: {context}\n  [y] Yes  [n] No  [a] Always  [v] Never"
   - render_plan_panel() — "Plan: {summary}\n  ☐ 1. step...\n  ☑ 2. step..."
   - render_system_message() — just the text itself

4. Ensure all callers of _mount_transcript_widget() pass a plain_text= kwarg. Many already do — verify all paths in _handle_tui_event() do as well.

5. Auto-scroll: After appending text, scroll the TextArea to the bottom:
     transcript.move_cursor_to_end_of_text()  # This scrolls to bottom in TextArea
   OR use scroll_end() if available.

6. Text selection and copy:
   - TextArea with read_only=True supports mouse text selection natively in Textual.
   - Ctrl+C / Cmd+C copies selected text via Textual's built-in clipboard support.
   - Update action_copy_selection() to use transcript.selected_text (this is a real TextArea property, unlike the nonexistent RichLog equivalent).

7. Keep the /copy commands working — they should pull from transcript.text instead of the fallback buffer.

8. Remove the _transcript_fallback_lines buffer and all related code (_record_transcript_fallback, _TRANSCRIPT_MAX_LINES) since the TextArea IS the single source of truth now.

9. Update _get_transcript_text() to simply return transcript.text.

10. Update _get_richlog_selection_text() — rename it to _get_transcript_selection_text() and just return transcript.selected_text.

TRADEOFFS:
- We lose Rich formatting (colors, bold, markdown rendering). The transcript will be plain monospace text.
- We GAIN text selection, which users have explicitly asked for twice.
- The overall output is still well-formatted — tool results show ⚙/✓/✗, plans show ☐/☑/▶, messages show YOU>, metadata shows model · timestamp. It's just not colorized.
- If rich formatting is wanted back later, consider a dual-pane approach: RichLog for display, TextArea as a hidden selectable overlay. But that's complexity for later.

PERFORMANCE:
- TextArea handles large text well but has limits. Cap at 200KB by trimming oldest content.
- Debounce scroll-to-bottom calls during rapid streaming (text_delta events can fire hundreds of times per response).
- For streaming assistant text: buffer deltas and flush every 200ms instead of appending each delta individually.

IMPORTANT CONSTRAINTS:
- Do NOT try to make RichLog selectable — it cannot be done with the current Textual version.
- Do NOT create a separate TextArea per message — one single TextArea for the entire transcript.
- Do NOT remove the render_*() functions from widgets.py — they are still used by the consent panel, plan panel, etc. for Rich rendering in the sidebar.
- Ensure keyboard shortcuts still work: Esc to interrupt, Ctrl+C to copy, Enter to submit from prompt.
- The TextArea must NOT steal focus from the prompt. Set focusable=False or handle focus carefully. Users should be able to click into the transcript to select text, then click back to the prompt to type. BUT the prompt should be the default focused widget.
```

---

### Prompt B3: Add a "copy mode" toggle as an interim selectability solution

**Why this matters:** If the TextArea approach in B2 proves too visually degraded (users love the formatting), here's an alternative: add a toggle that temporarily replaces the RichLog with a selectable TextArea showing the same content as plain text. Think of it like "View Source" in a browser.

```
TASK: Add a /text command and Ctrl+T keybinding that toggles the transcript between "rich mode" (RichLog, pretty but not selectable) and "text mode" (TextArea, plain but selectable).

CONTEXT:
- The transcript is currently a RichLog(id="transcript")
- _transcript_fallback_lines contains plain-text versions of all transcript content
- Users want to select text but also praised the rich formatting

REQUIREMENTS:

1. In compose(), yield BOTH widgets in the same position, one hidden:
   yield RichLog(id="transcript")
   yield TextArea(text="", read_only=True, show_line_numbers=False, id="transcript_text", soft_wrap=True)

   Default: RichLog visible (display: block), TextArea hidden (display: none).

2. Add a toggle method _toggle_transcript_mode() that:
   - If currently in rich mode:
     - Populate the TextArea with "\n".join(self._transcript_fallback_lines)
     - Hide RichLog (display: none), show TextArea (display: block)
     - Show toast: "Text mode: select text with mouse. /text to return."
     - self._transcript_mode = "text"
   - If currently in text mode:
     - Hide TextArea (display: none), show RichLog (display: block)
     - Scroll RichLog to bottom
     - Show toast: "Rich mode restored."
     - self._transcript_mode = "rich"

3. Bind Ctrl+T to _toggle_transcript_mode() in the key bindings.

4. Add /text to the command palette and slash command handler.

5. While in text mode, new events should STILL append to both:
   - RichLog gets Rich renderables (for when user switches back)
   - TextArea gets plain text appended
   - _transcript_fallback_lines gets updated (already happening)

6. In text mode, action_copy_selection() should use the TextArea's .selected_text property.

7. CSS for #transcript_text should match #transcript dimensions exactly (same width, height, border, scrollbar styling).

8. When entering text mode, move cursor to the end of the TextArea so it's scrolled to bottom.

CONSTRAINTS:
- The toggle must be instant — no re-rendering or re-processing of content.
- Both widgets share the same grid slot (overlay positioning, or swap visibility).
- The TextArea in text mode should NOT be editable (read_only=True).
```

---

### Prompt B4: Harden the RichLog-to-TextArea fallback for copy operations

**Why this matters regardless of which selectability approach is chosen:** The current `_get_richlog_selection_text()` method tries a shotgun of nonexistent API methods (`selected_text`, `selection_text`, `get_selection_text`, `export_selection`, etc.) hoping one works. None of them do. Meanwhile `_get_transcript_text()` tries `export_text`, `get_text`, `to_text` — also nonexistent. The fallback to `_transcript_fallback_lines` works but the code is fragile and misleading. Clean this up.

```
TASK: Clean up transcript text export and copy methods to use reliable APIs only.

CONTEXT:
- File: src/swarmee_river/tui/app.py
- _get_richlog_selection_text() at line 2678 — tries many nonexistent RichLog APIs
- _get_transcript_text() at line 2644 — tries many nonexistent RichLog APIs
- _transcript_fallback_lines is the actual reliable source of plain text
- action_copy_selection() at line 4240 — uses these broken methods

REQUIREMENTS:

1. Simplify _get_transcript_text():
   - Remove the attempts to call export_text/get_text/to_text on RichLog (they don't exist)
   - Remove the .lines attribute iteration (unreliable)
   - Simply return "\n".join(self._transcript_fallback_lines).rstrip() + "\n"
   - If fallback lines are empty, return ""

2. Simplify _get_richlog_selection_text() (or remove it entirely):
   - RichLog does not have selected_text, selection_text, get_selection_text, export_selection, or get_selected_text
   - Remove all these attempts
   - If the transcript is a TextArea (after B2 or B3), use transcript.selected_text which DOES exist
   - If the transcript is a RichLog, return "" (no selection possible)

3. Update action_copy_selection():
   - If transcript is TextArea: use transcript.selected_text
   - If transcript is RichLog: fall back to copying full transcript text via _get_transcript_text()
   - Remove the misleading "Select text first" warning for RichLog (users CAN'T select text in RichLog)

4. Ensure /copy, /copy all, /copy last all work reliably with the fallback lines.

CONSTRAINTS:
- Do NOT add new nonexistent API calls
- Test with: /copy, /copy all, /copy last, Ctrl+C when transcript is focused
```

---

### Prompt B5: Animate consent prompt appearance and add visual emphasis

**Why this matters:** Even after fixing the height (B1), the consent prompt needs to grab user attention. It appears during tool execution when the user may not be looking at the right area. Adding a visual entrance animation and distinct styling ensures users notice it.

```
TASK: Make the consent prompt visually prominent with entrance animation and attention-grabbing styling.

CONTEXT:
- File: src/swarmee_river/tui/widgets.py — ConsentPrompt class
- File: src/swarmee_river/tui/app.py — _show_consent_prompt() reveals it

REQUIREMENTS:

1. When ConsentPrompt.set_prompt() is called:
   - Add a Textual toast notification: self.app.notify("Tool consent required", severity="warning", timeout=10)
   - This ensures the user sees something even if the consent widget is outside their viewport

2. Add a brief CSS animation on reveal:
   - Use Textual's animation system to fade the border from transparent to yellow over 0.3s
   - Or simply toggle a "-highlight" CSS class that makes the border bold/bright for 2 seconds then reverts

3. Play a terminal bell (self.app.bell()) when consent is first shown — this is the standard terminal "attention needed" signal.

4. Make the context text more informative:
   - Show the FULL tool name and the first line of the tool input (e.g., "shell: git status")
   - Instead of just "Consent required: shell", show "Consent required: shell — Command: git status"
   - The tool input context is available in the consent_prompt event's "context" field which already includes it

5. Add a visual separator between the context and the buttons — a horizontal rule or empty line.

6. After the user makes a choice, show a brief confirmation in the consent widget area before hiding:
   - "✓ shell allowed" (green) for y/a
   - "✗ shell denied" (red) for n/v
   - Auto-hide after 1 second

CONSTRAINTS:
- The toast notification must work even if the consent widget is scrolled out of view.
- bell() should only fire once per consent prompt (not on every re-render).
- Keep the total widget height reasonable (max 6-7 lines with the confirmation).
```

---

### Prompt B6: Debounce and optimize streaming text rendering in the transcript

**Why this matters:** During assistant responses, text_delta events fire rapidly (potentially hundreds per second). Each delta currently triggers a TextArea text update (if B2 is applied) or a RichLog.write() call. This causes visible flicker and CPU load. A debounce buffer improves perceived smoothness.

```
TASK: Add a debounce buffer for streaming assistant text to reduce rendering overhead.

CONTEXT:
- File: src/swarmee_river/tui/app.py — _handle_tui_event() handles "text_delta" events
- Currently each delta is immediately rendered via AssistantMessage.append_delta() or _mount_transcript_widget()
- text_delta events can arrive many times per second during streaming

REQUIREMENTS:

1. Add a streaming buffer mechanism:
   - Instance variable: _streaming_buffer: list[str] = []
   - Instance variable: _streaming_flush_timer: Timer | None = None

2. When a text_delta event arrives:
   - Append the delta text to _streaming_buffer
   - If no flush timer is running, start one: self.set_timer(0.15, self._flush_streaming_buffer)

3. _flush_streaming_buffer():
   - Join all buffered chunks: text = "".join(self._streaming_buffer)
   - Clear the buffer
   - Append the joined text to the transcript (single write instead of many)
   - Reset the timer reference

4. On text_complete event:
   - Immediately flush any remaining buffer
   - Cancel the timer if running
   - Render the final assistant message

5. The debounce interval of 150ms gives ~7 renders per second, which is visually smooth for text streaming while dramatically reducing render calls.

6. Apply the same debounce to tool_progress content events.

CONSTRAINTS:
- The buffer must flush immediately on text_complete (don't wait for timer).
- The buffer must flush immediately if the user presses Esc (interrupt).
- Don't buffer consent_prompt or error events — those must render immediately.
- The timer should be a Textual timer (self.set_timer), not a threading.Timer.
```

---

## Execution Order

1. **B1** (consent visibility) — Quick CSS fix, immediate user impact, < 30 min work
2. **B2** (TextArea transcript) — Core selectability fix, medium effort, addresses #1 complaint
3. **B4** (copy cleanup) — Quick cleanup, prevents confusing error paths
4. **B6** (streaming debounce) — Performance fix needed after B2 (TextArea is slower than RichLog for rapid updates)
5. **B3** (dual mode toggle) — Optional if B2's visual regression is acceptable
6. **B5** (consent animation) — Polish after B1 makes consent visible

B1 and B4 are independent and can be sent to agents in parallel.
B2 should be done after B1 (so consent isn't broken by the transcript change).
B6 should be done after B2 (optimization for the new rendering path).
