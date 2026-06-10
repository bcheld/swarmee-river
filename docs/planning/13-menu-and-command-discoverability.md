# 13 — Menus & Commands: Visibility and Discoverability

**Date:** 2026-06-10
**Status:** Proposed
**Theme:** User complaints that "menu items are not visible."

---

## Problem Statement

The TUI exposes its commands through three surfaces: the footer key bar
(Textual `Footer`), the slash-command palette (`CommandPalette` in
`tui/widgets.py`), and the action sheet overlay (`ActionSheet`). All three have
visibility problems on common terminal sizes, and the binding table guarantees
footer truncation even on wide terminals.

---

## Findings

### F1 (Critical) — Footer is guaranteed to overflow: duplicate visible bindings

`BINDINGS` (`tui/app.py:2065-2089`) defines 23 bindings of which ~15 are
*visible* (`show=True`), including the same action shown multiple times:

- **Widen side**: `ctrl+left`, `ctrl+shift+left`, `f6` all visible (plus 2 hidden)
- **Widen transcript**: `ctrl+right`, `ctrl+shift+right`, `f7` all visible (plus 2 hidden)
- **Copy selection**: `ctrl+shift+c`, `ctrl+c`, `meta+c`, `super+c` — four visible entries
- **Actions**: `ctrl+p` visible, 2 hidden

Textual's `Footer` renders one key chip per visible binding and silently clips
whatever doesn't fit. With ~15 chips (~150+ columns of content), most
terminals never see "Search", "Toggle transcript mode", or anything past the
duplicates. This is the direct cause of "menu items are not visible": the
items past the fold simply never render, with no overflow indicator.

**Fix:** Exactly one visible binding per action (`show=False` for all
alternates) — that alone cuts visible chips from ~15 to ~9. Order chips by
importance (Actions, Send, Interrupt, Search, Quit first). Consider Textual's
compact footer mode for narrow widths.

### F2 (High) — No full keybinding reference for hidden alternates

Seven bindings are `show=False` (`app.py:2066, 2069, 2071, 2082-2085`) — e.g.
`f5` Send prompt, `ctrl+k`/`ctrl+space` Actions, `ctrl+h`/`ctrl+l` pane
resize. After F1 hides more alternates, even fewer are discoverable. There is
no in-app "all shortcuts" view.

**Fix:** Add a `/keys` (or `?`) help overlay listing every binding including
hidden alternates, generated from `BINDINGS` so it can't drift. Mention it in
the footer ("? Help") and in `/help` output.

### F3 (Medium) — Command palette vanishes silently on no match

`CommandPalette.filter()` (`tui/widgets.py:2383-2392`) sets
`display = "none"` when the typed prefix matches nothing. A user typing
`/setings` sees the palette disappear entirely and gets no signal that the
command system is still alive or what valid commands exist.

Note: bare `/` correctly shows all commands (every command starts with `/`),
so first-open browsing works — the failure mode is specifically typos/unknown
prefixes.

**Fix:** Keep the palette visible and render a "No matching commands —
backspace to see all" row; optionally fall back to substring/fuzzy matching
before declaring no-match.

### F4 (Medium) — Palette and overlay sizing breaks on narrow terminals

- `CommandPalette` (`tui/widgets.py:2329-2342` CSS, `:2412-2417` rendering)
  has no width constraint or truncation; rows are formatted as
  `{cmd:<14} {desc}` and long descriptions clip off-screen at <80 columns.
- `ActionSheet` panel (`tui/widgets.py:2432-2439`) is fixed `width: 72` with
  `max-width: 96%`; below ~75 columns labels and buttons clip. The action
  sheet hosts consent prompts and plan review controls, so clipping here can
  hide *the approve/deny buttons themselves*.

**Fix:** Width-aware rendering: truncate descriptions to available width with
an ellipsis; give ActionSheet `min-width` + percentage width and let rows wrap
or scroll. Add a Textual pilot test that opens both at 60×20 and asserts all
controls are within the visible region.

### F5 (Low) — Palette only appears while the prompt starts with `/`

`_update_command_palette()` (`tui/mixins/prompt_ui.py:193-200`) is the only
trigger. There is no keybinding to browse commands/actions when the prompt
already has text or focus is elsewhere — `ctrl+p` opens the *action sheet*,
which is a different, smaller list than the slash-command set.

**Fix:** Unify: include slash commands in the action sheet (or add a "All
commands…" entry that opens the palette pre-filtered to `/`). Long-term,
consider Textual's built-in `CommandPalette` provider system instead of the
custom widget, which gives fuzzy search and theming for free.

---

## Proposed Plan

| Phase | Work | Findings |
|-------|------|----------|
| 1 | Deduplicate visible bindings; importance-ordered footer | F1 |
| 2 | `/keys` help overlay generated from `BINDINGS`; footer "? Help" chip | F2 |
| 3 | Palette no-match state + width-aware truncation; ActionSheet responsive sizing | F3, F4 |
| 4 | Unify action sheet & slash commands; evaluate Textual command palette | F5 |

Phase 1 is a ~20-line diff and directly resolves the literal complaint.

## Test Strategy

- Unit: assert no action has more than one `show=True` binding (guards F1
  regressions permanently).
- Textual pilot at 80×24 and 60×20: footer chips for the top-priority actions
  are rendered; palette shows no-match row; ActionSheet approve/deny buttons
  on-screen.
