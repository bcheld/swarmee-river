# 11 — Settings: Reliability, Persistence, and UX Overhaul

**Date:** 2026-06-10
**Status:** In progress — see implementation status below
**Theme:** User complaints that settings "don't stick", silently revert, or don't take effect.

---

**Implementation status (2026-06-12):**
- DONE — F1/F3: atomic saves, corrupt-file backup, save-error surfacing (commit 3513779). Bonus fix: legacy `models.default_tier` no longer clobbered by the defaults template.
- DONE — F8 (scoped): restart-required warning toasts for daemon-spawn-time settings, burst-guarded.
- REMAINING — F2 lost-update coordinator (SettingsStore), F4 model-selection intent flow, F5/F6 cached loads + external-edit detection, F7 field validation, F9/F10 defaults consolidation.

## Problem Statement

The settings system spans three layers that have drifted apart:

1. `src/swarmee_river/settings.py` (~2,050 lines) — schema, defaults, load/save.
2. `src/swarmee_river/tui/mixins/settings.py` (~1,940 lines) — TUI forms, ~15 independent persist methods.
3. Daemon runtime state (`state.daemon.*`) — what is actually in effect.

Each layer holds its own copy of "current settings", and they synchronize via
ad-hoc load/refresh calls. Most reported complaints trace back to the layers
disagreeing or to silent failure paths.

---

## Findings

### F1 (Critical) — Corrupt settings file is silently replaced by defaults → data loss

`_load_project_settings_payload()` (`tui/mixins/settings.py:939-951`):

```python
with contextlib.suppress(OSError, ValueError):
    loaded = _json.loads(path.read_text(encoding="utf-8"))
```

If `.swarmee/settings.json` has a JSON syntax error (e.g., a hand edit with a
trailing comma), the error is suppressed, `raw` stays `{}`, and the function
returns pure defaults. Every persist method follows a *load → modify → save*
pattern against this payload, so the very next settings change **overwrites the
user's entire settings file with defaults**. The user sees all their
customizations vanish with no error.

**Fix:** On parse failure, surface an error notification, refuse to persist
(read-only mode for that path), and back up the unparseable file
(`settings.json.broken-<timestamp>`) before any rewrite.

### F2 (High) — Lost updates: ~15 uncoordinated load→modify→save paths

`tui/mixins/settings.py` has 10+ persist methods (`_persist_project_tui_shortcuts`
~line 342, `_persist_project_context_budget_tokens` ~line 380,
`_persist_project_aws_athena_settings` ~line 487,
`_persist_project_setting_env_override` ~line 1008, …), each doing its own
full-file read-modify-write. Two changes in quick succession (or a change racing
a background refresh) cause the classic lost-update problem: the second save is
based on a stale read and silently drops the first change.

**Fix:** Route all writes through a single `SettingsStore` coordinator that owns
the in-memory payload, serializes writes, and saves atomically
(write-to-temp + `os.replace`). This also gives one place to add error
handling, backups, and change events.

### F3 (High) — Saves have no error handling

`_save_project_settings_payload()` (`tui/mixins/settings.py:953-957`) calls
`save_settings()` with no try/except and no return status. Permission errors,
disk-full, or read-only filesystems fail silently — the UI shows the new value,
the file still has the old one, and the next restart "reverts" the setting.

Additionally, the payload is round-tripped through
`SwarmeeSettings.from_dict(...)` before saving — any key the schema doesn't
know about (hand-added by the user, or written by a newer version) is dropped
on the next save.

**Fix:** Make saves return success/failure, toast errors to the user, and
preserve unknown keys on round-trip (store extras, or merge the parsed model
back into the raw payload instead of replacing it).

### F4 (High) — Model selection state is split across three layers and desyncs

Tier/provider selection lives in:

- `state.daemon.model_tier_override` (runtime, in-memory; set around
  `tui/mixins/settings.py:1759-1819`),
- `models.default_tier` / `models.provider` in `settings.json` (persisted only
  via the Models settings form),
- the daemon's actual model (applied via a `set_model` command that can fail
  after the UI state was already updated).

Quick-select changes update runtime state but not always the persisted default;
a failed daemon command leaves the UI claiming a model that isn't active.
Users report "model selection doesn't persist" and "settings say X but it's
running Y".

**Fix:** Single source of truth: UI emits an *intent*, the coordinator applies
it to the daemon first, and only on acknowledgment updates state + persistence
together. Show an explicit "unsaved/override (session only)" badge when a quick
selection intentionally differs from the persisted default.

### F5 (Medium) — Blocking disk I/O on the UI thread, repeated per refresh

Every refresh method calls `load_settings()` independently
(`_refresh_settings_models` at `tui/mixins/settings.py:783-794`,
`_refresh_settings_bedrock_runtime_controls` ~line 176, general/env refreshes,
etc.), and a single save triggers several refreshes, each re-reading and
re-parsing the file synchronously on the Textual event loop. On slow/network
storage this is visible jank when opening or editing settings.

**Fix:** Load once per refresh cycle and pass the parsed settings down; cache
with mtime-based invalidation in the `SettingsStore` from F2. As a bonus,
mtime checking makes external edits (F6) visible.

### F6 (Medium) — External edits to settings.json are invisible until restart

The TUI never watches or re-checks the file; combined with F5's caching-by-
accident (stale widget state), users who edit `.swarmee/settings.json` in an
editor see no change in the UI — and a subsequent UI save clobbers their edit
(F2/F3).

**Fix:** mtime check on focus/refresh of the settings screen; offer
"file changed on disk — reload?" if dirty.

### F7 (Medium) — Silent validation/coercion hides user errors

`_as_int` / `_as_float` / `_as_bool` / `_normalized_choice`
(`settings.py:77-107, 235-257`) coerce invalid values to defaults with no
warning. A user typing `10O` into a numeric field gets the default applied with
no feedback. Same pattern for pricing overrides
(`tui/mixins/settings.py:1320-1348`) — negative or nonsense prices persist
unvalidated.

**Fix:** Validation returns `(value, error)`; settings forms display inline
errors and refuse to persist invalid fields. Log coercions at WARNING when they
happen during file load.

### F8 (Medium) — No "requires restart" indication

Provider, context-manager, and diagnostics settings only take effect after a
daemon restart, but the UI gives no hint. Users change the provider, run a
query, and the old provider answers.

**Fix:** Tag schema fields with `requires_restart`; render a badge and a
post-save toast ("takes effect after restart — restart now?") with a one-key
restart action.

### F9 (Low–Medium) — Duplicated defaults drift

Bedrock runtime defaults exist both in
`tui/mixins/settings.py:51-68` (`_BEDROCK_RUNTIME_DEFAULTS`) and in
`default_settings_template()` (`settings.py:1776-2049`). The 273-line hardcoded
template is also the only definition of providers/tiers/pricing, requiring a
code release to update model IDs.

**Fix:** Single source for defaults in `settings.py` (TUI imports them);
longer-term, move the template to a bundled `defaults.json` that deployments
can override.

### F10 (Low) — `env` section silently filtered

`filter_project_env_overrides()` (`settings.py:51-75`) drops nearly all keys
users add to the `env` section by hand, with no warning. Combined with unclear
deep-merge semantics for lists (`hidden_tiers`, etc.), hand-editing the file is
a minefield.

**Fix:** Warn (once, on load) about ignored `env` keys; document merge
semantics with examples in `docs/configuration/`.

---

## Proposed Plan

| Phase | Work | Findings addressed |
|-------|------|--------------------|
| 1 | `SettingsStore` coordinator: single owner, serialized atomic writes, error surfacing, corrupt-file backup, unknown-key preservation | F1, F2, F3 |
| 2 | Model-selection intent flow + unsaved/override badge | F4 |
| 3 | Cached loads w/ mtime invalidation, single load per refresh cycle, external-edit detection | F5, F6 |
| 4 | Field validation + restart-required badges | F7, F8 |
| 5 | Defaults consolidation + docs for hand-editing | F9, F10 |

Phase 1 is the highest-leverage change: it converts every "settings randomly
reverted" complaint class into either a fixed bug or a visible error.

## Test Strategy

- Unit: lost-update simulation (two interleaved persist calls), corrupt-file
  load/save round-trip, unknown-key preservation, atomic write crash safety.
- TUI (Textual pilot): change a setting → kill write permission → assert error
  toast and no silent revert; provider change → restart badge shown.
