# Sidebar UX Unification + Agent Studio Plan

> Living document — targets a production-grade, keyboard-first TUI sidebar that functions as the agent “control plane”.

## Snapshot (as of `a13ea9f`)

- TUI sidebar is a `TabbedContent` under `#side` in `/Users/bcheld/dev/vscode/swarmee-river/src/swarmee_river/tui/app.py` (compose around `#side_tabs`).
- TUI can now attach to the shared runtime broker first (fallback to subprocess daemon).
- Current sidebar tabs (implementation):

| Tab | Current UI | Interaction quality | Notes |
|---|---|---|---|
| Plan | read-only `TextArea` + `PlanActions` buttons | medium | plan text is not a structured list; step metadata not visible |
| Context | structured list + add/remove flows | good | missing URL add UI; no token/cost signal per item |
| SOPs | structured catalog w/ toggles + preview snippet | good | missing search/filter and “active stack” clarity |
| Artifacts | `SidebarHeader` + `SidebarList` + `SidebarDetail` | good | index-backed list + preview + actions (open/copy/add-as-context) |
| Session | Timeline + issues list/detail | good | timeline backed by `graph_index.json`; issues are actionable |
| Agent | Agent Studio (Profile/Tools&Safety/Team) | good | policy lens + session-only safety overrides + team presets v1 |

## Problem Statement

The sidebar is currently a mix of “real UI” and “text dumps”. It does not yet function as a coherent control plane for:
- **What will be injected into the next prompt** (context + SOP stack + profile changes),
- **What the agent is allowed to do** (tools/permissions/tier/packs),
- **How to recover when things go wrong** (errors, retries, escalations),
- **How to reuse and share agent configurations** (profiles + team composition).

## Design Principles (non-negotiable)

1. **Every pane is actionable**: list + detail + primary actions, never “just a text blob”.
2. **Single mental model**: consistent list/detail/action pattern across tabs.
3. **Cross-tab coherence**: “active” state (profile, SOP stack, context, KB) is visible everywhere it matters.
4. **Keyboard-first**: selection, search, open/preview, and primary actions have shortcuts; mouse is optional.
5. **Multi-client safe**: with the shared runtime service, assume *multiple clients may be attached*; display source-of-truth and avoid destructive silent overwrites.
6. **Progressive disclosure**: summary by default, expand-on-demand; avoid vertical bloat.

## Target Information Architecture (IA)

We keep the top-level tab count stable (6) and make each tab “strong” rather than adding more tabs.

### Tab 1: Plan
Goal: “what are we doing next, and what’s running now?”
- Structured step list (selectable), not markdown text.
- Step status driven by `plan_step_update` / `plan_complete` events.
- Primary actions: Approve / Replan / Clear.
- Secondary actions: Copy, Export to artifact, Jump-to-transcript (when available).

### Tab 2: Context
Goal: “what will the model see next turn?”
- Context sources as a sortable list with:
  - icon, label, source type, **token estimate**, sync state, remove, preview.
- Add flows for: file, note, SOP, KB id, URL.
- “Context Preview” (read-only) that shows the effective injected reminder (cache-safe), with truncation indicators.
- Global actions: Compact, Clear, Export context bundle as artifact.

### Tab 3: SOPs
Goal: “which procedures are shaping behavior right now?”
- Search + filter (source, pack, active-only).
- Clear “active SOP stack” at top (ordered list) with per-item deactivate.
- Preview is first-class (expand/collapse), and “promote to context” is explicit.

### Tab 4: Artifacts
Goal: “outputs you can act on”
- Artifact list driven by artifact index metadata (not just file paths).
- Preview and actions:
  - Add as context source
  - Open in editor
  - Promote to KB
  - Upload to S3 (when available)
  - Copy path
- Filter by kind + search by title/content (best-effort for text artifacts).

### Tab 5: Session
Goal: “observability + recovery”
- Timeline-lite MVP: last N notable events (plan generated/approved, tool failures, restores, compactions).
- Issues list becomes structured + actionable:
  - Retry tool / Skip tool (already supported by daemon)
  - Escalate tier / Continue
  - Dismiss (UI only; does not mutate daemon)
- Session controls:
  - restore available session, start fresh, show session id, show model summary, last cost/usage.

### Tab 6: Agent (Agent Studio)
Goal: “agent configuration as a product surface”

This tab is the Agent Studio. It has **three sub-views** inside a single tab (implemented as a segmented control or nested tabs):
1) **Profile**
2) **Tools & Safety**
3) **Team**

#### Agent Studio: Profile (MVP)
- Show current session “effective profile” (read-only summary):
  - provider/tier
  - system prompt snippet(s) (short preview)
  - active SOP stack
  - attached context sources summary
  - KB id (if set)
  - packs enabled (if supported)
- Allow saving/loading named profiles.
- “Apply to session” sends one atomic command to the daemon/broker (`set_profile`) to avoid partial state drift.

#### Agent Studio: Tools & Safety (MVP)
- Show effective safety posture:
  - tool consent mode (ask/allow/deny)
  - allowlist/blocklist (tier profile + session overrides)
  - notable permission rules
- Allow editing **session-only overrides** first (do not rewrite project settings by default).
- Provide “Explain why tool was blocked” UX using existing policy/consent error strings.

#### Agent Studio: Team (v1)
- Manage “team presets” (swarm/agent_graph specs) saved alongside profiles.
- One-click: “Run task with team” (invokes a `swarm` tool run with a controlled spec).
- Visibility: last team run results + artifacts created.

## Shared Widget System (Sidebar Foundation)

We standardize every tab around the same building blocks:

### `SidebarPane` anatomy
- `SidebarHeader`: title, badges, search input toggle, primary actions.
- `SidebarList`: selectable items, keyboard navigation, empty state.
- `SidebarDetail`: preview of selected item + secondary actions.

### `SidebarListItem` (unified item)
Properties:
- `id`, `kind`, `title`, `subtitle`, `badges[]`, `state` (`default|active|warning|error|syncing`)
- optional quick actions: remove/toggle/open/preview

### Tab badges (always visible)
Each tab title shows compact counters:
- Plan: `steps_done/steps_total` when running
- Context: item count + budget warning when >80%
- SOPs: active count
- Artifacts: new artifacts count (since last view)
- Session: warnings/errors count
- Agent: “dirty” dot when local edits not applied

## State, Sync, and Source-of-Truth

### Key concept: “session-effective config”
In a shared runtime world, the daemon session is authoritative for:
- active tier/provider
- active SOP overrides
- active context sources
- any profile currently applied

The TUI may keep local drafts (e.g., editing a profile) but must show:
- **Draft** vs **Applied**
- last-applied timestamp

### New daemon command: `set_profile` (atomic)
MVP `set_profile` payload:
```json
{
  "cmd": "set_profile",
  "profile": {
    "name": "my-profile",
    "provider": "openai",
    "tier": "balanced",
    "system_prompt_snippets": ["..."],
    "context_sources": [...],
    "active_sops": ["..."],
    "knowledge_base_id": "ABC123"
  }
}
```
Rules:
- daemon applies changes via the existing “single source of truth” paths already used today:
  - tier change -> existing tier setter
  - SOPs -> existing SOP override mechanism
  - context_sources -> existing context source mechanism
  - KB id -> existing KB override mechanism
  - system prompt snippet -> prompt-cache section (cache-safe reminder)
- daemon responds with a `profile_applied` event carrying the effective normalized profile.

### Broker pass-through
The runtime broker must proxy `set_profile` exactly like existing daemon proxy commands.

## Persistence Model

We need a dedicated store for profiles (project-local, not global):
- directory: `.swarmee/profiles/`
- files:
  - `profiles.json` (catalog + metadata)
  - `profiles/<id>.json` (full content) OR single-file only for MVP

Profiles are *shareable artifacts*:
- export/import supports copy/paste and file-based sharing
- optional: promote profile to KB

## Implementation Plan (Phases)

### Phase 1 — Sidebar foundation + upgrade “passive” tabs (done)
1. Implement shared sidebar widgets (`SidebarHeader`, `SidebarList`, `SidebarDetail`, `SidebarListItem`).
2. Convert **Artifacts** from `TextArea` to list + preview + actions.
3. Convert **Issues** into structured items inside a redesigned **Session** tab.
4. Remove dedicated Help tab (fold into Agent/Session).

### Phase 2 — Agent Studio (Profile MVP + daemon/broker command) (done)
1. Add Agent tab UI skeleton and Profile editor.
2. Implement profile persistence store under `.swarmee/profiles`.
3. Add daemon command `set_profile` and event `profile_applied`.
4. Add broker proxy for `set_profile`.
5. Wire TUI “Apply to session” across both transports (broker + subprocess).

### Phase 3 — Tools & Safety view + session-only overrides (done)
1. Show effective tool policies (session + tier profile) in Agent tab.
2. Implement session-only overrides first (avoid rewriting `.swarmee/settings.json` by default).
3. If pack enable/disable is supported in-session:
   - require explicit “restart agent session” action (tool set + system prompt rebuild implications).

### Phase 4 — Team presets + live composition (v1 done)
1. Profile includes `team_presets[]` with `swarm`/`agent_graph` specs.
2. Provide one-click “run with team” actions + results capture (v1: deterministic prompt insertion + optional auto-submit).

### Phase 5 — Event-sourced Session Graph (timeline, branching, export) (done)
Goal: make Session the connective tissue across TUI/CLI/Notebook by treating sessions as an event stream.

MVP deliverables:
1. A normalized per-session index (derived from JSONL logs + saved message snapshots).
2. TUI: “Timeline” pane inside Session tab (searchable list + detail).
3. CLI: `session export` (Markdown / JSON) and `session branch` (new session id truncated to N turns).
4. Broker: pass-through for session graph operations when attached (optional for MVP; CLI may read from disk directly).

### Phase 6 — “Industry-leading” refinement (next)
1. Timeline: search/filter + jump-to-transcript + “branch from here”.
2. Timeline: live ingest (append notable events during runs; no rebuild churn).
3. Export: “open as notebook” (ipynb) + copy/share bundles.
4. Multi-client: controller lease + conflict UX (draft vs applied vs remote changes).
5. Team: capture runs (artifacts + per-agent summary) + rerun with diff.

## Open Questions (answer before Phase 3+)

1. Should profiles be scoped to project only (recommended) or global + override?
2. How do we represent “system prompt snippets” safely (cache-friendly) vs a full prompt override?
3. What is the minimal pack toggling story that does not require hot-reloading complexity?
4. How do we handle conflicting edits when multiple clients are attached (last-write wins vs controller-only)?
5. Should Timeline data be derived-only (rebuildable) or partially canonical (append-only event store)?
