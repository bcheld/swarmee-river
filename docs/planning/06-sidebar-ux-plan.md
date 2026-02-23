# Sidebar UX Unification Plan

> Living document — intended for iteration across planning cycles.

## Current State

The TUI sidebar (`#side` vertical panel, `app.py` compose() line 1822) has six tabs:

| Tab | Content | Interactive? | Data Source |
|-----|---------|-------------|-------------|
| **Plan** | TextArea (read-only markdown) + PlanActions buttons | Yes (approve/replan/clear) | Generated plan from agent |
| **Context** | Active source list + add buttons (File/Note/SOP/KB) + input row | Yes (add/remove sources) | User-managed context sources |
| **SOPs** | Static header + VerticalScroll of SOP items | Browse only | Local/pack/strands-sops dirs |
| **Artifacts** | TextArea (read-only) | View only | `.swarmee/artifacts/index.jsonl` |
| **Issues** | TextArea (read-only) | View only | Tool errors from session |
| **Help** | TextArea (read-only) | View only | Static text |

### UX Problems Identified

1. **Inconsistent interaction models** — Plan and Context are interactive; Artifacts, Issues, and Help are passive text blobs. SOPs are browse-only when they should be activatable.
2. **No cross-tab awareness** — Adding a context source doesn't reflect in SOPs tab; activating an SOP doesn't show in Context tab. There's no unified "what's active" view.
3. **Artifact tab is underused** — Just lists file paths. Can't preview, promote to KB, or attach as context.
4. **Issues tab lacks actionability** — Lists errors but can't retry, dismiss, or escalate from the sidebar.
5. **Help is static** — Could be contextual (show relevant help based on current mode: planning, executing, idle).
6. **No session/history panel** — No way to browse previous sessions, branch points, or conversation timeline.
7. **No agent/tool status panel** — Active tools, loaded packs, model info, and token budget are scattered or hidden.

---

## Design Principles

1. **Every tab should be actionable** — Browse + act, not just browse.
2. **Cross-tab state should be visible** — Active SOPs shown in Context, context items available in Plan.
3. **Progressive disclosure** — Summary view by default, expandable detail on demand.
4. **Keyboard-first** — All sidebar actions reachable via shortcuts or command palette.
5. **Consistent widget patterns** — Use the same list/detail/action pattern across tabs.

---

## Proposed Tab Structure

### Tab 1: Plan (keep, enhance)

**Current**: TextArea + PlanActions
**Proposed**: Keep core layout, add step-level interaction.

- Each plan step becomes a clickable/focusable item
- Step status indicators: pending / in-progress / done / failed
- Click a step → shows expected tools, estimated complexity
- "Jump to step output" links to relevant transcript section
- PlanActions: Approve / Replan / Edit Step / Clear

### Tab 2: Context (keep, enhance)

**Current**: Source list + add buttons
**Proposed**: Unified "active context" view with richer source types.

- Source types: File, Note, SOP, KB Query, URL, Session History, Agent Output
- Each source shows: type icon, name/path, token estimate, remove button
- "Total context budget" bar at top (mirrors ContextBudgetBar)
- Quick-add: drag file from transcript, or right-click artifact → "Add as context"
- Active SOPs appear here automatically with "SOP" badge
- Sort by: recently added, token cost, type

### Tab 3: SOPs (keep, make interactive)

**Current**: Browse-only list
**Proposed**: Catalog with activate/deactivate toggles.

- Each SOP: name, source (local/pack), preview snippet, toggle switch
- Active SOPs highlighted, shown at top
- "Preview" action expands SOP content in-place (Collapsible)
- Filter/search bar at top
- Category grouping by pack source

### Tab 4: Artifacts (redesign)

**Current**: Flat text list
**Proposed**: Rich artifact browser with actions.

- List view: kind icon, title/name, timestamp, size
- Click to preview (text content or metadata for binary)
- Actions per artifact:
  - View full content (opens in transcript or modal)
  - Add as context source
  - Upload to S3
  - Promote to Knowledge Base
  - Copy path
- Filter by kind (plan, code, output, data)
- Search within artifacts

### Tab 5: Session (new, replaces Issues)

**Current**: Issues tab (passive error list)
**Proposed**: Session timeline + issue management.

- **Timeline view**: Chronological list of turns, tool calls, plans, errors
  - Each entry: timestamp, type icon, summary
  - Click to jump to transcript position
  - Branch points marked (where plans diverged)
- **Issues sub-section**: Collapsible, shows active errors
  - Each issue: retry / dismiss / escalate buttons
  - Resolved issues greyed out
- **Session info**: ID, model, duration, turn count, token usage summary
- **Session history**: Recent sessions list, resume/delete actions

### Tab 6: Tools & Config (new, replaces Help)

**Current**: Static help text
**Proposed**: Live system status + help.

- **Active tools list**: Name, source (core/pack/strands), trust level
  - Click tool → show description, recent invocations, avg duration
- **Loaded packs**: Name, path, enabled tools count, toggle
- **Model info**: Current provider, tier, model name, token limits
- **Help section**: Collapsible, context-sensitive
  - Shows relevant help based on app state (idle → commands, planning → plan workflow, executing → interrupt/consent)
  - Keyboard shortcut reference

---

## Widget Patterns

### Unified List Item Pattern

All list-based tabs should use a consistent `SidebarListItem` widget:

```
┌─────────────────────────────────┐
│ 📄 icon  Title/Name        [×] │
│          Subtitle/meta     [▸] │
└─────────────────────────────────┘
```

- Icon: type-specific (file, SOP, artifact, tool, etc.)
- Title: primary identifier
- Subtitle: secondary info (path, timestamp, size)
- Actions: context-dependent buttons (remove, expand, activate)
- States: default, selected, active, error

### Detail Panel Pattern

When an item is selected/expanded:

```
┌─────────────────────────────────┐
│ ▾ Title/Name                    │
│   Key: Value                    │
│   Key: Value                    │
│   ─────────────────────         │
│   Preview content...            │
│   ─────────────────────         │
│   [Action1] [Action2] [Action3] │
└─────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Batch 4-5)
- Create `SidebarListItem` and `SidebarDetailPanel` base widgets
- Refactor SOPs tab to use interactive toggles (activate/deactivate)
- Add artifact preview and "Add as context" action
- Move Issues into a collapsible section within a new Session tab

### Phase 2: Cross-Tab Integration (Batch 5-6)
- Active SOPs auto-appear in Context tab
- Artifacts promotable to context
- Plan steps linked to transcript positions
- Session timeline with turn/tool/error entries

### Phase 3: Tools & Config Tab (Batch 6-7)
- Replace Help tab with live Tools & Config panel
- Tool catalog with usage stats
- Pack management toggles
- Context-sensitive help

### Phase 4: Advanced Features (Batch 7+)
- Session branching and history navigation
- Drag-and-drop between tabs
- Artifact preview for non-text formats (after MS Office support lands)
- Agent studio integration (see separate planning item)

---

## Dependencies

- **MS Office file support** (Batch 4 prompt D1) → enables artifact preview for Office files
- **S3 session logs** (Batch 4 architecture) → enables session history persistence
- **Event-sourced session graph** (Batch 4 architecture) → enables timeline and branching in Session tab
- **Production data tools** (Batch 4 prompt D2) → enables KB/S3/database entries in Context tab

---

## Open Questions

1. Should the sidebar be resizable? Currently uses fixed CSS width.
2. Should tab order be user-configurable?
3. Should there be a "pinned" or "favorites" section across tabs?
4. How should we handle sidebar state persistence across sessions?
5. Is a floating/detachable sidebar desirable for power users?
