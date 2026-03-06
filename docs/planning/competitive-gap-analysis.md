# Competitive Gap Analysis: Swarmee River vs Claude Code & OpenCode

## Context

Swarmee River is an enterprise analytics + coding assistant built on the Strands SDK. This analysis identifies the largest UX and capability shortcomings compared to Claude Code (Anthropic's CLI) and OpenCode (open-source alternative), based on actual code audit of the codebase.

## What Already Works Well

These features are **fully implemented** and competitive:

| Feature | Implementation | Comparable to |
|---------|---------------|---------------|
| **Cost/token tracking** | StatusBar displays in/out/cache tokens + USD cost per turn. Pricing lookup per provider/model. | Claude Code `/cost` |
| **Command palette** | 27 `/` commands with fuzzy filtering, triggered by typing `/` in prompt | Claude Code slash commands, OpenCode `/` commands |
| **Context compaction** | Auto + manual (`/compact`). `BudgetedSummarizingConversationManager` auto-triggers at token budget. Per-tier strategy (balanced/cache_safe/long_running). | Claude Code auto-compaction |
| **Streaming reasoning** | Live `ThinkingMixin` streams reasoning blocks as they arrive. Animated indicator, elapsed time, char count, `/thinking` command. | Claude Code Ctrl+O verbose mode |
| **Plan/approve/execute** | Structured `WorkPlan` with steps, risks, file lists. `:y`/`:n`/`:replan`. Read-only tools during planning. | Unique — neither competitor has structured planning |
| **Multi-agent orchestration** | `strand`/`swarm`/`agent_graph`/`use_agent` with team presets, agent profiles, tool policies | Unique — significantly ahead of both competitors |
| **Enterprise data tools** | Athena, Snowflake, S3, Knowledge Base, Office docs | Unique to Swarmee River |

---

## Priority 1: Critical Gaps (High Impact, Daily Friction)

### 1. No Visual Diff Display for File Edits

**Claude Code**: Multi-file visual diffs with side-by-side view, line commenting, +X/-Y statistics. Desktop app renders diffs inline in the editor.

**OpenCode**: Diff rendering in TUI with "auto" (adapts to width) or "stacked" mode.

**Swarmee River**: File edits produce only text confirmations ("Replaced X with Y", "Wrote 500 chars"). `.diff`/`.patch` files are recognized as artifact types but have **no rendering engine**. Users must manually `git diff` to review changes.

**Impact**: Every coding session involves file edits. Users cannot review what changed without leaving the tool. This is the single largest daily-friction gap.

**Recommended approach**: After every file mutation (editor replace/insert, file_write, patch_apply), capture a unified diff and render it inline in the transcript using Rich syntax highlighting. Add a `/diff` command showing cumulative session changes.

---

### 2. No Undo / Rewind Capability

**Claude Code**: Double-Esc instantly rewinds to last checkpoint. Full undo of file changes.

**OpenCode**: `/undo` and `/redo` commands.

**Swarmee River**: Zero implementation — no undo, rewind, revert, or rollback references anywhere in the codebase. `patch_apply` creates backup artifacts, but there's no unified revert mechanism. Users must manually `git checkout` or `git stash`.

**Impact**: Users are reluctant to let the agent make changes autonomously without a safety net.

**Recommended approach**: Track all file mutations in a session-level changeset (the artifact system already backs up pre-edit state for patch_apply). Add `:undo` (revert last tool's changes) and `:undo all` (revert all session changes). Extend the backup pattern to `editor` and `file_write`.

---

### 3. No @ File Reference or Fuzzy File Picker

**Claude Code**: `@filename` syntax for referencing files. Tab-completion for paths.

**OpenCode**: `@` prefix triggers fuzzy file search picker. Selected files injected as context.

**Swarmee River**: No `@` syntax handling in the prompt input. No file picker widget. Users must type full paths or rely on the agent to discover files.

**Impact**: Every prompt referencing specific files requires more typing and is error-prone.

**Recommended approach**: In the TUI prompt input, detect `@` and open a fuzzy file picker overlay (Textual has `OptionList` for this). Inject selected file content into the prompt context. In REPL, support tab-completion on `@` prefix.

---

## Priority 2: Significant Gaps (Weekly Friction)

### 4. No MCP (Model Context Protocol) Support

**Claude Code**: 75+ MCP tools. Dynamic tool loading. `@`-mention MCP resources.

**OpenCode**: Full MCP integration for external tools.

**Swarmee River**: Zero MCP references in the codebase. Tool ecosystem limited to built-in tools, `tools/` hot-loading, and Strands Tools pack.

**Impact**: Users can't integrate with Jira, Confluence, Notion, custom APIs, or any MCP-compatible service. This is a significant limitation for enterprise environments where external tool integration is essential.

**Recommended approach**: Investigate whether the Strands SDK has MCP primitives. If not, add an MCP client that registers external tools into the existing tool registry. MCP tools would participate in the same permission system.

---

### 5. No Persistent Memory / Auto-Learning

**Claude Code**: CLAUDE.md as project constitution read at session start. Auto memory learns build commands, debugging insights, user preferences across sessions.

**OpenCode**: No auto memory.

**Swarmee River**: Has `.swarmee/settings.json`, agent profiles, SOPs, and `welcome` tool — but no equivalent to CLAUDE.md or auto-learning. Each session starts without memory of prior sessions' discoveries (successful build commands, error resolutions, user corrections).

**Impact**: Repeated context-setting across sessions. The agent re-discovers the same things.

**Recommended approach**: Add `.swarmee/memory.md` read at session start (similar to CLAUDE.md). Auto-capture: successful build commands discovered via `run_checks`, error resolutions, user corrections. The `journal` tool from Strands Tools could be the foundation.

---

### 6. No User-Facing Hooks / Lifecycle Automation

**Claude Code**: Shell commands run at lifecycle points (post-edit formatting, pre-commit linting). `/hooks` command for management.

**OpenCode**: No hooks system.

**Swarmee River**: Internal Strands SDK hooks exist (tool_policy, tool_consent, result_limiter, jsonl_logger) but there is **no user-facing hook system**. Users can't auto-format after edits or auto-lint before commits.

**Impact**: Manual post-edit cleanup steps that could be automated. Less relevant for analytics workflows but significant for code-heavy sessions.

**Recommended approach**: Add a `hooks` section to `.swarmee/settings.json` with lifecycle events: `post_file_edit`, `pre_commit`, `post_tool_call`. Each hook runs a shell command. Leverage the existing Strands `BeforeToolCallEvent`/`AfterToolCallEvent` infrastructure.

---

### 7. No LSP / Language Server Integration

**Claude Code**: Automatic LSP diagnostic integration.

**OpenCode**: 30+ built-in LSP servers with goToDefinition, findReferences, hover, etc.

**Swarmee River**: No LSP support. Agent relies on `file_search` (ripgrep) and `project_context` for code understanding.

**Impact**: The agent can't know about type errors or broken references until it runs the linter. Longer iteration cycles. Less critical for Swarmee's analytics focus (SQL, notebooks) than for pure coding assistants.

**Recommended approach**: Lower priority given enterprise analytics positioning. If pursued, start with Python (pyright) diagnostic integration — feed errors into agent context automatically after edits.

---

## Priority 3: Polish Gaps (Quality-of-Life)

### 8. No Background Agent Management UX

**Claude Code**: Ctrl+B backgrounds sub-agents. Visual indicator of progress.

**Swarmee River**: Has `strand`/`swarm`/`agent_graph` but no user-facing management. Can't see running sub-agents, their progress, or interrupt individuals. The multi-agent capability is powerful but opaque.

---

### 9. No Instant Session Continuity Across Surfaces

**Claude Code**: Remote Control (QR code, phone/tablet access). `/teleport` between terminal and web.

**Swarmee River**: Has shared runtime daemon with `swarmee attach`, but no cross-device or cross-surface continuity.

---

### 10. Compaction UX Could Be Improved

Context compaction exists and works, but the user gets no visibility into:
- What was compacted (no summary shown)
- How much space was recovered
- When auto-compaction triggers

**Recommended**: Surface compaction events in transcript ("Context compacted: 45K → 18K tokens, preserved last 10 turns").

---

## Summary: Prioritized Gap Table

| # | Gap | Impact | Effort | Competitors |
|---|-----|--------|--------|-------------|
| 1 | **Visual diff display** for file edits | Every edit session | Medium | Both have it |
| 2 | **Undo/rewind** for file changes | User confidence | Medium | Both have it |
| 3 | **@ file references** with fuzzy picker | Every prompt | Medium | Both have it |
| 4 | **MCP support** | Enterprise integration | High | Both have it |
| 5 | **Persistent memory** / auto-learning | Cross-session continuity | Medium | Claude Code has it |
| 6 | **User-facing hooks** | Post-edit automation | Medium | Claude Code has it |
| 7 | **LSP integration** | Code intelligence | High | OpenCode leads |
| 8 | **Background agent UX** | Multi-agent visibility | Low | Claude Code has it |
| 9 | **Cross-surface continuity** | Device mobility | High | Claude Code has it |
| 10 | **Compaction UX polish** | Long session clarity | Low | Incremental |

Items 1-3 would close the biggest daily-friction gaps with medium effort and no architectural changes. Items 4-6 are higher-effort but would significantly differentiate for enterprise use cases. Items 7-9 are longer-term strategic.

## Swarmee River's Unique Strengths (Not to Lose)

These are areas where Swarmee River is **ahead** of both competitors:
- **Structured plan/approve/execute** — neither competitor has anything like WorkPlan
- **Multi-agent orchestration** — strand/swarm/agent_graph/team presets far beyond either
- **Enterprise data connectivity** — Athena, Snowflake, S3, Knowledge Base, Office docs
- **Agent profiles + SOPs** — declarative workflow customization without code
- **Multi-provider support** — Bedrock, OpenAI, Ollama, GitHub Copilot with tier switching
