# Core Concepts

Before building custom workflows, it's worth understanding how Swarmee River is structured. This page gives you the mental model.

---

## The Two-Process Model

When you run `swarmee tui` or `swarmee`, two processes communicate over a local pipe:

```
You
 |
 | (keystrokes, commands)
 v
TUI / REPL  ──JSON lines (commands)──>  Daemon (the agent)
            <──JSON lines (events)────  |
                                        | runs tools, streams output
                                        v
                                    Tool results, artifacts
```

The **daemon** owns the AI model connection and executes all tool calls. The **TUI/REPL** is purely a display and control layer. This separation means you can attach multiple clients to the same running daemon, and the agent keeps running even if you detach.

**Key files:**
- `src/swarmee_river/swarmee.py` — daemon / orchestrator entry point
- `src/swarmee_river/tui/app.py` — TUI entry point
- `src/swarmee_river/tui/transport.py` — subprocess and socket transport implementations

---

## The Orchestrator

The orchestrator is the special top-level agent that is always present. It:

- Owns the **plan/approve/execute** workflow
- Manages **tool consent** (asking you before high-risk operations)
- Coordinates **multi-agent delegation** (strand, swarm)
- Has a fixed agent ID: `orchestrator`

You cannot delete it, but you can customize its system prompt via the `orchestrator_base` prompt asset (Tooling tab > Prompts, or `.swarmee/prompts.json`).

---

## Tools

Tools are Python functions decorated with `@tool` from the Strands SDK. The docstring becomes the tool's description (shown to the model), typed parameters become the schema, and the return value is handed back to the agent.

Every tool declares **permissions**:

| Permission | Meaning | Effect |
|-----------|---------|--------|
| `read` | Safe, non-mutating | Allowed during plan mode |
| `write` | Modifies files or state | Blocked during plan mode; may require consent |
| `execute` | Runs processes or calls APIs | Blocked during plan mode; may require consent |

Built-in tools live in the `tools/` directory at the project root. You can add your own there too — they are hot-loaded at startup.

Full reference: [../tool_permissions.md](../tool_permissions.md)

Preferred coding loop:

- Inspect with `file_list`, `file_search`, and `file_read`
- Edit with `editor` for single-file changes or `patch_apply` for multi-file/hunk patches
- Verify with `run_checks` first, then use `shell` only when you need something `run_checks` does not cover

---

## SOPs (Standard Operating Procedures)

An SOP is a Markdown file with a YAML frontmatter header. When an SOP is active, its full content is injected into the agent's system prompt, guiding behavior without changing any code.

Minimal SOP format:

```markdown
---
name: my-workflow
version: 0.1.0
description: One-line summary of the SOP.
---

# SOP: My Workflow

## Steps
1. Do this first.
2. Then do this.
3. Always finish with verification.
```

SOPs live in `./sops/` (project root) or inside a pack's `sops/` directory. You activate them via the TUI (Tooling > SOPs tab), the REPL (`:sop enable my-workflow`), or in an agent profile (`sop_names`).

See the existing examples: `sops/bugfix.sop.md`, `sops/swarm.sop.md`, `sops/code-change.sop.md`.

---

## Agent Profiles

A profile is a named bundle that defines everything about how an agent runs for a particular workflow:

| Field | Purpose |
|-------|---------|
| `id` | Unique identifier |
| `name` | Display name |
| `prompt` | Custom system prompt snippet (appended to the base) |
| `prompt_refs` | Named prompt assets to include (e.g., `["orchestrator_base"]`) |
| `sop_names` | Active SOPs |
| `tool_names` | Explicit tool allowlist (empty = all tools) |
| `knowledge_base_id` | Bedrock Knowledge Base to attach |
| `context_sources` | Files, notes, URLs, KBs to inject as context |

Profiles are stored in `.swarmee/settings.json` under the `agents` array. You can also create and edit them in the TUI's Agents tab.

---

## Packs

A pack is a bundle of tools, SOPs, and system prompt sections that you install as a unit:

```bash
swarmee pack install ./path/to/my-pack
swarmee pack enable my-pack
```

Pack tools live in `packs/<name>/tools/`, SOP files in `packs/<name>/sops/`. Packs are a convenient way to share and distribute custom tooling.

---

## Sessions and Context

Every conversation is a **session**. Sessions are persisted to `.swarmee/logs/` as JSONL files. Within a session, the agent maintains a **conversation context** — the history of messages, tool calls, and results.

When the context gets large, a **context manager** trims it automatically:
- **`summarize`** (default) — generates a summary of older turns and replaces them
- **`sliding`** — drops the oldest turns
- **none** — no trimming (will hit token limits eventually)

Long tool results are stored as **artifacts** in `.swarmee/artifacts/` and referenced by ID in the conversation, keeping the context lean.

---

## The `.swarmee/` Folder

Everything Swarmee River persists locally lives here:

```
.swarmee/
  settings.json          # All configuration (agent profiles, tool policy, context settings, etc.)
  prompts.json           # Named prompt assets (orchestrator_base and custom prompts)
  tool_metadata.json     # User-overridden tool tags and access hints
  logs/                  # JSONL event logs, one file per session
    session_<id>.jsonl
  artifacts/             # Truncated tool results stored by artifact ID
  runtime.json           # Broker socket address (written by `swarmee serve`)
```

Do not commit `.swarmee/` — it contains session data and local overrides. It is in `.gitignore` by default.

---

## Next Steps

- [first-workflow.md](first-workflow.md) — Build a complete agentic workflow end-to-end
- [custom-tools.md](custom-tools.md) — Write and register a custom tool
- [sops-and-profiles.md](sops-and-profiles.md) — Deep dive on SOPs and agent profiles
