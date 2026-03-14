# SOPs and Agent Profiles

SOPs and agent profiles are the two primary mechanisms for customizing agent behavior in Swarmee River. Neither requires writing code — they're configuration and Markdown.

---

## Standard Operating Procedures (SOPs)

### What an SOP Does

When an SOP is active, its full Markdown content is injected into the agent's system prompt before each invocation. The agent reads it as part of its instructions and follows the procedure naturally. SOPs are the lightest-weight way to change how an agent approaches a task.

### File Format

An SOP is a `.sop.md` file with a YAML frontmatter block:

```markdown
---
name: my-workflow       # Required. Used to reference the SOP in profiles and commands.
version: 0.1.0          # Optional but recommended.
description: One-line summary shown in the TUI SOPs tab.
---

# SOP: My Workflow

## Context
Brief description of when to use this SOP.

## Steps
1. **First step** — detailed instructions.
2. **Second step** — detailed instructions.
3. **Verification** — how to confirm success.

## Rules
- Always do X before Y.
- Never skip the verification step.
```

The `name` field in the frontmatter is what you use everywhere else (in profiles, REPL commands, etc.).

### Where SOPs Live

| Location | Description |
|----------|-------------|
| `./sops/` | Project-level SOPs — checked in with the repo |
| `packs/<name>/sops/` | Pack-bundled SOPs — installed with the pack |

Swarmee River discovers SOPs from both locations automatically.

### Activating SOPs

**TUI:** Tooling tab > SOPs sub-tab > check the checkbox next to the SOP name

**REPL:**
```
> :sop enable my-workflow
> :sop disable my-workflow
> :sop list
```

**Agent profile** (`sop_names` field — see Profiles section below): the SOP is activated automatically when the profile loads.

**Session persistence:** Active SOPs are saved with the session and restored on reload.

### Writing Effective SOPs

Study the existing examples in `sops/` — particularly `bugfix.sop.md` and `code-change.sop.md`:

- **Use numbered steps** — the agent follows them sequentially and can reference step numbers
- **Be explicit about outputs** — tell the agent what format to use for its response
- **Include a "Rules" section** — things that must always or never happen are well-suited for explicit rules
- **Keep it actionable** — avoid vague language; specific instructions produce specific behavior
- **Add an examples section** if the output format is complex — models learn well from examples

### Injection Order

SOPs are injected after the base system prompt and before any context sources. See [../agent_context_lifecycle.md](../agent_context_lifecycle.md) for the full system prompt assembly order.

---

## Agent Profiles

### What a Profile Contains

A profile is a named configuration bundle. When a profile is active, all of its settings apply to every invocation in that session.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (used in commands and delegation) |
| `name` | string | Display name shown in the TUI |
| `prompt` | string | Custom system prompt snippet appended to the base |
| `prompt_refs` | list of strings | Named prompt assets to include (e.g., `["orchestrator_base"]`) |
| `sop_names` | list of strings | SOPs to activate when this profile loads |
| `tool_names` | list of strings | Explicit tool allowlist (empty = all available tools) |
| `knowledge_base_id` | string | Bedrock Knowledge Base ID to attach |
| `context_sources` | list | Files, notes, URLs, KBs to inject as context |

### Profile Storage

Profiles live in `.swarmee/settings.json` under the `agents` array:

```json
{
  "agents": [
    {
      "id": "orchestrator",
      "name": "Orchestrator",
      "prompt_refs": ["orchestrator_base"]
    },
    {
      "id": "code-reviewer",
      "name": "Code Reviewer",
      "prompt": "You are a careful code reviewer. Be specific and actionable.",
      "sop_names": ["code-review"],
      "tool_names": ["git", "file_read", "file_list", "file_search"]
    }
  ]
}
```

### The Orchestrator Profile

The `orchestrator` profile is special — it always exists, cannot be deleted, and owns the top-level agent session. Its default system prompt comes from the `orchestrator_base` prompt asset.

To customize the orchestrator's base prompt:
- TUI: Tooling tab > Prompts > select `orchestrator_base` > edit content > Save
- Direct edit: local `.swarmee/prompts.json` → find the `orchestrator_base` entry (`.swarmee/prompts.json.example` is the tracked template)

### Creating a Profile via the TUI

1. Open the TUI (`swarmee tui`)
2. Navigate to the **Agents** tab
3. Click **New Agent** (or press `N`)
4. Fill in the fields:
   - **ID** — a short slug (e.g., `data-analyst`)
   - **Name** — display name (e.g., `Data Analyst`)
   - **System prompt** — custom snippet; leave blank to use only the base prompt
5. In the **SOPs** section, check any SOPs to activate
6. In the **Tools** section, list tool names (comma-separated) or leave blank for all tools
7. Click **Save**

The profile is written to `.swarmee/settings.json` immediately.

### Activating a Profile

Profiles apply to the **current session**. Switching profiles mid-session updates the system prompt for subsequent invocations (the conversation history is preserved).

**TUI:** Agents tab > click a profile row > it becomes active for the session

**REPL:** Profiles are activated by the TUI. From the REPL, you can activate SOPs (`:sop enable`) and configure context sources (`:context`).

### Context Sources

The `context_sources` field injects additional content into every invocation:

```json
"context_sources": [
  {"type": "file", "path": "/path/to/design-doc.md"},
  {"type": "note", "text": "Always prefer Python 3.11+ features."},
  {"type": "url", "url": "https://example.com/api-reference"},
  {"type": "kb", "id": "my-bedrock-kb-id"}
]
```

**TUI:** Agents tab > Context Sources section > add/remove sources interactively.

---

## Example Profiles

### Data Analyst

```json
{
  "id": "data-analyst",
  "name": "Data Analyst",
  "prompt": "You are a data analyst. Prefer pandas for data manipulation. Always explain your methodology before running code. Show intermediate results.",
  "sop_names": ["data-analysis"],
  "tool_names": ["python_repl", "file_read", "file_list", "editor"]
}
```

Pair with a `sops/data-analysis.sop.md` that specifies: load data → validate → transform → visualize → summarize findings.

### Security Reviewer

```json
{
  "id": "security-reviewer",
  "name": "Security Reviewer",
  "prompt": "You are a security-focused code reviewer. Prioritize OWASP Top 10 vulnerabilities. Never suggest workarounds that reduce security posture.",
  "sop_names": ["security-review"],
  "tool_names": ["git", "file_read", "file_list", "file_search"]
}
```

Uses only read-only tools so the agent can only inspect, never modify.

### DevOps Engineer

```json
{
  "id": "devops",
  "name": "DevOps Engineer",
  "prompt": "You are a DevOps engineer. Prefer idempotent operations. Always check the current state before making changes.",
  "sop_names": ["infrastructure-change"],
  "tool_names": ["shell", "file_read", "editor", "git", "http_request"],
  "context_sources": [
    {"type": "file", "path": "./infrastructure/README.md"}
  ]
}
```

---

## Next Steps

- [delegation.md](delegation.md) — compose multiple agent profiles for parallel multi-agent workflows
- [../agent_context_lifecycle.md](../agent_context_lifecycle.md) — how the system prompt is assembled from all active sources
- [../configuration/settings_inventory.md](../configuration/settings_inventory.md) — full settings.json schema reference
