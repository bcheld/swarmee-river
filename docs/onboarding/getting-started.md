# Getting Started with Swarmee River

This guide gets you from zero to a running local environment in about 30 minutes.

---

## What is Swarmee River?

Swarmee River is a multi-agent AI orchestration platform built on the [Strands Agents SDK](https://strandsagents.com). It provides a full-screen terminal UI (TUI), an interactive REPL, and a shared daemon that multiple clients can attach to simultaneously. You use it to build agentic workflows — composed of tools, SOPs, agent profiles, and multi-agent delegation — and run them interactively or in one-shot mode.

---

## Prerequisites

- **Python 3.10 or newer**
- **git**
- **`pipx`** (for end-users) or **`hatch`** (for contributors)
- An API key for at least one supported provider (OpenAI, AWS Bedrock, GitHub Copilot, or Ollama)

---

## Installation

### Option A: Install as a tool (end-user / evaluator)

```bash
pipx install "swarmee-river[tui]"
```

This gives you the `swarmee` command with full TUI support.

### Option B: Install for development (contributor)

```bash
git clone <repo-url>
cd swarmee-river
pipx install hatch
hatch shell          # creates a virtual env + installs all deps
```

From inside the `hatch shell`, the `swarmee` command runs from source.

---

## Configure Credentials

Swarmee River reads credentials from a `.env` file (secrets only — all other config lives in `.swarmee/settings.json`).

```bash
cp env.example .env
```

Then edit `.env` and uncomment the relevant block:

```bash
# OpenAI
OPENAI_API_KEY="sk-..."

# GitHub Copilot
GITHUB_TOKEN="..."

# AWS Bedrock (credentials managed by the AWS SDK)
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
```

---

## Your First Session

### Interactive REPL

```bash
swarmee
```

You'll see a prompt. Try:

```
> List the files in this directory
```

The agent will generate a plan, show you what it intends to do, and wait for approval. Type `:approve` to execute, or `:cancel` to abort.

### One-shot mode

```bash
swarmee "Summarize the README"
```

### Full-screen TUI

```bash
swarmee tui
```

The TUI has five main areas:
- **Transcript** — streaming agent output, tool calls, reasoning
- **Run tab** — plan approval, session timeline, issues panel
- **Agents tab** — agent profile builder
- **Tooling tab** — tool browser, prompt templates, SOPs, knowledge bases
- **Settings tab** — model selection, diagnostics

Press `Ctrl+C` or `Escape` to exit.

---

## The Five Things You'll Use Most

| Thing | What it is | Where to find it |
|-------|-----------|-----------------|
| **REPL commands** | `:help`, `:approve`, `:cancel`, `:plan`, `:sop`, `:session` | Type `:help` inside the REPL |
| **TUI tabs** | Run, Agents, Tooling, Settings | Top of the TUI screen |
| **`.swarmee/settings.json`** | Local non-secret config: agents, tool policy, context settings | Created when you save local settings |
| **`.swarmee/settings.json.example`** | Tracked workspace settings template | Committed with the repo |
| **`.swarmee/` folder** | Logs, artifacts, prompts, session state | `.swarmee/logs/`, `.swarmee/artifacts/` |
| **`tools/` directory** | Where you drop custom `@tool` files | Project root |

---

## Running Tests

```bash
# From hatch shell
pytest tests/ -v

# Or directly
.venv/bin/pytest tests/ -v
```

---

## Where to Go Next

| I want to... | Read this |
|-------------|-----------|
| Understand the core architecture | [core-concepts.md](core-concepts.md) |
| Build my first complete agentic workflow | [first-workflow.md](first-workflow.md) |
| Write a custom tool | [custom-tools.md](custom-tools.md) |
| Create an SOP or agent profile | [sops-and-profiles.md](sops-and-profiles.md) |
| Use strand or swarm for multi-agent work | [delegation.md](delegation.md) |
| Debug something that went wrong | [debugging.md](debugging.md) |
| Understand configuration in depth | [../configuration/settings_inventory.md](../configuration/settings_inventory.md) |
| Understand how context is assembled | [../agent_context_lifecycle.md](../agent_context_lifecycle.md) |
| Add permissions to a tool | [../tool_permissions.md](../tool_permissions.md) |
