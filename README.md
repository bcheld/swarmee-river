<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Swarmee River
  </h1>

  <h2>
    Enterprise analytics + coding assistant built on the Strands Agents SDK.
  </h2>

  <div align="center">
    <a href="https://github.com/strands-agents/agent-builder/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/strands-agents/agent-builder"/></a>
    <a href="https://github.com/strands-agents/agent-builder/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/strands-agents/agent-builder"/></a>
    <a href="https://github.com/strands-agents/agent-builder/pulls"><img alt="GitHub open pull requests" src="https://img.shields.io/github/issues-pr/strands-agents/agent-builder"/></a>
    <a href="https://github.com/strands-agents/agent-builder/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-agents/agent-builder"/></a>
    <a href="https://pypi.org/project/swarmee-river/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/swarmee-river"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/swarmee-river"/></a>
  </div>

  <p>
    <a href="https://strandsagents.com/">Documentation</a>
    ◆ <a href="https://github.com/strands-agents/samples">Samples</a>
    ◆ <a href="https://github.com/strands-agents/sdk-python">Python SDK</a>
    ◆ <a href="https://github.com/strands-agents/tools">Tools</a>
    ◆ <a href="https://github.com/strands-agents/agent-builder">Agent Builder</a>
    ◆ <a href="https://github.com/strands-agents/mcp-server">MCP Server</a>
  </p>
</div>

Swarmee River is an interactive, enterprise-oriented analytics + coding assistant built on the Strands Agents SDK. It keeps what works (simple packaging, hot-loaded tools, AWS connectivity) while adding better context management, agent profiles, SOPs, plan/approve/execute workflows, and a full-featured terminal UI.

## Quick Start

```bash
# Install (recommended)
pipx install swarmee-river

# --- Choose a model provider ---
#
# OpenAI:
export OPENAI_API_KEY="..."
export SWARMEE_MODEL_PROVIDER="openai"
#
# Bedrock:
# export SWARMEE_MODEL_PROVIDER="bedrock"
# export AWS_REGION="us-east-2"
# export STRANDS_MODEL_ID="us.anthropic.claude-sonnet-4-20250514-v1:0"
#
# Ollama:
# export SWARMEE_MODEL_PROVIDER="ollama"
# export SWARMEE_OLLAMA_HOST="http://localhost:11434"
# export SWARMEE_OLLAMA_MODEL_ID="llama3.1"
#
# GitHub Copilot (enterprise):
# export SWARMEE_MODEL_PROVIDER="github_copilot"
# export GITHUB_TOKEN="..."
# export SWARMEE_GITHUB_COPILOT_MODEL_ID="gpt-4o"

# Run interactive REPL
swarmee

# Run the full-screen terminal UI
pipx install "swarmee-river[tui]"
swarmee tui

# One-shot prompt
swarmee "Summarize the open PRs in this repo and suggest which to close"

# Pipe content in
cat agent-spec.txt | swarmee "Build a specialized agent based on these specifications"
```

## Interfaces

### Interactive CLI

The default `swarmee` command opens an interactive REPL with plan/approve/execute support, tool consent prompts, in-session model tier switching, and built-in commands (`:plan`, `:tier`, `:session`, `:sop`, etc.).

### Terminal UI

```bash
pip install "swarmee-river[tui]"
swarmee tui
```

The TUI is a full-screen Textual application with a multi-panel layout:

- **Transcript** — streaming assistant output with reasoning blocks, tool call expansion, and rich/text display modes
- **Run tab** — plan panel with step-by-step status, approve/replan/cancel actions, and session timeline
- **Agents tab** — agent profile builder: define system prompt snippets, context sources, active SOPs, tool policy, knowledge base, and agent team presets; apply profiles to the live session
- **Tools tab** — browse available tools with access-class metadata; manage prompt templates; import from S3
- **Settings tab** — model tier configuration, diagnostics controls (Settings > Advanced), environment overrides, orchestrator status

The TUI connects to a shared runtime daemon so multiple clients can attach to the same session.

### Shared Runtime

Run a local shared runtime broker (localhost TCP) and attach lightweight clients:

```bash
# Start broker (writes .swarmee/runtime.json)
swarmee serve --port 0

# Attach from the same repo
swarmee attach

# Tail-only mode (no prompt input loop)
swarmee attach --tail

# Daemon management
swarmee daemon status
swarmee daemon start
swarmee daemon stop

# Diagnostics
swarmee diagnostics tail
swarmee diagnostics doctor
swarmee diagnostics bundle
```

Discovery file lives at `.swarmee/runtime.json` (or under `SWARMEE_STATE_DIR`). `swarmee attach` defaults to `SWARMEE_SESSION_ID` when set; otherwise it derives a stable session ID from the current working directory.

## Key Features

**Context management**
- Summarizing conversation manager that trims old turns before hitting token limits
- Tool result limiting: large outputs are truncated in-prompt and persisted to `.swarmee/artifacts/`
- Preflight snapshot: lightweight repo summary / tree / file content injected at startup (`SWARMEE_PREFLIGHT_LEVEL=summary|summary+tree|summary+files`)
- Project map injection for persistent repo awareness

**Agent control**
- **Agent profiles**: named session configurations combining system prompt snippets, context sources, active SOPs, tool policy (allowlist/blocklist/consent), knowledge base, and agent team presets — apply any profile to the live session from the TUI or CLI
- **Plan / approve / execute**: for "do work" prompts Swarmee generates a structured plan and waits for approval before executing
- **Model tier switching**: `fast`, `balanced`, `deep`, `long` tiers with per-provider model ID overrides; switch in-session without restarting
- **SOPs** (Standard Operating Procedures): Markdown-based procedure files injected into the system prompt when active
- **Packs**: install/enable bundles of tools + SOPs + system prompt sections

**Tool ecosystem**
- 30+ built-in tools covering file ops, shell, Python, HTTP, git, artifacts, knowledge base, and multi-agent delegation
- Hot-load custom tools from `./tools/` at runtime
- Optional [Strands Tools](https://github.com/strands-agents/tools) pack for Slack, image generation, video, memory, and more

**Observability**
- JSONL event logs under `.swarmee/logs/` (optional S3 upload)
- Diagnostics files and support bundles under `.swarmee/diagnostics/`
- Artifact store under `.swarmee/artifacts/` indexed by `index.jsonl`
- Session persistence: save/load full conversation state across runs
- REPL replay: `:replay <invocation_id>` reconstructs any logged invocation

**Providers**: Bedrock, OpenAI, Ollama, GitHub Copilot, and custom providers via `.models/`

## Configuration

Configuration precedence (highest → lowest):
1. CLI flags (e.g., `--model-provider`)
2. Environment variables / `.env`
3. Project settings file: `.swarmee/settings.json` (optional; great for teams/repos)
4. Built-in defaults

For the full list of configuration knobs, see [`env.example`](env.example).
For an end-to-end walkthrough of context assembly, trimming, and persistence, see [`docs/agent_context_lifecycle.md`](docs/agent_context_lifecycle.md).

## Developer Onboarding

New to Swarmee River? Start here:

| Guide | What it covers |
|-------|---------------|
| [Getting Started](docs/onboarding/getting-started.md) | Install, configure credentials, run your first session |
| [Core Concepts](docs/onboarding/core-concepts.md) | Architecture, tools, SOPs, profiles, sessions |
| [First Workflow](docs/onboarding/first-workflow.md) | Build a code review agent end-to-end |
| [Custom Tools](docs/onboarding/custom-tools.md) | Write, annotate, and test a custom `@tool` |
| [SOPs and Profiles](docs/onboarding/sops-and-profiles.md) | SOP format, agent profile schema, examples |
| [Delegation](docs/onboarding/delegation.md) | strand, swarm, and use_agent patterns |
| [Debugging](docs/onboarding/debugging.md) | Logs, diagnostics, common errors and fixes |

## Model Providers

Swarmee is packaged with `bedrock`, `openai`, `ollama`, and `github_copilot` providers. See `docs/configuration/` for default model configs and reasoning settings.

### Bedrock

```bash
export SWARMEE_MODEL_PROVIDER="bedrock"
export AWS_REGION="us-east-2"
# Optional: override model
export STRANDS_MODEL_ID="us.anthropic.claude-sonnet-4-20250514-v1:0"
swarmee
```

Bedrock auth is credential-chain-first (env/profile/process/IMDS). Use `AWS_PROFILE` only when you want explicit profile-based login.

### OpenAI

```bash
echo "OPENAI_API_KEY=..." > .env
echo "OPENAI_BASE_URL=https://api.openai.com/v1" >> .env
swarmee --model-provider openai "Hello"
```

Tip: if you see a `max_tokens` / output token limit error, increase the output cap:

```bash
swarmee --model-provider openai --max-output-tokens 1024 "List your tools (briefly)"
```

### Ollama

```bash
swarmee --model-provider ollama --model-config '{"model_id": "llama3.1"}' "Hello"
```

### GitHub Copilot (enterprise)

```bash
echo "SWARMEE_MODEL_PROVIDER=github_copilot" > .env
echo "GITHUB_TOKEN=..." >> .env
echo "SWARMEE_GITHUB_COPILOT_MODEL_ID=gpt-4o" >> .env
swarmee
```

For OpenCode-like auth UX:

```bash
# Device-code login (opens browser, stores auth in ~/.local/share/swarmee/auth.json)
swarmee connect

# Manage credentials
swarmee auth list
swarmee auth login github_copilot
swarmee auth logout github_copilot
```

In REPL/TUI, use `:connect` / `:auth ...` or `/connect` / `/auth ...`.

### Custom Model Provider

```bash
# .models/custom_model.py — expose an `instance` function
swarmee --model-provider custom_model --model-config '{"key": "value"}'
```

See the [Strands custom provider guide](https://strandsagents.com/latest/user-guide/concepts/model-providers/custom_model_provider/) for the interface.

### Orchestrator Prompt Asset

```bash
# Manage orchestrator prompt refs from the fixed Orchestrator row in Agents > Builder.
# Local prompts are stored in .swarmee/prompts.json.
# The tracked template lives at .swarmee/prompts.json.example.
```

## Integrated Tools

Swarmee includes 30+ built-in tools across these categories:

- **File operations** — read, write, list, search, glob, patch
- **Shell / code execution** — shell, Python REPL, editor
- **HTTP / APIs** — http_request, use_aws
- **Git** — git operations, project_context
- **Artifacts & memory** — artifact store, todoread/todowrite, knowledge base retrieve/store, journal
- **Multi-agent delegation** — strand (nested agent), swarm (multi-agent), use_agent/use_llm (summary-only)
- **Utilities** — calculator, current_time, environment, welcome, stop
- **Optional (strands_tools pack)** — Slack, generate_image, image_reader, nova_reels, memory, workflow, cron, speak

Every tool declares `read`, `write`, and/or `execute` **permissions** used for plan-mode gating (only read tools allowed during planning), TUI access-class badges, and safety policy enforcement. See [`docs/tool_permissions.md`](docs/tool_permissions.md) for the full reference, permission table, and guide to annotating new tools.

Hot-load your own tools by placing Python files in `./tools/`. Full tool catalog: [`docs/opencode-port/tool_catalog.md`](docs/opencode-port/tool_catalog.md).

Install the optional Strands Tools pack:

```bash
pip install "swarmee-river[strands_tools]"
```

## Integrations

### Knowledge Base (Amazon Bedrock)

Swarmee can retrieve previously stored tools, agent configurations, and conversation summaries from an Amazon Bedrock Knowledge Base, and save new content back to it.

```bash
# Pass KB ID at runtime
swarmee --kb YOUR_KB_ID "Load my data_visualizer tool and add 3D plotting"

# Or set a default
export STRANDS_KNOWLEDGE_BASE_ID="YOUR_KB_ID"
swarmee "Find my most recent agent configuration and make it more efficient"
```

For KB setup, see the [AWS Bedrock Knowledge Base documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html). In the TUI, set the KB via an agent profile's `knowledge_base_id` field.

### Jupyter Notebook

```bash
pip install "swarmee-river[jupyter]"
```

```python
%load_ext swarmee_river.jupyter

%%swarmee
Review this notebook and suggest improvements to the code.
```

Flags: `--yes` (auto-approve), `--plan` (force plan mode), `--no-context` (skip notebook injection).

Example notebook: `examples/notebooks/swarmee_magic_demo.ipynb`

### Office Documents

```bash
pip install "swarmee-river[office]"
```

Enables reading and writing Word, Excel, and PowerPoint files via the bundled office tools.

### Snowflake

```bash
pip install "swarmee-river[snowflake]"
```

Enables the Snowflake connector for data query tools.

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Reporting bugs & features
- Development setup
- Contributing via Pull Requests
- Code of Conduct
- Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
