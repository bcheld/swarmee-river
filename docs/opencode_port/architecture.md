# OpenCode architecture (high level)

This document is a working, “good enough” architecture sketch to support feature mapping into Swarmee River.

## Repository shape

OpenCode is a TypeScript monorepo (Bun workspaces). Key top-level directories you’ll see in `dev`:

- `packages/` — product code (CLI/TUI, web, desktop, SDKs, plugins)
- `specs/` — schemas/specs (OpenAPI, config schema, etc.)
- `sdks/` — editor/IDE SDKs (e.g., VS Code)
- `script/` — build/release tooling
- `infra/` — deployment/infra definitions

## Client/server split

OpenCode is designed around a local server that can be driven by multiple clients:

- TUI client (interactive terminal UI)
- Web client (`opencode web`)
- Headless server (`opencode serve`)
- “One-shot” CLI (`opencode run ...`) that can optionally attach to a running server

This is conceptually similar to “agent runtime + clients” rather than a single monolithic CLI loop.

## Core request loop (conceptual)

At a high level, OpenCode runs:

1. Build session context (project + conversation)
2. Call model provider (selected model)
3. Model requests tools (read/search/edit/bash/…)
4. Tool execution returns results
5. Model continues until final response
6. Persist session + logs for replay/debugging

## Persistence model

OpenCode stores:

- logs (platform-specific user data directory)
- project/session state (keyed by a project identifier; different behavior for git vs non-git folders)

Swarmee River intentionally diverges here for this stage: project-only persistence under `.swarmee/`.

## Extensibility

OpenCode supports multiple extension surfaces:

- config-driven permissions (allow/ask/deny, with granular patterns)
- plugins
- MCP servers (tool sources)
- custom agents/commands/modes via config directories

Swarmee River maps this to:

- settings + env overrides
- packs (prompt sections + tools)
- consent + policy hooks

