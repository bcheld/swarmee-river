# OpenCode commands and config (inventory for porting)

This is a curated inventory of OpenCode user-facing controls that matter for parity.

## CLI entrypoints (common)

- `opencode` — launch interactive TUI
- `opencode run [message..]` — non-interactive run (one-shot)
- `opencode models [provider]` — list models (optionally filter)
- `opencode serve` — headless HTTP server (OpenAPI)
- `opencode web` — web UI (server + browser client)
- `opencode attach <url>` — attach a TUI client to a running server
- `opencode mcp ...` — MCP server auth / debug helpers

## Config file

OpenCode uses `opencode.json` / `opencode.jsonc` with a published JSON schema:

- Schema: `https://opencode.ai/config.json`
- Locations (merged, not replaced):
  - remote org defaults (`.well-known/opencode`)
  - global config (`~/.config/opencode/opencode.json`)
  - custom path (`OPENCODE_CONFIG`)
  - per-project config (`opencode.json` in project root)
  - config directories (e.g. `.opencode/agents`, `.opencode/commands`, plugins, etc.)
  - inline overrides (`OPENCODE_CONFIG_CONTENT`)

## Permissions

OpenCode controls tool behavior through a `permission` config:

- `"allow"`, `"ask"`, `"deny"`
- can be global `permission: "allow"` or an object keyed by tool name
- supports granular rules via pattern matching (e.g., for `bash` commands or file paths)

Swarmee River equivalent concepts:

- Plan gating: explicit plan approval before execution for “work” intent
- Tool consent: session-scoped allow/deny decisions (ToolConsentHooks)
- Tool policy: allow/deny tool names + plan-mode read-only allowlist (ToolPolicyHooks)
- Optional per-tier tool profiles (settings `harness.tier_profiles`)

