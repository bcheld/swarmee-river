# Migrating from OpenCode → Swarmee River

This guide maps common OpenCode workflows/config into the Swarmee River harness.

## Quick mapping

| OpenCode | Swarmee River |
|---|---|
| `opencode` (TUI) | `swarmee` (incremental REPL) |
| `opencode run "…"` | `swarmee "…"` |
| `/connect` | provider env vars (`OPENAI_API_KEY`, AWS credentials) |
| `opencode.json(c)` | `.swarmee/settings.json` + `.env` |
| `permission` | `safety.tool_consent` + `safety.tool_rules` |
| built-in tools (`bash`, `edit`, `read`, …) | `shell`, `editor`/`file_write`/`file_read`, `patch_apply`, `project_context`, `git`, `run_checks` |

## Commands

Swarmee River uses `:` commands in the REPL:

- `:help` — list commands
- `:session new|save|load|list|rm|info` — session management (project-local)
- `:tier list|set|auto` — tier switching
- `:status` — git status summary
- `:diff [--staged]` — git diff
- `:config show` — effective config (redacts secrets)
- `:artifact list|get` — inspect `.swarmee/artifacts`
- `:log tail` — tail `.swarmee/logs/*.jsonl`
- `:replay <invocation_id>` — reconstruct a run from logs

## Permissions / approvals

OpenCode uses `permission` rules (`allow`/`ask`/`deny`, optionally granular patterns).

Swarmee River equivalents:

- Plan gating: “work” requests generate a plan; execution requires `:y`/`:approve` (or `--yes`)
- Tool consent: interactive prompts per tool (session-scoped decisions)
- Tool policy: hard blocks in plan-mode and optional allowlists/denylists

### Recommended baseline

- Keep tool consent default `ask` for `shell`, `editor`, `file_write`, `http_request`, `git`, `patch_apply`, `run_checks`
- Use `deep` tier for harder reasoning and richer preflight context

