# OpenCode → Swarmee River capability matrix (core parity)

This table is the source-of-truth for what “core parity” means for this stage.

Legend:
- ✅ implemented
- 🟡 partial
- ❌ not yet

| OpenCode capability | Swarmee River equivalent | Status | Notes / gap |
|---|---|---:|---|
| One-shot run (`opencode run …`) | One-shot CLI query (`swarmee "…"`) | ✅ | Uses intent classifier to choose plan/execute. |
| Incremental interactive UX | REPL with `:` commands | ✅ | Not a full-screen TUI by design. |
| Session persistence | `.swarmee/sessions/<id>/…` | ✅ | Project-local only. |
| Model/provider switching | `SessionModelManager` tiers + provider resolution | ✅ | Env + settings precedence. |
| “Think harder” / deep reasoning | Guided tier reasoning depth (`low` / `medium` / `high`) | ✅ | OpenAI is Responses-only and stores reasoning effort in tier config; Bedrock maps guided depth to provider-native thinking controls. |
| Read/search primitives | `file_read`, `file_list`, `file_search`, `project_context` (+ `read`/`grep` aliases) | 🟡 | `file_list`/`file_search` prefer `rg` but fall back to pure Python when `rg` is unavailable. `project_context` covers summary/tree/files/search/read. |
| Glob search (`glob`) | `glob` tool | ✅ | Pure-Python glob with skip-dirs guardrails. |
| Directory listing (`list`) | `list` tool | ✅ | Cross-platform, no shell required; hidden entries excluded by default. |
| Shell execution | `shell` tool (fallback) + `bash` alias | ✅ | Consent-gated by default; alias uses the same safety gates. |
| Patch apply tool | `patch_apply` tool + `patch` alias | ✅ | Uses `git apply`; consent-gated by default; alias uses the same safety gates. |
| File write/edit | `file_write` / `editor` (+ `write` / `edit` aliases) | ✅ | Mutating tools are blocked in plan mode and remain consent-gated in execute mode. |
| Git workflows | `git` tool + `:status` / `:diff` | ✅ | Tool supports read + mutating actions; consent-gated. |
| Run checks / tests | `run_checks` tool | ✅ | Captures output as artifacts when large. |
| Permissions / approvals | Tool policy + tool consent + plan gating | ✅ | Plan-mode blocks mutating tools. |
| Log / replay | JSONL hooks + `:log tail` + `:replay <id>` | ✅ | Usage capture is best-effort (provider dependent). |
| Web/server UI | n/a | ❌ | Out of scope for this stage. |
| MCP servers | n/a | ❌ | Pack system covers local tools only (for now). |
| LSP tool | n/a | ❌ | Not implemented yet. |
| Todo tool | `todoread` / `todowrite` | ✅ | Project-local persistence under `<state_dir>/todo.md` with plan-mode read-only gating. |
| Share links | n/a | ❌ | Not implemented yet. |
