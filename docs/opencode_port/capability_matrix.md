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
| “Think harder” / deep reasoning | `deep` tier + provider-specific knobs | ✅ | Bedrock deep tier raises thinking budget; OpenAI optional env for `reasoning_effort`. |
| Read/search primitives | `file_read`, `file_list`, `file_search`, `project_context` | 🟡 | `file_list`/`file_search` prefer `rg` but fall back to pure Python when `rg` is unavailable. `project_context` covers summary/tree/files/search/read. |
| Glob search (`glob`) | `glob` tool | ✅ | Pure-Python glob with skip-dirs guardrails. |
| Directory listing (`list`) | `list` tool | ✅ | Cross-platform, no shell required; hidden entries excluded by default. |
| Shell execution | `shell` tool (fallback) | ✅ | Consent-gated by default. |
| Patch apply tool | `patch_apply` tool | ✅ | Uses `git apply`; consent-gated by default. |
| Git workflows | `git` tool + `:status` / `:diff` | ✅ | Tool supports read + mutating actions; consent-gated. |
| Run checks / tests | `run_checks` tool | ✅ | Captures output as artifacts when large. |
| Permissions / approvals | Tool policy + tool consent + plan gating | ✅ | Plan-mode blocks mutating tools. |
| Log / replay | JSONL hooks + `:log tail` + `:replay <id>` | ✅ | Usage capture is best-effort (provider dependent). |
| Web/server UI | n/a | ❌ | Out of scope for this stage. |
| MCP servers | n/a | ❌ | Pack system covers local tools only (for now). |
| LSP tool | n/a | ❌ | Not implemented yet. |
| Todo tool | n/a | ❌ | Not implemented yet. |
| Share links | n/a | ❌ | Not implemented yet. |
