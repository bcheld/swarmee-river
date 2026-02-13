# OpenCode built-in tool catalog (for parity mapping)

This catalog is based on OpenCode’s public tool documentation and is intended for mapping to Swarmee River primitives.

## Built-in tools (names)

- `bash` — execute shell commands
- `edit` — modify existing files via exact replacements
- `write` — create/overwrite files (covered by `edit` permission umbrella)
- `read` — read files (often supports ranges)
- `grep` — regex search
- `glob` — glob search
- `list` — list directory contents
- `lsp` (experimental) — LSP queries (definitions, references, hover, etc.)
- `patch` — apply unified diffs/patches (covered by `edit` umbrella)
- `skill` — load a `SKILL.md`
- `todowrite` / `todoread` — maintain a todo list
- `webfetch` — fetch URL content
- `websearch` — search the web (provider/feature gated)
- `question` — ask the user for clarification during execution

## Risk buckets (recommended)

- High risk (mutating / remote / exec): `bash`, `edit`/`write`/`patch`, `webfetch`
- Medium risk (broad context): `read`, `grep`, `glob`, `list`
- Higher-trust / platform-specific: `lsp`, `skill`
- Coordination: `question`, `todowrite`, `todoread`

Swarmee River uses:

- Plan gating (plan → approve → execute)
- Tool consent rules (ask/allow/deny per tool)
- Policy hooks (block in plan-mode, block outside approved plan, allowlists)

