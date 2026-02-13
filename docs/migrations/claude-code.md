# Migrating from Claude Code → Swarmee River

Claude Code users typically care about: approvals, safety defaults, fast repo-aware planning, and tight edit/test loops.

## Trust model

Swarmee River approximates a Claude Code–style trust model with:

- Plan-first gating for “work” intent (plan → approve → execute)
- Consent prompts for high-risk tools (session-scoped allow/deny)
- Optional hard policy blocks (plan-mode read-only, allowlists/denylists)

## Common workflows

- “Plan it first” → `:plan` then enter your request
- Approve plan → `:y` / `:approve`
- Cancel → `:n`
- Replan → `:replan`
- Inspect repo → `:status`, `:diff`, `:artifact list`, `:log tail`
- Resume work → `:session save` / `:session load <id>`

## Key differences

- Swarmee River persists state under `.swarmee/` (project-local), not a global user store.
- Swarmee River is provider-agnostic via Strands model providers (Bedrock/OpenAI/Ollama).

