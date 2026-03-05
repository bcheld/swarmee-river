# Prompt cache guidance

This document describes how Swarmee River keeps long-running agent sessions cache-friendly,
especially for OpenAI Responses-backed tiers.

## Core rules

- Keep the prompt prefix stable.
- Keep tool ordering deterministic.
- Prefer discovery over changing the active tool set.
- Compact with tool state preserved, not flattened away.

## Responses-first defaults

OpenAI tiers are Responses-only and now pair model choice with guided context behavior:

- `context.strategy=balanced`
  - Default for interactive analysis and coding work.
- `context.strategy=cache_safe`
  - Strongest prompt-prefix stability. Use this when long sessions, repeated turns, or provider-side caching
    matter more than immediate prompt richness.
- `context.strategy=long_running`
  - Best for tool-heavy work where compaction and artifact references need to stay coherent over time.

Compaction is controlled separately:

- `context.compaction=auto`
  - Compact automatically when the budget is exceeded.
- `context.compaction=manual`
  - Do not compact until the operator asks for it.

## Stable tool registry

Prompt caches break easily when the tool list changes. Swarmee therefore treats tool inventory stability as
part of context management.

What to prefer:

- Keep a stable catalog of `core`, `pack`, `native`, and `connector-backed` tools for the session.
- Keep tool ordering deterministic.
- Use tool search/discovery to reveal detail when needed instead of removing and re-adding tools.
- Use policy and consent to control usage, not ad hoc tool-set mutations.

What to avoid:

- Switching between different tool sets mid-session.
- Reordering tools non-deterministically.
- Treating plan mode as a different tool catalog.
- For Bedrock Claude sessions, forcing a specific tool while also expecting reasoning output.

## Compaction guidance

Compaction should preserve orchestration state, not just user-visible prose.

The tool-aware summarizing manager now accounts for:

- assistant reasoning fragments
- tool call IDs, names, and inputs
- tool result IDs, status, and result content
- artifact references

That matters because a cache-friendly summary still needs to let the agent continue an in-flight
multi-tool workflow without losing the thread.

For Bedrock Claude models, this also avoids a common failure mode: Bedrock reasoning is compatible with normal
tool availability, but not with forced tool choice on the same request.

Use these strategies:

- `balanced`
  - Good default. Summarizes normal conversation growth while preserving recent turns.
- `cache_safe`
  - Best when the stable prefix is the priority. Keep tool presence fixed and avoid loading detailed schemas
    unless the model explicitly discovers them.
- `long_running`
  - Best when tool outputs are large. Persist bulky outputs to artifacts, then preserve references during
    compaction rather than replaying raw text.

## System prompt and reminder guidance

Keep the API-level system prompt stable. Put changing information into follow-up context or reminders
instead of rewriting the system prompt every turn.

Swarmee already follows this pattern for:

- runtime environment details
- project map and preflight snapshots
- active SOP context
- approved plan context during execution

## Operational checklist

- Prefer one tier per session; changing models can invalidate cache reuse.
- Prefer `context.strategy=cache_safe` for long OpenAI Responses sessions.
- Keep `SWARMEE_FREEZE_TOOLS=true` when you want to avoid accidental tool-registry churn.
- Keep large tool outputs in artifacts and references rather than raw prompt text.
- Review JSONL logs if your provider exposes cache usage metrics:
  - `python3 scripts/prompt_cache_stats.py .swarmee/logs`
