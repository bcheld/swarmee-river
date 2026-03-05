# Reasoning configuration

Swarmee River now exposes reasoning as a guided model-tier choice instead of ad hoc provider arguments.

## OpenAI

OpenAI tiers are Responses-only. Each OpenAI tier stores a guided reasoning profile in `.swarmee/settings.json`:

```json
{
  "models": {
    "providers": {
      "openai": {
        "tiers": {
          "deep": {
            "transport": "responses",
            "reasoning": { "effort": "high" },
            "tooling": { "mode": "tool-heavy", "discovery": "search" },
            "context": { "strategy": "cache_safe", "compaction": "auto" }
          }
        }
      }
    }
  }
}
```

The model manager writes these fields for you. Users should not need to hand-edit `params`, `client_args`, or provider-specific request arguments.

### Guided reasoning depths

- `low` — fastest, lowest-cost reasoning profile
- `medium` — default day-to-day reasoning profile
- `high` — deeper repo reasoning and multi-step tool orchestration

## Bedrock

Bedrock is now modeled as `Claude-first on Bedrock`. Users still pick guided reasoning depth in the model manager, but Swarmee maps that choice to the correct Bedrock request shape for the selected Claude family:

- `fast`
  - Claude Haiku 4.5
  - low reasoning
  - minimal tools
- `balanced`
  - Claude Sonnet 4.5
  - medium reasoning
  - standard tool use
- `deep`
  - Claude Opus 4.6
  - adaptive reasoning
  - cache-safe tool/context defaults
- `long`
  - Claude Opus 4.6
  - adaptive reasoning
  - long-running context defaults

### Bedrock request mapping

The primary UI does not expose raw Bedrock request fields. Swarmee derives them internally from tier config and model family:

- Claude Opus 4.6 tiers use adaptive thinking.
- Claude Haiku 4.5 and Sonnet 4.5 tiers use extended thinking.
- Non-Claude Bedrock models fall back to no Bedrock-specific reasoning fields unless explicitly supported.
- Forced tool choice disables Bedrock reasoning fields for that request because Bedrock does not allow reasoning with forced tool use.

Provider-level Bedrock extras such as AWS profile, timeouts, retries, and guardrail wiring remain valid infrastructure settings, but raw reasoning payloads are not part of the primary model-manager flow.

## Tooling and context are part of reasoning

Reasoning depth is no longer treated as an isolated knob. Each tier can also express:

- `tooling.mode`: `minimal`, `standard`, or `tool-heavy`
- `tooling.discovery`: currently `off` or `search`
- `context.strategy`: `balanced`, `cache_safe`, or `long_running`
- `context.compaction`: `auto` or `manual`

This keeps the UI focused on behavior, not raw API vocabulary.

## Verify

- `:tier list` shows the active tier catalog.
- `:config show` shows the effective provider and tier.
- `.swarmee/logs/*.jsonl` captures per-invocation metadata, including guided reasoning/context selections when available.
