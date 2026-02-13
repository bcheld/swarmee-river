# Reasoning configuration (tiers + “think harder”)

Swarmee River exposes “reasoning intensity” primarily through model tiers.

## Tiers

- `fast` — minimal latency/cost defaults
- `balanced` — default for most work
- `deep` — stronger reasoning defaults (the “think harder” tier)
- `long` — long output defaults

Use `:tier list` and `:tier set deep` to switch in an interactive session.

## Bedrock (Anthropic via Bedrock)

- Base configuration uses Bedrock “thinking” fields via `additional_request_fields.thinking`.
- `deep` tier increases the default thinking budget to `8192` tokens.

To override, customize `.swarmee/settings.json` for the Bedrock deep tier:

```json
{
  "models": {
    "providers": {
      "bedrock": {
        "tiers": {
          "deep": {
            "additional_request_fields": {
              "thinking": { "type": "enabled", "budget_tokens": 8192 }
            }
          }
        }
      }
    }
  }
}
```

## OpenAI

- Primary lever: pick a stronger deep-tier model ID.
- Optional lever: set `SWARMEE_OPENAI_REASONING_EFFORT=high` to request higher reasoning effort in the deep tier.

Note: `reasoning_effort` is provider/model dependent; if your provider rejects the parameter, remove the env var.

## Verify

- `:config show` — shows effective tier/provider + relevant env vars (secrets redacted)
- `.swarmee/logs/*.jsonl` — per-invocation traces (redacted by default)

