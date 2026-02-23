# GitHub Copilot Provider Integration Plan

## Goal

Add a packaged `github_copilot` model provider so Swarmee can run in enterprise environments
that already license GitHub Copilot, alongside existing Bedrock/OpenAI/Ollama support.

## Scope

- Add a packaged provider module: `src/swarmee_river/models/github_copilot.py`
- Wire provider defaults and env parsing in `src/swarmee_river/utils/model_utils.py`
- Add provider normalization + fallback handling in `src/swarmee_river/utils/provider_utils.py`
- Add tier/config support in `src/swarmee_river/settings.py` and `src/swarmee_river/session/models.py`
- Surface env keys in diagnostics and notebook runtime fingerprinting
- Document usage in `README.md` and `env.example`
- Add test coverage for provider discovery, defaults, and tier env overrides

## Provider Contract

- Provider name: `github_copilot`
- Aliases accepted: `ghcp`, `github-copilot`, `githubcopilot`, `copilot`
- Token precedence:
  1. `SWARMEE_GITHUB_COPILOT_API_KEY`
  2. `GITHUB_TOKEN`
  3. `GH_TOKEN`
- Base URL default: `https://api.githubcopilot.com`

## Tier Defaults

- `fast`: `gpt-4o-mini`
- `balanced`: `gpt-4o`
- `deep`: `gpt-5`
- `long`: `gpt-5`

All tiers support per-tier overrides via:

- `SWARMEE_GITHUB_COPILOT_FAST_MODEL_ID`
- `SWARMEE_GITHUB_COPILOT_BALANCED_MODEL_ID`
- `SWARMEE_GITHUB_COPILOT_DEEP_MODEL_ID`
- `SWARMEE_GITHUB_COPILOT_LONG_MODEL_ID`

## Safety + Fallback Behavior

- Explicit CLI/env provider choice is always respected.
- Auto provider selection order:
  1. OpenAI if `OPENAI_API_KEY` exists
  2. GitHub Copilot if Copilot token exists
  3. Bedrock otherwise
- If provider resolves to Bedrock but AWS credentials are unavailable:
  1. fall back to OpenAI if available
  2. else fall back to GitHub Copilot if available

## Verification

- `ruff check` on updated files
- `pytest` for:
  - `tests/utils/test_model_utils.py`
  - `tests/utils/test_provider_utils.py`
  - `tests/test_settings_and_tiers.py`
