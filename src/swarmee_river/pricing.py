from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPricing:
    """USD prices per 1M tokens."""

    input_per_1m: float | None = None
    output_per_1m: float | None = None
    cached_input_per_1m: float | None = None

    def with_fallbacks(self) -> "TokenPricing":
        cached = self.cached_input_per_1m if self.cached_input_per_1m is not None else self.input_per_1m
        return TokenPricing(
            input_per_1m=self.input_per_1m,
            output_per_1m=self.output_per_1m,
            cached_input_per_1m=cached,
        )


# -----------------------------------------------------------------------------
# Optional, code-defined pricing defaults.
#
# This project supports provider/model-specific overrides via env vars, but when
# those are not set, the TUI can fall back to these defaults.
#
# Keep these values aligned with your org's current pricing/contract terms.
# If a value is None, cost display is suppressed.
# -----------------------------------------------------------------------------

DEFAULT_PROVIDER_PRICING: dict[str, TokenPricing] = {
    # Fill these in if you want cost estimates without env vars.
    "openai": TokenPricing(input_per_1m=1.75, output_per_1m=14, cached_input_per_1m=0.175),
    "bedrock": TokenPricing(input_per_1m=3, output_per_1m=15, cached_input_per_1m=0.3),
    "ollama": TokenPricing(input_per_1m=0.0, output_per_1m=0.0, cached_input_per_1m=0.0),
}

# Exact model_id overrides (highest priority).
DEFAULT_MODEL_PRICING: dict[str, TokenPricing] = {
    # OpenAI list pricing (openai.com/api/pricing, checked 2026-03-03).
    "gpt-5-nano": TokenPricing(input_per_1m=0.05, output_per_1m=0.4, cached_input_per_1m=0.005),
    "gpt-5-mini": TokenPricing(input_per_1m=0.25, output_per_1m=2.0, cached_input_per_1m=0.025),
    "gpt-5.2": TokenPricing(input_per_1m=1.75, output_per_1m=14.0, cached_input_per_1m=0.175),
    # gpt-5.3-codex is mapped to GPT-5.2 rates until OpenAI publishes distinct pricing.
    "gpt-5.3-codex": TokenPricing(input_per_1m=1.75, output_per_1m=14.0, cached_input_per_1m=0.175),
    # Bedrock Anthropic defaults for configured model IDs.
    # AWS model IDs from docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
    # Pricing baselines from Anthropic published rates until AWS exposes per-model entries for these revisions.
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": TokenPricing(
        input_per_1m=1.0,
        output_per_1m=5.0,
        cached_input_per_1m=0.1,
    ),
    "us.anthropic.claude-opus-4-6-v1:0": TokenPricing(
        input_per_1m=15.0,
        output_per_1m=75.0,
        cached_input_per_1m=1.5,
    ),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": TokenPricing(input_per_1m=1, output_per_1m=4),
}


def resolve_pricing(*, provider: str | None, model_id: str | None) -> TokenPricing | None:
    model_key = (model_id or "").strip()
    if model_key and model_key in DEFAULT_MODEL_PRICING:
        return DEFAULT_MODEL_PRICING[model_key].with_fallbacks()

    provider_key = (provider or "").strip().lower()
    if provider_key and provider_key in DEFAULT_PROVIDER_PRICING:
        pricing = DEFAULT_PROVIDER_PRICING[provider_key].with_fallbacks()
        if pricing.input_per_1m is None and pricing.output_per_1m is None:
            return None
        return pricing

    return None
