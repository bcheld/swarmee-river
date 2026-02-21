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
    # Examples (set to your actual defaults):
    # "gpt-5-mini": TokenPricing(input_per_1m=..., output_per_1m=14, cached_input_per_1m=0.175),
    # "us.anthropic.claude-sonnet-4-20250514-v1:0": TokenPricing(input_per_1m=..., output_per_1m=...),
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

