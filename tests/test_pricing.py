from __future__ import annotations

from swarmee_river.pricing import resolve_pricing


def test_resolve_pricing_openai_gpt5_nano_has_model_override() -> None:
    pricing = resolve_pricing(provider="openai", model_id="gpt-5-nano")
    assert pricing is not None
    assert pricing.input_per_1m == 0.05
    assert pricing.output_per_1m == 0.4
    assert pricing.cached_input_per_1m == 0.005


def test_resolve_pricing_openai_gpt53_codex_uses_model_specific_rate() -> None:
    pricing = resolve_pricing(provider="openai", model_id="gpt-5.3-codex")
    assert pricing is not None
    assert pricing.input_per_1m == 1.75
    assert pricing.output_per_1m == 14.0
    assert pricing.cached_input_per_1m == 0.175


def test_resolve_pricing_bedrock_new_model_ids_have_defaults() -> None:
    haiku = resolve_pricing(provider="bedrock", model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    opus = resolve_pricing(provider="bedrock", model_id="us.anthropic.claude-opus-4-6-v1:0")
    assert haiku is not None
    assert opus is not None
    assert haiku.input_per_1m == 1.0
    assert haiku.output_per_1m == 5.0
    assert haiku.cached_input_per_1m == 0.1
    assert opus.input_per_1m == 15.0
    assert opus.output_per_1m == 75.0
    assert opus.cached_input_per_1m == 1.5
