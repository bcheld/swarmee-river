from __future__ import annotations

import contextlib
import os
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterInvocationEvent, AfterModelCallEvent

from swarmee_river.hooks._compat import register_hook_callback
from swarmee_river.pricing import resolve_pricing
from swarmee_river.settings import PricingConfig, PricingOverride
from swarmee_river.utils.stdio_utils import write_stdout_jsonl


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _write_stdout_jsonl(event: dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        write_stdout_jsonl(event)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _extract_usage_payload(event: Any) -> dict[str, Any] | None:
    usage = getattr(event, "usage", None)
    if isinstance(usage, dict):
        return usage

    response = getattr(event, "response", None)
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        return response.get("usage")

    stop_response = getattr(event, "stop_response", None)
    if stop_response is not None:
        stop_usage = getattr(stop_response, "usage", None)
        if isinstance(stop_usage, dict):
            return stop_usage

    return None


def _extract_token_counts(usage: dict[str, Any]) -> tuple[int, int, int]:
    input_tokens = _as_int(usage.get("input_tokens")) or _as_int(usage.get("prompt_tokens")) or 0
    output_tokens = _as_int(usage.get("output_tokens")) or _as_int(usage.get("completion_tokens")) or 0

    cached = 0
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _as_int(details.get("cached_tokens")) or 0
    cached = cached or (_as_int(usage.get("cache_read_input_tokens")) or 0)
    return input_tokens, output_tokens, cached


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def _resolve_pricing_override(pricing: PricingConfig, provider: str | None) -> PricingOverride:
    if not provider:
        return pricing.default
    provider_key = str(provider).strip().lower()
    override = pricing.providers.get(provider_key)
    return override if override is not None else pricing.default


def _compute_cost_usd(
    *,
    usage: dict[str, Any],
    provider: str | None,
    model_id: str | None,
    pricing: PricingConfig,
) -> float | None:
    default_pricing = resolve_pricing(provider=provider, model_id=model_id)
    override = _resolve_pricing_override(pricing, provider)

    input_rate = override.input_per_1m if override.input_per_1m is not None else (
        default_pricing.input_per_1m if default_pricing else None
    )
    output_rate = override.output_per_1m if override.output_per_1m is not None else (
        default_pricing.output_per_1m if default_pricing else None
    )
    cached_rate = (
        override.cached_input_per_1m
        if override.cached_input_per_1m is not None
        else (default_pricing.cached_input_per_1m if default_pricing else None)
    )

    return _estimate_cost_usd(
        usage=usage,
        input_rate_per_1m=input_rate,
        output_rate_per_1m=output_rate,
        cached_input_rate_per_1m=cached_rate,
    )


def _estimate_cost_usd(
    *,
    usage: dict[str, Any],
    input_rate_per_1m: float | None = None,
    output_rate_per_1m: float | None = None,
    cached_input_rate_per_1m: float | None = None,
) -> float | None:
    """
    Provider-agnostic cost estimate.

    If pricing config does not provide any rates and there is no built-in default, returns None.
    """
    input_rate = input_rate_per_1m
    output_rate = output_rate_per_1m
    if input_rate is None and output_rate is None:
        return None
    cached_input_rate = (
        cached_input_rate_per_1m if cached_input_rate_per_1m is not None else None
    )
    if cached_input_rate is None:
        cached_input_rate = input_rate

    input_tokens, output_tokens, cached = _extract_token_counts(usage)
    billable_input = max(0, input_tokens - cached)
    cost = 0.0
    if input_rate is not None:
        cost += (billable_input / 1_000_000.0) * input_rate
    if cached_input_rate is not None and cached:
        cost += (cached / 1_000_000.0) * cached_input_rate
    if output_rate is not None:
        cost += (output_tokens / 1_000_000.0) * output_rate
    return round(cost, 6)


class TuiMetricsHooks(HookProvider):
    """Emit token usage / cost events to stdout for the Textual TUI."""

    def __init__(self, *, pricing: PricingConfig | None = None) -> None:
        self.enabled = _truthy_env("SWARMEE_TUI_EVENTS", False)
        self._pricing = pricing or PricingConfig()

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        register_hook_callback(registry, AfterInvocationEvent, self.after_invocation)
        register_hook_callback(registry, AfterModelCallEvent, self.after_model_call)

    def after_invocation(self, event: AfterInvocationEvent) -> None:
        """
        Prefer emitting usage from `AgentResult.metrics`, since Strands' AfterModelCallEvent
        does not currently carry usage payloads directly.
        """
        if not self.enabled:
            return
        result = getattr(event, "result", None)
        metrics = getattr(result, "metrics", None) if result is not None else None
        latest_invocation = getattr(metrics, "latest_agent_invocation", None) if metrics is not None else None
        usage_obj = getattr(latest_invocation, "usage", None) if latest_invocation is not None else None
        if not isinstance(usage_obj, dict):
            return

        provider: str | None = None
        model_id: str | None = None
        sw = event.invocation_state.get("swarmee") if isinstance(event.invocation_state, dict) else None
        if isinstance(sw, dict):
            raw_provider = sw.get("provider")
            provider = str(raw_provider).strip().lower() if isinstance(raw_provider, str) else None
            raw_model = sw.get("model_id")
            model_id = str(raw_model).strip() if isinstance(raw_model, str) else None

        normalized: dict[str, Any] = {
            "input_tokens": _as_int(usage_obj.get("inputTokens")) or 0,
            "output_tokens": _as_int(usage_obj.get("outputTokens")) or 0,
            "total_tokens": _as_int(usage_obj.get("totalTokens")) or 0,
        }
        cache_read = _as_int(usage_obj.get("cacheReadInputTokens"))
        cache_write = _as_int(usage_obj.get("cacheWriteInputTokens"))
        if cache_read is not None:
            normalized["cache_read_input_tokens"] = cache_read
        if cache_write is not None:
            normalized["cache_write_input_tokens"] = cache_write

        payload: dict[str, Any] = {"event": "usage", "usage": normalized, "provider": provider, "model_id": model_id}
        cost = _compute_cost_usd(usage=normalized, provider=provider, model_id=model_id, pricing=self._pricing)
        if cost is not None:
            payload["cost_usd"] = cost
        _write_stdout_jsonl(payload)

    def after_model_call(self, event: AfterModelCallEvent) -> None:
        if not self.enabled:
            return
        usage = _extract_usage_payload(event)
        if not isinstance(usage, dict):
            return
        provider: str | None = None
        model_id: str | None = None
        invocation_state = getattr(event, "invocation_state", None)
        if isinstance(invocation_state, dict):
            sw = invocation_state.get("swarmee")
            if isinstance(sw, dict):
                raw_provider = sw.get("provider")
                provider = str(raw_provider).strip().lower() if isinstance(raw_provider, str) else None
                raw_model = sw.get("model_id")
                model_id = str(raw_model).strip() if isinstance(raw_model, str) else None

        payload: dict[str, Any] = {"event": "usage", "usage": usage, "provider": provider, "model_id": model_id}
        cost = _compute_cost_usd(usage=usage, provider=provider, model_id=model_id, pricing=self._pricing)
        if cost is not None:
            payload["cost_usd"] = cost
        _write_stdout_jsonl(payload)
