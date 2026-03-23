"""Automatic retry with escalated token limits when Bedrock hits max_tokens."""

from __future__ import annotations

import logging
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterModelCallEvent

from swarmee_river.hooks._compat import register_hook_callback
from swarmee_river.utils.model_utils import bedrock_max_output_tokens

_LOGGER = logging.getLogger(__name__)

_RETRY_STATE_KEY = "_swarmee_max_tokens_retries"
_MAX_RETRIES = 2
_SCALE_FACTOR = 2


class MaxTokensRetryHooks(HookProvider):
    """Retry model calls with increased token limits when stop_reason is max_tokens."""

    def register_hooks(self, registry: HookRegistry) -> None:
        register_hook_callback(registry, AfterModelCallEvent, self._after_model_call)

    def _after_model_call(self, event: AfterModelCallEvent) -> None:
        if event.stop_response is None or event.stop_response.stop_reason != "max_tokens":
            return

        retries = event.invocation_state.get(_RETRY_STATE_KEY, 0)
        if retries >= _MAX_RETRIES:
            _LOGGER.info("max_tokens retry limit reached (%d attempts), not retrying", retries)
            return

        model = getattr(event.agent, "model", None)
        if model is None:
            return

        config: dict[str, Any] = getattr(model, "config", {})
        if not isinstance(config, dict):
            return

        current_max = config.get("max_tokens")
        if not isinstance(current_max, int) or current_max <= 0:
            current_max = 32_768

        model_id = config.get("model_id")
        new_max = min(current_max * _SCALE_FACTOR, bedrock_max_output_tokens(model_id))
        if new_max <= current_max:
            _LOGGER.info("max_tokens already at cap (%d), not retrying", current_max)
            return

        config["max_tokens"] = new_max
        _LOGGER.info("max_tokens retry %d: escalating max_tokens %d -> %d", retries + 1, current_max, new_max)

        additional = config.get("additional_request_fields")
        if isinstance(additional, dict):
            thinking = additional.get("thinking")
            if isinstance(thinking, dict) and thinking.get("type") in ("enabled",):
                current_budget = thinking.get("budget_tokens")
                if isinstance(current_budget, int) and current_budget > 0:
                    new_budget = min(current_budget * _SCALE_FACTOR, new_max // 2)
                    if new_budget > current_budget:
                        thinking["budget_tokens"] = new_budget
                        _LOGGER.info(
                            "max_tokens retry %d: escalating thinking budget %d -> %d",
                            retries + 1,
                            current_budget,
                            new_budget,
                        )

        event.invocation_state[_RETRY_STATE_KEY] = retries + 1
        event.retry = True
