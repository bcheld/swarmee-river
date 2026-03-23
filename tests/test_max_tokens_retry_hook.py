from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.hooks.max_tokens_retry import MaxTokensRetryHooks


def test_max_tokens_retry_hook_clamps_bedrock_sonnet_retries_to_model_cap() -> None:
    hook = MaxTokensRetryHooks()
    model = SimpleNamespace(
        config={
            "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "max_tokens": 20_000,
            "additional_request_fields": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 12_000,
                }
            },
        }
    )
    event = SimpleNamespace(
        stop_response=SimpleNamespace(stop_reason="max_tokens"),
        invocation_state={},
        agent=SimpleNamespace(model=model),
        retry=False,
    )

    hook._after_model_call(event)

    assert model.config["max_tokens"] == 32_768
    assert model.config["additional_request_fields"]["thinking"]["budget_tokens"] == 16_384
    assert event.invocation_state["_swarmee_max_tokens_retries"] == 1
    assert event.retry is True
