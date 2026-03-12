from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.hooks.tui_metrics import TuiMetricsHooks


class _UsageObject:
    inputTokens = 1200
    outputTokens = 300
    totalTokens = 1500
    cacheReadInputTokens = 900


def test_after_invocation_normalizes_attribute_usage_payload(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        "swarmee_river.hooks.tui_metrics._write_stdout_jsonl",
        lambda payload: emitted.append(dict(payload)),
    )

    hook = TuiMetricsHooks()
    hook.enabled = True
    event = SimpleNamespace(
        result=SimpleNamespace(metrics=SimpleNamespace(latest_agent_invocation=SimpleNamespace(usage=_UsageObject()))),
        invocation_state={"swarmee": {"provider": "bedrock", "model_id": "us.anthropic.claude-opus-4-6-v1"}},
    )

    hook.after_invocation(event)

    assert emitted
    payload = emitted[-1]
    assert payload["event"] == "usage"
    assert payload["usage"] == {
        "input_tokens": 1200,
        "output_tokens": 300,
        "total_tokens": 1500,
        "cache_read_input_tokens": 900,
    }
    assert payload["cost_usd"] is not None


def test_after_model_call_normalizes_attribute_usage_payload(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        "swarmee_river.hooks.tui_metrics._write_stdout_jsonl",
        lambda payload: emitted.append(dict(payload)),
    )

    hook = TuiMetricsHooks()
    hook.enabled = True
    event = SimpleNamespace(
        usage=_UsageObject(),
        invocation_state={"swarmee": {"provider": "openai", "model_id": "gpt-5-mini"}},
    )

    hook.after_model_call(event)

    assert emitted
    payload = emitted[-1]
    assert payload["usage"]["input_tokens"] == 1200
    assert payload["usage"]["output_tokens"] == 300
