from __future__ import annotations

import logging
from typing import Any

import pytest

from swarmee_river.models import bedrock as bedrock_model


class _FakeBedrockModel:
    BedrockConfig = dict

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _reset_warning_cache(monkeypatch) -> None:
    monkeypatch.setattr(bedrock_model, "_MISSING_REGION_WARNING_KEYS", set())


def test_bedrock_instance_warns_for_unprefixed_model_id(monkeypatch, caplog) -> None:
    _reset_warning_cache(monkeypatch)
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeBedrockModel)
    caplog.set_level(logging.WARNING, logger="swarmee_river.models.bedrock")

    model = bedrock_model.instance(model_id="anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")

    assert isinstance(model, _FakeBedrockModel)
    assert any("unprefixed" in record.getMessage() for record in caplog.records)


def test_bedrock_instance_warns_for_prefix_region_mismatch(monkeypatch, caplog) -> None:
    _reset_warning_cache(monkeypatch)
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeBedrockModel)
    caplog.set_level(logging.WARNING, logger="swarmee_river.models.bedrock")

    model = bedrock_model.instance(model_id="eu.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")

    assert isinstance(model, _FakeBedrockModel)
    assert any("may not match resolved region" in record.getMessage() for record in caplog.records)


def test_bedrock_instance_no_warning_for_matching_prefixed_model(monkeypatch, caplog) -> None:
    _reset_warning_cache(monkeypatch)
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeBedrockModel)
    caplog.set_level(logging.WARNING, logger="swarmee_river.models.bedrock")

    model = bedrock_model.instance(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")

    assert isinstance(model, _FakeBedrockModel)
    assert not caplog.records


def test_bedrock_instance_uses_aws_region_env_without_warning(monkeypatch, caplog) -> None:
    _reset_warning_cache(monkeypatch)
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeBedrockModel)
    # Botocore region inference honors AWS_DEFAULT_REGION.
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    caplog.set_level(logging.WARNING, logger="swarmee_river.models.bedrock")

    model = bedrock_model.instance(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")

    assert isinstance(model, _FakeBedrockModel)
    assert not caplog.records


def test_bedrock_instance_uses_inferred_region_without_warning(monkeypatch, caplog) -> None:
    _reset_warning_cache(monkeypatch)
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeBedrockModel)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(bedrock_model, "resolve_aws_region_source", lambda: ("us-east-1", "profile_or_config"))
    caplog.set_level(logging.WARNING, logger="swarmee_river.models.bedrock")

    model = bedrock_model.instance(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")

    assert isinstance(model, _FakeBedrockModel)
    assert not caplog.records


def test_bedrock_instance_warns_once_per_process_for_missing_region(monkeypatch, caplog) -> None:
    _reset_warning_cache(monkeypatch)
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeBedrockModel)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(bedrock_model, "resolve_aws_region_source", lambda: (None, "unknown"))
    caplog.set_level(logging.WARNING, logger="swarmee_river.models.bedrock")

    model_a = bedrock_model.instance(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    model_b = bedrock_model.instance(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")

    assert isinstance(model_a, _FakeBedrockModel)
    assert isinstance(model_b, _FakeBedrockModel)
    matching = [record for record in caplog.records if "is prefixed but AWS region is not set" in record.getMessage()]
    assert len(matching) == 1


def test_validate_converse_stream_request_accepts_valid_payload() -> None:
    model = bedrock_model.instance(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")
    request = {
        "modelId": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "system": [],
        "inferenceConfig": {},
    }

    bedrock_model._validate_converse_stream_request(request, client=model.client)


def test_validate_converse_stream_request_rejects_invalid_payload() -> None:
    model = bedrock_model.instance(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")
    request = {
        "modelId": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "messages": "not-a-list",
        "system": [],
        "inferenceConfig": {},
    }

    with pytest.raises(bedrock_model.BedrockConverseStreamValidationError, match="request validation failed"):
        bedrock_model._validate_converse_stream_request(request, client=model.client)


class _FakeFormattingBedrockModel:
    BedrockConfig = dict

    def __init__(self, **kwargs: Any) -> None:
        self.config = dict(kwargs)
        self.client = type("Client", (), {"meta": type("Meta", (), {"service_model": None})()})()

    def _format_request(
        self,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]] | None = None,
        system_prompt_content: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = tool_choice
        request: dict[str, Any] = {
            "modelId": self.config["model_id"],
            "messages": messages,
            "system": list(system_prompt_content or []),
            "inferenceConfig": {},
        }
        if tool_specs:
            request["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": spec["name"],
                            "description": spec["description"],
                            "inputSchema": spec["inputSchema"],
                        }
                    }
                    for spec in tool_specs
                ]
            }
            if self.config.get("cache_tools"):
                request["toolConfig"]["tools"].append({"cachePoint": {"type": self.config["cache_tools"]}})
        return request


def _cache_locations(request: dict[str, Any]) -> list[str]:
    locations: list[str] = []
    for item in request.get("toolConfig", {}).get("tools", []):
        if "cachePoint" in item:
            locations.append("tools")
    for item in request.get("system", []):
        if "cachePoint" in item:
            locations.append("system")
    for idx, message in enumerate(request.get("messages", [])):
        for item in message.get("content", []):
            if "cachePoint" in item:
                locations.append(f"messages[{idx}]")
    return locations


def test_swarmee_cache_policy_places_explicit_checkpoints(monkeypatch) -> None:
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeFormattingBedrockModel)
    model = bedrock_model.instance(
        model_id="us.anthropic.claude-opus-4-7",
        region_name="us-east-1",
        cache_tools="default",
        swarmee_cache_policy={
            "enabled": True,
            "cache_type": "default",
            "max_checkpoints": 4,
        },
    )

    request = model._format_request(
        [
            {"role": "user", "content": [{"text": "first"}]},
            {"role": "assistant", "content": [{"text": "stable older"}]},
            {"role": "user", "content": [{"text": "second"}]},
            {"role": "assistant", "content": [{"text": "stable newer"}]},
            {"role": "user", "content": [{"text": "<system-reminder>dynamic</system-reminder>\n\ncurrent query"}]},
        ],
        tool_specs=[{"name": "read", "description": "Read", "inputSchema": {"json": {"type": "object"}}}],
        system_prompt_content=[{"text": "stable system"}],
    )

    locations = _cache_locations(request)
    assert locations == ["tools", "system", "messages[1]", "messages[3]"]
    assert len(locations) == 4
    assert request["messages"][-1]["content"] == [
        {"text": "<system-reminder>dynamic</system-reminder>\n\ncurrent query"}
    ]
    assert model._swarmee_last_cache_diagnostics["bedrock_cache_checkpoint_count"] == 4


def test_swarmee_cache_policy_emits_one_hour_ttl_for_long_running(monkeypatch) -> None:
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeFormattingBedrockModel)
    model = bedrock_model.instance(
        model_id="us.anthropic.claude-opus-4-7",
        region_name="us-east-1",
        cache_tools="default",
        swarmee_cache_policy={
            "enabled": True,
            "cache_type": "default",
            "ttl": "1h",
            "max_checkpoints": 4,
        },
    )

    request = model._format_request(
        [{"role": "assistant", "content": [{"text": "stable"}]}, {"role": "user", "content": [{"text": "now"}]}],
        tool_specs=[{"name": "read", "description": "Read", "inputSchema": {"json": {"type": "object"}}}],
        system_prompt_content=[{"text": "stable system"}],
    )

    cache_points: list[dict[str, Any]] = []
    for item in request["toolConfig"]["tools"]:
        if "cachePoint" in item:
            cache_points.append(item["cachePoint"])
    for item in request["system"]:
        if "cachePoint" in item:
            cache_points.append(item["cachePoint"])
    for message in request["messages"]:
        for item in message["content"]:
            if "cachePoint" in item:
                cache_points.append(item["cachePoint"])

    assert cache_points
    assert {point.get("ttl") for point in cache_points} == {"1h"}
