from __future__ import annotations

import logging
from typing import Any

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
