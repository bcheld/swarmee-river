from __future__ import annotations

import logging
from typing import Any

from swarmee_river.models import bedrock as bedrock_model


class _FakeBedrockModel:
    BedrockConfig = dict

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_bedrock_instance_warns_for_unprefixed_model_id(monkeypatch, caplog) -> None:
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeBedrockModel)
    caplog.set_level(logging.WARNING, logger="swarmee_river.models.bedrock")

    model = bedrock_model.instance(model_id="anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")

    assert isinstance(model, _FakeBedrockModel)
    assert any("unprefixed" in record.getMessage() for record in caplog.records)


def test_bedrock_instance_warns_for_prefix_region_mismatch(monkeypatch, caplog) -> None:
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeBedrockModel)
    caplog.set_level(logging.WARNING, logger="swarmee_river.models.bedrock")

    model = bedrock_model.instance(model_id="eu.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")

    assert isinstance(model, _FakeBedrockModel)
    assert any("may not match resolved region" in record.getMessage() for record in caplog.records)


def test_bedrock_instance_no_warning_for_matching_prefixed_model(monkeypatch, caplog) -> None:
    monkeypatch.setattr(bedrock_model, "BedrockModel", _FakeBedrockModel)
    caplog.set_level(logging.WARNING, logger="swarmee_river.models.bedrock")

    model = bedrock_model.instance(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")

    assert isinstance(model, _FakeBedrockModel)
    assert not caplog.records
