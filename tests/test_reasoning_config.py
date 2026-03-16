from __future__ import annotations

from pathlib import Path

import pytest

from swarmee_river.session.models import SessionModelManager
from swarmee_river.settings import load_settings
from swarmee_river.utils import model_utils


def _tier_status_by_provider_and_name(
    manager: SessionModelManager,
    *,
    provider: str,
    name: str,
):
    for item in manager.list_tiers():
        if item.provider == provider and item.name == name:
            return item
    raise AssertionError(f"Missing tier status for {provider}/{name}")


def test_bedrock_deep_tier_sets_adaptive_reasoning_for_opus_46(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(tmp_path / "settings.json")
    manager = SessionModelManager(settings, fallback_provider="bedrock")

    captured: dict[str, object] = {}

    monkeypatch.setattr(model_utils, "load_path", lambda _provider: Path("dummy.py"))

    def fake_load_model(_path: Path, config: dict) -> object:
        captured["config"] = config
        return object()

    monkeypatch.setattr(model_utils, "load_model", fake_load_model)

    manager.build_model("deep")

    config = captured.get("config")
    assert isinstance(config, dict)
    assert config["additional_request_fields"]["thinking"] == {"type": "adaptive"}
    assert config["additional_request_fields"]["output_config"] == {"effort": "high"}
    assert config["cache_tools"] == "default"


def test_openai_guided_reasoning_applies_to_deep_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        '{"models":{"providers":{"openai":{"tiers":{"deep":{"reasoning":{"effort":"high"}}}}}}}\n',
        encoding="utf-8",
    )
    settings = load_settings(settings_path)
    manager = SessionModelManager(settings, fallback_provider="openai")

    captured: dict[str, object] = {}

    monkeypatch.setattr(model_utils, "load_path", lambda _provider: Path("dummy.py"))

    def fake_load_model(_path: Path, config: dict) -> object:
        captured["config"] = config
        return object()

    monkeypatch.setattr(model_utils, "load_model", fake_load_model)

    manager.build_model("deep")

    config = captured.get("config")
    assert isinstance(config, dict)
    assert config["transport"] == "responses"
    assert config["params"]["reasoning"] == {"effort": "high"}


def test_openai_balanced_gpt5_mini_uses_responses_without_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(tmp_path / "settings.json")
    manager = SessionModelManager(settings, fallback_provider="openai")

    captured: dict[str, object] = {}

    monkeypatch.setattr(model_utils, "load_path", lambda _provider: Path("dummy.py"))

    def fake_load_model(_path: Path, config: dict) -> object:
        captured["config"] = config
        return object()

    monkeypatch.setattr(model_utils, "load_model", fake_load_model)

    manager.build_model("balanced")

    config = captured.get("config")
    assert isinstance(config, dict)
    assert config["model_id"] == "gpt-5-mini"
    assert config["transport"] == "responses"
    assert "reasoning" not in config["params"]
    tier = _tier_status_by_provider_and_name(manager, provider="openai", name="balanced")
    assert tier.reasoning_effort is None
    assert tier.reasoning_mode == "none"


def test_bedrock_deep_tier_strips_thinking_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"models":{"providers":{"bedrock":{"thinking_type":"disable"}}}}\n', encoding="utf-8")
    settings = load_settings(settings_path)
    manager = SessionModelManager(settings, fallback_provider="bedrock")

    captured: dict[str, object] = {}

    monkeypatch.setattr(model_utils, "load_path", lambda _provider: Path("dummy.py"))

    def fake_load_model(_path: Path, config: dict) -> object:
        captured["config"] = config
        return object()

    monkeypatch.setattr(model_utils, "load_model", fake_load_model)

    manager.build_model("deep")

    config = captured.get("config")
    assert isinstance(config, dict)
    additional = config.get("additional_request_fields")
    if isinstance(additional, dict):
        assert "thinking" not in additional
    else:
        assert additional is None


def test_bedrock_balanced_tier_uses_extended_thinking_and_capabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(tmp_path / "settings.json")
    manager = SessionModelManager(settings, fallback_provider="bedrock")

    captured: dict[str, object] = {}

    monkeypatch.setattr(model_utils, "load_path", lambda _provider: Path("dummy.py"))

    def fake_load_model(_path: Path, config: dict) -> object:
        captured["config"] = config
        return object()

    monkeypatch.setattr(model_utils, "load_model", fake_load_model)

    tier = _tier_status_by_provider_and_name(manager, provider="bedrock", name="balanced")
    manager.build_model("balanced")

    config = captured.get("config")
    assert isinstance(config, dict)
    assert config["additional_request_fields"]["thinking"]["type"] == "enabled"
    assert config["additional_request_fields"]["thinking"]["budget_tokens"] == 8192
    assert tier.reasoning_mode == "extended"
    assert tier.supports_cache_tools is True
    assert tier.supports_forced_tool_with_reasoning is False
