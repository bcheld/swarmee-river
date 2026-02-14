from __future__ import annotations

from pathlib import Path

import pytest

from swarmee_river.session.models import SessionModelManager
from swarmee_river.settings import load_settings
from swarmee_river.utils import model_utils


def test_bedrock_deep_tier_sets_higher_thinking_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    thinking = config["additional_request_fields"]["thinking"]
    assert thinking["type"] == "enabled"
    assert thinking["budget_tokens"] == 8192


def test_openai_reasoning_effort_env_applies_to_deep_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWARMEE_OPENAI_REASONING_EFFORT", "high")
    settings = load_settings(tmp_path / "settings.json")
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
    assert config["params"]["reasoning_effort"] == "high"
