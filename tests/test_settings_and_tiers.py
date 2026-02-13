from __future__ import annotations

from pathlib import Path

from swarmee_river.session.models import SessionModelManager
from swarmee_river.settings import load_settings


def test_load_settings_uses_builtins_when_file_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_MODEL_PROVIDER", raising=False)
    settings_path = tmp_path / "settings.json"

    settings = load_settings(settings_path)

    assert settings.models.provider is None
    assert "openai" in settings.models.providers
    assert settings.models.providers["openai"].tiers["fast"].model_id == "gpt-5-nano"
    assert settings.models.providers["openai"].tiers["fast"].display_name


def test_load_settings_env_overrides_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_MODEL_PROVIDER", "openai")
    settings_path = tmp_path / "settings.json"

    settings = load_settings(settings_path)

    assert settings.models.provider == "openai"


def test_tier_model_id_env_overrides_win_over_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_MODEL_PROVIDER", "openai")
    monkeypatch.delenv("SWARMEE_OPENAI_MODEL_ID", raising=False)
    monkeypatch.setenv("SWARMEE_OPENAI_FAST_MODEL_ID", "gpt-4o-mini")
    settings = load_settings(tmp_path / "settings.json")

    manager = SessionModelManager(settings, fallback_provider="openai")
    tiers = {t.name: t for t in manager.list_tiers()}

    assert tiers["fast"].provider == "openai"
    assert tiers["fast"].model_id == "gpt-4o-mini"


def test_provider_level_env_model_id_overrides_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("SWARMEE_OPENAI_MODEL_ID", "gpt-4o-mini")
    monkeypatch.delenv("SWARMEE_OPENAI_FAST_MODEL_ID", raising=False)
    settings = load_settings(tmp_path / "settings.json")

    manager = SessionModelManager(settings, fallback_provider="openai")
    tiers = {t.name: t for t in manager.list_tiers()}

    assert tiers["fast"].model_id == "gpt-4o-mini"
    assert tiers["balanced"].model_id == "gpt-4o-mini"


def test_tier_env_model_id_beats_provider_env_model_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("SWARMEE_OPENAI_MODEL_ID", "gpt-4o-mini")
    monkeypatch.setenv("SWARMEE_OPENAI_FAST_MODEL_ID", "gpt-5-nano")
    settings = load_settings(tmp_path / "settings.json")

    manager = SessionModelManager(settings, fallback_provider="openai")
    tiers = {t.name: t for t in manager.list_tiers()}

    assert tiers["fast"].model_id == "gpt-5-nano"
    assert tiers["balanced"].model_id == "gpt-4o-mini"

