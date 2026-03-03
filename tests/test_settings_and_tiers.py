from __future__ import annotations

import json
import os
from pathlib import Path

from swarmee_river.session.models import SessionModelManager
from swarmee_river.settings import apply_project_env_overrides, load_project_env_overrides, load_settings


def test_load_settings_uses_builtins_when_file_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_MODEL_PROVIDER", raising=False)
    settings_path = tmp_path / "settings.json"

    settings = load_settings(settings_path)

    assert settings.models.provider is None
    assert "openai" in settings.models.providers
    assert "github_copilot" in settings.models.providers
    assert settings.models.providers["openai"].tiers["fast"].model_id == "gpt-5-nano"
    assert settings.models.providers["openai"].tiers["fast"].display_name
    assert settings.models.default_selection.tier == settings.models.default_tier


def test_load_project_env_overrides_parses_env_section(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "env": {
                    "AWS_PROFILE": " ds-pr ",
                    "EMPTY_VALUE": "   ",
                    "NULL_VALUE": None,
                    "  ": "ignored",
                    "SWARMEE_MODEL_PROVIDER": "bedrock",
                }
            }
        ),
        encoding="utf-8",
    )

    overrides = load_project_env_overrides(settings_path)
    assert overrides == {
        "AWS_PROFILE": "ds-pr",
        "SWARMEE_MODEL_PROVIDER": "bedrock",
    }


def test_apply_project_env_overrides_honors_overwrite_flag(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"env": {"AWS_PROFILE": "profile-from-settings"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_PROFILE", "existing-profile")

    applied_no_overwrite = apply_project_env_overrides(settings_path, overwrite=False)
    assert applied_no_overwrite == {}
    assert (os.getenv("AWS_PROFILE") or "") == "existing-profile"

    applied_overwrite = apply_project_env_overrides(settings_path, overwrite=True)
    assert applied_overwrite == {"AWS_PROFILE": "profile-from-settings"}
    assert (os.getenv("AWS_PROFILE") or "") == "profile-from-settings"


def test_load_settings_env_overrides_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_MODEL_PROVIDER", "openai")
    settings_path = tmp_path / "settings.json"

    settings = load_settings(settings_path)

    assert settings.models.provider == "openai"
    assert settings.models.default_selection.provider == "openai"


def test_load_settings_prefers_default_selection_over_legacy_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_MODEL_PROVIDER", raising=False)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "models": {
                    "provider": "bedrock",
                    "default_tier": "balanced",
                    "default_selection": {"provider": "openai", "tier": "coding"},
                }
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings.models.provider == "openai"
    assert settings.models.default_tier == "coding"
    assert settings.models.default_selection.provider == "openai"
    assert settings.models.default_selection.tier == "coding"


def test_auto_escalation_uses_explicit_order_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_MODEL_PROVIDER", raising=False)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "models": {
                    "provider": "openai",
                    "auto_escalation": {"enabled": True, "order": ["coding", "deep"]},
                }
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(settings_path)
    manager = SessionModelManager(settings, fallback_provider="openai")
    assert manager.auto_escalation_enabled is True
    assert settings.models.auto_escalation.order == ["coding", "deep"]


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


def test_fallback_provider_overrides_settings_provider_for_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_MODEL_PROVIDER", raising=False)

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"models": {"provider": "bedrock"}}), encoding="utf-8")
    settings = load_settings(settings_path)

    manager = SessionModelManager(settings, fallback_provider="openai")
    tiers = {t.name: t for t in manager.list_tiers()}

    assert tiers["balanced"].provider == "openai"


def test_provider_level_env_model_id_overrides_settings_for_github_copilot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_MODEL_PROVIDER", "github_copilot")
    monkeypatch.setenv("SWARMEE_GITHUB_COPILOT_MODEL_ID", "gpt-4.1")
    monkeypatch.delenv("SWARMEE_GITHUB_COPILOT_FAST_MODEL_ID", raising=False)
    settings = load_settings(tmp_path / "settings.json")

    manager = SessionModelManager(settings, fallback_provider="github_copilot")
    tiers = {t.name: t for t in manager.list_tiers()}

    assert tiers["fast"].model_id == "gpt-4.1"
    assert tiers["balanced"].model_id == "gpt-4.1"


def test_tier_env_model_id_beats_provider_env_for_github_copilot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_MODEL_PROVIDER", "github_copilot")
    monkeypatch.setenv("SWARMEE_GITHUB_COPILOT_MODEL_ID", "gpt-4.1")
    monkeypatch.setenv("SWARMEE_GITHUB_COPILOT_FAST_MODEL_ID", "gpt-4o-mini")
    settings = load_settings(tmp_path / "settings.json")

    manager = SessionModelManager(settings, fallback_provider="github_copilot")
    tiers = {t.name: t for t in manager.list_tiers()}

    assert tiers["fast"].model_id == "gpt-4o-mini"
    assert tiers["balanced"].model_id == "gpt-4.1"


def test_default_safety_rules_include_opencode_alias_entries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_MODEL_PROVIDER", raising=False)
    settings = load_settings(tmp_path / "settings.json")

    rules = {rule.tool: rule.default for rule in settings.safety.tool_rules}

    for alias_name in ["bash", "patch", "write", "edit"]:
        assert rules[alias_name] == "ask"
    for alias_name in ["grep", "read"]:
        assert rules[alias_name] == "allow"
    assert rules["todoread"] == "allow"
    assert rules["todowrite"] == "ask"
    assert rules["shell"] == "ask"
    assert rules["patch_apply"] == "ask"


def test_load_settings_filters_hidden_tiers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_MODEL_PROVIDER", raising=False)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"models": {"hidden_tiers": ["openai|balanced"]}}),
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    openai_tiers = settings.models.providers["openai"].tiers
    assert "balanced" not in openai_tiers
    assert "fast" in openai_tiers


def test_load_settings_restores_hidden_tier_when_key_removed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_MODEL_PROVIDER", raising=False)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"models": {"hidden_tiers": ["openai|balanced"]}}),
        encoding="utf-8",
    )
    hidden_settings = load_settings(settings_path)
    assert "balanced" not in hidden_settings.models.providers["openai"].tiers

    settings_path.write_text(
        json.dumps({"models": {"hidden_tiers": []}}),
        encoding="utf-8",
    )
    restored_settings = load_settings(settings_path)
    assert "balanced" in restored_settings.models.providers["openai"].tiers


def test_load_settings_hidden_tier_overrides_are_ignored(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_MODEL_PROVIDER", raising=False)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "models": {
                    "hidden_tiers": ["openai|balanced"],
                    "providers": {
                        "openai": {
                            "tiers": {
                                "balanced": {
                                    "provider": "openai",
                                    "model_id": "gpt-override-hidden",
                                }
                            }
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert "balanced" not in settings.models.providers["openai"].tiers
