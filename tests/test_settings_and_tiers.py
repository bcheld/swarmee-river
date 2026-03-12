from __future__ import annotations

import json
import os
from pathlib import Path

from swarmee_river.session.models import SessionModelManager
from swarmee_river.settings import apply_project_env_overrides, load_project_env_overrides, load_settings
from swarmee_river.utils.model_utils import OpenAIResponsesTransportStatus


def _tiers_by_provider_and_name(manager: SessionModelManager) -> dict[tuple[str, str], object]:
    return {(t.provider, t.name): t for t in manager.list_tiers()}


def test_load_settings_uses_builtins_when_file_missing(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"

    settings = load_settings(settings_path)

    assert settings.models.provider is None
    assert "openai" in settings.models.providers
    assert "github_copilot" in settings.models.providers
    assert settings.models.providers["openai"].tiers["fast"].model_id == "gpt-5-nano"
    assert settings.models.providers["openai"].tiers["fast"].display_name
    assert settings.models.default_selection.tier == settings.models.default_tier
    assert settings.notebook.default_selection.provider is None
    assert settings.notebook.default_selection.tier == "fast"
    assert settings.tui.shortcuts.toggle_transcript_mode == ["f8"]
    assert settings.tui.shortcuts.copy_selection == ["ctrl+shift+c", "ctrl+c", "meta+c", "super+c"]


def test_load_settings_parses_tui_shortcuts(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "tui": {
                    "shortcuts": {
                        "toggle_transcript_mode": ["f9", "f9"],
                        "copy_selection": ["ctrl+shift+c", "meta+c", ""],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings.tui.shortcuts.toggle_transcript_mode == ["f9"]
    assert settings.tui.shortcuts.copy_selection == ["ctrl+shift+c", "meta+c"]


def test_load_project_env_overrides_parses_env_section(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "env": {
                    "AWS_PROFILE": " ds-pr ",
                    "PYTHONWARNINGS": " default::DeprecationWarning ",
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
    # Project env overrides are migration-only and restricted to internal wiring keys.
    assert overrides == {"PYTHONWARNINGS": "default::DeprecationWarning"}


def test_apply_project_env_overrides_honors_overwrite_flag(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"env": {"PYTHONWARNINGS": "default::DeprecationWarning"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONWARNINGS", "error")

    applied_no_overwrite = apply_project_env_overrides(settings_path, overwrite=False)
    assert applied_no_overwrite == {}
    assert (os.getenv("PYTHONWARNINGS") or "") == "error"

    applied_overwrite = apply_project_env_overrides(settings_path, overwrite=True)
    assert applied_overwrite == {"PYTHONWARNINGS": "default::DeprecationWarning"}
    assert (os.getenv("PYTHONWARNINGS") or "") == "default::DeprecationWarning"


def test_load_settings_env_overrides_provider(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"models": {"provider": "openai"}}), encoding="utf-8")

    settings = load_settings(settings_path)

    assert settings.models.provider == "openai"
    assert settings.models.default_selection.provider == "openai"


def test_load_settings_prefers_default_selection_over_legacy_keys(tmp_path: Path) -> None:
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


def test_load_settings_parses_notebook_default_selection(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "notebook": {
                    "default_selection": {
                        "provider": "openai",
                        "tier": "fast",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings.notebook.default_selection.provider == "openai"
    assert settings.notebook.default_selection.tier == "fast"


def test_auto_escalation_uses_explicit_order_only(tmp_path: Path) -> None:
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


def test_tier_model_id_settings_overrides_apply(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"models": {"providers": {"openai": {"tiers": {"fast": {"model_id": "gpt-4o-mini"}}}}}}),
        encoding="utf-8",
    )
    settings = load_settings(settings_path)

    manager = SessionModelManager(settings, fallback_provider="openai")
    tiers = _tiers_by_provider_and_name(manager)

    assert tiers[("openai", "fast")].provider == "openai"
    assert tiers[("openai", "fast")].model_id == "gpt-4o-mini"


def test_tier_model_id_settings_overrides_can_apply_to_multiple_tiers(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "models": {
                    "providers": {
                        "openai": {
                            "tiers": {
                                "fast": {"model_id": "gpt-4o-mini"},
                                "balanced": {"model_id": "gpt-4o-mini"},
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(settings_path)

    manager = SessionModelManager(settings, fallback_provider="openai")
    tiers = _tiers_by_provider_and_name(manager)

    assert tiers[("openai", "fast")].model_id == "gpt-4o-mini"
    assert tiers[("openai", "balanced")].model_id == "gpt-4o-mini"


def test_tier_model_id_override_is_per_tier(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "models": {
                    "providers": {
                        "openai": {
                            "tiers": {
                                "fast": {"model_id": "gpt-5-nano"},
                                "balanced": {"model_id": "gpt-4o-mini"},
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(settings_path)

    manager = SessionModelManager(settings, fallback_provider="openai")
    tiers = _tiers_by_provider_and_name(manager)

    assert tiers[("openai", "fast")].model_id == "gpt-5-nano"
    assert tiers[("openai", "balanced")].model_id == "gpt-4o-mini"


def test_fallback_provider_overrides_settings_provider_for_session(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"models": {"provider": "bedrock"}}), encoding="utf-8")
    settings = load_settings(settings_path)

    manager = SessionModelManager(settings, fallback_provider="openai")
    tiers = _tiers_by_provider_and_name(manager)

    assert tiers[("openai", "balanced")].provider == "openai"


def test_openai_tiers_report_unavailable_when_responses_transport_missing(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"models": {"provider": "openai"}}), encoding="utf-8")
    settings = load_settings(settings_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "swarmee_river.session.models.model_utils.probe_openai_responses_transport",
        lambda: OpenAIResponsesTransportStatus(
            available=False,
            strands_version="1.26.0",
            openai_version="2.0.0",
            reason="Installed strands-agents==1.26.0 is below Swarmee's supported runtime version.",
        ),
    )

    manager = SessionModelManager(settings, fallback_provider="openai")
    tiers = _tiers_by_provider_and_name(manager)

    assert tiers[("openai", "balanced")].available is False
    assert "strands-agents==1.26.0" in str(tiers[("openai", "balanced")].reason)


def test_tier_model_id_settings_overrides_apply_for_github_copilot(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"models": {"providers": {"github_copilot": {"tiers": {"fast": {"model_id": "gpt-4.1"}}}}}}),
        encoding="utf-8",
    )
    settings = load_settings(settings_path)

    manager = SessionModelManager(settings, fallback_provider="github_copilot")
    tiers = _tiers_by_provider_and_name(manager)

    assert tiers[("github_copilot", "fast")].model_id == "gpt-4.1"
    assert tiers[("github_copilot", "balanced")].model_id is not None


def test_github_copilot_tier_overrides_are_per_tier(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "models": {
                    "providers": {"github_copilot": {"tiers": {"fast": {"model_id": "gpt-4o-mini"}}}},
                }
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(settings_path)

    manager = SessionModelManager(settings, fallback_provider="github_copilot")
    tiers = _tiers_by_provider_and_name(manager)

    assert tiers[("github_copilot", "fast")].model_id == "gpt-4o-mini"
    assert tiers[("github_copilot", "balanced")].model_id != "gpt-4o-mini"


def test_default_safety_rules_include_canonical_tool_entries(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "settings.json")

    rules = {rule.tool: rule.default for rule in settings.safety.tool_rules}

    assert rules["todoread"] == "allow"
    assert rules["todowrite"] == "ask"
    assert rules["shell"] == "ask"
    assert rules["editor"] == "ask"
    assert rules["patch_apply"] == "ask"
    assert rules["file_search"] == "allow"
    assert rules["file_read"] == "allow"


def test_load_settings_filters_hidden_tiers(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"models": {"hidden_tiers": ["openai|balanced"]}}),
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    openai_tiers = settings.models.providers["openai"].tiers
    assert "balanced" not in openai_tiers
    assert "fast" in openai_tiers


def test_load_settings_restores_hidden_tier_when_key_removed(tmp_path: Path) -> None:
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


def test_load_settings_hidden_tier_overrides_are_ignored(tmp_path: Path) -> None:
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


def test_default_context_budget_resolves_from_shipped_tier_limits(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "settings.json")

    openai = SessionModelManager(settings, fallback_provider="openai")
    bedrock = SessionModelManager(settings, fallback_provider="bedrock")

    assert openai.resolve_effective_context_budget(tier_name="fast") == 400000
    assert openai.resolve_effective_context_budget(tier_name="deep") == 200000
    assert bedrock.resolve_effective_context_budget(tier_name="balanced") == 200000


def test_custom_context_budget_is_clamped_to_provider_cap(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"context": {"max_prompt_tokens": 999999}, "models": {"provider": "bedrock"}}),
        encoding="utf-8",
    )
    settings = load_settings(settings_path)
    manager = SessionModelManager(settings, fallback_provider="bedrock")

    assert manager.resolve_effective_context_budget(tier_name="deep") == 200000
