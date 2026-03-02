from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from swarmee_river.tui.mixins.settings import SettingsMixin
from swarmee_river.tui.views import settings as settings_view


def test_env_var_specs_include_model_provider() -> None:
    specs = settings_view.env_var_specs()
    provider_spec = next((spec for spec in specs if spec.key == "SWARMEE_MODEL_PROVIDER"), None)
    assert provider_spec is not None
    assert provider_spec.category
    assert "openai" in provider_spec.choices
    assert "bedrock" in provider_spec.choices


def test_build_env_sidebar_items_includes_default_and_current(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_MODEL_PROVIDER", "openai")
    items = settings_view.build_env_sidebar_items()
    target = next((item for item in items if item["id"] == "SWARMEE_MODEL_PROVIDER"), None)
    assert target is not None
    assert "current: openai" in target["subtitle"]
    assert "default:" in target["subtitle"]


def test_env_var_specs_include_bedrock_and_interrupt_runtime_controls() -> None:
    specs = settings_view.env_var_specs()
    keys = {spec.key for spec in specs}
    assert "SWARMEE_BEDROCK_READ_TIMEOUT_SEC" in keys
    assert "SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC" in keys
    assert "SWARMEE_BEDROCK_MAX_RETRIES" in keys
    assert "SWARMEE_INTERRUPT_TIMEOUT_SEC" in keys
    assert "SWARMEE_INTERRUPT_FORCE_RESTART" in keys


class _Widget:
    def __init__(self, value: str = "") -> None:
        self.value = value


class _SettingsHarness(SettingsMixin):
    def __init__(self, payload: dict[str, Any], *, selected_id: str | None = None) -> None:
        self._payload = deepcopy(payload)
        self.saved_payload: dict[str, Any] | None = None
        self._settings_models_selected_id = selected_id
        self._settings_env_selected_key = None
        self._refresh_model_select_calls = 0
        self._refresh_settings_models_calls = 0
        self._refresh_settings_general_calls = 0
        self._refresh_agent_summary_calls = 0
        self.messages: list[str] = []
        self._widgets: dict[str, _Widget] = {
            "#settings_models_form_provider": _Widget("openai"),
            "#settings_models_form_tier": _Widget("balanced"),
            "#settings_models_form_model_id": _Widget("gpt-5-mini"),
            "#settings_models_form_display_name": _Widget(""),
            "#settings_models_form_description": _Widget(""),
            "#settings_models_form_price_input": _Widget(""),
            "#settings_models_form_price_output": _Widget(""),
            "#settings_models_form_price_cached": _Widget(""),
            "#settings_bedrock_read_timeout_input": _Widget("60"),
            "#settings_bedrock_connect_timeout_input": _Widget("10"),
            "#settings_bedrock_max_retries_input": _Widget("1"),
            "#settings_interrupt_timeout_input": _Widget("2.0"),
            "#settings_interrupt_force_restart_select": _Widget("true"),
        }
        self._settings_bedrock_read_timeout_input = self._widgets["#settings_bedrock_read_timeout_input"]
        self._settings_bedrock_connect_timeout_input = self._widgets["#settings_bedrock_connect_timeout_input"]
        self._settings_bedrock_max_retries_input = self._widgets["#settings_bedrock_max_retries_input"]
        self._settings_interrupt_timeout_input = self._widgets["#settings_interrupt_timeout_input"]
        self._settings_interrupt_force_restart_select = self._widgets["#settings_interrupt_force_restart_select"]

    def query_one(self, selector: str, _widget_type: Any = None) -> _Widget:
        return self._widgets[selector]

    def _load_project_settings_payload(self) -> tuple[dict[str, Any], Path]:
        return deepcopy(self._payload), Path("/tmp/settings.json")

    def _save_project_settings_payload(self, payload: dict[str, Any], path: Path) -> None:
        del path
        self._payload = deepcopy(payload)
        self.saved_payload = deepcopy(payload)

    def _refresh_model_select(self) -> None:
        self._refresh_model_select_calls += 1

    def _refresh_settings_models(self) -> None:
        self._refresh_settings_models_calls += 1

    def _refresh_settings_general(self) -> None:
        self._refresh_settings_general_calls += 1

    def _refresh_settings_env_list(self) -> None:
        return None

    def _refresh_settings_env_detail(self, _selected_key: str | None) -> None:
        return None

    def _refresh_agent_summary(self) -> None:
        self._refresh_agent_summary_calls += 1

    def _write_transcript_line(self, text: str) -> None:
        self.messages.append(text)


def test_delete_model_hides_row_and_removes_override() -> None:
    harness = _SettingsHarness(
        {
            "models": {
                "providers": {
                    "openai": {
                        "tiers": {
                            "balanced": {"provider": "openai", "model_id": "custom-model"},
                            "fast": {"provider": "openai", "model_id": "gpt-5-nano"},
                        }
                    }
                }
            }
        },
        selected_id="openai|balanced",
    )

    harness._delete_model_form_selection()

    assert harness.saved_payload is not None
    assert "openai|balanced" in harness.saved_payload["models"]["hidden_tiers"]
    tiers = harness.saved_payload["models"]["providers"]["openai"]["tiers"]
    assert "balanced" not in tiers
    assert harness._refresh_settings_models_calls == 1


def test_restore_defaults_unhides_row_and_clears_override() -> None:
    harness = _SettingsHarness(
        {
            "models": {
                "hidden_tiers": ["openai|balanced"],
                "providers": {
                    "openai": {
                        "tiers": {
                            "balanced": {"provider": "openai", "model_id": "custom-model"},
                        }
                    }
                },
            }
        },
        selected_id="openai|balanced",
    )

    harness._restore_model_form_selection()

    assert harness.saved_payload is not None
    assert "openai|balanced" not in harness.saved_payload["models"]["hidden_tiers"]
    openai_provider = harness.saved_payload["models"]["providers"].get("openai", {})
    openai_tiers = openai_provider.get("tiers", {})
    assert "balanced" not in openai_tiers
    assert harness._settings_models_selected_id == "openai|balanced"
    assert harness._refresh_settings_models_calls == 1


def test_save_model_unhides_hidden_tier_and_updates_row() -> None:
    harness = _SettingsHarness(
        {"models": {"hidden_tiers": ["openai|balanced"], "providers": {}}},
        selected_id="openai|balanced",
    )
    harness.query_one("#settings_models_form_model_id").value = "gpt-5-custom"
    harness.query_one("#settings_models_form_display_name").value = "Custom"
    harness.query_one("#settings_models_form_description").value = "Custom tier description"

    harness._save_model_form()

    assert harness.saved_payload is not None
    assert "openai|balanced" not in harness.saved_payload["models"]["hidden_tiers"]
    balanced = harness.saved_payload["models"]["providers"]["openai"]["tiers"]["balanced"]
    assert balanced["model_id"] == "gpt-5-custom"
    assert balanced["display_name"] == "Custom"
    assert balanced["description"] == "Custom tier description"


def test_apply_bedrock_runtime_settings_persists_values() -> None:
    harness = _SettingsHarness({"models": {}, "env": {}}, selected_id=None)
    harness._settings_bedrock_read_timeout_input.value = "75"
    harness._settings_bedrock_connect_timeout_input.value = "12"
    harness._settings_bedrock_max_retries_input.value = "2"

    harness._apply_bedrock_runtime_settings()

    assert harness.saved_payload is not None
    env_payload = harness.saved_payload.get("env", {})
    assert env_payload["SWARMEE_BEDROCK_READ_TIMEOUT_SEC"] == "75.0"
    assert env_payload["SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC"] == "12.0"
    assert env_payload["SWARMEE_BEDROCK_MAX_RETRIES"] == "2"
    assert harness._refresh_settings_models_calls == 1


def test_apply_interrupt_control_settings_persists_values() -> None:
    harness = _SettingsHarness({"models": {}, "env": {}}, selected_id=None)
    harness._settings_interrupt_timeout_input.value = "1.5"
    harness._settings_interrupt_force_restart_select.value = "false"

    harness._apply_interrupt_control_settings()

    assert harness.saved_payload is not None
    env_payload = harness.saved_payload.get("env", {})
    assert env_payload["SWARMEE_INTERRUPT_TIMEOUT_SEC"] == "1.5"
    assert env_payload["SWARMEE_INTERRUPT_FORCE_RESTART"] == "false"
    assert harness._refresh_settings_general_calls == 1


def test_apply_bedrock_runtime_settings_rejects_invalid_values() -> None:
    harness = _SettingsHarness({"models": {}, "env": {}}, selected_id=None)
    harness._settings_bedrock_read_timeout_input.value = "bad"
    harness._settings_bedrock_connect_timeout_input.value = "10"
    harness._settings_bedrock_max_retries_input.value = "1"

    harness._apply_bedrock_runtime_settings()

    assert harness.saved_payload is None
    assert any("invalid Bedrock read timeout" in msg for msg in harness.messages)
