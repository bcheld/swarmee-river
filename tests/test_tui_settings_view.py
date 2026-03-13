from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from swarmee_river.settings import default_settings_template
from swarmee_river.tui.mixins.settings import SettingsMixin
from swarmee_river.tui.views import settings as settings_view


def test_env_var_specs_include_supported_secrets() -> None:
    specs = settings_view.env_var_specs()
    keys = {spec.key for spec in specs}
    assert "OPENAI_API_KEY" in keys
    assert "SWARMEE_GITHUB_COPILOT_API_KEY" in keys
    assert "GITHUB_TOKEN" in keys
    assert "GH_TOKEN" in keys

    openai = settings_view.env_spec_by_key("OPENAI_API_KEY")
    assert openai is not None
    assert openai.category == "Supported Secrets (Read By Swarmee)"


def test_build_env_sidebar_items_includes_default_and_current(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890abcdef")
    items = settings_view.build_env_sidebar_items(category="Supported Secrets (Read By Swarmee)")
    target = next((item for item in items if item["id"] == "OPENAI_API_KEY"), None)
    assert target is not None
    # Masked value in UI.
    assert "current: sk-1...cdef" in target["subtitle"]
    assert "default:" in target["subtitle"]


def test_env_category_options_include_expected_sections() -> None:
    categories = {value for _label, value in settings_view.env_category_options()}
    assert "Supported Secrets (Read By Swarmee)" in categories
    assert "External Provider Credentials (SDK-Managed)" in categories


class _Widget:
    def __init__(self, value: str = "") -> None:
        self.value = value
        self.label = value
        self.variant = "default"
        self.text = ""
        self.options: list[tuple[str, str]] = []
        self.styles = SimpleNamespace(display="block")

    def update(self, text: str) -> None:
        self.text = text

    def set_options(self, options: list[tuple[str, str]]) -> None:
        self.options = list(options)


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
            "#settings_general_context_budget_mode": _Widget("auto"),
            "#settings_general_context_budget_input": _Widget(""),
            "#settings_models_provider_select": _Widget("__auto__"),
            "#settings_models_default_tier_select": _Widget("balanced"),
            "#settings_notebook_models_provider_select": _Widget("__auto__"),
            "#settings_notebook_models_default_tier_select": _Widget("fast"),
        }
        self._settings_bedrock_read_timeout_input = self._widgets["#settings_bedrock_read_timeout_input"]
        self._settings_bedrock_connect_timeout_input = self._widgets["#settings_bedrock_connect_timeout_input"]
        self._settings_bedrock_max_retries_input = self._widgets["#settings_bedrock_max_retries_input"]
        self._settings_interrupt_timeout_input = self._widgets["#settings_interrupt_timeout_input"]
        self._settings_general_context_budget_mode_select = self._widgets["#settings_general_context_budget_mode"]
        self._settings_general_context_budget_input = self._widgets["#settings_general_context_budget_input"]
        self._settings_interrupt_force_restart_select = None
        self._settings_diag_level_select = None
        self._settings_diag_redact_toggle = None
        self._settings_diag_retention_input = None
        self._settings_diag_max_bytes_input = None
        self._settings_diag_status = None
        self.state = SimpleNamespace(
            daemon=SimpleNamespace(
                model_provider_override=None,
                model_tier_override=None,
                provider=None,
                proc=None,
                ready=False,
            )
        )
        self.state.settings_model_select_syncing = False

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

    def _add_artifact_paths(self, _paths: list[str]) -> None:
        return None


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


def test_model_env_overrides_merge_project_env_and_model_overrides() -> None:
    harness = _SettingsHarness(
        {
            "models": {},
            "env": {
                "PYTHONWARNINGS": "default",
                "SWARMEE_MODEL_PROVIDER": "bedrock",
            },
        }
    )
    harness.state.daemon.model_provider_override = "openai"
    harness.state.daemon.model_tier_override = "deep"

    overrides = harness._model_env_overrides()

    # Secrets-only env policy: project settings `env.*` keeps internal wiring keys only.
    assert overrides == {"PYTHONWARNINGS": "default"}


def test_model_env_overrides_inherit_aws_region_from_process_env(monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    harness = _SettingsHarness({"models": {}, "env": {}})

    overrides = harness._model_env_overrides()

    # Secrets-only env policy: region settings should not be injected into daemon env overrides.
    assert "AWS_REGION" not in overrides
    assert "AWS_DEFAULT_REGION" not in overrides


def test_model_env_overrides_project_aws_region_wins_over_process_env(monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    harness = _SettingsHarness({"models": {}, "env": {"AWS_REGION": "eu-west-1"}})

    overrides = harness._model_env_overrides()

    assert "AWS_REGION" not in overrides


def test_apply_bedrock_runtime_settings_persists_values() -> None:
    harness = _SettingsHarness({"models": {}, "env": {}}, selected_id=None)
    harness._settings_bedrock_read_timeout_input.value = "75"
    harness._settings_bedrock_connect_timeout_input.value = "12"
    harness._settings_bedrock_max_retries_input.value = "2"

    harness._apply_bedrock_runtime_settings()

    assert harness.saved_payload is not None
    bedrock = harness.saved_payload["models"]["providers"]["bedrock"]
    assert bedrock["read_timeout_sec"] == 75.0
    assert bedrock["connect_timeout_sec"] == 12.0
    assert bedrock["max_retries"] == 2
    assert "env" not in harness.saved_payload or (
        "SWARMEE_BEDROCK_READ_TIMEOUT_SEC" not in harness.saved_payload.get("env", {})
    )
    assert harness._refresh_settings_models_calls == 1


def test_apply_interrupt_control_settings_persists_values() -> None:
    harness = _SettingsHarness({"models": {}, "env": {}}, selected_id=None)
    harness._settings_interrupt_timeout_input.value = "1.5"

    harness._apply_interrupt_control_settings()

    assert harness.saved_payload is not None
    assert harness.saved_payload["runtime"]["interrupt_timeout_sec"] == 1.5
    assert "env" not in harness.saved_payload or (
        "SWARMEE_INTERRUPT_TIMEOUT_SEC" not in harness.saved_payload.get("env", {})
    )
    assert harness._refresh_settings_general_calls == 1


def test_apply_bedrock_runtime_settings_rejects_invalid_values() -> None:
    harness = _SettingsHarness({"models": {}, "env": {}}, selected_id=None)
    harness._settings_bedrock_read_timeout_input.value = "bad"
    harness._settings_bedrock_connect_timeout_input.value = "10"
    harness._settings_bedrock_max_retries_input.value = "1"

    harness._apply_bedrock_runtime_settings()

    assert harness.saved_payload is None
    assert any("invalid Bedrock read timeout" in msg for msg in harness.messages)


def test_apply_settings_model_manager_result_persists_models_and_env() -> None:
    harness = _SettingsHarness({"models": {"providers": {}}, "env": {}}, selected_id=None)

    harness._apply_settings_model_manager_result(
        {
            "models": {
                "provider": "openai",
                "default_tier": "coding",
                "default_selection": {"provider": "openai", "tier": "coding"},
                "providers": {
                    "openai": {
                        "tiers": {
                            "coding": {
                                "provider": "openai",
                                "model_id": "gpt-5.3-codex",
                            }
                        }
                    }
                },
            },
            "env": {
                "SWARMEE_BEDROCK_READ_TIMEOUT_SEC": "60",
            },
        }
    )

    assert harness.saved_payload is not None
    assert harness.saved_payload["models"]["default_selection"]["tier"] == "coding"
    assert harness.saved_payload["models"]["providers"]["openai"]["tiers"]["coding"]["model_id"] == "gpt-5.3-codex"
    bedrock = harness.saved_payload["models"]["providers"]["bedrock"]
    assert bedrock["read_timeout_sec"] == 60.0
    assert "env" not in harness.saved_payload or (
        "SWARMEE_BEDROCK_READ_TIMEOUT_SEC" not in harness.saved_payload.get("env", {})
    )


def test_apply_context_budget_setting_persists_custom_value() -> None:
    harness = _SettingsHarness({"context": {}, "models": {}, "env": {}}, selected_id=None)

    harness._apply_context_budget_setting("250000")

    assert harness.saved_payload is not None
    assert harness.saved_payload["context"]["max_prompt_tokens"] == 250000
    assert any("250,000" in message for message in harness.messages)


def test_apply_context_budget_setting_can_reset_to_auto() -> None:
    harness = _SettingsHarness({"context": {"max_prompt_tokens": 40000}, "models": {}, "env": {}}, selected_id=None)

    harness._apply_context_budget_setting("")

    assert harness.saved_payload is not None
    assert "max_prompt_tokens" not in harness.saved_payload.get("context", {})
    assert any("reset to auto" in message for message in harness.messages)


def test_settings_model_tier_options_for_specific_provider_exclude_other_provider_tiers() -> None:
    settings = default_settings_template()

    values = [
        value
        for _label, value in SettingsMixin._settings_model_tier_options(
            settings,
            provider_value="bedrock",
        )
    ]

    assert "fast" in values
    assert "balanced" in values
    assert "deep" in values
    assert "long" in values
    assert "coding" not in values


def test_settings_model_tier_options_for_auto_include_union_of_provider_tiers() -> None:
    settings = default_settings_template()

    values = [
        value
        for _label, value in SettingsMixin._settings_model_tier_options(
            settings,
            provider_value="__auto__",
        )
    ]

    assert "coding" in values
    assert "fast" in values


def test_save_notebook_models_default_selection_persists_values() -> None:
    harness = _SettingsHarness({"models": {}, "notebook": {}, "env": {}}, selected_id=None)
    harness.query_one("#settings_notebook_models_provider_select").value = "openai"
    harness.query_one("#settings_notebook_models_default_tier_select").value = "fast"

    harness._save_notebook_models_default_selection()

    assert harness.saved_payload is not None
    assert harness.saved_payload["notebook"]["default_selection"] == {
        "provider": "openai",
        "tier": "fast",
    }


def test_save_notebook_models_default_selection_persists_auto_provider() -> None:
    harness = _SettingsHarness({"models": {}, "notebook": {}, "env": {}}, selected_id=None)
    harness.query_one("#settings_notebook_models_provider_select").value = "__auto__"
    harness.query_one("#settings_notebook_models_default_tier_select").value = "fast"

    harness._save_notebook_models_default_selection()

    assert harness.saved_payload is not None
    assert harness.saved_payload["notebook"]["default_selection"] == {
        "provider": None,
        "tier": "fast",
    }
