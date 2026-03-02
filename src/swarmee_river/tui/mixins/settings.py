from __future__ import annotations

import contextlib
import json as _json
import os
import time
from pathlib import Path
from typing import Any

from swarmee_river.tui.commands import _MODEL_USAGE_TEXT
from swarmee_river.tui.model_select import (
    choose_model_summary_parts,
    daemon_model_select_options,
    model_select_options,
    parse_model_select_value,
    resolve_model_config_summary,
)
from swarmee_river.tui.views.settings import (
    build_models_table_rows,
    build_env_table_rows,
    env_category_options,
    env_spec_by_key,
)

_RUN_ACTIVE_TIER_WARNING = "[model] cannot change tier while a run is active."


def _should_ignore_stale_model_info_update(
    *,
    incoming_value: str | None,
    target_value: str | None,
    target_until_mono: float | None,
    now_mono: float,
) -> bool:
    incoming = str(incoming_value or "").strip().lower()
    target = str(target_value or "").strip().lower()
    if not incoming or not target:
        return False
    if incoming == target:
        return False
    if target_until_mono is None:
        return False
    return now_mono < float(target_until_mono)


class SettingsMixin:
    _BEDROCK_RUNTIME_DEFAULTS: dict[str, str] = {
        "SWARMEE_BEDROCK_READ_TIMEOUT_SEC": "60",
        "SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC": "10",
        "SWARMEE_BEDROCK_MAX_RETRIES": "1",
    }
    _INTERRUPT_CONTROL_DEFAULTS: dict[str, str] = {
        "SWARMEE_INTERRUPT_TIMEOUT_SEC": "2.0",
        "SWARMEE_INTERRUPT_FORCE_RESTART": "true",
    }

    def _set_settings_view_mode(self, mode: str) -> None:
        normalized = mode.strip().lower()
        if normalized not in {"general", "models", "advanced"}:
            normalized = "general"
        self.state.settings_view_mode = normalized
        if self._settings_general_view:
            self._settings_general_view.styles.display = "block" if normalized == "general" else "none"
        if self._settings_models_view:
            self._settings_models_view.styles.display = "block" if normalized == "models" else "none"
        if self._settings_advanced_view:
            self._settings_advanced_view.styles.display = "block" if normalized == "advanced" else "none"
        if self._settings_view_general_button:
            self._settings_view_general_button.variant = "primary" if normalized == "general" else "default"
        if self._settings_view_models_button:
            self._settings_view_models_button.variant = "primary" if normalized == "models" else "default"
        if self._settings_view_advanced_button:
            self._settings_view_advanced_button.variant = "primary" if normalized == "advanced" else "default"
        if normalized == "general":
            self._refresh_settings_general()
        if normalized == "models":
            self._refresh_settings_models()
        if normalized == "advanced":
            self._refresh_settings_env_list()
            self._refresh_settings_env_detail(self._settings_env_selected_key)
            self._set_agent_tools_override_form_values(self.state.agent_studio.session_safety_overrides)

    def _refresh_orchestrator_status(self) -> None:
        """Update the orchestrator status line in the Engage tab."""
        widget = self._engage_orchestrator_status
        if widget is None:
            return
        summary = self._current_model_summary()
        widget.update(f"Orchestrator: {summary}" if summary else "Orchestrator")

    def _refresh_settings_bedrock_runtime_controls(self) -> None:
        read_widget = self._settings_bedrock_read_timeout_input
        connect_widget = self._settings_bedrock_connect_timeout_input
        retries_widget = self._settings_bedrock_max_retries_input
        if read_widget is not None:
            with contextlib.suppress(Exception):
                read_widget.value = (
                    os.environ.get("SWARMEE_BEDROCK_READ_TIMEOUT_SEC")
                    or self._BEDROCK_RUNTIME_DEFAULTS["SWARMEE_BEDROCK_READ_TIMEOUT_SEC"]
                )
        if connect_widget is not None:
            with contextlib.suppress(Exception):
                connect_widget.value = (
                    os.environ.get("SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC")
                    or self._BEDROCK_RUNTIME_DEFAULTS["SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC"]
                )
        if retries_widget is not None:
            with contextlib.suppress(Exception):
                retries_widget.value = (
                    os.environ.get("SWARMEE_BEDROCK_MAX_RETRIES")
                    or self._BEDROCK_RUNTIME_DEFAULTS["SWARMEE_BEDROCK_MAX_RETRIES"]
                )

    def _refresh_settings_interrupt_control_controls(self) -> None:
        timeout_widget = self._settings_interrupt_timeout_input
        restart_widget = self._settings_interrupt_force_restart_select
        if timeout_widget is not None:
            with contextlib.suppress(Exception):
                timeout_widget.value = (
                    os.environ.get("SWARMEE_INTERRUPT_TIMEOUT_SEC")
                    or os.environ.get("SWARMEE_INTERRUPT_TIMEOUT")
                    or self._INTERRUPT_CONTROL_DEFAULTS["SWARMEE_INTERRUPT_TIMEOUT_SEC"]
                )
        if restart_widget is not None:
            raw = (
                os.environ.get("SWARMEE_INTERRUPT_FORCE_RESTART")
                or self._INTERRUPT_CONTROL_DEFAULTS["SWARMEE_INTERRUPT_FORCE_RESTART"]
            ).strip().lower()
            value = "false" if raw in {"false", "0", "no", "off", "disabled"} else "true"
            with contextlib.suppress(Exception):
                restart_widget.value = value

    def _parse_positive_float_setting(self, raw: str, *, label: str) -> float | None:
        token = str(raw or "").strip()
        try:
            value = float(token)
        except ValueError:
            self._write_transcript_line(f"[settings] invalid {label}: {token or '(blank)'}")
            return None
        if value <= 0:
            self._write_transcript_line(f"[settings] {label} must be > 0.")
            return None
        return value

    def _parse_non_negative_int_setting(self, raw: str, *, label: str) -> int | None:
        token = str(raw or "").strip()
        if not token.isdigit():
            self._write_transcript_line(f"[settings] invalid {label}: {token or '(blank)'}")
            return None
        value = int(token)
        if value < 0:
            self._write_transcript_line(f"[settings] {label} must be >= 0.")
            return None
        return value

    def _apply_bedrock_runtime_settings(self) -> None:
        read_raw = str(getattr(self._settings_bedrock_read_timeout_input, "value", "")).strip()
        connect_raw = str(getattr(self._settings_bedrock_connect_timeout_input, "value", "")).strip()
        retries_raw = str(getattr(self._settings_bedrock_max_retries_input, "value", "")).strip()
        read_timeout = self._parse_positive_float_setting(read_raw, label="Bedrock read timeout")
        if read_timeout is None:
            return
        connect_timeout = self._parse_positive_float_setting(connect_raw, label="Bedrock connect timeout")
        if connect_timeout is None:
            return
        max_retries = self._parse_non_negative_int_setting(retries_raw, label="Bedrock max retries")
        if max_retries is None:
            return
        self._persist_project_setting_env_override("SWARMEE_BEDROCK_READ_TIMEOUT_SEC", str(read_timeout))
        self._persist_project_setting_env_override("SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC", str(connect_timeout))
        self._persist_project_setting_env_override("SWARMEE_BEDROCK_MAX_RETRIES", str(max_retries))
        self._refresh_settings_models()
        self._refresh_settings_env_list()
        self._refresh_settings_env_detail(self._settings_env_selected_key)
        self._write_transcript_line(
            f"[settings] bedrock runtime updated (read={read_timeout}s connect={connect_timeout}s retries={max_retries})."
        )

    def _reset_bedrock_runtime_settings(self) -> None:
        for key, value in self._BEDROCK_RUNTIME_DEFAULTS.items():
            self._persist_project_setting_env_override(key, value)
        self._refresh_settings_models()
        self._refresh_settings_env_list()
        self._refresh_settings_env_detail(self._settings_env_selected_key)
        self._write_transcript_line("[settings] bedrock runtime reset to defaults.")

    def _apply_interrupt_control_settings(self) -> None:
        timeout_raw = str(getattr(self._settings_interrupt_timeout_input, "value", "")).strip()
        timeout_s = self._parse_positive_float_setting(timeout_raw, label="interrupt timeout")
        if timeout_s is None:
            return
        restart_raw = str(getattr(self._settings_interrupt_force_restart_select, "value", "true")).strip().lower()
        restart_value = "false" if restart_raw in {"false", "0", "no", "off", "disabled"} else "true"
        self._persist_project_setting_env_override("SWARMEE_INTERRUPT_TIMEOUT_SEC", str(timeout_s))
        self._persist_project_setting_env_override("SWARMEE_INTERRUPT_FORCE_RESTART", restart_value)
        self._refresh_settings_general()
        self._refresh_settings_env_list()
        self._refresh_settings_env_detail(self._settings_env_selected_key)
        self._write_transcript_line(
            f"[settings] interrupt control updated (timeout={timeout_s}s force_restart={restart_value})."
        )

    def _reset_interrupt_control_settings(self) -> None:
        for key, value in self._INTERRUPT_CONTROL_DEFAULTS.items():
            self._persist_project_setting_env_override(key, value)
        self._refresh_settings_general()
        self._refresh_settings_env_list()
        self._refresh_settings_env_detail(self._settings_env_selected_key)
        self._write_transcript_line("[settings] interrupt control reset to defaults.")

    def _refresh_plan_actions_visibility(self) -> None:
        """Show plan action buttons only when a plan is pending approval."""
        import contextlib as _ctx

        with _ctx.suppress(Exception):
            pa = self.query_one("#plan_actions")
            if self.state.plan.pending_prompt:
                pa.styles.display = "block"
            else:
                pa.styles.display = "none"

    def _refresh_settings_env_list(self) -> None:
        """Populate the Settings env list from env.example catalog."""
        table = self._settings_env_table
        if table is None:
            return
        category_widget = self._settings_env_category_select
        category = str(getattr(category_widget, "value", "")).strip() if category_widget is not None else ""
        if not category:
            options = env_category_options()
            category = options[0][1] if options else ""
            if category_widget is not None and category:
                with contextlib.suppress(Exception):
                    category_widget.set_options(options)
                    category_widget.value = category
        if not table.columns:
            table.add_column("Key", key="key")
            table.add_column("Current", key="current")
            table.add_column("Default", key="default")
            table.add_column("State", key="state", width=10)

        rows = build_env_table_rows(category=category or None)
        table.clear()
        for key, current, default, state in rows:
            table.add_row(key, current, default, state, key=key)

        selected_id = str(self._settings_env_selected_key or "").strip()
        if selected_id and rows:
            for idx, (row_key, _current, _default, _state) in enumerate(rows):
                if row_key == selected_id:
                    with contextlib.suppress(Exception):
                        table.move_cursor(row=idx)
                    break
        elif rows:
            with contextlib.suppress(Exception):
                table.move_cursor(row=0)

        cursor_coordinate = getattr(table, "cursor_coordinate", None)
        row_index = int(getattr(cursor_coordinate, "row", -1) or -1)
        if 0 <= row_index < len(rows):
            self._settings_env_selected_key = rows[row_index][0]
        else:
            self._settings_env_selected_key = None

    def _settings_auth_status_lines(self) -> list[str]:
        from swarmee_river.utils.provider_utils import has_aws_credentials, has_github_copilot_token

        active_provider = self.state.daemon.model_provider_override or self.state.daemon.provider or "(auto)"
        aws_profile = (os.getenv("AWS_PROFILE") or "").strip() or "default"
        aws_ok = has_aws_credentials()
        copilot_ok = has_github_copilot_token()
        return [
            f"Active provider: {active_provider}",
            f"AWS (profile={aws_profile}): {'connected' if aws_ok else 'not connected'}",
            f"GitHub Copilot: {'connected' if copilot_ok else 'not connected'}",
        ]

    def _set_settings_env_value_controls(self, *, key: str, current_value: str, default_value: str) -> None:
        spec = env_spec_by_key(key)
        has_choices = spec is not None and bool(spec.choices)
        select_widget = self._settings_env_value_select
        input_widget = self._settings_env_value_input
        # Show only the relevant control: select for constrained, input for free-form
        if select_widget is not None:
            options: list[tuple[str, str]] = [("Select constrained value...", "__none__")]
            if has_choices:
                options.extend((choice, choice) for choice in spec.choices)
            with contextlib.suppress(Exception):
                select_widget.set_options(options)
                selected_value = current_value or (
                    default_value if default_value and default_value != "(unset)" else ""
                )
                select_widget.value = (
                    selected_value if selected_value in {value for _label, value in options} else "__none__"
                )
            with contextlib.suppress(Exception):
                select_widget.styles.display = "block" if has_choices else "none"
        if input_widget is not None:
            candidate = current_value or (default_value if default_value != "(unset)" else "")
            with contextlib.suppress(Exception):
                input_widget.value = candidate
            with contextlib.suppress(Exception):
                input_widget.styles.display = "none" if has_choices else "block"

    def _refresh_settings_env_detail(self, selected_key: str | None) -> None:
        detail_widget = self._settings_env_detail
        if detail_widget is None:
            return
        key = str(selected_key or "").strip()
        spec = env_spec_by_key(key)
        if spec is None:
            with contextlib.suppress(Exception):
                detail_widget.update("Select a variable to view details and edit its value.")
            return

        current = os.environ.get(spec.key, "").strip()
        sensitive = spec.key in {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "SWARMEE_GITHUB_COPILOT_API_KEY",
            "GITHUB_TOKEN",
            "GH_TOKEN",
        }

        def _mask(raw: str) -> str:
            if not sensitive or len(raw) <= 8:
                return raw
            return raw[:4] + "..." + raw[-4:]

        current_display = _mask(current) if current else "(unset)"
        default_display = _mask(spec.default) if spec.default else "(unset)"
        choices_text = ", ".join(spec.choices) if spec.choices else "free-form"
        lines = [
            spec.key,
            f"  Current:  {current_display}",
            f"  Default:  {default_display}",
        ]
        if spec.choices:
            lines.append(f"  Allowed:  {choices_text}")
        lines.append(f"  {spec.description}")
        detail_text = "\n".join(lines)
        with contextlib.suppress(Exception):
            detail_widget.update(detail_text)
        self._set_settings_env_value_controls(key=spec.key, current_value=current, default_value=spec.default)

    def _refresh_settings_models(self) -> None:
        from textual.widgets import Select

        from swarmee_river.settings import load_settings

        table = self._settings_models_table
        settings = load_settings()
        provider_default = (settings.models.provider or "__auto__").strip().lower() or "__auto__"
        tier_default = (settings.models.default_tier or "balanced").strip().lower() or "balanced"

        self.state.daemon.model_select_syncing = True
        try:
            provider_select = self.query_one("#settings_models_provider_select", Select)
            tier_select = self.query_one("#settings_models_default_tier_select", Select)
            provider_options: list[tuple[str, str]] = [
                ("Provider: auto", "__auto__"),
                ("bedrock", "bedrock"),
                ("openai", "openai"),
                ("ollama", "ollama"),
                ("github_copilot", "github_copilot"),
            ]
            provider_select.set_options(provider_options)
            with contextlib.suppress(Exception):
                provider_select.value = provider_default if provider_default else "__auto__"
            selected_provider = (
                provider_default
                if provider_default != "__auto__"
                else str(getattr(provider_select, "value", "__auto__")).strip().lower()
            )
            tier_options: list[tuple[str, str]] = []
            if selected_provider and selected_provider != "__auto__":
                provider_cfg = settings.models.providers.get(selected_provider)
                tier_names = sorted(provider_cfg.tiers.keys()) if provider_cfg and provider_cfg.tiers else []
                tier_options = [(f"Tier: {tier_name}", tier_name) for tier_name in tier_names]
            if not tier_options:
                tier_options = [("Tier: (none configured)", "__none__")]
            tier_select.set_options(tier_options)
            with contextlib.suppress(Exception):
                tier_select.value = tier_default if tier_default in {value for _label, value in tier_options} else tier_options[0][1]
        finally:
            self.state.daemon.model_select_syncing = False

        if table is not None:
            rows = build_models_table_rows(settings)
            if not rows:
                rows = [("__none__", "No model tiers configured", "(unset)", "")]
            if not table.columns:
                table.add_column("Provider/Tier", key="provider_tier")
                table.add_column("Model ID", key="model_id")
                table.add_column("Pricing", key="pricing")
            table.clear()
            for row_id, provider_tier, model_id, pricing_label in rows:
                table.add_row(provider_tier, model_id, pricing_label, key=row_id)

            selected_id = str(self._settings_models_selected_id or "").strip()
            cursor_row = -1
            if selected_id:
                for idx, (row_id, _provider_tier, _model_id, _pricing_label) in enumerate(rows):
                    if row_id == selected_id:
                        cursor_row = idx
                        break
            if cursor_row < 0:
                cursor_row = 0 if rows else -1
            if cursor_row >= 0:
                with contextlib.suppress(Exception):
                    table.move_cursor(row=cursor_row)
                self._settings_models_selected_id = rows[cursor_row][0]
            else:
                self._settings_models_selected_id = None

        summary_widget = self._settings_models_summary
        if summary_widget is not None:
            provider_label = provider_default if provider_default != "__auto__" else "auto"
            provider_count = len(settings.models.providers)
            tier_count = sum(len(provider.tiers) for provider in settings.models.providers.values())
            with contextlib.suppress(Exception):
                summary_widget.update(
                    f"Default provider: {provider_label} | Default tier: {tier_default} | Providers: {provider_count} | Tiers: {tier_count}"
                )

        auth_widget = self._settings_auth_status
        if auth_widget is not None:
            with contextlib.suppress(Exception):
                auth_widget.update("\n".join(self._settings_auth_status_lines()))

        aws_input = self._settings_aws_profile_input
        if aws_input is not None:
            current_profile = (os.getenv("AWS_PROFILE") or "").strip()
            if current_profile and not str(getattr(aws_input, "value", "")).strip():
                with contextlib.suppress(Exception):
                    aws_input.value = current_profile

        self._refresh_settings_model_detail()

    def _refresh_settings_model_detail(self) -> None:
        from swarmee_river.pricing import resolve_pricing
        from swarmee_river.settings import load_settings

        detail_widget = self._settings_models_detail
        if detail_widget is None:
            return
        selected = str(self._settings_models_selected_id or "").strip().lower()
        if "|" not in selected:
            with contextlib.suppress(Exception):
                detail_widget.update("Select a model row to inspect, or use Manage Models to edit.")
            return
        provider_name, tier_name = selected.split("|", 1)
        settings = load_settings()
        provider = settings.models.providers.get(provider_name)
        tier = provider.tiers.get(tier_name) if provider is not None else None
        if tier is None:
            with contextlib.suppress(Exception):
                detail_widget.update("Selected model tier is no longer available.")
            return
        pricing = resolve_pricing(provider=provider_name, model_id=tier.model_id)
        pricing_label = "Pricing: unavailable"
        if pricing is not None and pricing.input_per_1m is not None and pricing.output_per_1m is not None:
            cached = pricing.cached_input_per_1m if pricing.cached_input_per_1m is not None else pricing.input_per_1m
            pricing_label = (
                "Pricing: "
                f"${pricing.input_per_1m}/1M input, ${pricing.output_per_1m}/1M output, ${cached}/1M cached input"
            )
        with contextlib.suppress(Exception):
            detail_widget.update(
                "\n".join(
                    [
                        f"Provider: {provider_name}",
                        f"Tier: {tier_name}",
                        f"Model ID: {tier.model_id or '(unset)'}",
                        f"Display: {tier.display_name or '(unset)'}",
                        f"Description: {tier.description or '(unset)'}",
                        pricing_label,
                    ]
                )
            )

    def _load_project_settings_payload(self) -> tuple[dict[str, Any], Path]:
        from swarmee_river.settings import deep_merge_dict, default_settings_template

        path = Path.cwd() / ".swarmee" / "settings.json"
        raw: dict[str, Any] = {}
        if path.exists() and path.is_file():
            with contextlib.suppress(OSError, ValueError):
                loaded = _json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
        defaults = default_settings_template().to_dict()
        merged = deep_merge_dict(defaults, raw) if raw else defaults
        return merged, path

    def _save_project_settings_payload(self, payload: dict[str, Any], path: Path) -> None:
        from swarmee_river.settings import SwarmeeSettings, save_settings

        parsed = SwarmeeSettings.from_dict(payload if isinstance(payload, dict) else {})
        save_settings(parsed, path=path)

    def _project_settings_env_overrides(self) -> dict[str, str]:
        payload, _path = self._load_project_settings_payload()
        env_payload = payload.get("env")
        if not isinstance(env_payload, dict):
            return {}
        resolved: dict[str, str] = {}
        for raw_key, raw_value in env_payload.items():
            key = str(raw_key).strip()
            if not key:
                continue
            if raw_value is None:
                continue
            resolved[key] = str(raw_value).strip()
        return resolved

    def _apply_project_settings_env_overrides(self) -> None:
        env_overrides = self._project_settings_env_overrides()
        for key, value in env_overrides.items():
            if value:
                os.environ[key] = value

    def _persist_project_setting_env_override(self, key: str, value: str | None) -> None:
        normalized_key = str(key or "").strip()
        if not normalized_key:
            return
        payload, path = self._load_project_settings_payload()
        env_payload = payload.get("env")
        if not isinstance(env_payload, dict):
            env_payload = {}
        normalized_value = str(value).strip() if value is not None else ""
        if normalized_value:
            env_payload[normalized_key] = normalized_value
            os.environ[normalized_key] = normalized_value
        else:
            env_payload.pop(normalized_key, None)
            os.environ.pop(normalized_key, None)
        if env_payload:
            payload["env"] = env_payload
        else:
            payload.pop("env", None)
        self._save_project_settings_payload(payload, path)

    def _model_tier_key(self, provider: str, tier: str) -> str:
        return f"{str(provider or '').strip().lower()}|{str(tier or '').strip().lower()}"

    def _normalized_hidden_tiers(self, models_payload: dict[str, Any]) -> list[str]:
        raw_hidden = models_payload.get("hidden_tiers")
        normalized: list[str] = []
        seen: set[str] = set()
        if isinstance(raw_hidden, list):
            for item in raw_hidden:
                token = str(item or "").strip().lower()
                if "|" not in token:
                    continue
                provider_name, tier_name = token.split("|", 1)
                key = self._model_tier_key(provider_name, tier_name)
                if "|" not in key or key in seen:
                    continue
                seen.add(key)
                normalized.append(key)
        models_payload["hidden_tiers"] = normalized
        return normalized

    def _set_model_hidden_tier(self, models_payload: dict[str, Any], provider: str, tier: str, *, hidden: bool) -> None:
        key = self._model_tier_key(provider, tier)
        if "|" not in key:
            return
        hidden_tiers = self._normalized_hidden_tiers(models_payload)
        if hidden:
            if key not in hidden_tiers:
                hidden_tiers.append(key)
        else:
            hidden_tiers = [item for item in hidden_tiers if item != key]
        models_payload["hidden_tiers"] = hidden_tiers

    def _remove_model_tier_override(self, models_payload: dict[str, Any], provider: str, tier: str) -> None:
        provider_name = str(provider or "").strip().lower()
        tier_name = str(tier or "").strip().lower()
        if not provider_name or not tier_name:
            return
        providers = models_payload.get("providers")
        if not isinstance(providers, dict):
            return
        provider_dict = providers.get(provider_name)
        if not isinstance(provider_dict, dict):
            return
        tiers = provider_dict.get("tiers")
        if not isinstance(tiers, dict):
            return
        tiers.pop(tier_name, None)
        if not tiers:
            providers.pop(provider_name, None)

    def _resolve_model_provider_tier_selection(self) -> tuple[str, str] | None:
        selected = str(self._settings_models_selected_id or "").strip().lower()
        if "|" in selected:
            provider, tier = selected.split("|", 1)
            provider = provider.strip().lower()
            tier = tier.strip().lower()
            if provider and tier:
                return provider, tier
        from textual.widgets import Select

        with contextlib.suppress(Exception):
            provider_widget = self.query_one("#settings_models_form_provider", Select)
            tier_widget = self.query_one("#settings_models_form_tier", Select)
            provider = str(getattr(provider_widget, "value", "")).strip().lower()
            tier = str(getattr(tier_widget, "value", "")).strip().lower()
            if provider and tier:
                return provider, tier
        return None

    def _save_models_default_selection(self) -> None:
        from textual.widgets import Select

        payload, path = self._load_project_settings_payload()
        models = payload.setdefault("models", {})
        provider_widget = self.query_one("#settings_models_provider_select", Select)
        tier_widget = self.query_one("#settings_models_default_tier_select", Select)
        provider_value = str(getattr(provider_widget, "value", "__auto__")).strip().lower()
        tier_value = str(getattr(tier_widget, "value", "balanced")).strip().lower() or "balanced"
        if tier_value == "__none__":
            tier_value = str(models.get("default_tier", "balanced")).strip().lower() or "balanced"
        if provider_value in {"", "__auto__"}:
            models["provider"] = None
            models["default_selection"] = {"provider": None, "tier": tier_value}
        else:
            models["provider"] = provider_value
            models["default_selection"] = {"provider": provider_value, "tier": tier_value}
        models["default_tier"] = tier_value or "balanced"
        self._save_project_settings_payload(payload, path)

    def _open_settings_model_manager(self) -> None:
        from swarmee_river.tui.widgets import ModelConfigManagerScreen

        payload, _path = self._load_project_settings_payload()
        self.push_screen(
            ModelConfigManagerScreen(payload),
            self._apply_settings_model_manager_result,
        )

    def _apply_settings_model_manager_result(self, result: dict[str, Any] | None) -> None:
        if not isinstance(result, dict):
            return
        payload, path = self._load_project_settings_payload()
        incoming_models = result.get("models")
        if isinstance(incoming_models, dict):
            payload["models"] = incoming_models
        incoming_env = result.get("env")
        if isinstance(incoming_env, dict):
            payload["env"] = incoming_env
            for key, value in incoming_env.items():
                normalized_key = str(key).strip()
                normalized_value = str(value).strip()
                if normalized_key and normalized_value:
                    os.environ[normalized_key] = normalized_value
                elif normalized_key:
                    os.environ.pop(normalized_key, None)
        self._save_project_settings_payload(payload, path)
        self._refresh_model_select()
        self._refresh_settings_models()
        self._refresh_settings_env_list()
        self._refresh_settings_env_detail(self._settings_env_selected_key)
        self._refresh_agent_summary()
        self._write_transcript_line("[settings] saved model manager changes.")

    def _save_model_form(self) -> None:
        from textual.widgets import Input, Select

        provider_widget = self.query_one("#settings_models_form_provider", Select)
        tier_widget = self.query_one("#settings_models_form_tier", Select)
        model_id_input = self.query_one("#settings_models_form_model_id", Input)
        display_input = self.query_one("#settings_models_form_display_name", Input)
        description_input = self.query_one("#settings_models_form_description", Input)
        price_input_widget = self.query_one("#settings_models_form_price_input", Input)
        price_output_widget = self.query_one("#settings_models_form_price_output", Input)
        price_cached_widget = self.query_one("#settings_models_form_price_cached", Input)
        provider = str(getattr(provider_widget, "value", "")).strip().lower()
        tier = str(getattr(tier_widget, "value", "")).strip().lower()
        model_id = str(getattr(model_id_input, "value", "")).strip()
        display_name = str(getattr(display_input, "value", "")).strip()
        description = str(getattr(description_input, "value", "")).strip()
        price_input = str(getattr(price_input_widget, "value", "")).strip()
        price_output = str(getattr(price_output_widget, "value", "")).strip()
        price_cached = str(getattr(price_cached_widget, "value", "")).strip()
        if not provider or not tier or not model_id:
            self._write_transcript_line("[settings] provider, tier, and model_id are required.")
            return

        payload, path = self._load_project_settings_payload()
        models = payload.setdefault("models", {})
        providers = models.setdefault("providers", {})
        provider_dict = providers.setdefault(provider, {})
        tiers = provider_dict.setdefault("tiers", {})
        self._set_model_hidden_tier(models, provider, tier, hidden=False)
        tier_dict: dict[str, Any] = dict(tiers.get(tier, {})) if isinstance(tiers.get(tier), dict) else {}
        tier_dict["provider"] = provider
        tier_dict["model_id"] = model_id
        if display_name:
            tier_dict["display_name"] = display_name
        else:
            tier_dict.pop("display_name", None)
        if description:
            tier_dict["description"] = description
        else:
            tier_dict.pop("description", None)
        tiers[tier] = tier_dict

        provider_key = provider.upper()
        env_payload = payload.get("env")
        if not isinstance(env_payload, dict):
            env_payload = {}
        for env_key, raw_value in (
            (f"SWARMEE_PRICE_{provider_key}_INPUT_PER_1M", price_input),
            (f"SWARMEE_PRICE_{provider_key}_OUTPUT_PER_1M", price_output),
            (f"SWARMEE_PRICE_{provider_key}_CACHED_INPUT_PER_1M", price_cached),
        ):
            if raw_value:
                try:
                    float(raw_value)
                except ValueError:
                    self._write_transcript_line(f"[settings] invalid numeric price for {env_key}: {raw_value}")
                    return
                env_payload[env_key] = raw_value
                os.environ[env_key] = raw_value
            else:
                env_payload.pop(env_key, None)
                os.environ.pop(env_key, None)
        if env_payload:
            payload["env"] = env_payload
        else:
            payload.pop("env", None)

        self._save_project_settings_payload(payload, path)
        self._settings_models_selected_id = f"{provider}|{tier}"
        self._refresh_model_select()
        self._refresh_settings_models()
        self._refresh_settings_env_list()
        self._refresh_settings_env_detail(self._settings_env_selected_key)
        self._refresh_agent_summary()
        self._write_transcript_line(f"[settings] saved model {provider}/{tier} -> {model_id}")

    def _delete_model_form_selection(self) -> None:
        selected = self._resolve_model_provider_tier_selection()
        if selected is None:
            self._write_transcript_line("[settings] select a model tier to delete.")
            return
        provider, tier = selected
        payload, path = self._load_project_settings_payload()
        models = payload.setdefault("models", {})
        self._set_model_hidden_tier(models, provider, tier, hidden=True)
        self._remove_model_tier_override(models, provider, tier)
        self._save_project_settings_payload(payload, path)
        self._settings_models_selected_id = None
        self._refresh_model_select()
        self._refresh_settings_models()
        self._refresh_agent_summary()
        self._write_transcript_line(f"[settings] hid model tier {provider}/{tier}")

    def _restore_model_form_selection(self) -> None:
        selected = self._resolve_model_provider_tier_selection()
        if selected is None:
            self._write_transcript_line("[settings] select a model tier to restore.")
            return
        provider, tier = selected
        payload, path = self._load_project_settings_payload()
        models = payload.setdefault("models", {})
        self._set_model_hidden_tier(models, provider, tier, hidden=False)
        self._remove_model_tier_override(models, provider, tier)
        self._save_project_settings_payload(payload, path)
        self._settings_models_selected_id = f"{provider}|{tier}"
        self._refresh_model_select()
        self._refresh_settings_models()
        self._refresh_agent_summary()
        self._write_transcript_line(f"[settings] restored defaults for {provider}/{tier}")

    def _clear_model_form(self) -> None:
        from textual.widgets import Input, Select

        for selector_id, default_value in (
            ("#settings_models_form_provider", "bedrock"),
            ("#settings_models_form_tier", "balanced"),
        ):
            with contextlib.suppress(Exception):
                self.query_one(selector_id, Select).value = default_value
        for input_id in (
            "#settings_models_form_model_id",
            "#settings_models_form_display_name",
            "#settings_models_form_description",
            "#settings_models_form_price_input",
            "#settings_models_form_price_output",
            "#settings_models_form_price_cached",
        ):
            with contextlib.suppress(Exception):
                self.query_one(input_id, Input).value = ""
        self._settings_models_selected_id = None
        self._refresh_settings_model_detail()

    def _refresh_settings_general(self) -> None:
        summary_widget = self._settings_general_summary
        if summary_widget is not None:
            with contextlib.suppress(Exception):
                summary_widget.update(
                    "Workspace settings and runtime behavior.\n"
                    "Use Models for provider auth and model catalog management."
                )

        aws_input = self._settings_aws_profile_input
        if aws_input is not None:
            current_profile = (os.getenv("AWS_PROFILE") or "").strip()
            if current_profile and not str(getattr(aws_input, "value", "")).strip():
                with contextlib.suppress(Exception):
                    aws_input.value = current_profile

        # -- Auto-Approve toggle --
        toggle = self._settings_toggle_auto_approve_button
        if toggle is not None:
            enabled = bool(self._default_auto_approve)
            toggle.label = f"Auto-Approve: {'On' if enabled else 'Off'}"
            toggle.variant = "warning" if enabled else "default"

        # -- Bypass Consent toggle --
        btn = self._settings_toggle_bypass_consent_button
        if btn is not None:
            val = (os.environ.get("BYPASS_TOOL_CONSENT") or "").strip().lower()
            on = val in {"true", "1", "yes", "on"}
            btn.label = f"Bypass Consent: {'On' if on else 'Off'}"
            btn.variant = "warning" if on else "default"

        # -- ESC Interrupt toggle --
        btn = self._settings_toggle_esc_interrupt_button
        if btn is not None:
            val = (os.environ.get("SWARMEE_ESC_INTERRUPT") or "enabled").strip().lower()
            on = val != "disabled"
            btn.label = f"ESC Interrupt: {'On' if on else 'Off'}"
            btn.variant = "success" if on else "default"
        self._refresh_settings_interrupt_control_controls()

        # -- Context Manager select --
        sel = self._settings_general_context_manager_select
        if sel is not None:
            val = (os.environ.get("SWARMEE_CONTEXT_MANAGER") or "summarize").strip().lower()
            if val not in {"summarize", "sliding", "none"}:
                val = "summarize"
            with contextlib.suppress(Exception):
                sel.value = val

        # -- Preflight select --
        sel = self._settings_general_preflight_select
        if sel is not None:
            val = (os.environ.get("SWARMEE_PREFLIGHT") or "enabled").strip().lower()
            if val not in {"enabled", "disabled"}:
                val = "enabled"
            with contextlib.suppress(Exception):
                sel.value = val

        # -- Swarm toggle --
        btn = self._settings_toggle_swarm_button
        if btn is not None:
            val = (os.environ.get("SWARMEE_SWARM_ENABLED") or "true").strip().lower()
            on = val not in {"false", "0", "no", "off", "disabled"}
            btn.label = f"Swarm: {'On' if on else 'Off'}"
            btn.variant = "success" if on else "default"

        # -- Log Events toggle --
        btn = self._settings_toggle_log_events_button
        if btn is not None:
            val = (os.environ.get("SWARMEE_LOG_EVENTS") or "").strip().lower()
            on = val in {"true", "1", "yes", "on"}
            btn.label = f"Log Events: {'On' if on else 'Off'}"
            btn.variant = "success" if on else "default"

        # -- Project Map toggle --
        btn = self._settings_toggle_project_map_button
        if btn is not None:
            val = (os.environ.get("SWARMEE_PROJECT_MAP") or "enabled").strip().lower()
            on = val != "disabled"
            btn.label = f"Project Map: {'On' if on else 'Off'}"
            btn.variant = "success" if on else "default"

        # -- Preflight Level select --
        sel = self._settings_general_preflight_level_select
        if sel is not None:
            val = (os.environ.get("SWARMEE_PREFLIGHT_LEVEL") or "summary").strip().lower()
            if val not in {"summary", "summary+tree", "summary+files"}:
                val = "summary"
            with contextlib.suppress(Exception):
                sel.value = val

        # -- Limit Tool Results toggle --
        btn = self._settings_toggle_limit_tool_results_button
        if btn is not None:
            val = (os.environ.get("SWARMEE_LIMIT_TOOL_RESULTS") or "true").strip().lower()
            on = val not in {"false", "0", "no", "off"}
            btn.label = f"Limit Results: {'On' if on else 'Off'}"
            btn.variant = "success" if on else "default"

        # -- Truncate Results toggle --
        btn = self._settings_toggle_truncate_results_button
        if btn is not None:
            val = (os.environ.get("SWARMEE_TRUNCATE_RESULTS") or "true").strip().lower()
            on = val not in {"false", "0", "no", "off"}
            btn.label = f"Truncate: {'On' if on else 'Off'}"
            btn.variant = "success" if on else "default"

        # -- Log Redact toggle --
        btn = self._settings_toggle_log_redact_button
        if btn is not None:
            val = (os.environ.get("SWARMEE_LOG_REDACT") or "true").strip().lower()
            on = val not in {"false", "0", "no", "off"}
            btn.label = f"Redact Logs: {'On' if on else 'Off'}"
            btn.variant = "success" if on else "default"

        # -- Freeze Tools toggle --
        btn = self._settings_toggle_freeze_tools_button
        if btn is not None:
            val = (os.environ.get("SWARMEE_FREEZE_TOOLS") or "").strip().lower()
            on = val in {"true", "1", "yes", "on"}
            btn.label = f"Freeze Tools: {'On' if on else 'Off'}"
            btn.variant = "warning" if on else "default"

        # -- Workspace scope display --
        self._refresh_settings_scope_display()

    def _refresh_settings_scope_display(self) -> None:
        """Update the current scope path display in Settings."""
        from swarmee_river.state_paths import state_dir

        widget = self._settings_scope_current
        if widget is None:
            return
        import contextlib as _ctx

        with _ctx.suppress(Exception):
            path = state_dir()
            widget.update(f"Current scope: {path}")

    def _update_header_status(self) -> None:
        counts = []
        if self.state.session.warning_count:
            counts.append(f"warn={self.state.session.warning_count}")
        if self.state.session.error_count:
            counts.append(f"err={self.state.session.error_count}")
        suffix = (" | " + " ".join(counts)) if counts else ""
        self.sub_title = f"{self._current_model_summary()}{suffix}"
        if self._status_bar is not None:
            self._status_bar.set_counts(
                warnings=self.state.session.warning_count, errors=self.state.session.error_count
            )
        self._refresh_session_header()
        self._refresh_orchestrator_status()

    def _current_model_summary(self) -> str:
        provider_name, tier_name, model_id = choose_model_summary_parts(
            daemon_provider=self.state.daemon.provider,
            daemon_tier=self.state.daemon.tier,
            daemon_model_id=self.state.daemon.model_id,
            daemon_tiers=self.state.daemon.tiers,
            pending_value=self.state.daemon.pending_model_select_value,
            override_provider=self.state.daemon.model_provider_override,
            override_tier=self.state.daemon.model_tier_override,
        )
        if provider_name and tier_name:
            suffix = f" ({model_id})" if model_id else ""
            return f"Model: {provider_name}/{tier_name}{suffix}"
        return resolve_model_config_summary(
            provider_override=self.state.daemon.model_provider_override,
            tier_override=self.state.daemon.model_tier_override,
        )

    def _model_env_overrides(self) -> dict[str, str]:
        overrides: dict[str, str] = {}
        if self.state.daemon.model_provider_override:
            overrides["SWARMEE_MODEL_PROVIDER"] = self.state.daemon.model_provider_override
        if self.state.daemon.model_tier_override:
            overrides["SWARMEE_MODEL_TIER"] = self.state.daemon.model_tier_override
        return overrides

    def _refresh_model_select(self) -> None:
        if self.state.daemon.provider and self.state.daemon.tier and self.state.daemon.tiers:
            self._refresh_model_select_from_daemon(
                provider=self.state.daemon.provider,
                tier=self.state.daemon.tier,
                tiers=self.state.daemon.tiers,
            )
            self._refresh_orchestrator_status()
            self._refresh_settings_models()
            return

        options, selected_value = self._model_select_options()
        self._apply_model_select_options(options, selected_value)
        self._refresh_orchestrator_status()
        self._refresh_settings_models()

    def _apply_model_select_options(self, options: list[tuple[str, str]], selected_value: str) -> None:
        from textual.widgets import Select

        self.state.daemon.model_select_syncing = True
        self._model_select_programmatic_value = str(selected_value or "").strip().lower() or None
        try:
            with contextlib.suppress(Exception):
                selector = self.query_one("#model_select", Select)
                selector.set_options(options)
                selector.value = selected_value
        finally:
            with contextlib.suppress(Exception):
                self.call_after_refresh(self._release_model_select_syncing)
            if self.state.daemon.model_select_syncing:
                self._release_model_select_syncing()

    def _release_model_select_syncing(self) -> None:
        self.state.daemon.model_select_syncing = False

    def _refresh_model_select_from_daemon(
        self,
        *,
        provider: str,
        tier: str,
        tiers: list[dict[str, Any]],
    ) -> None:
        options, selected_value = daemon_model_select_options(
            provider=provider,
            tier=tier,
            tiers=tiers,
            pending_value=self.state.daemon.pending_model_select_value,
            override_provider=self.state.daemon.model_provider_override,
            override_tier=self.state.daemon.model_tier_override,
        )
        self._apply_model_select_options(options, selected_value)

    def _pin_model_select_target(self, provider: str, tier: str, *, seconds: float = 2.5) -> None:
        provider_name = str(provider or "").strip().lower()
        tier_name = str(tier or "").strip().lower()
        if not provider_name or not tier_name:
            self._model_select_target_value = None
            self._model_select_target_until_mono = None
            return
        self._model_select_target_value = f"{provider_name}|{tier_name}"
        self._model_select_target_until_mono = time.monotonic() + max(0.1, float(seconds))

    def _model_select_options(self) -> tuple[list[tuple[str, str]], str]:
        if self.state.daemon.tiers and self.state.daemon.provider:
            return daemon_model_select_options(
                provider=self.state.daemon.provider,
                tier=(self.state.daemon.tier or ""),
                tiers=self.state.daemon.tiers,
                pending_value=self.state.daemon.pending_model_select_value,
                override_provider=self.state.daemon.model_provider_override,
                override_tier=self.state.daemon.model_tier_override,
            )
        return model_select_options(
            provider_override=self.state.daemon.model_provider_override,
            tier_override=self.state.daemon.model_tier_override,
        )

    def _handle_model_info(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self._handle_connect_model_info_event(event)
        provider = str(event.get("provider", "")).strip().lower()
        tier = str(event.get("tier", "")).strip().lower()
        tool_names_raw = event.get("tool_names")
        if isinstance(tool_names_raw, list):
            self._refresh_agent_tool_catalog([str(item) for item in tool_names_raw])
        else:
            self._refresh_agent_tool_catalog(None)
        incoming_value = f"{provider}|{tier}" if provider and tier else ""
        now_mono = time.monotonic()
        if self._model_select_target_until_mono is not None and now_mono >= self._model_select_target_until_mono:
            self._model_select_target_value = None
            self._model_select_target_until_mono = None
        if _should_ignore_stale_model_info_update(
            incoming_value=incoming_value,
            target_value=self._model_select_target_value,
            target_until_mono=self._model_select_target_until_mono,
            now_mono=now_mono,
        ):
            return
        if self._model_select_target_value and incoming_value == self._model_select_target_value:
            self._model_select_target_value = None
            self._model_select_target_until_mono = None
        model_id = event.get("model_id")
        tiers = event.get("tiers")

        self.state.daemon.provider = provider or None
        self.state.daemon.tier = tier or None
        self.state.daemon.model_id = str(model_id).strip() if model_id is not None and str(model_id).strip() else None
        self.state.daemon.tiers = tiers if isinstance(tiers, list) else []
        self.state.daemon.current_model = self.state.daemon.model_id or (
            f"{self.state.daemon.provider}/{self.state.daemon.tier}"
            if self.state.daemon.provider and self.state.daemon.tier
            else None
        )
        pending_value = (self.state.daemon.pending_model_select_value or "").strip().lower()
        if pending_value and "|" in pending_value:
            pending_provider, pending_tier = pending_value.split("|", 1)
            if pending_provider == provider and pending_tier == tier:
                self.state.daemon.pending_model_select_value = None
                pending_value = ""

        if not pending_value and self.state.daemon.provider and self.state.daemon.tier:
            self.state.daemon.model_provider_override = self.state.daemon.provider
            self.state.daemon.model_tier_override = self.state.daemon.tier

        if self.state.daemon.provider and self.state.daemon.tier:
            self._refresh_model_select_from_daemon(
                provider=self.state.daemon.provider,
                tier=self.state.daemon.tier,
                tiers=self.state.daemon.tiers,
            )
        else:
            self._refresh_model_select()

        self._update_header_status()
        if self._status_bar is not None:
            self._status_bar.set_model(self._current_model_summary())
        self._refresh_agent_summary()
        self._render_agent_builder_panel()

    def _warn_run_active_tier_change_once(self) -> None:
        if self.state.daemon.run_active_tier_warning_emitted:
            return
        self.state.daemon.run_active_tier_warning_emitted = True
        self._write_transcript_line(_RUN_ACTIVE_TIER_WARNING)

    def _sync_settings_sidebar_autosize(self, pane_id: str | None) -> None:
        """Auto-expand sidebar in Settings and restore prior ratio when leaving."""
        import contextlib as _ctx

        from textual.widgets import TabbedContent

        current_pane_id = str(pane_id or "").strip()
        if not current_pane_id:
            with _ctx.suppress(Exception):
                tabs = self.query_one("#side_tabs", TabbedContent)
                current_pane_id = str(getattr(tabs, "active", "")).strip()
        is_settings = current_pane_id == "tab_settings"
        if is_settings:
            if self._pre_settings_split_ratio is None:
                self._pre_settings_split_ratio = self._split_ratio
            if self._split_ratio > 1:
                self._split_ratio = 1
                self._apply_split_ratio()
            return
        if self._pre_settings_split_ratio is not None:
            restored_ratio = max(1, min(4, int(self._pre_settings_split_ratio)))
            self._split_ratio = restored_ratio
            self._pre_settings_split_ratio = None
            self._apply_split_ratio()

    def _set_model_tier_from_value(self, value: str) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        parsed = parse_model_select_value(value)
        if parsed is None:
            return
        requested_provider, requested_tier = parsed
        self.state.daemon.pending_model_select_value = None
        self.state.daemon.model_provider_override = requested_provider or None
        self.state.daemon.model_tier_override = requested_tier or None
        self._refresh_model_select()
        self._update_header_status()
        self._update_prompt_placeholder()
        if (
            self.state.daemon.ready
            and self.state.daemon.proc is not None
            and self.state.daemon.proc.poll() is None
            and not self.state.daemon.query_active
        ):
            if not send_daemon_command(self.state.daemon.proc, {"cmd": "set_tier", "tier": requested_tier}):
                self._write_transcript_line("[model] failed to send tier change to daemon.")
            else:
                self.state.daemon.pending_model_select_value = f"{requested_provider}|{requested_tier}"
        if self._status_bar is not None:
            self._status_bar.set_model(self._current_model_summary())

    def _sync_selected_model_before_run(self) -> None:
        import contextlib as _ctx

        from textual.widgets import Select

        selected_value: str | None = None
        with _ctx.suppress(Exception):
            selector = self.query_one("#model_select", Select)
            selected_value = str(getattr(selector, "value", "")).strip()
        parsed = parse_model_select_value(selected_value)
        if parsed is None:
            return

        requested_provider, requested_tier = parsed
        self.state.daemon.model_provider_override = requested_provider or None
        self.state.daemon.model_tier_override = requested_tier or None

        current_provider = (self.state.daemon.provider or "").strip().lower()
        current_tier = (self.state.daemon.tier or "").strip().lower()
        if (
            current_provider
            and current_tier
            and requested_provider == current_provider
            and requested_tier == current_tier
        ):
            self.state.daemon.pending_model_select_value = None
            return
        self.state.daemon.pending_model_select_value = f"{requested_provider}|{requested_tier}"

    def _handle_model_command(self, normalized: str) -> bool:
        from swarmee_river.tui.commands import classify_model_command
        from swarmee_river.tui.transport import send_daemon_command

        command = classify_model_command(normalized)
        if command is None:
            return False
        action, argument = command

        if action == "help":
            self._write_transcript_line(self._current_model_summary())
            self._write_transcript_line(_MODEL_USAGE_TEXT)
            return True

        if action == "show":
            self._write_transcript_line(self._current_model_summary())
            return True

        if action == "list":
            options, _ = self._model_select_options()
            for label, _value in options:
                self._write_transcript_line(f"- {label}")
            return True

        if action == "reset":
            self.state.daemon.pending_model_select_value = None
            self.state.daemon.model_provider_override = None
            self.state.daemon.model_tier_override = None
            self._refresh_model_select()
            self._update_header_status()
            self._write_transcript_line(f"[model] reset. {self._current_model_summary()}")
            return True

        if action == "provider":
            provider = (argument or "").strip()
            if not provider:
                self._write_transcript_line("Usage: /model provider <name>")
                return True
            self.state.daemon.pending_model_select_value = None
            self.state.daemon.model_provider_override = provider
            self._refresh_model_select()
            self._update_header_status()
            self._write_transcript_line(f"[model] provider set to {provider}.")
            if self.state.daemon.ready:
                self._write_transcript_line("[model] restart daemon to apply provider changes.")
            self._write_transcript_line(self._current_model_summary())
            return True

        if action == "tier":
            tier = (argument or "").strip()
            if not tier:
                self._write_transcript_line("Usage: /model tier <name>")
                return True
            if (
                self.state.daemon.ready
                and self.state.daemon.proc is not None
                and self.state.daemon.proc.poll() is None
                and self.state.daemon.query_active
            ):
                self._warn_run_active_tier_change_once()
                return True
            self.state.daemon.pending_model_select_value = None
            self.state.daemon.model_tier_override = tier
            self._refresh_model_select()
            self._update_header_status()
            self._write_transcript_line(f"[model] tier set to {tier}.")
            if (
                self.state.daemon.ready
                and self.state.daemon.proc is not None
                and self.state.daemon.proc.poll() is None
                and not self.state.daemon.query_active
            ):
                requested_provider = (
                    (self.state.daemon.model_provider_override or self.state.daemon.provider or "").strip().lower()
                )
                requested_tier = tier.strip().lower()
                if not send_daemon_command(self.state.daemon.proc, {"cmd": "set_tier", "tier": tier}):
                    self.state.daemon.pending_model_select_value = None
                    self._write_transcript_line("[model] failed to send tier change to daemon.")
                elif requested_provider and requested_tier:
                    self.state.daemon.pending_model_select_value = f"{requested_provider}|{requested_tier}"
            self._write_transcript_line(self._current_model_summary())
            return True

        return False

    def _settings_aws_profile_value(self) -> str:
        widget = self._settings_aws_profile_input
        if widget is None:
            return ""
        return str(getattr(widget, "value", "")).strip()

    def _apply_settings_aws_profile(self, profile: str, *, announce: bool = True) -> None:
        normalized = profile.strip()
        if normalized:
            self._persist_project_setting_env_override("AWS_PROFILE", normalized)
            if announce:
                self._write_transcript_line(f"[settings] AWS profile set to {normalized}")
        else:
            self._persist_project_setting_env_override("AWS_PROFILE", None)
            if announce:
                self._write_transcript_line("[settings] AWS profile cleared (using default credential chain).")
        self._refresh_settings_env_list()
        self._refresh_settings_env_detail(self._settings_env_selected_key)
        self._refresh_settings_models()
