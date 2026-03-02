"""Settings sidebar tab UI composition and widget wiring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, DirectoryTree, Input, Select, Static, TabPane


class SettingsDirectoryTree(DirectoryTree):
    """DirectoryTree variant that avoids async watch warnings before mount."""

    _pending_initial_reload: bool = False

    def watch_path(self) -> None:  # type: ignore[override]
        # `DirectoryTree.watch_path` is async in Textual. During plain compose-time
        # test construction it may be invoked without an event loop and emit an
        # un-awaited coroutine warning. Defer until mount instead.
        if not self.is_mounted:
            self._pending_initial_reload = True
            return
        self._pending_initial_reload = False
        self.run_worker(
            self._reload_for_path_change(),
            group="settings-directory-tree-watch-path",
            exclusive=True,
        )

    def on_mount(self) -> None:
        if self._pending_initial_reload or not self.root.children:
            self._pending_initial_reload = False
            self.run_worker(
                self._reload_for_path_change(),
                group="settings-directory-tree-watch-path",
                exclusive=True,
            )

    async def _reload_for_path_change(self) -> None:
        has_cursor = self.cursor_node is not None
        root_data = self.root.data
        root_data_type = type(root_data) if root_data is not None else None
        next_data = root_data_type(self.PATH(self.path)) if root_data_type is not None else None
        self.reset_node(self.root, str(self.path), next_data)
        await self.reload()
        if has_cursor:
            self.cursor_line = 0
        self.scroll_to(0, 0, animate=False)


@dataclass(frozen=True)
class EnvVarSpec:
    key: str
    category: str
    default: str
    description: str
    choices: tuple[str, ...] = ()


# Keys whose values should be masked in the UI.
_MASKED_KEYS = {
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "SWARMEE_GITHUB_COPILOT_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
}

_CHOICE_MAP: dict[str, tuple[str, ...]] = {
    "SWARMEE_CONTEXT_MANAGER": ("summarize", "sliding", "none"),
    "SWARMEE_SUMMARIZE_CONTEXT": ("true", "false"),
    "SWARMEE_TRUNCATE_RESULTS": ("true", "false"),
    "SWARMEE_LIMIT_TOOL_RESULTS": ("true", "false"),
    "SWARMEE_ESC_INTERRUPT": ("enabled", "disabled"),
    "SWARMEE_NOTEBOOK_NO_CONTEXT": ("false", "true"),
    "SWARMEE_JUPYTER_MODEL_PROVIDER": ("bedrock", "openai", "ollama", "github_copilot"),
    "SWARMEE_SWARM_ENABLED": ("true", "false"),
    "SWARMEE_LOG_EVENTS": ("true", "false"),
    "SWARMEE_LOG_REDACT": ("true", "false"),
    "SWARMEE_SESSION_S3_AUTO_EXPORT": ("false", "true"),
    "SWARMEE_SESSION_KB_PROMOTE_ON_COMPLETE": ("false", "true"),
    "SWARMEE_MODEL_PROVIDER": ("bedrock", "openai", "ollama", "github_copilot"),
    "SWARMEE_OPENAI_REASONING_EFFORT": ("low", "medium", "high"),
    "SWARMEE_AUTO_APPROVE": ("false", "true"),
    "SWARMEE_PREFLIGHT": ("enabled", "disabled"),
    "SWARMEE_PREFLIGHT_LEVEL": ("summary", "summary+tree", "summary+files"),
    "SWARMEE_PREFLIGHT_PRINT": ("disabled", "enabled"),
    "SWARMEE_PROJECT_MAP": ("enabled", "disabled"),
    "SWARMEE_FREEZE_TOOLS": ("true", "false"),
    "SWARMEE_CACHE_SAFE_SUMMARY": ("false", "true"),
    "SWARMEE_MODEL_TIER": ("fast", "balanced", "deep", "long"),
    "SWARMEE_TIER_AUTO": ("false", "true"),
    "STRANDS_THINKING_TYPE": ("enabled", "disabled"),
    "STRANDS_TOOL_CONSOLE_MODE": ("enabled", "disabled"),
    "BYPASS_TOOL_CONSENT": ("false", "true"),
}


def _mask_value(key: str, value: str) -> str:
    if key in _MASKED_KEYS and len(value) > 8:
        return value[:4] + "..." + value[-4:]
    return value


def _env_example_path() -> Path:
    return Path(__file__).resolve().parents[4] / "env.example"


@lru_cache(maxsize=1)
def env_var_specs() -> tuple[EnvVarSpec, ...]:
    """Parse env.example and return an ordered catalog of configurable vars."""
    path = _env_example_path()
    if not path.exists():
        return tuple()

    lines = path.read_text(encoding="utf-8").splitlines()
    category = "General"
    description_buffer: list[str] = []
    items_by_key: dict[str, EnvVarSpec] = {}
    ordered_keys: list[str] = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        body = stripped[1:].strip()

        prev = lines[idx - 1].strip() if idx > 0 else ""
        nxt = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if body and "=" not in body and prev.startswith("###") and nxt.startswith("###"):
            category = body
            description_buffer = []
            continue

        if not body:
            continue
        if body.startswith("-") or body.startswith("("):
            description_buffer.append(body)
            continue
        if "=" not in body:
            description_buffer.append(body)
            continue

        key, _, raw_default = body.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key.upper() != key:
            description_buffer = []
            continue

        default_value = raw_default.strip()
        if default_value.startswith('"') and default_value.endswith('"') and len(default_value) >= 2:
            default_value = default_value[1:-1]
        if not default_value:
            default_value = "(unset)"

        description = " ".join(part.strip() for part in description_buffer if part.strip()).strip() or "No description."
        description_buffer = []

        spec = EnvVarSpec(
            key=key,
            category=category,
            default=default_value,
            description=description,
            choices=_CHOICE_MAP.get(key, ()),
        )
        if key not in items_by_key:
            ordered_keys.append(key)
            items_by_key[key] = spec
            continue

        existing = items_by_key[key]
        merged_desc = existing.description
        if description and description not in merged_desc:
            merged_desc = f"{merged_desc} {description}".strip()
        merged_default = existing.default if existing.default != "(unset)" else spec.default
        items_by_key[key] = EnvVarSpec(
            key=key,
            category=existing.category,
            default=merged_default,
            description=merged_desc,
            choices=existing.choices or spec.choices,
        )

    return tuple(items_by_key[key] for key in ordered_keys)


def env_category_options() -> list[tuple[str, str]]:
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for spec in env_var_specs():
        if spec.category in seen:
            continue
        seen.add(spec.category)
        options.append((spec.category, spec.category))
    return options or [("General", "General")]


def env_spec_by_key(key: str) -> EnvVarSpec | None:
    lookup = str(key or "").strip()
    if not lookup:
        return None
    for spec in env_var_specs():
        if spec.key == lookup:
            return spec
    return None


def build_env_sidebar_items(category: str | None = None) -> list[dict[str, str]]:
    """Build SidebarList items for configurable environment variables."""
    selected_category = (category or "").strip()
    specs = env_var_specs()
    if selected_category:
        specs = tuple(spec for spec in specs if spec.category == selected_category)

    items: list[dict[str, str]] = []
    for spec in specs:
        current_value = os.environ.get(spec.key, "").strip()
        shown_current = _mask_value(spec.key, current_value) if current_value else "(unset)"
        shown_default = _mask_value(spec.key, spec.default) if spec.default != "(unset)" else "(unset)"
        items.append(
            {
                "id": spec.key,
                "title": spec.key,
                "subtitle": f"current: {shown_current} | default: {shown_default}",
                "state": "success" if current_value else "default",
            }
        )
    if not items:
        items.append(
            {
                "id": "__no_env__",
                "title": "No variables in this category",
                "subtitle": "Select another category or verify env.example parsing.",
                "state": "default",
            }
        )
    return items


def build_env_table_rows(category: str | None = None) -> list[tuple[str, str, str, str]]:
    """Build DataTable rows for configurable environment variables."""
    selected_category = (category or "").strip()
    specs = env_var_specs()
    if selected_category:
        specs = tuple(spec for spec in specs if spec.category == selected_category)

    rows: list[tuple[str, str, str, str]] = []
    for spec in specs:
        current_value = os.environ.get(spec.key, "").strip()
        shown_current = _mask_value(spec.key, current_value) if current_value else "(unset)"
        shown_default = _mask_value(spec.key, spec.default) if spec.default != "(unset)" else "(unset)"
        state = "set" if current_value else "unset"
        rows.append((spec.key, shown_current, shown_default, state))
    return rows


def build_models_table_rows(settings: Any) -> list[tuple[str, str, str, str]]:
    """Build DataTable rows for model provider/tier catalog."""
    from swarmee_river.pricing import resolve_pricing

    rows: list[tuple[str, str, str, str]] = []
    providers = getattr(getattr(settings, "models", None), "providers", {}) or {}
    for provider_name, provider in providers.items():
        tiers = getattr(provider, "tiers", {}) or {}
        for tier_name, tier in tiers.items():
            model_id = str(getattr(tier, "model_id", "") or "").strip() or "(unset)"
            pricing = resolve_pricing(provider=provider_name, model_id=getattr(tier, "model_id", None))
            pricing_label = ""
            if pricing is not None and pricing.input_per_1m is not None and pricing.output_per_1m is not None:
                cached = (
                    pricing.cached_input_per_1m if pricing.cached_input_per_1m is not None else pricing.input_per_1m
                )
                pricing_label = (
                    f" | ${pricing.input_per_1m}/1M in, ${pricing.output_per_1m}/1M out, ${cached}/1M cached"
                )
            row_id = f"{provider_name}|{tier_name}"
            rows.append((row_id, f"{provider_name}/{tier_name}", model_id, pricing_label))
    return sorted(rows, key=lambda item: item[0])


def compose_settings_tab() -> Iterator[Any]:
    """Yield the Settings tab pane."""
    with TabPane("Settings", id="tab_settings"):
        with Vertical(id="settings_panel"):
            with Horizontal(id="settings_view_switch"):
                yield Button("General", id="settings_view_general", compact=True, variant="primary")
                yield Button("Models", id="settings_view_models", compact=True, variant="default")
                yield Button("Advanced", id="settings_view_advanced", compact=True, variant="default")

            # -- General sub-view --------------------------------------------
            with VerticalScroll(id="settings_general_view"):
                yield Static("General Configuration", id="settings_general_header")
                yield Static("", id="settings_general_summary")

                yield Static("Runtime", classes="settings-section-label")
                with Horizontal(id="settings_general_runtime_row"):
                    yield Button(
                        "Auto-Approve: Off", id="settings_toggle_auto_approve", compact=True, variant="default"
                    )
                    yield Button(
                        "Bypass Consent: Off", id="settings_toggle_bypass_consent", compact=True, variant="default"
                    )
                    yield Button(
                        "ESC Interrupt: On", id="settings_toggle_esc_interrupt", compact=True, variant="default"
                    )

                yield Static("Context", classes="settings-section-label")
                with Horizontal(id="settings_general_context_row"):
                    yield Select(
                        options=[
                            ("Context: summarize", "summarize"),
                            ("Context: sliding", "sliding"),
                            ("Context: none", "none"),
                        ],
                        allow_blank=False,
                        id="settings_general_context_manager",
                        compact=True,
                    )
                    yield Select(
                        options=[
                            ("Preflight: enabled", "enabled"),
                            ("Preflight: disabled", "disabled"),
                        ],
                        allow_blank=False,
                        id="settings_general_preflight",
                        compact=True,
                    )
                    yield Select(
                        options=[
                            ("Level: summary", "summary"),
                            ("Level: summary+tree", "summary+tree"),
                            ("Level: summary+files", "summary+files"),
                        ],
                        allow_blank=False,
                        id="settings_general_preflight_level",
                        compact=True,
                    )

                yield Static("Features", classes="settings-section-label")
                with Horizontal(id="settings_general_features_row"):
                    yield Button("Swarm: On", id="settings_toggle_swarm", compact=True, variant="default")
                    yield Button("Log Events: Off", id="settings_toggle_log_events", compact=True, variant="default")
                    yield Button("Project Map: On", id="settings_toggle_project_map", compact=True, variant="default")

                yield Static("Guardrails", classes="settings-section-label")
                with Horizontal(id="settings_general_guardrails_row"):
                    yield Button(
                        "Limit Results: On", id="settings_toggle_limit_tool_results", compact=True, variant="default"
                    )
                    yield Button("Truncate: On", id="settings_toggle_truncate_results", compact=True, variant="default")
                    yield Button("Redact Logs: On", id="settings_toggle_log_redact", compact=True, variant="default")
                    yield Button(
                        "Freeze Tools: Off", id="settings_toggle_freeze_tools", compact=True, variant="default"
                    )

                yield Static("Workspace", classes="settings-section-label")
                yield Static("", id="settings_scope_current")
                with Horizontal(id="settings_scope_row"):
                    yield Input(placeholder="Enter path or browse below", id="settings_scope_path_input")
                    yield Button("Set Scope", id="settings_set_scope", compact=True, variant="primary")
                yield SettingsDirectoryTree(str(Path.home()), id="settings_directory_tree")

            # -- Models sub-view ---------------------------------------------
            with Vertical(id="settings_models_view"):
                yield Static("Model Catalog", id="settings_models_header")
                yield Static("", id="settings_models_summary")
                with Horizontal(id="settings_models_defaults_row"):
                    yield Select(
                        options=[
                            ("Provider: auto", "__auto__"),
                            ("bedrock", "bedrock"),
                            ("openai", "openai"),
                            ("ollama", "ollama"),
                            ("github_copilot", "github_copilot"),
                        ],
                        allow_blank=False,
                        id="settings_models_provider_select",
                        compact=True,
                    )
                    yield Select(
                        options=[
                            ("Tier: fast", "fast"),
                            ("Tier: balanced", "balanced"),
                            ("Tier: deep", "deep"),
                            ("Tier: long", "long"),
                        ],
                        allow_blank=False,
                        id="settings_models_default_tier_select",
                        compact=True,
                    )
                with Horizontal(id="settings_auth_row"):
                    yield Button("Connect Copilot", id="settings_auth_connect_copilot", compact=True, variant="success")
                    yield Button("Connect AWS", id="settings_auth_connect_aws", compact=True, variant="primary")
                    yield Button("Refresh Status", id="settings_auth_refresh", compact=True, variant="default")
                with Horizontal(id="settings_aws_profile_row"):
                    yield Input(placeholder="AWS profile (default)", id="settings_aws_profile_input")
                    yield Button("Apply", id="settings_aws_profile_apply", compact=True, variant="default")
                yield Static("", id="settings_auth_status")
                yield DataTable(id="settings_models_table", cursor_type="row")
                yield Static("", id="settings_models_detail")
                with Horizontal(id="settings_models_form_row_1"):
                    yield Select(
                        options=[
                            ("bedrock", "bedrock"),
                            ("openai", "openai"),
                            ("ollama", "ollama"),
                            ("github_copilot", "github_copilot"),
                        ],
                        allow_blank=False,
                        id="settings_models_form_provider",
                        compact=True,
                    )
                    yield Select(
                        options=[
                            ("fast", "fast"),
                            ("balanced", "balanced"),
                            ("deep", "deep"),
                            ("long", "long"),
                        ],
                        allow_blank=False,
                        id="settings_models_form_tier",
                        compact=True,
                    )
                    yield Input(placeholder="model_id", id="settings_models_form_model_id")
                with Horizontal(id="settings_models_form_row_2"):
                    yield Input(placeholder="display_name", id="settings_models_form_display_name")
                    yield Input(placeholder="description", id="settings_models_form_description")
                with Horizontal(id="settings_models_form_row_3"):
                    yield Input(placeholder="input $ / 1M", id="settings_models_form_price_input")
                    yield Input(placeholder="output $ / 1M", id="settings_models_form_price_output")
                    yield Input(placeholder="cached input $ / 1M", id="settings_models_form_price_cached")
                with Horizontal(id="settings_models_form_actions"):
                    yield Button("New", id="settings_models_new", compact=True, variant="default")
                    yield Button("Save Model", id="settings_models_save", compact=True, variant="success")
                    yield Button(
                        "Restore Defaults",
                        id="settings_models_restore_defaults",
                        compact=True,
                        variant="default",
                    )
                    yield Button("Delete", id="settings_models_delete", compact=True, variant="error")

            # -- Advanced sub-view (all env vars by category) ----------------
            with Vertical(id="settings_advanced_view"):
                yield Static("Advanced Configuration", id="settings_env_header")
                yield Select(options=env_category_options(), allow_blank=False, id="settings_env_category")
                yield DataTable(id="settings_env_table", cursor_type="row")
                yield Static("", id="settings_env_detail")
                with Horizontal(id="settings_env_edit_row"):
                    yield Select(
                        options=[("Select constrained value...", "__none__")],
                        allow_blank=False,
                        id="settings_env_value_select",
                        compact=True,
                    )
                    yield Input(placeholder="Enter value", id="settings_env_value_input")
                with Horizontal(id="settings_env_actions"):
                    yield Button("Apply Value", id="settings_env_apply", compact=True, variant="success")
                    yield Button("Use Default", id="settings_env_default", compact=True, variant="default")
                    yield Button("Unset", id="settings_env_unset", compact=True, variant="warning")
                yield Static("Session Safety Overrides", id="settings_safety_header")
                yield Input(
                    placeholder="tool_consent: ask | allow | deny (blank = inherit)",
                    id="settings_safety_tool_consent",
                )
                yield Input(
                    placeholder="tool_allowlist: tool1, tool2, ... (blank = inherit)",
                    id="settings_safety_tool_allowlist",
                )
                yield Input(
                    placeholder="tool_blocklist: tool1, tool2, ... (blank = inherit)",
                    id="settings_safety_tool_blocklist",
                )
                with Horizontal(id="settings_safety_actions"):
                    yield Button("Apply", id="settings_safety_apply", compact=True, variant="success")
                    yield Button("Reset", id="settings_safety_reset", compact=True, variant="default")
                yield Static("", id="settings_safety_status")


def wire_settings_widgets(app: Any) -> None:
    """Bind Settings tab widgets onto app fields used by event handlers."""
    app._settings_view_general_button = app.query_one("#settings_view_general", Button)
    app._settings_view_models_button = app.query_one("#settings_view_models", Button)
    app._settings_view_advanced_button = app.query_one("#settings_view_advanced", Button)
    app._settings_general_view = app.query_one("#settings_general_view", VerticalScroll)
    app._settings_models_view = app.query_one("#settings_models_view", Vertical)
    app._settings_advanced_view = app.query_one("#settings_advanced_view", Vertical)
    app._settings_general_summary = app.query_one("#settings_general_summary", Static)
    app._settings_models_summary = app.query_one("#settings_models_summary", Static)
    app._settings_models_table = app.query_one("#settings_models_table", DataTable)
    app._settings_models_detail = app.query_one("#settings_models_detail", Static)
    app._settings_auth_status = app.query_one("#settings_auth_status", Static)
    app._settings_aws_profile_input = app.query_one("#settings_aws_profile_input", Input)
    app._settings_env_category_select = app.query_one("#settings_env_category", Select)
    app._settings_env_detail = app.query_one("#settings_env_detail", Static)
    app._settings_env_value_select = app.query_one("#settings_env_value_select", Select)
    app._settings_env_value_input = app.query_one("#settings_env_value_input", Input)
    app._settings_safety_tool_consent_input = app.query_one("#settings_safety_tool_consent", Input)
    app._settings_safety_tool_allowlist_input = app.query_one("#settings_safety_tool_allowlist", Input)
    app._settings_safety_tool_blocklist_input = app.query_one("#settings_safety_tool_blocklist", Input)
    app._settings_safety_status = app.query_one("#settings_safety_status", Static)
    app._settings_toggle_auto_approve_button = app.query_one("#settings_toggle_auto_approve", Button)
    app._settings_toggle_bypass_consent_button = app.query_one("#settings_toggle_bypass_consent", Button)
    app._settings_toggle_esc_interrupt_button = app.query_one("#settings_toggle_esc_interrupt", Button)
    app._settings_general_context_manager_select = app.query_one("#settings_general_context_manager", Select)
    app._settings_general_preflight_select = app.query_one("#settings_general_preflight", Select)
    app._settings_general_preflight_level_select = app.query_one("#settings_general_preflight_level", Select)
    app._settings_toggle_swarm_button = app.query_one("#settings_toggle_swarm", Button)
    app._settings_toggle_log_events_button = app.query_one("#settings_toggle_log_events", Button)
    app._settings_toggle_project_map_button = app.query_one("#settings_toggle_project_map", Button)
    app._settings_toggle_limit_tool_results_button = app.query_one("#settings_toggle_limit_tool_results", Button)
    app._settings_toggle_truncate_results_button = app.query_one("#settings_toggle_truncate_results", Button)
    app._settings_toggle_log_redact_button = app.query_one("#settings_toggle_log_redact", Button)
    app._settings_toggle_freeze_tools_button = app.query_one("#settings_toggle_freeze_tools", Button)
    app._settings_env_table = app.query_one("#settings_env_table", DataTable)
    app._settings_scope_current = app.query_one("#settings_scope_current", Static)
    app._settings_directory_tree = app.query_one("#settings_directory_tree", SettingsDirectoryTree)
    # Backward-compatible aliases for agent safety override helpers.
    app._agent_tools_override_consent_input = app._settings_safety_tool_consent_input
    app._agent_tools_override_allowlist_input = app._settings_safety_tool_allowlist_input
    app._agent_tools_override_blocklist_input = app._settings_safety_tool_blocklist_input
    app._agent_tools_override_status = app._settings_safety_status


__all__ = [
    "EnvVarSpec",
    "build_models_table_rows",
    "build_env_table_rows",
    "build_env_sidebar_items",
    "compose_settings_tab",
    "env_category_options",
    "env_spec_by_key",
    "env_var_specs",
    "wire_settings_widgets",
]
