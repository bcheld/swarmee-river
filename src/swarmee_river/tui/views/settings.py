"""Settings sidebar tab UI composition and widget wiring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DirectoryTree, Input, Static, TabPane

from swarmee_river.tui.widgets import SidebarList


# Environment variable keys to display (in order).
_ENV_KEYS = [
    "SWARMEE_MODEL_PROVIDER",
    "SWARMEE_MODEL_TIER",
    "SWARMEE_STATE_DIR",
    "SWARMEE_ENABLE_TOOLS",
    "SWARMEE_DISABLE_TOOLS",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "AWS_PROFILE",
    "AWS_REGION",
]

# Keys whose values should be masked in the UI.
_MASKED_KEYS = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}


def _mask_value(key: str, value: str) -> str:
    if key in _MASKED_KEYS and len(value) > 8:
        return value[:4] + "..." + value[-4:]
    return value


def build_env_sidebar_items() -> list[dict[str, str]]:
    """Build SidebarList items for relevant environment variables."""
    items: list[dict[str, str]] = []
    for key in _ENV_KEYS:
        value = os.environ.get(key, "")
        if not value:
            continue
        items.append({
            "id": key,
            "title": key,
            "subtitle": _mask_value(key, value),
            "state": "default",
        })
    if not items:
        items.append({
            "id": "__no_env__",
            "title": "No environment variables set",
            "subtitle": "Add variables below or set them in your shell profile.",
            "state": "default",
        })
    return items


def compose_settings_tab() -> Iterator[Any]:
    """Yield the Settings tab pane."""
    with TabPane("Settings", id="tab_settings"):
        with Vertical(id="settings_panel"):
            with Horizontal(id="settings_view_switch"):
                yield Button("Environment", id="settings_view_env", compact=True, variant="primary")
                yield Button("Scoping", id="settings_view_scoping", compact=True, variant="default")

            # -- Environment sub-view ----------------------------------------
            with Vertical(id="settings_env_view"):
                yield Static("Environment Variables", id="settings_env_header")
                yield SidebarList(id="settings_env_list")
                with Horizontal(id="settings_env_add_row"):
                    yield Input(placeholder="KEY", id="settings_env_key")
                    yield Input(placeholder="VALUE", id="settings_env_value")
                    yield Button("Set", id="settings_env_add", compact=True, variant="success")

            # -- Scoping sub-view --------------------------------------------
            with Vertical(id="settings_scoping_view"):
                yield Static("Session Directory Scoping", id="settings_scoping_header")
                yield Static("", id="settings_scope_current")
                yield Input(placeholder="Enter path or browse below", id="settings_scope_path_input")
                yield DirectoryTree(str(Path.home()), id="settings_directory_tree")
                yield Button("Set Scope", id="settings_set_scope", variant="primary")


def wire_settings_widgets(app: Any) -> None:
    """Bind Settings tab widgets onto app fields used by event handlers."""
    app._settings_view_env_button = app.query_one("#settings_view_env", Button)
    app._settings_view_scoping_button = app.query_one("#settings_view_scoping", Button)
    app._settings_env_view = app.query_one("#settings_env_view", Vertical)
    app._settings_scoping_view = app.query_one("#settings_scoping_view", Vertical)
    app._settings_env_list = app.query_one("#settings_env_list", SidebarList)
    app._settings_scope_current = app.query_one("#settings_scope_current", Static)

__all__ = ["compose_settings_tab", "wire_settings_widgets", "build_env_sidebar_items"]
