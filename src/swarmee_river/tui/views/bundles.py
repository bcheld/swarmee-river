"""Bundles tab UI composition and widget wiring."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Static, TabPane

from swarmee_river.tui.widgets import SidebarDetail, SidebarHeader


def compose_bundles_tab() -> Iterator[Any]:
    """Yield the Bundles tab pane."""
    with TabPane("Bundles", id="tab_bundles"):
        with Vertical(id="bundles_panel"):
            yield SidebarHeader("Bundles", id="bundles_header")
            yield Static(
                "Runtime presets catalog. Agent bundles are editable; path packs are listed read-only.",
                id="bundles_help",
            )
            yield DataTable(id="bundles_table", cursor_type="row")
            yield SidebarDetail(id="bundles_detail")
            with Horizontal(id="bundles_meta_row"):
                yield Input(placeholder="Bundle id", id="bundle_id")
                yield Input(placeholder="Bundle name", id="bundle_name")
            with Horizontal(id="bundles_actions_primary"):
                yield Button("New", id="bundle_new", compact=True, variant="default")
                yield Button("Save Bundle", id="bundle_save", compact=True, variant="success")
                yield Button("Delete Bundle", id="bundle_delete", compact=True, variant="warning")
            with Horizontal(id="bundles_actions_secondary"):
                yield Button("Load Draft", id="bundle_load_draft", compact=True, variant="default")
                yield Button("Apply Bundle", id="bundle_apply", compact=True, variant="primary")
            yield Static("", id="bundles_status")


def wire_bundles_widgets(app: Any) -> None:
    """Bind Bundles tab widgets onto app fields."""
    app._bundles_panel = app.query_one("#bundles_panel", Vertical)
    app._bundles_header = app.query_one("#bundles_header", SidebarHeader)
    app._bundles_table = app.query_one("#bundles_table", DataTable)
    app._bundles_detail = app.query_one("#bundles_detail", SidebarDetail)
    app._bundle_id_input = app.query_one("#bundle_id", Input)
    app._bundle_name_input = app.query_one("#bundle_name", Input)
    app._bundles_status = app.query_one("#bundles_status", Static)


__all__ = ["compose_bundles_tab", "wire_bundles_widgets"]
