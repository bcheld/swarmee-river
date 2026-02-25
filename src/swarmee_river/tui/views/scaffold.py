"""Scaffold sidebar tab UI composition and widget wiring."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Select, Static, TabPane

from swarmee_river.tui.widgets import SidebarDetail, SidebarHeader, SidebarList


def compose_scaffold_tab(*, context_select_placeholder: str) -> Iterator[Any]:
    """Yield the Scaffold tab pane."""
    with TabPane("Scaffold", id="tab_scaffold"):
        with Vertical(id="scaffold_panel"):
            with Horizontal(id="scaffold_view_switch"):
                yield Button("Context", id="scaffold_view_context", compact=True, variant="primary")
                yield Button("SOPs", id="scaffold_view_sops", compact=True, variant="default")
                yield Button("KBs", id="scaffold_view_kbs", compact=True, variant="default")
                yield Button("Artifacts", id="scaffold_view_artifacts", compact=True, variant="default")

            # -- Context sub-view (default) ----------------------------------
            with Vertical(id="scaffold_context_view"):
                yield Static("Active Context Sources", id="context_header")
                yield VerticalScroll(id="context_sources_list")
                with Horizontal(id="context_add_row"):
                    yield Button("File", id="context_add_file", compact=True, variant="default")
                    yield Button("Note", id="context_add_note", compact=True, variant="default")
                    yield Button("SOP", id="context_add_sop", compact=True, variant="default")
                    yield Button("KB", id="context_add_kb", compact=True, variant="default")
                with Horizontal(id="context_input_row"):
                    yield Input(placeholder="Enter context value", id="context_input")
                    yield Button("Add", id="context_add_commit", compact=True, variant="success")
                    yield Button("Cancel", id="context_add_cancel", compact=True, variant="default")
                with Horizontal(id="context_sop_row"):
                    yield Select(
                        options=[("Select SOP...", context_select_placeholder)],
                        allow_blank=False,
                        id="context_sop_select",
                        compact=True,
                    )
                    yield Button("Add", id="context_sop_commit", compact=True, variant="success")
                    yield Button("Cancel", id="context_sop_cancel", compact=True, variant="default")

            # -- SOPs sub-view -----------------------------------------------
            with Vertical(id="scaffold_sops_view"):
                yield Static("Available SOPs", id="sops_header")
                yield VerticalScroll(id="sop_list")

            # -- KBs sub-view -----------------------------------------------
            with Vertical(id="scaffold_kbs_view"):
                yield SidebarHeader("Knowledge Bases", id="kbs_header")
                yield Static(
                    "No knowledge bases connected.\nUse /kb to connect one.",
                    id="kbs_empty_state",
                )
                yield SidebarList(id="kbs_list")
                yield SidebarDetail(id="kbs_detail")

            # -- Artifacts sub-view ------------------------------------------
            with Vertical(id="scaffold_artifacts_view"):
                yield SidebarHeader("Artifacts", id="artifacts_header")
                yield SidebarList(id="artifacts_list")
                yield SidebarDetail(id="artifacts_detail")


def wire_scaffold_widgets(app: Any) -> None:
    """Bind Scaffold tab widgets onto app fields used by event handlers."""
    app._scaffold_view_context_button = app.query_one("#scaffold_view_context", Button)
    app._scaffold_view_sops_button = app.query_one("#scaffold_view_sops", Button)
    app._scaffold_view_kbs_button = app.query_one("#scaffold_view_kbs", Button)
    app._scaffold_view_artifacts_button = app.query_one("#scaffold_view_artifacts", Button)
    app._scaffold_context_view = app.query_one("#scaffold_context_view", Vertical)
    app._scaffold_sops_view = app.query_one("#scaffold_sops_view", Vertical)
    app._scaffold_kbs_view = app.query_one("#scaffold_kbs_view", Vertical)
    app._scaffold_artifacts_view = app.query_one("#scaffold_artifacts_view", Vertical)
    app._artifacts_header = app.query_one("#artifacts_header", SidebarHeader)
    app._artifacts_list = app.query_one("#artifacts_list", SidebarList)
    app._artifacts_detail = app.query_one("#artifacts_detail", SidebarDetail)
    app._context_sources_list = app.query_one("#context_sources_list", VerticalScroll)
    app._sop_list = app.query_one("#sop_list", VerticalScroll)
    app._context_input = app.query_one("#context_input", Input)
    app._context_sop_select = app.query_one("#context_sop_select", Select)

__all__ = ["compose_scaffold_tab", "wire_scaffold_widgets"]
