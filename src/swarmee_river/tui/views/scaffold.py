"""Tooling sidebar tab UI composition and widget wiring.

Formerly the "Context / Scaffold" tab — renamed to **Tooling** with subtabs:
Tools, Prompts, SOPs, KBs.
"""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Static, TabPane, TextArea

from swarmee_river.tui.widgets import SidebarDetail, SidebarHeader


def compose_tooling_tab() -> Iterator[Any]:
    """Yield the Tooling tab pane."""
    with TabPane("Tooling", id="tab_tooling"):
        with Vertical(id="tooling_panel"):
            with Horizontal(id="tooling_view_switch"):
                yield Button("Tools", id="tooling_view_tools", compact=True, variant="primary")
                yield Button("Prompts", id="tooling_view_prompts", compact=True, variant="default")
                yield Button("SOPs", id="tooling_view_sops", compact=True, variant="default")
                yield Button("KBs", id="tooling_view_kbs", compact=True, variant="default")

            # -- Tools sub-view (default) --------------------------------------
            with Vertical(id="tooling_tools_view"):
                yield SidebarHeader(
                    "Tool Catalog",
                    id="tooling_tools_header",
                    actions=[{"id": "tooling_tools_s3_import", "label": "S3 Import", "variant": "default"}],
                )
                yield DataTable(id="tooling_tools_table", cursor_type="row")
                yield SidebarDetail(id="tooling_tools_detail")

            # -- Prompts sub-view ----------------------------------------------
            with Vertical(id="tooling_prompts_view"):
                yield SidebarHeader(
                    "Prompt Assets",
                    id="tooling_prompts_header",
                )
                yield DataTable(id="tooling_prompts_table", cursor_type="row")
                yield TextArea(
                    text="",
                    id="tooling_prompt_content_input",
                    soft_wrap=True,
                )
                with Horizontal(id="tooling_prompt_actions"):
                    yield Button("New", id="tooling_prompt_new", compact=True, variant="default")
                    yield Button("Save", id="tooling_prompt_save", compact=True, variant="success")
                    yield Button("Delete", id="tooling_prompt_delete", compact=True, variant="warning")

            # -- SOPs sub-view -------------------------------------------------
            with Vertical(id="tooling_sops_view"):
                yield SidebarHeader(
                    "SOPs",
                    id="tooling_sops_header",
                    actions=[{"id": "tooling_sops_s3_import", "label": "S3 Import", "variant": "default"}],
                )
                yield DataTable(id="tooling_sops_table", cursor_type="row")
                yield SidebarDetail(id="tooling_sops_detail")

            # -- KBs sub-view --------------------------------------------------
            with Vertical(id="tooling_kbs_view"):
                yield SidebarHeader(
                    "Knowledge Bases",
                    id="tooling_kbs_header",
                    actions=[{"id": "tooling_kbs_s3_import", "label": "S3 Import", "variant": "default"}],
                )
                yield Static(
                    "No knowledge bases connected.\nUse /kb to connect one.",
                    id="kbs_empty_state",
                )
                yield DataTable(id="tooling_kbs_table", cursor_type="row")
                yield SidebarDetail(id="kbs_detail")


def wire_tooling_widgets(app: Any) -> None:
    """Bind Tooling tab widgets onto app fields used by event handlers."""
    # Sub-view switch buttons
    app._tooling_view_prompts_button = app.query_one("#tooling_view_prompts", Button)
    app._tooling_view_tools_button = app.query_one("#tooling_view_tools", Button)
    app._tooling_view_sops_button = app.query_one("#tooling_view_sops", Button)
    app._tooling_view_kbs_button = app.query_one("#tooling_view_kbs", Button)

    # Sub-view containers
    app._tooling_prompts_view = app.query_one("#tooling_prompts_view", Vertical)
    app._tooling_tools_view = app.query_one("#tooling_tools_view", Vertical)
    app._tooling_sops_view = app.query_one("#tooling_sops_view", Vertical)
    app._tooling_kbs_view = app.query_one("#tooling_kbs_view", Vertical)

    # Prompts widgets
    app._tooling_prompts_header = app.query_one("#tooling_prompts_header", SidebarHeader)
    app._tooling_prompts_table = app.query_one("#tooling_prompts_table", DataTable)
    app._tooling_prompt_content_input = app.query_one("#tooling_prompt_content_input", TextArea)

    # Tools widgets
    app._tooling_tools_header = app.query_one("#tooling_tools_header", SidebarHeader)
    app._tooling_tools_table = app.query_one("#tooling_tools_table", DataTable)
    app._tooling_tools_detail = app.query_one("#tooling_tools_detail", SidebarDetail)

    # SOPs widgets
    app._tooling_sops_header = app.query_one("#tooling_sops_header", SidebarHeader)
    app._tooling_sops_table = app.query_one("#tooling_sops_table", DataTable)
    app._tooling_sops_detail = app.query_one("#tooling_sops_detail", SidebarDetail)

    # KBs widgets
    app._tooling_kbs_header = app.query_one("#tooling_kbs_header", SidebarHeader)
    app._tooling_kbs_table = app.query_one("#tooling_kbs_table", DataTable)
    app._kbs_detail = app.query_one("#kbs_detail", SidebarDetail)

    # Backward-compat: old scaffold widget names that app.py may still reference
    # during transition.  These will be cleaned up in later phases.
    app._scaffold_view_context_button = None
    app._scaffold_view_sops_button = app._tooling_view_sops_button
    app._scaffold_view_kbs_button = app._tooling_view_kbs_button
    app._scaffold_view_artifacts_button = None
    app._scaffold_context_view = None
    app._scaffold_sops_view = app._tooling_sops_view
    app._scaffold_kbs_view = app._tooling_kbs_view
    app._scaffold_artifacts_view = None
    app._artifacts_header = None
    app._artifacts_table = None
    app._artifacts_detail = None
    app._context_sources_list = None
    app._context_input = None
    app._context_sop_select = None


# Keep old names importable for backward compat during transition.
compose_scaffold_tab = compose_tooling_tab
wire_scaffold_widgets = wire_tooling_widgets

__all__ = ["compose_tooling_tab", "wire_tooling_widgets", "compose_scaffold_tab", "wire_scaffold_widgets"]
