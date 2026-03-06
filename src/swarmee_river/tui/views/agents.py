"""Agents sidebar tab UI composition and widget wiring."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Select, Static, TabPane, TextArea

from swarmee_river.tui.widgets import SidebarDetail, SidebarHeader


def compose_agents_tab() -> Iterator[Any]:
    """Yield the Agents tab pane with Overview + Builder sub-views."""
    with TabPane("Agents", id="tab_agents"):
        with Vertical(id="agent_panel"):
            with Horizontal(id="agent_view_switch"):
                yield Button("Overview", id="agent_view_overview", compact=True, variant="primary")
                yield Button("Builder", id="agent_view_builder", compact=True, variant="default")

            with Vertical(id="agent_overview_view"):
                yield Static("Orchestrator Agent", id="agent_summary_header")
                yield Static(
                    "Activated agents are available for delegation and swarm runs.",
                    id="agent_overview_help",
                )
                with Horizontal(id="agent_overview_model_row"):
                    yield Static("Runtime Model", id="agent_overview_model_label")
                    yield Select(
                        options=[("Loading model info...", "__loading__")],
                        allow_blank=False,
                        id="model_select",
                        compact=True,
                    )
                yield Static(
                    "Applies to orchestrator runs; set provider/tier before sending a prompt.",
                    id="agent_overview_model_help",
                )
                yield TextArea(
                    text="",
                    read_only=True,
                    show_cursor=False,
                    id="agent_summary",
                    soft_wrap=True,
                )
                yield SidebarHeader("Activated Agents", id="agent_overview_header")
                yield DataTable(id="agent_overview_table", cursor_type="row")
                yield SidebarDetail(id="agent_overview_detail")
                yield Static("", id="agent_overview_status")

            with Vertical(id="agent_builder_view"):
                with VerticalScroll(id="agent_builder_scroll"):
                    yield Static(
                        "Focused roster editor. Save and apply runtime bundles in the Bundles tab.",
                        id="agent_builder_help",
                    )
                    yield SidebarHeader(
                        "Agent Roster",
                        id="agent_builder_header",
                        actions=[{"id": "agent_builder_open_manager", "label": "Agent Manager", "variant": "primary"}],
                    )
                    yield DataTable(id="agent_builder_table", cursor_type="row")
                    yield SidebarDetail(id="agent_builder_agent_detail")
                    yield Static("", id="agent_builder_status")


def wire_agents_widgets(app: Any) -> None:
    """Bind Agents tab widgets onto app fields."""
    app._agent_view_overview_button = app.query_one("#agent_view_overview", Button)
    app._agent_view_builder_button = app.query_one("#agent_view_builder", Button)
    app._agent_overview_view = app.query_one("#agent_overview_view", Vertical)
    app._agent_builder_view = app.query_one("#agent_builder_view", Vertical)
    app._agent_builder_scroll = app.query_one("#agent_builder_scroll", VerticalScroll)
    app._agent_summary = app.query_one("#agent_summary", TextArea)
    app._agent_overview_header = app.query_one("#agent_overview_header", SidebarHeader)
    app._agent_overview_table = app.query_one("#agent_overview_table", DataTable)
    app._agent_overview_detail = app.query_one("#agent_overview_detail", SidebarDetail)
    app._agent_overview_status = app.query_one("#agent_overview_status", Static)

    app._agent_builder_table = app.query_one("#agent_builder_table", DataTable)
    app._agent_builder_detail = app.query_one("#agent_builder_agent_detail", SidebarDetail)
    app._agent_builder_status = app.query_one("#agent_builder_status", Static)
    app._agent_builder_agent_id_input = None
    app._agent_builder_agent_name_input = None
    app._agent_builder_agent_summary_input = None
    app._agent_builder_agent_prompt_input = None
    app._agent_builder_agent_prompt_refs_input = None
    app._agent_builder_prompt_asset_name_input = None
    app._agent_builder_prompt_asset_id_input = None
    app._agent_builder_prompt_asset_tags_input = None
    app._agent_builder_agent_provider_select = None
    app._agent_builder_agent_tier_select = None
    app._agent_builder_tools_summary = None
    app._agent_builder_sops_summary = None
    app._agent_builder_kb_summary = None
    app._agent_builder_agent_activated_checkbox = None

    # Keep compatibility with older app.py attribute names.
    app._agent_view_profile_button = app._agent_view_overview_button
    app._agent_view_tools_button = None
    app._agent_view_team_button = None
    app._agent_profile_view = app._agent_overview_view
    app._agent_tools_view = None
    app._agent_team_view = None
    app._agent_profile_list = None
    app._agent_profile_select = None
    app._agent_profile_id_input = None
    app._agent_profile_name_input = None
    app._agent_profile_status = app._agent_builder_status
    app._agent_builder_auto_delegate_checkbox = None
    app._agent_tools_header = None
    app._agent_tools_list = None
    app._agent_tools_detail = None
    app._agent_team_header = None
    app._agent_team_list = None
    app._agent_team_detail = None
    app._agent_team_preset_id_input = None
    app._agent_team_preset_name_input = None
    app._agent_team_preset_description_input = None
    app._agent_team_preset_spec_input = None
    app._agent_team_status = None


__all__ = ["compose_agents_tab", "wire_agents_widgets"]
