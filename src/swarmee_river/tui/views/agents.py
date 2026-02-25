"""Agents sidebar tab UI composition and widget wiring."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static, TabPane, TextArea

from swarmee_river.tui.widgets import AgentProfileActions, SidebarDetail, SidebarHeader, SidebarList


def compose_agents_tab() -> Iterator[Any]:
    """Yield the Agents tab pane.

    Sub-views reuse existing Agent Studio widget IDs so all app.py
    logic continues to work unchanged.
    """
    with TabPane("Agents", id="tab_agents"):
        with Vertical(id="agent_panel"):
            with Horizontal(id="agent_view_switch"):
                yield Button("Overview", id="agent_view_profile", compact=True, variant="primary")
                yield Button("Tools & Safety", id="agent_view_tools", compact=True, variant="default")
                yield Button("Team", id="agent_view_team", compact=True, variant="default")

            # -- Overview sub-view (default) ---------------------------------
            with Vertical(id="agent_profile_view"):
                yield Static("Orchestrator Agent", id="agent_summary_header")
                yield Static(
                    "The orchestrator processes your prompts and coordinates tools.",
                    id="agent_overview_help",
                )
                yield TextArea(
                    text="",
                    read_only=True,
                    show_cursor=False,
                    id="agent_summary",
                    soft_wrap=True,
                )
                yield Static("Saved Profiles", id="agent_profiles_header")
                yield SidebarList(id="agent_profile_list")
                with Horizontal(id="agent_profile_meta_row"):
                    yield Input(placeholder="Profile id", id="agent_profile_id")
                    yield Input(placeholder="Profile name", id="agent_profile_name")
                yield AgentProfileActions(id="agent_profile_actions")
                yield Static("", id="agent_profile_status")

            # -- Tools & Safety sub-view -------------------------------------
            with Vertical(id="agent_tools_view"):
                yield Static(
                    "Control which tools the orchestrator can use and set safety policies.",
                    id="agent_tools_help",
                )
                yield SidebarHeader("Tools & Safety", id="agent_tools_header")
                yield SidebarList(id="agent_tools_list")
                yield SidebarDetail(id="agent_tools_detail")
                yield Static("Session Overrides", id="agent_tools_overrides_header")
                yield Input(
                    placeholder="tool_consent: ask | allow | deny",
                    id="agent_tools_override_consent",
                )
                yield Input(
                    placeholder="tool_allowlist: tool1, tool2, ...",
                    id="agent_tools_override_allowlist",
                )
                yield Input(
                    placeholder="tool_blocklist: tool1, tool2, ...",
                    id="agent_tools_override_blocklist",
                )
                with Horizontal(id="agent_tools_override_actions"):
                    yield Button("Apply", id="agent_tools_overrides_apply", compact=True, variant="success")
                    yield Button("Reset", id="agent_tools_overrides_reset", compact=True, variant="default")
                yield Static("", id="agent_tools_override_status")

            # -- Team sub-view -----------------------------------------------
            with Vertical(id="agent_team_view"):
                yield Static(
                    "Define multi-agent presets the orchestrator can invoke via the swarm tool.",
                    id="agent_team_help",
                )
                yield SidebarHeader("Team Presets", id="agent_team_header")
                yield SidebarList(id="agent_team_list")
                yield SidebarDetail(id="agent_team_detail")
                yield Static("Preset Editor", id="agent_team_editor_header")
                with Horizontal(id="agent_team_meta_row"):
                    yield Input(placeholder="Preset id", id="agent_team_preset_id")
                    yield Input(placeholder="Preset name", id="agent_team_preset_name")
                yield Input(placeholder="Description (optional)", id="agent_team_preset_description")
                yield TextArea(
                    text="{}",
                    language="json",
                    id="agent_team_preset_spec",
                    soft_wrap=True,
                )
                with Horizontal(id="agent_team_actions"):
                    yield Button("New", id="agent_team_new", compact=True, variant="default")
                    yield Button("Save", id="agent_team_save", compact=True, variant="success")
                    yield Button("Delete", id="agent_team_delete", compact=True, variant="warning")
                    yield Button("Insert Run Prompt", id="agent_team_insert_prompt", compact=True, variant="primary")
                    yield Button("Run Now", id="agent_team_run_now", compact=True, variant="default")
                yield Static("", id="agent_team_status")


def wire_agents_widgets(app: Any) -> None:
    """Bind Agents tab widgets onto app fields."""
    app._agent_view_profile_button = app.query_one("#agent_view_profile", Button)
    app._agent_view_tools_button = app.query_one("#agent_view_tools", Button)
    app._agent_view_team_button = app.query_one("#agent_view_team", Button)
    app._agent_profile_view = app.query_one("#agent_profile_view", Vertical)
    app._agent_tools_view = app.query_one("#agent_tools_view", Vertical)
    app._agent_team_view = app.query_one("#agent_team_view", Vertical)
    app._agent_summary = app.query_one("#agent_summary", TextArea)
    app._agent_profile_list = app.query_one("#agent_profile_list", SidebarList)
    app._agent_tools_header = app.query_one("#agent_tools_header", SidebarHeader)
    app._agent_tools_list = app.query_one("#agent_tools_list", SidebarList)
    app._agent_tools_detail = app.query_one("#agent_tools_detail", SidebarDetail)
    app._agent_tools_override_consent_input = app.query_one("#agent_tools_override_consent", Input)
    app._agent_tools_override_allowlist_input = app.query_one("#agent_tools_override_allowlist", Input)
    app._agent_tools_override_blocklist_input = app.query_one("#agent_tools_override_blocklist", Input)
    app._agent_tools_override_status = app.query_one("#agent_tools_override_status", Static)
    app._agent_team_header = app.query_one("#agent_team_header", SidebarHeader)
    app._agent_team_list = app.query_one("#agent_team_list", SidebarList)
    app._agent_team_detail = app.query_one("#agent_team_detail", SidebarDetail)
    app._agent_team_preset_id_input = app.query_one("#agent_team_preset_id", Input)
    app._agent_team_preset_name_input = app.query_one("#agent_team_preset_name", Input)
    app._agent_team_preset_description_input = app.query_one("#agent_team_preset_description", Input)
    app._agent_team_preset_spec_input = app.query_one("#agent_team_preset_spec", TextArea)
    app._agent_team_status = app.query_one("#agent_team_status", Static)
    app._agent_profile_id_input = app.query_one("#agent_profile_id", Input)
    app._agent_profile_name_input = app.query_one("#agent_profile_name", Input)
    app._agent_profile_status = app.query_one("#agent_profile_status", Static)

__all__ = ["compose_agents_tab", "wire_agents_widgets"]
