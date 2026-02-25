"""Agents sidebar tab UI composition and widget wiring."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Select, Static, TabPane, TextArea

from swarmee_river.tui.widgets import SidebarDetail, SidebarHeader, SidebarList


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
                yield TextArea(
                    text="",
                    read_only=True,
                    show_cursor=False,
                    id="agent_summary",
                    soft_wrap=True,
                )
                yield SidebarHeader("Activated Agents", id="agent_overview_header")
                yield SidebarList(id="agent_overview_list")
                yield SidebarDetail(id="agent_overview_detail")
                yield Static("", id="agent_overview_status")

            with Vertical(id="agent_builder_view"):
                yield Static("Profile", id="agent_builder_profile_header")
                with Horizontal(id="agent_builder_profile_row"):
                    yield Select(
                        options=[("Current draft / session", "__agent_profile_none__")],
                        allow_blank=False,
                        id="agent_profile_select",
                    )
                    yield Button(
                        "New from Current Settings",
                        id="agent_profile_new_from_current",
                        compact=True,
                        variant="default",
                    )
                with Horizontal(id="agent_profile_meta_row"):
                    yield Input(placeholder="Profile id", id="agent_profile_id")
                    yield Input(placeholder="Profile name", id="agent_profile_name")
                with Horizontal(id="agent_profile_actions"):
                    yield Button("New Draft", id="agent_profile_new", compact=True, variant="default")
                    yield Button("Save Profile", id="agent_profile_save", compact=True, variant="success")
                    yield Button("Delete Profile", id="agent_profile_delete", compact=True, variant="warning")
                    yield Button("Apply Profile", id="agent_profile_apply", compact=True, variant="primary")
                yield Checkbox("Assistive auto delegation", value=True, id="agent_builder_auto_delegate")
                yield Static("", id="agent_profile_status")

                yield Static("Agent Roster", id="agent_builder_agents_header")
                yield SidebarList(id="agent_builder_agent_list")
                yield SidebarDetail(id="agent_builder_agent_detail")
                with Horizontal(id="agent_builder_agent_meta_row"):
                    yield Input(placeholder="Agent id", id="agent_builder_agent_id")
                    yield Input(placeholder="Agent name", id="agent_builder_agent_name")
                yield Input(placeholder="Summary", id="agent_builder_agent_summary")
                yield TextArea(
                    text="",
                    id="agent_builder_agent_prompt",
                    soft_wrap=True,
                )
                with Horizontal(id="agent_builder_model_row"):
                    yield Select(
                        options=[
                            ("Model provider: inherit", "__inherit__"),
                            ("bedrock", "bedrock"),
                            ("openai", "openai"),
                            ("ollama", "ollama"),
                            ("github_copilot", "github_copilot"),
                        ],
                        allow_blank=False,
                        id="agent_builder_agent_provider",
                        compact=True,
                    )
                    yield Select(
                        options=[
                            ("Model tier: inherit", "__inherit__"),
                            ("fast", "fast"),
                            ("balanced", "balanced"),
                            ("deep", "deep"),
                            ("long", "long"),
                        ],
                        allow_blank=False,
                        id="agent_builder_agent_tier",
                        compact=True,
                    )
                yield Input(
                    placeholder="Tools (comma-separated, blank = inherit)",
                    id="agent_builder_agent_tools",
                )
                yield Input(
                    placeholder="SOPs (comma-separated)",
                    id="agent_builder_agent_sops",
                )
                yield Input(
                    placeholder="Knowledge base id (optional)",
                    id="agent_builder_agent_kb",
                )
                yield Checkbox("Activated", value=False, id="agent_builder_agent_activated")
                with Horizontal(id="agent_builder_agent_actions"):
                    yield Button("New Agent", id="agent_builder_agent_new", compact=True, variant="default")
                    yield Button("Save Agent", id="agent_builder_agent_save", compact=True, variant="success")
                    yield Button("Delete Agent", id="agent_builder_agent_delete", compact=True, variant="warning")
                    yield Button("Insert Run Prompt", id="agent_builder_insert_prompt", compact=True, variant="primary")
                    yield Button("Run Now", id="agent_builder_run_now", compact=True, variant="default")
                yield Static("", id="agent_builder_status")


def wire_agents_widgets(app: Any) -> None:
    """Bind Agents tab widgets onto app fields."""
    app._agent_view_overview_button = app.query_one("#agent_view_overview", Button)
    app._agent_view_builder_button = app.query_one("#agent_view_builder", Button)
    app._agent_overview_view = app.query_one("#agent_overview_view", Vertical)
    app._agent_builder_view = app.query_one("#agent_builder_view", Vertical)
    app._agent_summary = app.query_one("#agent_summary", TextArea)
    app._agent_overview_header = app.query_one("#agent_overview_header", SidebarHeader)
    app._agent_overview_list = app.query_one("#agent_overview_list", SidebarList)
    app._agent_overview_detail = app.query_one("#agent_overview_detail", SidebarDetail)
    app._agent_overview_status = app.query_one("#agent_overview_status", Static)

    app._agent_profile_select = app.query_one("#agent_profile_select", Select)
    app._agent_profile_id_input = app.query_one("#agent_profile_id", Input)
    app._agent_profile_name_input = app.query_one("#agent_profile_name", Input)
    app._agent_profile_status = app.query_one("#agent_profile_status", Static)
    app._agent_builder_auto_delegate_checkbox = app.query_one("#agent_builder_auto_delegate", Checkbox)

    app._agent_builder_list = app.query_one("#agent_builder_agent_list", SidebarList)
    app._agent_builder_detail = app.query_one("#agent_builder_agent_detail", SidebarDetail)
    app._agent_builder_agent_id_input = app.query_one("#agent_builder_agent_id", Input)
    app._agent_builder_agent_name_input = app.query_one("#agent_builder_agent_name", Input)
    app._agent_builder_agent_summary_input = app.query_one("#agent_builder_agent_summary", Input)
    app._agent_builder_agent_prompt_input = app.query_one("#agent_builder_agent_prompt", TextArea)
    app._agent_builder_agent_provider_select = app.query_one("#agent_builder_agent_provider", Select)
    app._agent_builder_agent_tier_select = app.query_one("#agent_builder_agent_tier", Select)
    app._agent_builder_agent_tools_input = app.query_one("#agent_builder_agent_tools", Input)
    app._agent_builder_agent_sops_input = app.query_one("#agent_builder_agent_sops", Input)
    app._agent_builder_agent_kb_input = app.query_one("#agent_builder_agent_kb", Input)
    app._agent_builder_agent_activated_checkbox = app.query_one("#agent_builder_agent_activated", Checkbox)
    app._agent_builder_status = app.query_one("#agent_builder_status", Static)

    # Keep compatibility with older app.py attribute names.
    app._agent_view_profile_button = app._agent_view_overview_button
    app._agent_view_tools_button = None
    app._agent_view_team_button = None
    app._agent_profile_view = app._agent_overview_view
    app._agent_tools_view = None
    app._agent_team_view = None
    app._agent_profile_list = None
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
