"""Sidebar UI composition helpers for the TUI app."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Select, Static, TabbedContent, TabPane, TextArea

from swarmee_river.tui.views.agent_studio import compose_agent_studio_tab
from swarmee_river.tui.views.session import compose_session_tab
from swarmee_river.tui.widgets import PlanActions, SidebarDetail, SidebarHeader, SidebarList


def compose_sidebar(*, context_select_placeholder: str) -> Iterator[Any]:
    """Yield the right-hand sidebar, including all tab panes."""
    with Vertical(id="side"):
        with TabbedContent(id="side_tabs"):
            with TabPane("Plan", id="tab_plan"):
                yield TextArea(
                    text="",
                    language="markdown",
                    read_only=True,
                    show_cursor=False,
                    id="plan",
                    soft_wrap=True,
                )
                yield PlanActions(id="plan_actions")
            with TabPane("Context", id="tab_context"):
                with Vertical(id="context_panel"):
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
            with TabPane("SOPs", id="tab_sops"):
                with Vertical(id="sops_panel"):
                    yield Static("Available SOPs", id="sops_header")
                    yield VerticalScroll(id="sop_list")
            with TabPane("Artifacts", id="tab_artifacts"):
                with Vertical(id="artifacts_panel"):
                    yield SidebarHeader("Artifacts", id="artifacts_header")
                    yield SidebarList(id="artifacts_list")
                    yield SidebarDetail(id="artifacts_detail")
            yield from compose_session_tab()
            yield from compose_agent_studio_tab()


__all__ = ["compose_sidebar"]
