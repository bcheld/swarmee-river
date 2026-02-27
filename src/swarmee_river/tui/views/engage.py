"""Run sidebar tab UI composition and widget wiring."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, TabPane, TextArea, Static

from swarmee_river.tui.widgets import PlanActions, SidebarDetail, SidebarHeader, SidebarList


def compose_engage_tab() -> Iterator[Any]:
    """Yield the Run tab pane.

    Sub-views:
    - Execution: orchestrator status + plan display (default).
    - Planning: interactive plan development (start/continue).
    - Session: timeline and issues.
    """
    with TabPane("Run", id="tab_engage"):
        with Vertical(id="engage_panel"):
            with Horizontal(id="engage_view_switch"):
                yield Button("Execution", id="engage_view_execution", compact=True, variant="primary")
                yield Button("Planning", id="engage_view_planning", compact=True, variant="default")
                yield Button("Session", id="engage_view_session", compact=True, variant="default")

            # -- Execution sub-view (default) --------------------------------
            with Vertical(id="engage_execution_view"):
                yield Static("Orchestrator", id="engage_orchestrator_status")
                yield TextArea(
                    text="No active plan. Enter a prompt to get started,\nor switch to Planning to develop a plan interactively.",
                    language="markdown",
                    read_only=True,
                    show_cursor=False,
                    id="plan",
                    soft_wrap=True,
                )
                yield PlanActions(id="plan_actions")

            # -- Planning sub-view -------------------------------------------
            with Vertical(id="engage_planning_view"):
                yield Static(
                    "Describe what you want to build. The orchestrator will\n"
                    "develop a plan you can review and refine.",
                    id="engage_planning_header",
                )
                yield Button("Start Plan", id="engage_start_plan", variant="success", compact=True)
                yield Static("", id="engage_plan_summary")
                with VerticalScroll(id="engage_plan_items"):
                    pass
                yield Button("Continue", id="engage_continue_plan", variant="primary", compact=True)

            # -- Session sub-view --------------------------------------------
            with Vertical(id="engage_session_view"):
                with Vertical(id="session_panel"):
                    with Horizontal(id="session_view_switch"):
                        yield Button("Timeline", id="session_view_timeline", compact=True, variant="primary")
                        yield Button("Issues", id="session_view_issues", compact=True, variant="default")
                    with Vertical(id="session_timeline_view"):
                        yield SidebarHeader("Timeline", id="session_timeline_header")
                        yield SidebarList(id="session_timeline_list")
                        yield SidebarDetail(id="session_timeline_detail")
                    with Vertical(id="session_issues_view"):
                        yield SidebarHeader("Issues", id="session_issues_header")
                        yield SidebarList(id="session_issue_list")
                        yield SidebarDetail(id="session_issue_detail")


def wire_engage_widgets(app: Any) -> None:
    """Bind Run tab widgets onto app fields used by event handlers."""
    from textual.containers import Vertical, VerticalScroll
    from textual.widgets import Button

    app._engage_view_execution_button = app.query_one("#engage_view_execution", Button)
    app._engage_view_planning_button = app.query_one("#engage_view_planning", Button)
    app._engage_view_session_button = app.query_one("#engage_view_session", Button)
    app._engage_execution_view = app.query_one("#engage_execution_view", Vertical)
    app._engage_planning_view = app.query_one("#engage_planning_view", Vertical)
    app._engage_session_view = app.query_one("#engage_session_view", Vertical)
    app._engage_orchestrator_status = app.query_one("#engage_orchestrator_status", Static)
    app._engage_plan_summary = app.query_one("#engage_plan_summary", Static)
    app._engage_plan_items = app.query_one("#engage_plan_items", VerticalScroll)

    # Session widgets
    app._session_header = app.query_one("#session_issues_header", SidebarHeader)
    app._session_view_timeline_button = app.query_one("#session_view_timeline", Button)
    app._session_view_issues_button = app.query_one("#session_view_issues", Button)
    app._session_timeline_view = app.query_one("#session_timeline_view", Vertical)
    app._session_issues_view = app.query_one("#session_issues_view", Vertical)
    app._session_timeline_header = app.query_one("#session_timeline_header", SidebarHeader)
    app._session_timeline_list = app.query_one("#session_timeline_list", SidebarList)
    app._session_timeline_detail = app.query_one("#session_timeline_detail", SidebarDetail)
    app._session_issue_list = app.query_one("#session_issue_list", SidebarList)
    app._session_issue_detail = app.query_one("#session_issue_detail", SidebarDetail)

__all__ = ["compose_engage_tab", "wire_engage_widgets"]
