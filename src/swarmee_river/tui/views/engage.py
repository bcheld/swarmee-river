"""Run sidebar tab UI composition and widget wiring."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Static, TabPane, TextArea

from swarmee_river.tui.widgets import SidebarDetail, SidebarHeader


def compose_engage_tab() -> Iterator[Any]:
    """Yield the Run tab pane.

    Sub-views:
    - Plan: orchestrator status + plan display + interactive planning controls (default).
    - Session: timeline and artifacts.
    """
    with TabPane("Run", id="tab_engage"):
        with Vertical(id="engage_panel"):
            with Horizontal(id="engage_view_switch"):
                yield Button("Plan", id="engage_view_plan", compact=True, variant="primary")
                yield Button("Session", id="engage_view_session", compact=True, variant="default")

            # -- Plan sub-view (default) -------------------------------------
            with Vertical(id="engage_plan_view"):
                yield Static("Orchestrator", id="engage_orchestrator_status")
                yield TextArea(
                    text="",
                    placeholder="Type your initial planning query here, then press Start Plan.",
                    read_only=False,
                    show_cursor=True,
                    id="plan",
                    soft_wrap=True,
                )
                yield Static(
                    "Planning controls",
                    id="engage_planning_header",
                )
                yield Button("Start Plan", id="engage_start_plan", variant="success", compact=True)
                yield Static("", id="engage_plan_summary")
                with VerticalScroll(id="engage_plan_items"):
                    pass
                with VerticalScroll(id="engage_plan_questions"):
                    pass
                with Horizontal(id="engage_plan_actions_row"):
                    yield Button("Continue", id="engage_continue_plan", variant="primary", compact=True)
                    yield Button("Clear", id="engage_clear_plan", variant="default", compact=True)
                    yield Button("Cancel", id="engage_cancel_plan", variant="warning", compact=True)

            # -- Session sub-view --------------------------------------------
            with Vertical(id="engage_session_view"):
                with Vertical(id="session_panel"):
                    with Horizontal(id="session_view_switch"):
                        yield Button("Timeline", id="session_view_timeline", compact=True, variant="primary")
                        yield Button("Artifacts", id="session_view_artifacts", compact=True, variant="default")
                    with Vertical(id="session_timeline_view"):
                        yield SidebarHeader("Timeline", id="session_timeline_header")
                        yield DataTable(id="session_timeline_table", cursor_type="row")
                        yield SidebarDetail(id="session_timeline_detail")
                    with Vertical(id="session_artifacts_view"):
                        yield SidebarHeader("Artifacts", id="session_artifacts_header")
                        yield DataTable(id="session_artifacts_table", cursor_type="row")
                        yield SidebarDetail(id="session_artifacts_detail")


def wire_engage_widgets(app: Any) -> None:
    """Bind Run tab widgets onto app fields used by event handlers."""
    from textual.containers import Vertical, VerticalScroll
    from textual.widgets import Button

    app._engage_view_plan_button = app.query_one("#engage_view_plan", Button)
    app._engage_view_session_button = app.query_one("#engage_view_session", Button)
    app._engage_plan_view = app.query_one("#engage_plan_view", Vertical)
    app._engage_session_view = app.query_one("#engage_session_view", Vertical)
    app._engage_orchestrator_status = app.query_one("#engage_orchestrator_status", Static)
    app._engage_plan_summary = app.query_one("#engage_plan_summary", Static)
    app._engage_plan_questions = app.query_one("#engage_plan_questions", VerticalScroll)
    app._engage_plan_items = app.query_one("#engage_plan_items", VerticalScroll)

    # Session widgets
    app._session_view_timeline_button = app.query_one("#session_view_timeline", Button)
    app._session_view_artifacts_button = app.query_one("#session_view_artifacts", Button)
    app._session_timeline_view = app.query_one("#session_timeline_view", Vertical)
    app._session_artifacts_view = app.query_one("#session_artifacts_view", Vertical)
    app._session_timeline_header = app.query_one("#session_timeline_header", SidebarHeader)
    app._session_timeline_table = app.query_one("#session_timeline_table", DataTable)
    app._session_timeline_detail = app.query_one("#session_timeline_detail", SidebarDetail)
    app._session_artifacts_header = app.query_one("#session_artifacts_header", SidebarHeader)
    app._session_artifacts_table = app.query_one("#session_artifacts_table", DataTable)
    app._session_artifacts_detail = app.query_one("#session_artifacts_detail", SidebarDetail)

    # Backward-compat aliases for app.py code that still references old names.
    # Artifact widgets now live in session; these map to the new locations.
    app._artifacts_header = app._session_artifacts_header
    app._artifacts_table = app._session_artifacts_table
    app._artifacts_detail = app._session_artifacts_detail
    # Issues widgets are removed; set to None so guarded accesses pass.
    app._session_header = app._session_timeline_header
    app._session_view_issues_button = None
    app._session_issues_view = None
    app._session_issue_list = None
    app._session_issue_detail = None


__all__ = ["compose_engage_tab", "wire_engage_widgets"]
