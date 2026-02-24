"""Session sidebar tab UI composition and widget wiring."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, TabPane

from swarmee_river.tui.widgets import SidebarDetail, SidebarHeader, SidebarList


def compose_session_tab() -> Iterator[Any]:
    """Yield the Session tab pane with timeline/issues subviews."""
    with TabPane("Session", id="tab_session"):
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


def wire_session_widgets(app: Any) -> None:
    """Bind Session tab widgets onto app fields used by event handlers."""
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


__all__ = ["compose_session_tab", "wire_session_widgets"]
