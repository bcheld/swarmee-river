"""Sidebar UI composition helpers for the TUI app."""

from __future__ import annotations

from typing import Any, Iterator

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Select, Static, TabbedContent, TabPane, TextArea

from swarmee_river.tui.views.engage import compose_engage_tab
from swarmee_river.tui.views.agents import compose_agents_tab
from swarmee_river.tui.views.scaffold import compose_scaffold_tab
from swarmee_river.tui.views.settings import compose_settings_tab


def compose_sidebar(*, context_select_placeholder: str) -> Iterator[Any]:
    """Yield the right-hand sidebar, including all tab panes."""
    with Vertical(id="side"):
        with TabbedContent(id="side_tabs"):
            yield from compose_engage_tab()
            yield from compose_agents_tab()
            yield from compose_scaffold_tab(context_select_placeholder=context_select_placeholder)
            yield from compose_settings_tab()


__all__ = ["compose_sidebar"]
