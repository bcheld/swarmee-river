"""ActionSheet visibility on constrained terminals (doc 13 F4).

The action sheet hosts consent and plan-review controls; clipping it off
screen hides the approve/deny buttons themselves.
"""

from __future__ import annotations

import pytest

from tests.tui_harness import tui_app_factory  # noqa: F401  (fixture)


async def _wait_for(condition, *, pilot, attempts: int = 40, delay: float = 0.05) -> bool:
    for _ in range(attempts):
        if condition():
            return True
        await pilot.pause(delay=delay)
    return condition()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 20), (80, 24)], ids=["60x20", "80x24"])
async def test_action_sheet_panel_fits_on_screen(tui_app_factory, size):
    async with tui_app_factory(size=size) as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        app.action_open_action_sheet()
        await pilot.pause(delay=0.2)

        sheet = app.query_one("ActionSheet")
        assert sheet.is_visible

        panel = app.query_one("#action_sheet_panel")
        region = panel.region
        width, height = size
        assert region.width > 0 and region.height > 0
        assert region.x >= 0 and region.y >= 0
        assert region.x + region.width <= width, "panel clipped horizontally"
        assert region.y + region.height <= height, "panel clipped vertically"
