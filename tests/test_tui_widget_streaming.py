"""Performance regression tests for streaming transcript widgets.

Guards the streaming render coalescing: re-rendering the full accumulated
markdown on every delta is O(n²) per message and must not come back.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from swarmee_river.tui.widgets import (
    AssistantMessage,
    AssistantStreamBlock,
    ReasoningBlock,
)


class _HostApp(App):
    def __init__(self, widget) -> None:
        super().__init__()
        self._widget = widget

    def compose(self) -> ComposeResult:
        yield self._widget


def _count_renders(widget) -> list[int]:
    counter = [0]
    original = widget._render_stream_content

    def counting() -> None:
        counter[0] += 1
        original()

    widget._render_stream_content = counting
    return counter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [AssistantMessage, AssistantStreamBlock, ReasoningBlock],
    ids=["assistant_message", "assistant_stream_block", "reasoning_block"],
)
async def test_streaming_renders_are_coalesced(factory) -> None:
    """200 rapid deltas must trigger only a handful of renders, not 200."""
    widget = factory()
    app = _HostApp(widget)
    async with app.run_test() as pilot:
        renders = _count_renders(widget)
        for index in range(200):
            widget.append_delta(f"word{index} ")
        # Allow the trailing coalesced render to flush.
        await pilot.pause(widget.STREAM_RENDER_INTERVAL_S * 2)
        # One immediate render plus a small number of interval flushes.
        assert renders[0] <= 5, f"expected coalesced renders, got {renders[0]}"
        assert "".join(widget._buffer) == "".join(f"word{index} " for index in range(200))


@pytest.mark.asyncio
async def test_finalize_renders_full_content_after_coalescing() -> None:
    """finalize() must render the complete text even if a flush was pending."""
    widget = AssistantMessage()
    app = _HostApp(widget)
    async with app.run_test() as pilot:
        renders = _count_renders(widget)
        widget.append_delta("hello ")
        widget.append_delta("world")
        full = widget.finalize()
        assert full == "hello world"
        # The pending coalesced flush was cancelled; no late re-render fires.
        before = renders[0]
        await pilot.pause(widget.STREAM_RENDER_INTERVAL_S * 2)
        assert renders[0] == before


def test_append_delta_works_without_event_loop() -> None:
    """Unmounted widgets (no timers available) fall back to direct rendering."""
    widget = AssistantMessage()
    widget.append_delta("alpha ")
    widget.append_delta("beta")
    assert widget.full_text == "alpha beta"
