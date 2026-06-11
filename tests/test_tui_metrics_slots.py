"""Usage/context indicator delivery tests (doc 12 F3).

Indicator events are last-write-wins state, not transcript content: a burst
of output must never leave the indicators stuck on a stale value, and the
final displayed value must equal the last event on the wire.
"""

from __future__ import annotations

import json

import pytest

from swarmee_river.tui.mixins.output import metrics_event_kind
from tests.tui_harness import tui_app_factory  # noqa: F401  (fixture)


async def _wait_for(condition, *, pilot, attempts: int = 60, delay: float = 0.05) -> bool:
    for _ in range(attempts):
        if condition():
            return True
        await pilot.pause(delay=delay)
    return condition()


def test_metrics_event_kind_detection() -> None:
    assert metrics_event_kind('{"event": "usage", "usage": {"input_tokens": 1}}') == "usage"
    assert metrics_event_kind('{"event": "context", "prompt_tokens_est": 5}') == "context"
    # Lines that merely mention the words must not be misrouted.
    assert metrics_event_kind('{"event": "text_delta", "text": "check \\"usage\\" docs"}') is None
    assert metrics_event_kind("plain text mentioning usage") is None
    assert metrics_event_kind('{"event": "usage"') is None  # malformed JSON


@pytest.mark.asyncio
async def test_indicator_shows_last_usage_event_after_output_flood(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready()
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        # Interleave a large transcript flood with usage updates.
        for index in range(50):
            for _ in range(20):
                transport.emit("tool output line\n")
            transport.emit(
                json.dumps(
                    {
                        "event": "usage",
                        "usage": {"input_tokens": 100 + index, "output_tokens": index},
                        "cost_usd": 0.001 * (index + 1),
                    }
                )
                + "\n"
            )

        reached = await _wait_for(
            lambda: isinstance(app.state.daemon.last_usage, dict)
            and app.state.daemon.last_usage.get("input_tokens") == 149,
            pilot=pilot,
        )
        assert reached, (
            f"indicator stuck on stale usage: {app.state.daemon.last_usage!r}"
        )
        assert app.state.daemon.last_cost_usd == pytest.approx(0.05)
