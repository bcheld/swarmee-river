"""Tests for the logged ui_guard suppression helper (doc 17 F1)."""

from __future__ import annotations

import logging

import pytest

from swarmee_river.tui.ui_guard import (
    reset_ui_guard_failure_counts,
    ui_guard,
    ui_guard_failure_counts,
)


@pytest.fixture(autouse=True)
def _clean_counts():
    reset_ui_guard_failure_counts()
    yield
    reset_ui_guard_failure_counts()


def test_suppresses_and_counts_exceptions() -> None:
    for _ in range(3):
        with ui_guard("test.label"):
            raise ValueError("boom")
    assert ui_guard_failure_counts() == {"test.label": 3}


def test_derives_label_from_failure_site() -> None:
    with ui_guard():
        raise RuntimeError("nope")
    (label,) = ui_guard_failure_counts()
    assert label.startswith("test_ui_guard.py:test_derives_label_from_failure_site:")


def test_logs_warning_on_first_failure_then_debug(caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="swarmee_river.tui.ui_guard"):
        with ui_guard("noisy"):
            raise ValueError("first")
        with ui_guard("noisy"):
            raise ValueError("second")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(warnings) == 1
    assert len(debugs) == 1


def test_does_not_swallow_base_exceptions() -> None:
    with pytest.raises(KeyboardInterrupt):
        with ui_guard("signals"):
            raise KeyboardInterrupt
    assert ui_guard_failure_counts() == {}


def test_no_failure_counts_on_success() -> None:
    with ui_guard("fine"):
        pass
    assert ui_guard_failure_counts() == {}


def test_diagnostics_ui_command_reports_counters(monkeypatch) -> None:
    from swarmee_river.tui.mixins.daemon import DaemonMixin

    with ui_guard("plan.visibility"):
        raise ValueError("widget missing")

    class _Host:
        _thread_dispatch_dropped_total = 2
        _output_lines_dropped_total = 7
        _pending_output_lines = None

        def __init__(self) -> None:
            self.lines: list[str] = []

        _write_ui_diagnostics = DaemonMixin._write_ui_diagnostics

        def _write_transcript_line(self, text: str) -> None:
            self.lines.append(text)

    host = _Host()
    host._write_ui_diagnostics()
    output = "\n".join(host.lines)
    assert "plan.visibility" in output
    assert "Dropped cross-thread dispatches: 2" in output
    assert "Output lines dropped under load: 7" in output
