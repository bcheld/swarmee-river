"""Logged, counted replacement for blanket ``contextlib.suppress(Exception)``.

The TUI historically suppressed widget-query and rendering failures
silently, which let selector typos and lifecycle races ship unnoticed
(e.g. the plan Continue button bug). ``ui_guard()`` keeps the defensive
behavior — the UI must not crash on a failed refresh — but every failure
now leaves a trail: a WARNING with traceback on first occurrence per call
site, a DEBUG line afterwards, and a per-site counter that a diagnostics
view can surface.

Usage::

    with ui_guard():            # label derived from the failure site
        self.query_one("#plan", TextArea).read_only = True

    with ui_guard("plan.visibility"):   # explicit label
        ...
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from pathlib import Path
from types import TracebackType

_LOGGER = logging.getLogger("swarmee_river.tui.ui_guard")

_counts: Counter[str] = Counter()
_counts_lock = threading.Lock()


def ui_guard_failure_counts() -> dict[str, int]:
    """Snapshot of suppressed-failure counts keyed by call-site label."""
    with _counts_lock:
        return dict(_counts)


def reset_ui_guard_failure_counts() -> None:
    with _counts_lock:
        _counts.clear()


def _label_from_traceback(tb: TracebackType | None) -> str:
    if tb is None:
        return "unknown"
    frame = tb.tb_frame
    return f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}:{tb.tb_lineno}"


class ui_guard:  # noqa: N801 - context manager used like contextlib.suppress
    __slots__ = ("_label",)

    def __init__(self, label: str | None = None) -> None:
        self._label = label

    def __enter__(self) -> "ui_guard":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is None:
            return False
        # Never swallow cancellation/exit signals.
        if not issubclass(exc_type, Exception):
            return False
        label = self._label or _label_from_traceback(tb)
        with _counts_lock:
            _counts[label] += 1
            occurrence = _counts[label]
        if occurrence == 1:
            _LOGGER.warning("suppressed UI failure at %s: %r", label, exc, exc_info=(exc_type, exc, tb))
        else:
            _LOGGER.debug("suppressed UI failure at %s (x%d): %r", label, occurrence, exc)
        return True
