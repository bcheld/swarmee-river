from __future__ import annotations

import threading
from unittest import mock

from swarmee_river import interrupts


def test_pause_active_interrupt_watcher_for_input_calls_pause_resume(monkeypatch) -> None:
    class _FakeWatcher:
        def __init__(self) -> None:
            self.pause_calls = 0
            self.resume_calls = 0

        def pause(self) -> None:
            self.pause_calls += 1

        def resume(self) -> None:
            self.resume_calls += 1

    watcher = _FakeWatcher()
    monkeypatch.setattr(interrupts, "_ACTIVE_WATCHER", watcher)

    with interrupts.pause_active_interrupt_watcher_for_input():
        pass

    assert watcher.pause_calls == 1
    assert watcher.resume_calls == 1


def test_watcher_pause_resume_transitions_terminal_mode_once() -> None:
    watcher = interrupts.EscInterruptWatcher(threading.Event(), enabled=True)
    restore = mock.Mock()
    enable = mock.Mock()

    watcher._restore_terminal_mode = restore  # type: ignore[method-assign]
    watcher._enable_cbreak_mode = enable  # type: ignore[method-assign]

    watcher.pause()
    watcher.pause()
    watcher.resume()
    watcher.resume()

    assert restore.call_count == 1
    assert enable.call_count == 1
