from __future__ import annotations

import os
import platform
import sys
import threading
import time
from contextlib import AbstractContextManager
from contextlib import contextmanager
from typing import Optional


class AgentInterruptedError(RuntimeError):
    """Raised to abort the current agent invocation (e.g., when Esc is pressed)."""


_ACTIVE_WATCHER_LOCK = threading.Lock()
_ACTIVE_WATCHER: "EscInterruptWatcher | None" = None


class EscInterruptWatcher(AbstractContextManager["EscInterruptWatcher"]):
    """
    Watches stdin for the Esc key while the agent is running and sets `interrupt_event` when detected.

    Implementation notes:
    - Windows: uses `msvcrt.kbhit/getwch`.
    - POSIX: temporarily switches stdin into cbreak mode and uses `select`.
    """

    def __init__(self, interrupt_event: threading.Event, *, enabled: bool = True, poll_interval_s: float = 0.05):
        self._interrupt_event = interrupt_event
        self._enabled = enabled
        self._poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pause_count = 0
        self._pause_lock = threading.Lock()

        self._is_windows = platform.system() == "Windows"
        self._stdin_fd: Optional[int] = None
        self._termios_old: Optional[object] = None

    def __enter__(self) -> "EscInterruptWatcher":
        global _ACTIVE_WATCHER
        if not self._enabled:
            return self

        self._stop_event.clear()

        if self._is_windows:
            self._thread = threading.Thread(target=self._watch_windows, daemon=True)
            self._thread.start()
            return self

        try:
            import termios
            import tty

            self._stdin_fd = sys.stdin.fileno()
            self._termios_old = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
        except Exception:
            # If we can't switch modes (non-tty, CI, etc), still run the watcher best-effort.
            self._stdin_fd = None
            self._termios_old = None

        self._thread = threading.Thread(target=self._watch_posix, daemon=True)
        self._thread.start()
        with _ACTIVE_WATCHER_LOCK:
            _ACTIVE_WATCHER = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _ACTIVE_WATCHER
        if not self._enabled:
            return None

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.2)

        if not self._is_windows and self._stdin_fd is not None and self._termios_old is not None:
            try:
                import termios

                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._termios_old)
            except Exception:
                pass
        with _ACTIVE_WATCHER_LOCK:
            if _ACTIVE_WATCHER is self:
                _ACTIVE_WATCHER = None

        return None

    def pause(self) -> None:
        with self._pause_lock:
            self._pause_count += 1

    def resume(self) -> None:
        with self._pause_lock:
            if self._pause_count > 0:
                self._pause_count -= 1

    def _is_paused(self) -> bool:
        with self._pause_lock:
            return self._pause_count > 0

    def _watch_windows(self) -> None:
        try:
            import msvcrt
        except Exception:
            return

        while not self._stop_event.is_set() and not self._interrupt_event.is_set():
            if self._is_paused():
                time.sleep(self._poll_interval_s)
                continue
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch == "\x1b":
                        self._interrupt_event.set()
                        return
            except Exception:
                return
            time.sleep(self._poll_interval_s)

    def _watch_posix(self) -> None:
        try:
            import select
        except Exception:
            return

        while not self._stop_event.is_set() and not self._interrupt_event.is_set():
            if self._is_paused():
                time.sleep(self._poll_interval_s)
                continue
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], self._poll_interval_s)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":
                        self._interrupt_event.set()
                        return
            except Exception:
                return


@contextmanager
def pause_active_interrupt_watcher_for_input():
    watcher: EscInterruptWatcher | None = None
    with _ACTIVE_WATCHER_LOCK:
        watcher = _ACTIVE_WATCHER
    if watcher is None:
        yield
        return

    watcher.pause()
    try:
        yield
    finally:
        watcher.resume()


def interrupt_watcher_from_env(interrupt_event: threading.Event) -> EscInterruptWatcher:
    enabled = (sys.stdin is not None) and sys.stdin.isatty() and (platform.system() in {"Windows", "Darwin", "Linux"})
    # Allow disabling in automation/CI, or when Esc conflicts with terminal flows.
    if os.getenv("SWARMEE_ESC_INTERRUPT", "enabled").strip().lower() in {
        "0",
        "false",
        "off",
        "disabled",
    }:
        enabled = False
    return EscInterruptWatcher(interrupt_event, enabled=enabled)
