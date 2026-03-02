"""
Agentic TUI test harness.

Provides MockTransport and TuiHarness, enabling scenario-based tests that drive
the real SwarmeeTUI app (via Textual's Pilot API) without spawning any subprocess.

Usage::

    @pytest.mark.asyncio
    async def test_something(tui_app_factory):
        async with tui_app_factory() as (app, pilot):
            # Emit a sequence of JSONL lines as if they came from the daemon
            app.state.daemon.proc.emit('{"type":"tui_event","kind":"ready"}\\n')
            await pilot.pause()
            ...

Fixtures exported
-----------------
tui_app_factory  – pytest fixture that yields an async context manager
"""

from __future__ import annotations

import json
import queue
import threading
from contextlib import asynccontextmanager
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# MockTransport
# ---------------------------------------------------------------------------


class MockTransport:
    """
    Drop-in replacement for _SubprocessTransport / _SocketTransport.

    The test drives the daemon output by calling ``emit(line)``; the TUI
    streaming thread reads those lines via ``read_line()``.  The transport
    stays "alive" (poll() → None) until ``close()`` is called.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._closed = False
        self._commands: list[dict[str, Any]] = []
        self._close_event = threading.Event()

    # --- _DaemonTransport interface ---

    @property
    def pid(self) -> int:
        return 0

    def poll(self) -> int | None:
        return None if not self._closed else 0

    def wait(self, timeout: float | None = None) -> int:
        self._close_event.wait(timeout=timeout)
        return 0

    def read_line(self) -> str:
        """Block until a line is available or the transport is closed."""
        while True:
            try:
                return self._q.get(timeout=0.05)
            except queue.Empty:
                if self._closed:
                    return ""

    def send_command(self, cmd_dict: dict[str, Any]) -> bool:
        self._commands.append(cmd_dict)
        return True

    def close(self) -> None:
        self._closed = True
        self._close_event.set()
        # Unblock any waiting read_line()
        self._q.put("")

    # --- Test helpers ---

    def emit(self, line: str) -> None:
        """Feed a raw line to the TUI's output reader thread."""
        self._q.put(line if line.endswith("\n") else line + "\n")

    def emit_event(self, event: dict[str, Any]) -> None:
        """Serialise *event* as JSONL and feed it to the TUI."""
        self.emit(json.dumps(event, ensure_ascii=False))

    def emit_ready(self, *, session_id: str = "test-session") -> None:
        """Emit the standard daemon-ready handshake."""
        self.emit_event({"event": "ready", "session_id": session_id})

    @property
    def sent_commands(self) -> list[dict[str, Any]]:
        return list(self._commands)


# ---------------------------------------------------------------------------
# App factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tui_app_factory(tmp_path: Any, monkeypatch: Any):
    """
    Pytest fixture that returns an async context manager.

    Example::

        @pytest.mark.asyncio
        async def test_basic(tui_app_factory):
            async with tui_app_factory() as (app, pilot, transport):
                transport.emit_ready()
                await pilot.pause(delay=0.1)
                assert app.state.daemon.ready
    """
    # Isolate filesystem state
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))
    monkeypatch.setenv("SWARMEE_PREFLIGHT", "disabled")
    monkeypatch.setenv("SWARMEE_PROJECT_MAP", "disabled")

    @asynccontextmanager
    async def _factory(*, size: tuple[int, int] = (200, 50)):
        # Import here so test collection doesn't fail if textual is missing.
        from swarmee_river.tui.app import get_swarmee_tui_class

        SwarmeeTUI = get_swarmee_tui_class()

        transport = MockTransport()

        class _TestApp(SwarmeeTUI):
            _test_transport_factory = staticmethod(lambda: transport)

        app = _TestApp()
        async with app.run_test(size=size) as pilot:
            yield app, pilot, transport

    return _factory
