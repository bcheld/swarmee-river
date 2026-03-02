from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from swarmee_river.runtime_service.client import RuntimeServiceClient
from swarmee_river.runtime_service.server import RuntimeServiceServer, SessionState


class _FakeStdout:
    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()
        self._closed = False

    def push_event(self, event: dict[str, Any]) -> None:
        self._lines.put(json.dumps(event, ensure_ascii=False) + "\n")

    def readline(self) -> str:
        while True:
            if self._closed and self._lines.empty():
                return ""
            try:
                return self._lines.get(timeout=0.1)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed = True


class _FakeDaemonProcess:
    _pid_seq = 30000

    class _Stdin:
        def __init__(self, owner: _FakeDaemonProcess) -> None:
            self._owner = owner
            self._buffer = ""

        def write(self, text: str) -> int:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                payload = line.strip()
                if not payload:
                    continue
                try:
                    command = json.loads(payload)
                except Exception:
                    self._owner.stdout.push_event({"event": "warning", "text": f"Invalid command JSON: {payload}"})
                    continue
                if isinstance(command, dict):
                    self._owner.handle_command(command)
            return len(text)

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    def __init__(self, session_id: str) -> None:
        type(self)._pid_seq += 1
        self.pid = type(self)._pid_seq
        self.session_id = session_id
        self.stdout = _FakeStdout()
        self.stdin = _FakeDaemonProcess._Stdin(self)
        self._return_code: int | None = None
        self._ready_emitted = False

    def poll(self) -> int | None:
        return self._return_code

    def wait(self, timeout: float | None = None) -> int:
        if self._return_code is not None:
            return self._return_code
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while self._return_code is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(cmd=["fake-daemon"], timeout=timeout)
            time.sleep(0.01)
        return self._return_code

    def terminate(self) -> None:
        self._shutdown()

    def kill(self) -> None:
        self._shutdown()

    def send_signal(self, _sig: int) -> None:
        self._shutdown()

    def _shutdown(self) -> None:
        self._return_code = 0
        self.stdout.close()

    def handle_command(self, command: dict[str, Any]) -> None:
        cmd = str(command.get("cmd", "")).strip().lower()
        if cmd == "query":
            text = str(command.get("text", "")).strip()
            if not self._ready_emitted:
                self._ready_emitted = True
                self.stdout.push_event({"event": "ready", "session_id": self.session_id})
            self.stdout.push_event({"event": "text_delta", "text": f"echo:{text}"})
            self.stdout.push_event({"event": "consent_prompt", "context": "Allow fake tool?"})
            return
        if cmd == "consent_response":
            choice = str(command.get("choice", "")).strip().lower()
            self.stdout.push_event({"event": "text_delta", "text": f"consent:{choice}"})
            self.stdout.push_event({"event": "turn_complete", "exit_status": "ok"})
            return
        if cmd == "shutdown":
            self._shutdown()
            return
        self.stdout.push_event({"event": "warning", "text": f"Unhandled fake cmd: {cmd}"})


class _ClientReader:
    def __init__(self, client: RuntimeServiceClient) -> None:
        self.client = client
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()
        self.errors: list[BaseException] = []
        self.seen: list[dict[str, Any]] = []

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self.client.read_event()
            except BaseException as exc:  # noqa: BLE001
                self.errors.append(exc)
                break
            if event is None:
                break
            self._queue.put(event)

    def wait_for(self, predicate: Callable[[dict[str, Any]], bool], *, timeout_s: float = 8.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.errors:
                raise AssertionError(f"Client reader error: {self.errors[0]}")
            remaining = max(0.01, deadline - time.monotonic())
            try:
                event = self._queue.get(timeout=min(0.2, remaining))
            except queue.Empty:
                continue
            self.seen.append(event)
            if predicate(event):
                return event
        raise AssertionError(f"Timed out waiting for event; seen={self.seen}")

    def close(self) -> None:
        self._stop.set()
        self.client.close()
        self._thread.join(timeout=1.0)


def test_runtime_broker_smoke_broadcast_and_controller_gating(monkeypatch, tmp_path: Path) -> None:
    async def _fake_start_session_process(self: RuntimeServiceServer, session: SessionState) -> None:
        process = _FakeDaemonProcess(session.session_id)
        session.process = process
        session.started_new_session = False
        session.stdout_task = asyncio.create_task(
            self._session_stdout_reader(session=session, process=process),
            name=f"test-runtime-session-{session.session_id}-stdout",
        )

    monkeypatch.setattr(RuntimeServiceServer, "_start_session_process", _fake_start_session_process)

    runtime_file = tmp_path / "runtime.json"
    token = "smoke-token"
    started = threading.Event()
    stop_requested = threading.Event()
    holder: dict[str, Any] = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server = RuntimeServiceServer(port=0, token=token, runtime_file=runtime_file)
        holder["loop"] = loop
        holder["server"] = server
        try:
            try:
                loop.run_until_complete(server.start())
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc
                started.set()
                return
            started.set()
            while not stop_requested.is_set():
                loop.run_until_complete(asyncio.sleep(0.05))
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(server.stop())
            loop.close()

    thread = threading.Thread(target=_worker, daemon=True, name="runtime-service-smoke")
    thread.start()
    assert started.wait(timeout=5.0), "runtime broker did not start in time"
    error = holder.get("error")
    if isinstance(error, PermissionError):
        pytest.skip(f"Socket bind blocked in this environment: {error}")
    if error is not None:
        raise AssertionError(f"runtime broker failed to start: {error}") from error
    server = holder["server"]
    assert isinstance(server, RuntimeServiceServer)

    client1 = RuntimeServiceClient.from_discovery_file(runtime_file)
    client2 = RuntimeServiceClient.from_discovery_file(runtime_file)
    reader1: _ClientReader | None = None
    reader2: _ClientReader | None = None
    session_id = "smoke-session"
    cwd = str(Path.cwd().resolve())
    try:
        client1.connect()
        client2.connect()

        hello1 = client1.hello(client_name="smoke-c1", surface="test")
        hello2 = client2.hello(client_name="smoke-c2", surface="test")
        assert hello1 is not None and str(hello1.get("event", "")).lower() == "hello_ack"
        assert hello2 is not None and str(hello2.get("event", "")).lower() == "hello_ack"

        attach1 = client1.attach(session_id=session_id, cwd=cwd)
        attach2 = client2.attach(session_id=session_id, cwd=cwd)
        assert attach1 is not None and str(attach1.get("event", "")).lower() == "attached"
        assert attach2 is not None and str(attach2.get("event", "")).lower() == "attached"

        reader1 = _ClientReader(client1)
        reader2 = _ClientReader(client2)
        reader1.start()
        reader2.start()

        client1.send_command({"cmd": "query", "text": "smoke"})

        ready1 = reader1.wait_for(lambda event: str(event.get("event", "")).lower() == "ready")
        ready2 = reader2.wait_for(lambda event: str(event.get("event", "")).lower() == "ready")
        assert ready1.get("session_id") == session_id
        assert ready2.get("session_id") == session_id

        delta1 = reader1.wait_for(lambda event: str(event.get("event", "")).lower() == "text_delta")
        delta2 = reader2.wait_for(lambda event: str(event.get("event", "")).lower() == "text_delta")
        assert delta1.get("text") == delta2.get("text")

        _ = reader1.wait_for(lambda event: str(event.get("event", "")).lower() == "consent_prompt")
        _ = reader2.wait_for(lambda event: str(event.get("event", "")).lower() == "consent_prompt")

        client2.send_command({"cmd": "consent_response", "choice": "y"})
        non_controller_error = reader2.wait_for(
            lambda event: (
                str(event.get("event", "")).lower() == "error"
                and str(event.get("code", "")).lower() == "not_controller"
            )
        )
        assert "controller" in str(non_controller_error.get("message", "")).lower()

        client1.send_command({"cmd": "consent_response", "choice": "y"})
        complete1 = reader1.wait_for(lambda event: str(event.get("event", "")).lower() == "turn_complete")
        complete2 = reader2.wait_for(lambda event: str(event.get("event", "")).lower() == "turn_complete")
        assert str(complete1.get("exit_status", "")).lower() == "ok"
        assert str(complete2.get("exit_status", "")).lower() == "ok"
    finally:
        if reader1 is not None:
            reader1.close()
        else:
            client1.close()
        if reader2 is not None:
            reader2.close()
        else:
            client2.close()
        stop_requested.set()
        loop = holder.get("loop")
        if loop is not None:
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(lambda: None)
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "runtime broker thread did not stop"
