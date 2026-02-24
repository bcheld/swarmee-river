from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from queue import Queue
from typing import Any

import pytest

from swarmee_river.runtime_service.server import RuntimeServiceServer


async def _read_event(reader: asyncio.StreamReader) -> dict[str, Any]:
    raw = await asyncio.wait_for(reader.readline(), timeout=2.0)
    assert raw, "expected event line from runtime service"
    parsed = json.loads(raw.decode("utf-8", errors="replace"))
    assert isinstance(parsed, dict)
    return parsed


async def _start_service_or_skip(service: RuntimeServiceServer) -> None:
    try:
        await service.start()
    except PermissionError as exc:
        pytest.skip(f"Socket bind blocked in this environment: {exc}")


class _FakeStdout:
    def __init__(self) -> None:
        self._queue: Queue[str] = Queue()
        self._closed = False

    def feed(self, line: str) -> None:
        if self._closed:
            return
        payload = line if line.endswith("\n") else f"{line}\n"
        self._queue.put(payload)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put("")

    def readline(self) -> str:
        return self._queue.get()


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.closed = False

    def write(self, data: str) -> int:
        if self.closed:
            raise BrokenPipeError("stdin closed")
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        if self.closed:
            raise BrokenPipeError("stdin closed")

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    def __init__(self, args: list[str], kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout()
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            if timeout is not None:
                raise subprocess.TimeoutExpired(self.args, timeout)
            self.returncode = 0
        return int(self.returncode)

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.close()

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()

    def send_signal(self, _sig: int) -> None:
        self.returncode = 0
        self.stdout.close()


class _FakePopenFactory:
    def __init__(self) -> None:
        self.instances: list[_FakePopen] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _FakePopen:
        command = list(args[0]) if args else []
        proc = _FakePopen(command, kwargs)
        self.instances.append(proc)
        return proc


def test_runtime_service_startup_and_basic_commands(monkeypatch, tmp_path: Path) -> None:
    state_root = tmp_path / ".swarmee"
    token = "test-token"

    async def _scenario() -> None:
        import swarmee_river.runtime_service.server as server_module

        monkeypatch.setattr(server_module, "state_dir", lambda: state_root)
        fake_popen = _FakePopenFactory()
        monkeypatch.setattr(server_module.subprocess, "Popen", fake_popen)
        service = RuntimeServiceServer(port=0, token=token)
        runtime_file = state_root / "runtime.json"
        await _start_service_or_skip(service)
        try:
            assert service.port > 0
            assert runtime_file.exists()
            discovery = json.loads(runtime_file.read_text(encoding="utf-8"))
            assert discovery["host"] == "127.0.0.1"
            assert int(discovery["port"]) == service.port
            assert discovery["token"] == token
            assert int(discovery["pid"]) > 0
            assert isinstance(discovery["started_at"], str) and discovery["started_at"]

            reader, writer = await asyncio.open_connection("127.0.0.1", service.port)
            try:
                writer.write(b"{bad json}\n")
                await writer.drain()
                event = await _read_event(reader)
                assert event["event"] == "error"
                assert event["code"] == "invalid_json"

                writer.write(json.dumps({"cmd": "ping"}).encode("utf-8") + b"\n")
                await writer.drain()
                event = await _read_event(reader)
                assert event["event"] == "error"
                assert event["code"] == "unauthorized"

                writer.write(
                    json.dumps(
                        {"cmd": "hello", "token": token, "client_name": "unit-test", "surface": "tests"},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                await writer.drain()
                event = await _read_event(reader)
                assert event["event"] == "hello_ack"
                assert event["status"] == "ok"

                writer.write(
                    json.dumps({"cmd": "attach", "session_id": "s-1", "cwd": str(tmp_path)}).encode("utf-8") + b"\n"
                )
                await writer.drain()
                event = await _read_event(reader)
                assert event["event"] == "attached"
                assert event["session_id"] == "s-1"

                writer.write(json.dumps({"cmd": "ping"}).encode("utf-8") + b"\n")
                await writer.drain()
                event = await _read_event(reader)
                assert event["event"] == "pong"
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            await service.stop()

        assert not runtime_file.exists()

    asyncio.run(_scenario())


def test_runtime_service_attach_tracks_per_session_clients(monkeypatch, tmp_path: Path) -> None:
    state_root = tmp_path / ".swarmee"
    token = "attach-token"
    session_id = "session-1"
    attach_cwd = tmp_path / "repo"
    attach_cwd.mkdir(parents=True, exist_ok=True)

    async def _hello_and_attach(
        *,
        port: int,
        cwd: str,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, dict[str, Any]]:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            json.dumps({"cmd": "hello", "token": token, "client_name": "test", "surface": "tests"}).encode("utf-8")
            + b"\n"
        )
        await writer.drain()
        hello_event = await _read_event(reader)
        assert hello_event["event"] == "hello_ack"

        writer.write(json.dumps({"cmd": "attach", "session_id": session_id, "cwd": cwd}).encode("utf-8") + b"\n")
        await writer.drain()
        attached_event = await _read_event(reader)
        assert attached_event["event"] == "attached"
        return reader, writer, attached_event

    async def _scenario() -> None:
        import swarmee_river.runtime_service.server as server_module

        monkeypatch.setattr(server_module, "state_dir", lambda: state_root)
        fake_popen = _FakePopenFactory()
        monkeypatch.setattr(server_module.subprocess, "Popen", fake_popen)
        service = RuntimeServiceServer(port=0, token=token)
        await _start_service_or_skip(service)
        try:
            reader1, writer1, attach1 = await _hello_and_attach(port=service.port, cwd=str(attach_cwd))
            assert attach1["session_id"] == session_id
            assert attach1["clients"] == 1

            reader2, writer2, attach2 = await _hello_and_attach(port=service.port, cwd=str(attach_cwd))
            assert attach2["session_id"] == session_id
            assert attach2["clients"] == 2
            assert session_id in service.sessions
            assert len(service.sessions[session_id].client_ids) == 2

            writer1.close()
            await writer1.wait_closed()
            writer2.close()
            await writer2.wait_closed()
            del reader1
            del reader2
        finally:
            await service.stop()

    asyncio.run(_scenario())


def test_runtime_service_query_broadcast_and_controller_consent(monkeypatch, tmp_path: Path) -> None:
    state_root = tmp_path / ".swarmee"
    token = "stream-token"
    session_id = "shared"
    attach_cwd = tmp_path / "workspace"
    attach_cwd.mkdir(parents=True, exist_ok=True)

    async def _connect_and_attach(
        *,
        port: int,
        client_name: str,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            json.dumps(
                {"cmd": "hello", "token": token, "client_name": client_name, "surface": "tests"},
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        await writer.drain()
        hello_event = await _read_event(reader)
        assert hello_event["event"] == "hello_ack"

        writer.write(
            json.dumps({"cmd": "attach", "session_id": session_id, "cwd": str(attach_cwd)}, ensure_ascii=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        await writer.drain()
        attached_event = await _read_event(reader)
        assert attached_event["event"] == "attached"
        return reader, writer

    async def _scenario() -> None:
        import swarmee_river.runtime_service.server as server_module

        monkeypatch.setattr(server_module, "state_dir", lambda: state_root)
        fake_popen = _FakePopenFactory()
        monkeypatch.setattr(server_module.subprocess, "Popen", fake_popen)

        service = RuntimeServiceServer(port=0, token=token)
        await _start_service_or_skip(service)
        try:
            reader1, writer1 = await _connect_and_attach(port=service.port, client_name="controller")
            reader2, writer2 = await _connect_and_attach(port=service.port, client_name="observer")

            assert len(fake_popen.instances) == 1
            proc = fake_popen.instances[0]

            writer1.write(json.dumps({"cmd": "query", "text": "hello runtime"}).encode("utf-8") + b"\n")
            await writer1.drain()
            await asyncio.sleep(0.05)
            assert proc.stdin.writes, "query command should be forwarded to daemon stdin"
            forwarded_query = json.loads(proc.stdin.writes[-1].strip())
            assert forwarded_query["cmd"] == "query"
            assert forwarded_query["text"] == "hello runtime"

            proc.stdout.feed(json.dumps({"event": "text_delta", "text": "hello"}))
            event1 = await _read_event(reader1)
            event2 = await _read_event(reader2)
            assert event1["event"] == "text_delta"
            assert event2["event"] == "text_delta"
            assert event1["session_id"] == session_id
            assert event2["session_id"] == session_id

            proc.stdout.feed("plain daemon output")
            warn1 = await _read_event(reader1)
            warn2 = await _read_event(reader2)
            assert warn1["event"] == "warning"
            assert warn2["event"] == "warning"
            assert "plain daemon output" in warn1["text"]
            assert "plain daemon output" in warn2["text"]

            proc.stdout.feed(json.dumps({"event": "consent_prompt", "context": "Allow tool?"}))
            consent1 = await _read_event(reader1)
            consent2 = await _read_event(reader2)
            assert consent1["event"] == "consent_prompt"
            assert consent2["event"] == "consent_prompt"

            writer2.write(json.dumps({"cmd": "consent_response", "choice": "y"}).encode("utf-8") + b"\n")
            await writer2.drain()
            denied = await _read_event(reader2)
            assert denied["event"] == "error"
            assert denied["code"] == "not_controller"

            writes_before = len(proc.stdin.writes)
            writer1.write(json.dumps({"cmd": "consent_response", "choice": "y"}).encode("utf-8") + b"\n")
            await writer1.drain()
            await asyncio.sleep(0.05)
            assert len(proc.stdin.writes) == writes_before + 1
            forwarded_consent = json.loads(proc.stdin.writes[-1].strip())
            assert forwarded_consent == {"cmd": "consent_response", "choice": "y"}

            proc.stdout.feed(json.dumps({"event": "turn_complete", "exit_status": "ok"}))
            done1 = await _read_event(reader1)
            done2 = await _read_event(reader2)
            assert done1["event"] == "turn_complete"
            assert done2["event"] == "turn_complete"
            assert service.sessions[session_id].controller_client_id is None
            assert service.sessions[session_id].consent_pending is False

            writer1.close()
            await writer1.wait_closed()
            writer2.close()
            await writer2.wait_closed()
        finally:
            await service.stop()

    asyncio.run(_scenario())


def test_runtime_service_set_profile_requires_idle_query_and_proxies(monkeypatch, tmp_path: Path) -> None:
    state_root = tmp_path / ".swarmee"
    token = "profile-token"
    session_id = "shared-profile"
    attach_cwd = tmp_path / "workspace"
    attach_cwd.mkdir(parents=True, exist_ok=True)

    async def _connect_and_attach(
        *,
        port: int,
        client_name: str,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            json.dumps(
                {"cmd": "hello", "token": token, "client_name": client_name, "surface": "tests"},
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        await writer.drain()
        hello_event = await _read_event(reader)
        assert hello_event["event"] == "hello_ack"

        writer.write(
            json.dumps({"cmd": "attach", "session_id": session_id, "cwd": str(attach_cwd)}, ensure_ascii=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        await writer.drain()
        attached_event = await _read_event(reader)
        assert attached_event["event"] == "attached"
        return reader, writer

    async def _scenario() -> None:
        import swarmee_river.runtime_service.server as server_module

        monkeypatch.setattr(server_module, "state_dir", lambda: state_root)
        fake_popen = _FakePopenFactory()
        monkeypatch.setattr(server_module.subprocess, "Popen", fake_popen)

        service = RuntimeServiceServer(port=0, token=token)
        await _start_service_or_skip(service)
        try:
            reader1, writer1 = await _connect_and_attach(port=service.port, client_name="controller")
            reader2, writer2 = await _connect_and_attach(port=service.port, client_name="observer")

            assert len(fake_popen.instances) == 1
            proc = fake_popen.instances[0]

            writer1.write(json.dumps({"cmd": "query", "text": "run"}).encode("utf-8") + b"\n")
            await writer1.drain()
            await asyncio.sleep(0.05)
            assert proc.stdin.writes
            forwarded_query = json.loads(proc.stdin.writes[-1].strip())
            assert forwarded_query["cmd"] == "query"

            writes_before = len(proc.stdin.writes)
            writer2.write(
                json.dumps({"cmd": "set_profile", "profile": {"id": "qa", "name": "QA"}}).encode("utf-8") + b"\n"
            )
            await writer2.drain()
            denied = await _read_event(reader2)
            assert denied["event"] == "error"
            assert denied["code"] == "query_active"
            assert len(proc.stdin.writes) == writes_before

            proc.stdout.feed(json.dumps({"event": "turn_complete", "exit_status": "ok"}))
            done1 = await _read_event(reader1)
            done2 = await _read_event(reader2)
            assert done1["event"] == "turn_complete"
            assert done2["event"] == "turn_complete"

            writes_before = len(proc.stdin.writes)
            writer2.write(
                json.dumps(
                    {
                        "cmd": "set_profile",
                        "profile": {"id": "qa", "name": "QA", "tier": "deep", "active_sops": ["review"]},
                    }
                ).encode("utf-8")
                + b"\n"
            )
            await writer2.drain()
            await asyncio.sleep(0.05)
            assert len(proc.stdin.writes) == writes_before + 1
            forwarded_profile = json.loads(proc.stdin.writes[-1].strip())
            assert forwarded_profile == {
                "cmd": "set_profile",
                "profile": {"id": "qa", "name": "QA", "tier": "deep", "active_sops": ["review"]},
            }

            writes_before = len(proc.stdin.writes)
            writer2.write(
                json.dumps(
                    {
                        "cmd": "set_safety_overrides",
                        "tool_consent": "deny",
                        "tool_allowlist": ["file_read"],
                        "tool_blocklist": ["shell"],
                    }
                ).encode("utf-8")
                + b"\n"
            )
            await writer2.drain()
            await asyncio.sleep(0.05)
            assert len(proc.stdin.writes) == writes_before + 1
            forwarded_safety = json.loads(proc.stdin.writes[-1].strip())
            assert forwarded_safety == {
                "cmd": "set_safety_overrides",
                "overrides": {
                    "tool_consent": "deny",
                    "tool_allowlist": ["file_read"],
                    "tool_blocklist": ["shell"],
                },
            }

            writer1.close()
            await writer1.wait_closed()
            writer2.close()
            await writer2.wait_closed()
        finally:
            await service.stop()

    asyncio.run(_scenario())
