from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from swarmee_river.runtime_service.client import (
    RuntimeDiscovery,
    RuntimeServiceClient,
    default_session_id_for_cwd,
    ensure_runtime_broker,
    load_runtime_discovery,
    runtime_discovery_path,
    shutdown_runtime_broker,
)


def test_default_session_id_for_cwd_is_stable(tmp_path: Path) -> None:
    session_a = default_session_id_for_cwd(tmp_path)
    session_b = default_session_id_for_cwd(tmp_path)
    assert session_a == session_b
    assert session_a.startswith("cwd-")
    assert len(session_a) > 8


def test_load_runtime_discovery_round_trip(tmp_path: Path) -> None:
    discovery_path = tmp_path / "runtime.json"
    payload = {
        "schema_version": "2",
        "host": "127.0.0.1",
        "port": 7342,
        "token": "abc123",
        "pid": 999,
        "started_at": "2026-02-23T00:00:00Z",
        "broker_log_path": "/tmp/broker.log",
        "session_events_path": "/tmp/sessions/{session_id}.jsonl",
    }
    discovery_path.write_text(json.dumps(payload), encoding="utf-8")

    discovery = load_runtime_discovery(discovery_path)
    assert discovery.host == "127.0.0.1"
    assert discovery.port == 7342
    assert discovery.token == "abc123"
    assert discovery.pid == 999
    assert discovery.started_at == "2026-02-23T00:00:00Z"
    assert discovery.schema_version == "2"
    assert discovery.broker_log_path == "/tmp/broker.log"
    assert discovery.session_events_path == "/tmp/sessions/{session_id}.jsonl"


def test_ensure_runtime_broker_returns_existing_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))
    monkeypatch.setattr("swarmee_river.runtime_service.client._discovery_is_reachable", lambda _path: True)
    monkeypatch.setattr("swarmee_river.runtime_service.client.subprocess.Popen", lambda *_a, **_k: None)

    discovery = ensure_runtime_broker(cwd=tmp_path)
    assert discovery == runtime_discovery_path(cwd=tmp_path)


def test_ensure_runtime_broker_spawns_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))
    reachable = iter([False, False, True])
    monkeypatch.setattr(
        "swarmee_river.runtime_service.client._discovery_is_reachable",
        lambda _path: next(reachable),
    )

    class _FakeProc:
        returncode = None

        def poll(self) -> int | None:
            return None

    spawned: dict[str, object] = {}

    def _fake_popen(cmd: list[str], **kwargs: object) -> _FakeProc:
        spawned["cmd"] = cmd
        spawned["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr("swarmee_river.runtime_service.client.subprocess.Popen", _fake_popen)

    discovery = ensure_runtime_broker(cwd=tmp_path, timeout_s=1.0, poll_interval_s=0.01)
    assert discovery == runtime_discovery_path(cwd=tmp_path)
    command = spawned["cmd"]
    assert isinstance(command, list)
    assert command[:1] == [sys.executable]
    assert command[1:] == ["-u", "-m", "swarmee_river.swarmee", "serve"]


def test_shutdown_runtime_broker_sends_shutdown_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))
    discovery_path = runtime_discovery_path(cwd=tmp_path)
    discovery_path.parent.mkdir(parents=True, exist_ok=True)
    discovery_path.write_text(
        json.dumps({"host": "127.0.0.1", "port": 1234, "token": "tok"}, ensure_ascii=False),
        encoding="utf-8",
    )

    reachable = iter([True, False])
    monkeypatch.setattr(
        "swarmee_river.runtime_service.client._discovery_is_reachable",
        lambda _path: next(reachable),
    )

    class _FakeClient:
        def __init__(self) -> None:
            self.commands: list[dict[str, object]] = []

        def connect(self) -> None:
            return None

        def close(self) -> None:
            return None

        def hello(self, *, client_name: str, surface: str) -> dict[str, str]:
            assert client_name == "swarmee-runtime-control"
            assert surface == "control"
            return {"event": "hello_ack"}

        def send_command(self, payload: dict[str, object]) -> None:
            self.commands.append(dict(payload))

    fake_client = _FakeClient()
    monkeypatch.setattr(
        "swarmee_river.runtime_service.client.RuntimeServiceClient.from_discovery_file",
        staticmethod(lambda _path, timeout_s=10.0: fake_client),
    )

    assert shutdown_runtime_broker(cwd=tmp_path, timeout_s=0.5) is True
    assert fake_client.commands == [{"cmd": "shutdown_service"}]


def test_runtime_client_close_does_not_block_on_stream_close() -> None:
    class _BlockingStream:
        def close(self) -> None:
            time.sleep(0.4)

    class _FakeSocket:
        def __init__(self) -> None:
            self.shutdown_called = False
            self.close_called = False

        def shutdown(self, _how: int) -> None:
            self.shutdown_called = True

        def close(self) -> None:
            self.close_called = True

    client = RuntimeServiceClient(
        discovery=RuntimeDiscovery(host="127.0.0.1", port=1, token="tok"),
        timeout_s=1.0,
    )
    fake_socket = _FakeSocket()
    client._sock = fake_socket  # type: ignore[assignment]
    client._reader = _BlockingStream()
    client._writer = _BlockingStream()

    started = time.monotonic()
    client.close()
    elapsed = time.monotonic() - started

    assert elapsed < 0.35
    assert fake_socket.shutdown_called is True
    assert fake_socket.close_called is True
    assert client._sock is None
    assert client._reader is None
    assert client._writer is None
