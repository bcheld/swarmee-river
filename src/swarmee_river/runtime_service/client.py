from __future__ import annotations

import hashlib
import json
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarmee_river.state_paths import state_dir


@dataclass(frozen=True)
class RuntimeDiscovery:
    host: str
    port: int
    token: str
    pid: int | None = None
    started_at: str | None = None


def default_session_id_for_cwd(cwd: Path) -> str:
    resolved = cwd.expanduser().resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"cwd-{digest}"


def runtime_discovery_path(*, cwd: Path | None = None) -> Path:
    return state_dir(cwd=cwd) / "runtime.json"


def load_runtime_discovery(path: Path) -> RuntimeDiscovery:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid runtime discovery payload in {path}")

    host = str(payload.get("host", "")).strip()
    token = str(payload.get("token", "")).strip()
    raw_port = payload.get("port")
    try:
        port = int(raw_port)
    except Exception as exc:
        raise ValueError(f"Invalid runtime port in {path}") from exc

    if not host:
        raise ValueError(f"Missing host in runtime discovery file: {path}")
    if not token:
        raise ValueError(f"Missing token in runtime discovery file: {path}")
    if port <= 0:
        raise ValueError(f"Invalid runtime port in runtime discovery file: {path}")

    raw_pid = payload.get("pid")
    pid: int | None = None
    if isinstance(raw_pid, int):
        pid = raw_pid
    elif isinstance(raw_pid, str) and raw_pid.strip().isdigit():
        pid = int(raw_pid.strip())

    started_at = str(payload.get("started_at", "")).strip() or None
    return RuntimeDiscovery(host=host, port=port, token=token, pid=pid, started_at=started_at)


class RuntimeServiceClient:
    def __init__(self, *, discovery: RuntimeDiscovery, timeout_s: float = 10.0) -> None:
        self.discovery = discovery
        self.timeout_s = max(0.1, float(timeout_s))
        self._sock: socket.socket | None = None
        self._reader: Any | None = None
        self._writer: Any | None = None
        self._write_lock = threading.Lock()

    @classmethod
    def from_discovery_file(cls, path: Path, *, timeout_s: float = 10.0) -> RuntimeServiceClient:
        return cls(discovery=load_runtime_discovery(path), timeout_s=timeout_s)

    @classmethod
    def from_repo_state(cls, *, cwd: Path | None = None, timeout_s: float = 10.0) -> RuntimeServiceClient:
        return cls.from_discovery_file(runtime_discovery_path(cwd=cwd), timeout_s=timeout_s)

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.discovery.host, self.discovery.port), timeout=self.timeout_s)
        sock.settimeout(None)
        self._sock = sock
        self._reader = sock.makefile("r", encoding="utf-8", errors="replace", newline="\n")
        self._writer = sock.makefile("w", encoding="utf-8", errors="replace", newline="\n")

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def send_command(self, payload: dict[str, Any]) -> None:
        writer = self._writer
        if writer is None:
            raise RuntimeError("Runtime client is not connected")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._write_lock:
            writer.write(line)
            writer.flush()

    def read_event(self) -> dict[str, Any] | None:
        reader = self._reader
        if reader is None:
            raise RuntimeError("Runtime client is not connected")
        raw_line = reader.readline()
        if raw_line == "":
            return None
        text = raw_line.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"event": "warning", "text": f"Invalid JSON event from runtime: {text[:400]}"}
        if isinstance(parsed, dict):
            return parsed
        return {"event": "warning", "text": f"Non-object runtime event: {text[:400]}"}

    def hello(self, *, client_name: str, surface: str) -> dict[str, Any] | None:
        self.send_command(
            {
                "cmd": "hello",
                "token": self.discovery.token,
                "client_name": client_name,
                "surface": surface,
            }
        )
        return self.read_event()

    def attach(self, *, session_id: str, cwd: str) -> dict[str, Any] | None:
        self.send_command({"cmd": "attach", "session_id": session_id, "cwd": cwd})
        return self.read_event()
