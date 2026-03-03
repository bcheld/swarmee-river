from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarmee_river.diagnostics import broker_log_path, session_events_template
from swarmee_river.state_paths import scope_root, state_dir


@dataclass(frozen=True)
class RuntimeDiscovery:
    host: str
    port: int
    token: str
    pid: int | None = None
    started_at: str | None = None
    schema_version: str | None = None
    broker_log_path: str | None = None
    session_events_path: str | None = None


def default_session_id_for_cwd(cwd: Path) -> str:
    resolved = scope_root(cwd=cwd)
    digest = hashlib.sha1(str(resolved).encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"cwd-{digest}"


def runtime_discovery_path(*, cwd: Path | None = None) -> Path:
    return state_dir(cwd=cwd) / "runtime.json"


def _runtime_start_lock_path(*, cwd: Path | None = None) -> Path:
    return state_dir(cwd=cwd) / "runtime.start.lock"


@contextlib.contextmanager
def _runtime_start_lock(*, cwd: Path | None = None) -> Any:
    path = _runtime_start_lock_path(cwd=cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    locked = False
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        elif os.name == "nt":
            import msvcrt

            with contextlib.suppress(Exception):
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            locked = True
        yield
    finally:
        if locked:
            if os.name == "posix":
                import fcntl

                with contextlib.suppress(Exception):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif os.name == "nt":
                import msvcrt

                with contextlib.suppress(Exception):
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        with contextlib.suppress(Exception):
            handle.close()


def _discovery_is_reachable(path: Path, *, timeout_s: float = 0.5) -> bool:
    if not path.exists():
        return False
    try:
        discovery = load_runtime_discovery(path)
    except Exception:
        return False
    try:
        sock = socket.create_connection((discovery.host, discovery.port), timeout=max(0.1, float(timeout_s)))
    except OSError:
        return False
    with contextlib.suppress(Exception):
        sock.close()
    return True


def _kill_stale_broker_process(discovery_path: Path) -> None:
    """Best-effort kill of the broker process recorded in a stale discovery file."""
    try:
        discovery = load_runtime_discovery(discovery_path)
    except Exception:
        return
    pid = discovery.pid
    if pid is None or pid <= 0:
        return
    try:
        os.kill(pid, 0)  # check if process exists
    except OSError:
        return
    import signal as _signal

    with contextlib.suppress(Exception):
        os.kill(pid, _signal.SIGTERM)
    # Give it a moment to exit before we proceed.
    for _ in range(10):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except OSError:
            return  # process exited
    # Still alive — force kill.
    with contextlib.suppress(Exception):
        os.kill(pid, _signal.SIGKILL)


def ensure_runtime_broker(
    *,
    cwd: Path | None = None,
    timeout_s: float = 6.0,
    poll_interval_s: float = 0.1,
) -> Path:
    """
    Ensure a runtime broker is running for the current state scope.

    Starts `python -m swarmee_river.swarmee serve` in the background when no
    reachable broker is discovered, then waits for discovery readiness.
    """
    discovery = runtime_discovery_path(cwd=cwd)
    if _discovery_is_reachable(discovery):
        return discovery

    with _runtime_start_lock(cwd=cwd):
        # Re-check inside lock in case another process already started it.
        if _discovery_is_reachable(discovery):
            return discovery

        # Kill the stale broker process before starting a new one.
        _kill_stale_broker_process(discovery)
        with contextlib.suppress(Exception):
            if discovery.exists():
                discovery.unlink()

        resolved_state_dir = state_dir(cwd=cwd)
        resolved_state_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["SWARMEE_STATE_DIR"] = str(resolved_state_dir)
        env["SWARMEE_BROKER_LOG_PATH"] = str(broker_log_path(cwd=cwd))
        env["SWARMEE_DIAG_SESSION_EVENTS_PATH"] = session_events_template(cwd=cwd)

        broker_log = broker_log_path(cwd=cwd)
        broker_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = broker_log.open("a", encoding="utf-8", errors="replace")
        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "env": env,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "swarmee_river.swarmee", "serve"],
                **popen_kwargs,
            )
        finally:
            with contextlib.suppress(Exception):
                log_handle.close()

        deadline = time.monotonic() + max(0.5, float(timeout_s))
        interval = max(0.02, float(poll_interval_s))
        while time.monotonic() < deadline:
            if _discovery_is_reachable(discovery):
                return discovery
            if process.poll() is not None:
                raise RuntimeError(
                    f"runtime broker exited during startup (code {process.returncode}); see {broker_log}"
                )
            time.sleep(interval)

        raise RuntimeError(f"runtime broker did not become ready within {timeout_s:.1f}s; see {broker_log}")


def shutdown_runtime_broker(*, cwd: Path | None = None, timeout_s: float = 6.0) -> bool:
    """
    Request runtime broker shutdown for the current scope.

    Returns True if shutdown is confirmed (broker no longer reachable), else False.
    """
    discovery_path = runtime_discovery_path(cwd=cwd)
    if not discovery_path.exists():
        return False
    if not _discovery_is_reachable(discovery_path):
        _kill_stale_broker_process(discovery_path)
        with contextlib.suppress(Exception):
            discovery_path.unlink()
        return True

    client = RuntimeServiceClient.from_discovery_file(discovery_path, timeout_s=2.0)
    try:
        client.connect()
        hello = client.hello(client_name="swarmee-runtime-control", surface="control")
        if not isinstance(hello, dict) or str(hello.get("event", "")).strip().lower() != "hello_ack":
            return False
        client.send_command({"cmd": "shutdown_service"})
    except Exception:
        return False
    finally:
        client.close()

    deadline = time.monotonic() + max(0.5, float(timeout_s))
    while time.monotonic() < deadline:
        if not _discovery_is_reachable(discovery_path):
            with contextlib.suppress(Exception):
                discovery_path.unlink()
            return True
        time.sleep(0.1)
    return False


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
    schema_version = str(payload.get("schema_version", "")).strip() or None
    broker_log = str(payload.get("broker_log_path", "")).strip() or None
    session_events = str(payload.get("session_events_path", "")).strip() or None
    return RuntimeDiscovery(
        host=host,
        port=port,
        token=token,
        pid=pid,
        started_at=started_at,
        schema_version=schema_version,
        broker_log_path=broker_log,
        session_events_path=session_events,
    )


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
        # Shutdown the socket first to interrupt any blocking recv() in reader
        # threads.  BufferedReader.close() needs the buffer lock, but a blocked
        # readline() holds it — shutdown(SHUT_RDWR) causes recv() to return
        # immediately, releasing the lock so close() can proceed.
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
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

    def attach(
        self,
        *,
        session_id: str,
        cwd: str,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {"cmd": "attach", "session_id": session_id, "cwd": cwd}
        if env_overrides:
            payload["env_overrides"] = dict(env_overrides)
        self.send_command(payload)
        return self.read_event()

    def shutdown_service(self) -> dict[str, Any] | None:
        self.send_command({"cmd": "shutdown_service"})
        return self.read_event()
