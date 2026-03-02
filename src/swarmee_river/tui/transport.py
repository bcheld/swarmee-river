"""Daemon transport and subprocess lifecycle helpers for the TUI."""

from __future__ import annotations

import contextlib
import json as _json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from swarmee_river.runtime_service.client import RuntimeServiceClient, runtime_discovery_path


class _DaemonTransport:
    @property
    def pid(self) -> int:
        raise NotImplementedError

    def poll(self) -> int | None:
        raise NotImplementedError

    def wait(self, timeout: float | None = None) -> int:
        raise NotImplementedError

    def read_line(self) -> str:
        raise NotImplementedError

    def send_command(self, cmd_dict: dict[str, Any]) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _SubprocessTransport(_DaemonTransport):
    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def poll(self) -> int | None:
        return self._proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        return int(self._proc.wait(timeout=timeout))

    def read_line(self) -> str:
        stdout = self._proc.stdout
        if stdout is None:
            return ""
        return stdout.readline()

    def send_command(self, cmd_dict: dict[str, Any]) -> bool:
        stdin = self._proc.stdin
        if stdin is None:
            return False
        try:
            payload = _json.dumps(cmd_dict, ensure_ascii=False) + "\n"
            stdin.write(payload)
            stdin.flush()
        except Exception:
            return False
        return True

    def close(self) -> None:
        stop_process(self._proc)


class _SocketTransport(_DaemonTransport):
    def __init__(
        self,
        *,
        client: RuntimeServiceClient,
        session_id: str,
        broker_pid: int | None = None,
        pending_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._broker_pid = int(broker_pid) if isinstance(broker_pid, int) else None
        self._pending_events = list(pending_events or [])
        self._closed = False
        self._poll_code: int | None = None

    @classmethod
    def connect(
        cls,
        *,
        session_id: str,
        cwd: Path,
        client_name: str,
        surface: str,
        runtime_discovery_path_fn: Callable[..., Path] = runtime_discovery_path,
        client_from_discovery_fn: Callable[[Path], RuntimeServiceClient] = RuntimeServiceClient.from_discovery_file,
    ) -> _SocketTransport:
        discovery = runtime_discovery_path_fn(cwd=cwd)
        if not discovery.exists():
            raise FileNotFoundError(f"Runtime discovery file not found: {discovery}")

        client = client_from_discovery_fn(discovery)
        client.connect()
        hello = client.hello(client_name=client_name, surface=surface) or {}
        if str(hello.get("event", "")).strip().lower() == "error":
            message = str(hello.get("message", hello)).strip() or "hello failed"
            client.close()
            raise RuntimeError(message)

        attach = client.attach(session_id=session_id, cwd=str(cwd)) or {}
        if str(attach.get("event", "")).strip().lower() == "error":
            message = str(attach.get("message", attach)).strip() or "attach failed"
            client.close()
            raise RuntimeError(message)

        broker_pid = hello.get("pid")
        return cls(
            client=client,
            session_id=session_id,
            broker_pid=(int(broker_pid) if isinstance(broker_pid, int) else None),
            pending_events=[attach] if isinstance(attach, dict) else [],
        )

    @property
    def pid(self) -> int:
        return int(self._broker_pid) if self._broker_pid is not None else -1

    def poll(self) -> int | None:
        return self._poll_code if self._closed else None

    def wait(self, timeout: float | None = None) -> int:
        start = time.monotonic()
        while not self._closed:
            if timeout is not None and (time.monotonic() - start) >= timeout:
                raise subprocess.TimeoutExpired(["runtime-socket"], timeout)
            time.sleep(0.01)
        return int(self._poll_code or 0)

    def read_line(self) -> str:
        if self._pending_events:
            event = self._pending_events.pop(0)
            return _json.dumps(event, ensure_ascii=False) + "\n"
        if self._closed:
            return ""
        event = self._client.read_event()
        if event is None:
            self._closed = True
            self._poll_code = 0
            return ""
        return _json.dumps(event, ensure_ascii=False) + "\n"

    def send_command(self, cmd_dict: dict[str, Any]) -> bool:
        if self._closed:
            return False
        payload = dict(cmd_dict)
        cmd = str(payload.get("cmd", "")).strip().lower()
        if cmd == "shutdown":
            payload = {"cmd": "shutdown_session"}
        try:
            self._client.send_command(payload)
        except Exception:
            self._closed = True
            self._poll_code = 1
            return False
        if str(payload.get("cmd", "")).strip().lower() == "shutdown_session":
            self.close()
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._poll_code = 0
        self._client.close()


def detect_consent_prompt(line: str) -> str | None:
    """Detect consent-related subprocess output lines."""
    normalized = line.strip().lower()
    if "~ consent>" in normalized:
        return "prompt"
    if "allow tool '" in normalized:
        return "header"
    return None


def update_consent_capture(
    consent_active: bool,
    consent_buffer: list[str],
    line: str,
    *,
    max_lines: int = 20,
) -> tuple[bool, list[str]]:
    """Update consent capture state from a single output line."""
    kind = detect_consent_prompt(line)
    if kind is None and not consent_active:
        return consent_active, consent_buffer

    updated = list(consent_buffer)
    updated.append(line.rstrip("\n"))
    if len(updated) > max_lines:
        updated = updated[-max_lines:]
    return True, updated


def write_to_proc(proc: Any, text: str) -> bool:
    """Write a response line to a subprocess stdin."""
    stdin = getattr(proc, "stdin", None)
    if stdin is None:
        return False

    payload = text if text.endswith("\n") else f"{text}\n"
    try:
        stdin.write(payload)
        stdin.flush()
    except Exception:
        return False
    return True


def send_daemon_command(
    proc: Any,
    cmd_dict: dict[str, Any],
    *,
    json_module: Any = _json,
    write_to_proc_fn: Callable[[Any, str], bool] = write_to_proc,
) -> bool:
    """Serialize and send a daemon command as JSONL."""
    sender = getattr(proc, "send_command", None)
    if callable(sender):
        try:
            return bool(sender(cmd_dict))
        except Exception:
            return False

    payload = json_module.dumps(cmd_dict, ensure_ascii=False) + "\n"
    return write_to_proc_fn(proc, payload)


def _build_swarmee_subprocess_env(
    *,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
    os_module: Any = os,
) -> dict[str, str]:
    env = dict(os_module.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # The CLI callback handler prints spinners using ANSI + carriage returns; disable for TUI subprocess capture.
    env["SWARMEE_SPINNERS"] = "0"
    # Enable structured JSONL event output for TUI consumption.
    env["SWARMEE_TUI_EVENTS"] = "1"
    # Ensure JSONL event logging is active so the Session Timeline has data.
    env["SWARMEE_LOG_EVENTS"] = "1"
    # Avoid macOS fsevents watcher crashes in sandboxed/PTY contexts.
    # Polling is slower but stable and prevents traceback spam from degrading TUI input responsiveness.
    env.setdefault("WATCHFILES_FORCE_POLLING", "1")
    env.setdefault("WATCHDOG_USE_POLLING", "1")
    existing_warning_filters = env.get("PYTHONWARNINGS", "").strip()
    tui_warning_filters = [
        # `PYTHONWARNINGS` is parsed via `warnings._setoption`, which `re.escape`s message+module.
        # Use exact (literal) values, not regex patterns.
        'ignore:Field name "json" in "Http_requestTool" shadows an attribute in parent "BaseModel"'
        ":UserWarning:pydantic.main",
    ]
    env["PYTHONWARNINGS"] = ",".join(
        [item for item in [*tui_warning_filters, existing_warning_filters] if isinstance(item, str) and item.strip()]
    )
    if env_overrides:
        env.update(env_overrides)
    if session_id:
        env["SWARMEE_SESSION_ID"] = session_id
    return env


def _default_build_swarmee_cmd(prompt: str, *, auto_approve: bool) -> list[str]:
    command = [sys.executable, "-u", "-m", "swarmee_river.swarmee"]
    if auto_approve:
        command.append("--yes")
    command.append(prompt)
    return command


def _default_build_swarmee_daemon_cmd() -> list[str]:
    return [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--tui-daemon"]


def _spawn_swarmee_process(
    command: list[str],
    *,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    subprocess_module: Any = subprocess,
    os_module: Any = os,
    env_builder: Callable[..., dict[str, str]] = _build_swarmee_subprocess_env,
) -> subprocess.Popen[str]:
    env = env_builder(session_id=session_id, env_overrides=env_overrides, os_module=os_module)
    return popen(
        command,
        stdin=subprocess_module.PIPE,
        stdout=subprocess_module.PIPE,
        stderr=subprocess_module.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        # Isolate child subprocesses from the interactive terminal session to
        # prevent terminal-title churn while tools (git/rg/etc.) execute.
        start_new_session=True,
    )


def spawn_swarmee(
    prompt: str,
    *,
    auto_approve: bool,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    subprocess_module: Any = subprocess,
    os_module: Any = os,
    build_swarmee_cmd_fn: Callable[..., list[str]] = _default_build_swarmee_cmd,
    env_builder: Callable[..., dict[str, str]] = _build_swarmee_subprocess_env,
) -> subprocess.Popen[str]:
    """Spawn Swarmee as a subprocess with line-buffered merged output."""
    return _spawn_swarmee_process(
        build_swarmee_cmd_fn(prompt, auto_approve=auto_approve),
        session_id=session_id,
        env_overrides=env_overrides,
        popen=popen,
        subprocess_module=subprocess_module,
        os_module=os_module,
        env_builder=env_builder,
    )


def spawn_swarmee_daemon(
    *,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    subprocess_module: Any = subprocess,
    os_module: Any = os,
    build_swarmee_daemon_cmd_fn: Callable[[], list[str]] = _default_build_swarmee_daemon_cmd,
    env_builder: Callable[..., dict[str, str]] = _build_swarmee_subprocess_env,
) -> subprocess.Popen[str]:
    """Spawn Swarmee daemon with line-buffered merged output."""
    return _spawn_swarmee_process(
        build_swarmee_daemon_cmd_fn(),
        session_id=session_id,
        env_overrides=env_overrides,
        popen=popen,
        subprocess_module=subprocess_module,
        os_module=os_module,
        env_builder=env_builder,
    )


def stop_process(
    proc: subprocess.Popen[str],
    *,
    timeout_s: float = 2.0,
    os_module: Any = os,
    signal_module: Any = signal,
    subprocess_module: Any = subprocess,
    contextlib_module: Any = contextlib,
) -> None:
    """Stop a running subprocess, escalating from interrupt to terminate to kill."""
    if proc.poll() is not None:
        return

    if os_module.name == "posix" and hasattr(signal_module, "SIGINT"):
        signaled = False
        if hasattr(os_module, "killpg"):
            with contextlib_module.suppress(Exception):
                os_module.killpg(proc.pid, signal_module.SIGINT)
                signaled = True
        if not signaled:
            with contextlib_module.suppress(Exception):
                proc.send_signal(signal_module.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess_module.TimeoutExpired:
            pass

    terminated = False
    if os_module.name == "posix" and hasattr(os_module, "killpg") and hasattr(signal_module, "SIGTERM"):
        with contextlib_module.suppress(Exception):
            os_module.killpg(proc.pid, signal_module.SIGTERM)
            terminated = True
    if not terminated:
        with contextlib_module.suppress(Exception):
            proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess_module.TimeoutExpired:
        pass
    except Exception:
        return

    killed = False
    if os_module.name == "posix" and hasattr(os_module, "killpg") and hasattr(signal_module, "SIGKILL"):
        with contextlib_module.suppress(Exception):
            os_module.killpg(proc.pid, signal_module.SIGKILL)
            killed = True
    if not killed:
        with contextlib_module.suppress(Exception):
            proc.kill()
    with contextlib_module.suppress(Exception):
        proc.wait(timeout=timeout_s)


__all__ = [
    "_DaemonTransport",
    "_SocketTransport",
    "_SubprocessTransport",
    "_build_swarmee_subprocess_env",
    "detect_consent_prompt",
    "send_daemon_command",
    "spawn_swarmee",
    "spawn_swarmee_daemon",
    "stop_process",
    "update_consent_capture",
    "write_to_proc",
]
