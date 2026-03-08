from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swarmee_river.settings import load_settings
from swarmee_river.state_paths import state_dir

from .protocol import encode_jsonl, make_error_event, parse_jsonl_command, utc_now_iso


@dataclass
class ClientState:
    client_id: str
    writer: asyncio.StreamWriter
    authenticated: bool = False
    client_name: str = ""
    surface: str = ""
    session_id: str | None = None
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_SESSION_IDLE_TIMEOUT_S: float = 3600.0
_BROKER_IDLE_TIMEOUT_S: float = 3600.0
_SESSION_STARTUP_OBSERVE_TIMEOUT_S: float = 1.0


def _interrupt_timeout_seconds(*, session_cwd: str) -> float:
    """Interrupt watchdog timeout (project-local settings)."""
    try:
        settings = load_settings(Path(session_cwd) / ".swarmee" / "settings.json")
        value = float(settings.runtime.interrupt_timeout_sec)
    except Exception:
        value = 2.0
    if value <= 0:
        return 2.0
    return value


def _bedrock_stall_warn_seconds() -> float:
    return 90.0


def _bedrock_stall_hard_fail_seconds() -> float | None:
    return None


def _normalize_env_overrides(raw_env: Any) -> dict[str, str]:
    if not isinstance(raw_env, dict):
        return {}
    resolved: dict[str, str] = {}
    for raw_key, raw_value in raw_env.items():
        key = str(raw_key or "").strip()
        if not key or "=" in key or "\x00" in key:
            continue
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value or "\x00" in value:
            continue
        resolved[key] = value
    return resolved


@dataclass
class SessionState:
    session_id: str
    cwd: str = ""
    env_overrides: dict[str, str] = field(default_factory=dict)
    client_ids: set[str] = field(default_factory=set)
    process: subprocess.Popen[str] | None = None
    stdout_task: asyncio.Task[None] | None = None
    stdin_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    query_active: bool = False
    consent_pending: bool = False
    controller_client_id: str | None = None
    started_new_session: bool = False
    idle_timeout_task: asyncio.Task[None] | None = None
    interrupt_watchdog_task: asyncio.Task[None] | None = None
    query_stall_watchdog_task: asyncio.Task[None] | None = None
    stall_restart_task: asyncio.Task[None] | None = None
    last_session_event_mono: float = field(default_factory=time.monotonic)
    last_query_start_mono: float = 0.0
    last_query_stall_warn_mono: float = 0.0
    pending_stall_restart: bool = False
    provider_hint: str = ""
    stdout_first_line_seen: bool = False
    ready_received: bool = False
    startup_future: asyncio.Future[bool] | None = None
    startup_failure_summary: str | None = None


class RuntimeServiceServer:
    """Minimal shared runtime broker (MVP) with token-authenticated JSONL over TCP."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str | None = None,
        runtime_file: Path | None = None,
        broker_log_path: str | Path | None = None,
        session_events_path_template: str | None = None,
    ) -> None:
        if host.strip() != "127.0.0.1":
            raise ValueError("Runtime service must bind to 127.0.0.1 for MVP")
        if int(port) < 0:
            raise ValueError("port must be >= 0")

        self.host = "127.0.0.1"
        self.requested_port = int(port)
        self.port: int = int(port)
        self.token = str(token).strip() if isinstance(token, str) and str(token).strip() else secrets.token_hex(32)
        self.pid = os.getpid()
        self.started_at = utc_now_iso()

        default_runtime_file = state_dir() / "runtime.json"
        self.runtime_file = runtime_file or default_runtime_file
        state_root = self.runtime_file.parent
        raw_broker_log = str(broker_log_path).strip() if broker_log_path is not None else ""
        if raw_broker_log:
            broker_path = Path(raw_broker_log).expanduser()
            if not broker_path.is_absolute():
                broker_path = (state_root / broker_path).resolve()
            self.broker_log_path = broker_path
        else:
            self.broker_log_path = state_root / "diagnostics" / "broker.log"
        raw_events_template = str(session_events_path_template).strip() if session_events_path_template else ""
        if raw_events_template:
            events_template = Path(raw_events_template).expanduser()
            if not events_template.is_absolute():
                events_template = (state_root / events_template).resolve()
            self.session_events_path_template = str(events_template)
        else:
            self.session_events_path_template = str(state_root / "diagnostics" / "sessions" / "{session_id}.jsonl")

        self._server: asyncio.AbstractServer | None = None
        self._stopped = asyncio.Event()
        self._clients: dict[str, ClientState] = {}
        self._sessions: dict[str, SessionState] = {}
        self._client_seq = 0
        self._running = False
        self._broker_idle_task: asyncio.Task[None] | None = None

    @property
    def sessions(self) -> dict[str, SessionState]:
        return self._sessions

    async def start(self) -> None:
        if self._running:
            return

        self.runtime_file.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            self.runtime_file.unlink()

        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.requested_port,
        )
        sockets = list(self._server.sockets or [])
        if not sockets:
            raise RuntimeError("Runtime service failed to bind a socket")
        self.port = int(sockets[0].getsockname()[1])
        self._write_discovery_file()
        self._running = True
        self._stopped.clear()

    async def stop(self) -> None:
        if not self._running and self._server is None:
            with contextlib.suppress(FileNotFoundError):
                self.runtime_file.unlink()
            self._stopped.set()
            return

        self._running = False
        self._cancel_broker_idle_timer()

        server = self._server
        self._server = None
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()

        for session in list(self._sessions.values()):
            self._cancel_session_idle_timer(session)
            await self._stop_session_process(session)

        client_states = list(self._clients.values())
        for client in client_states:
            await self._detach_and_close_client(client)

        self._clients.clear()
        self._sessions.clear()
        with contextlib.suppress(FileNotFoundError):
            self.runtime_file.unlink()
        self._stopped.set()

    async def serve_forever(self) -> None:
        if not self._running:
            await self.start()
        await self._stopped.wait()

    def _write_discovery_file(self) -> None:
        payload = {
            "schema_version": "2",
            "host": self.host,
            "port": int(self.port),
            "token": self.token,
            "pid": int(self.pid),
            "started_at": self.started_at,
            "broker_log_path": str(self.broker_log_path),
            "session_events_path": str(self.session_events_path_template),
        }
        self.runtime_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8")

    def _next_client_id(self) -> str:
        self._client_seq += 1
        return f"client-{self._client_seq}"

    @staticmethod
    def _truncate_text(value: str, *, max_chars: int = 4000) -> str:
        text = value.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "..."

    def _write_broker_log_line(self, message: str) -> None:
        line = str(message or "").strip()
        if not line:
            return
        payload = f"[runtime] {utc_now_iso()} {line}\n"
        with contextlib.suppress(Exception):
            self.broker_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.broker_log_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(payload)

    @staticmethod
    def _cwd_short_hash(cwd: str) -> str:
        normalized = str(cwd or "").strip()
        if not normalized:
            return "unknown"
        digest = hashlib.sha1(normalized.encode("utf-8", errors="replace")).hexdigest()[:10]
        tail = Path(normalized).name or normalized[-12:]
        return f"{tail}:{digest}"

    def _trace_session_transition(
        self,
        *,
        session_id: str,
        message: str,
        persist_session_warning: bool = False,
        **fields: Any,
    ) -> None:
        safe_session_id = str(session_id or "").strip() or "unknown"
        parts = [f"session={safe_session_id}", str(message or "").strip() or "event"]
        for key, value in fields.items():
            if value is None:
                continue
            rendered = str(value).strip()
            if not rendered:
                continue
            parts.append(f"{key}={rendered}")
        self._write_broker_log_line(" | ".join(parts))
        if persist_session_warning:
            payload: dict[str, Any] = {
                "event": "warning",
                "text": f"[broker] {message}",
                "source": "runtime_broker",
                "session_id": safe_session_id,
            }
            payload.update({key: value for key, value in fields.items() if value is not None})
            with contextlib.suppress(Exception):
                self._persist_session_event(safe_session_id, payload)

    def _resolve_attach_cwd(self, raw_cwd: Any) -> Path | None:
        if not isinstance(raw_cwd, str) or not raw_cwd.strip():
            return None
        candidate = Path(raw_cwd.strip()).expanduser()
        resolved = candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()
        with contextlib.suppress(Exception):
            resolved = resolved.resolve()
        if not resolved.exists() or not resolved.is_dir():
            return None
        return resolved

    async def _send_event(self, client: ClientState, payload: dict[str, Any]) -> None:
        if client.writer.is_closing():
            return
        data = encode_jsonl(payload).encode("utf-8", errors="replace")
        async with client.write_lock:
            try:
                client.writer.write(data)
                await client.writer.drain()
            except Exception:
                await self._detach_and_close_client(client)

    async def _send_error(self, client: ClientState, code: str, message: str, *, detail: str | None = None) -> None:
        await self._send_event(client, make_error_event(code, message, detail=detail))

    def _session_events_path(self, session_id: str) -> Path:
        safe_session_id = (
            "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "-" for ch in session_id) or "unknown"
        )
        template = str(self.session_events_path_template)
        if "{session_id}" in template:
            path = Path(template.replace("{session_id}", safe_session_id))
        else:
            candidate = Path(template)
            if candidate.suffix:
                path = candidate
            else:
                path = candidate / f"{safe_session_id}.jsonl"
        return path

    def _persist_session_event(self, session_id: str, payload: dict[str, Any]) -> None:
        path = self._session_events_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        event_payload = dict(payload)
        event_payload.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, ensure_ascii=False))
            handle.write("\n")

    async def _detach_and_close_client(self, client: ClientState) -> None:
        self._detach_client_from_session(client)
        self._clients.pop(client.client_id, None)
        if not client.writer.is_closing():
            client.writer.close()
            with contextlib.suppress(Exception):
                await client.writer.wait_closed()
        self._maybe_schedule_broker_idle()

    def _detach_client_from_session(self, client: ClientState) -> None:
        session_id = client.session_id
        if not session_id:
            return
        session = self._sessions.get(session_id)
        if session is None:
            client.session_id = None
            return
        session.client_ids.discard(client.client_id)
        client.session_id = None
        if session.controller_client_id == client.client_id:
            session.controller_client_id = None
            session.consent_pending = False
        if not session.client_ids and not session.query_active:
            self._schedule_session_idle_cleanup(session)

    # -- Idle cleanup --------------------------------------------------------

    def _cancel_session_idle_timer(self, session: SessionState) -> None:
        task = session.idle_timeout_task
        if task is not None and not task.done():
            task.cancel()
        session.idle_timeout_task = None

    def _schedule_session_idle_cleanup(self, session: SessionState) -> None:
        self._cancel_session_idle_timer(session)
        if _SESSION_IDLE_TIMEOUT_S <= 0:
            return
        session.idle_timeout_task = asyncio.ensure_future(self._idle_session_cleanup(session))

    def _cancel_interrupt_watchdog(self, session: SessionState) -> None:
        task = session.interrupt_watchdog_task
        current_task = asyncio.current_task()
        if task is not None and not task.done() and task is not current_task:
            task.cancel()
        session.interrupt_watchdog_task = None

    def _arm_interrupt_watchdog(self, session: SessionState) -> None:
        self._cancel_interrupt_watchdog(session)
        if not session.query_active:
            return
        timeout_s = _interrupt_timeout_seconds(session_cwd=session.cwd)
        if timeout_s <= 0:
            return
        session.interrupt_watchdog_task = asyncio.ensure_future(
            self._interrupt_watchdog(session, timeout_s=timeout_s)
        )

    async def _interrupt_watchdog(self, session: SessionState, *, timeout_s: float) -> None:
        try:
            await asyncio.sleep(timeout_s)
        except asyncio.CancelledError:
            return
        if not session.query_active:
            return
        await self._broadcast_session_event(
            session,
            {
                "event": "warning",
                "text": "Interrupt timeout reached; waiting for graceful shutdown.",
                "interrupt_timeout_sec": timeout_s,
                "force_restart": False,
                "forced_restart_triggered": False,
            },
        )

    def _cancel_query_stall_watchdog(self, session: SessionState) -> None:
        task = session.query_stall_watchdog_task
        current_task = asyncio.current_task()
        if task is not None and not task.done() and task is not current_task:
            task.cancel()
        session.query_stall_watchdog_task = None

    def _arm_query_stall_watchdog(self, session: SessionState) -> None:
        self._cancel_query_stall_watchdog(session)
        if not session.query_active:
            return
        warn_sec = _bedrock_stall_warn_seconds()
        hard_fail_sec = _bedrock_stall_hard_fail_seconds()
        if warn_sec <= 0 and hard_fail_sec is None:
            return
        now = time.monotonic()
        session.last_query_start_mono = now
        session.last_session_event_mono = now
        session.last_query_stall_warn_mono = 0.0
        session.query_stall_watchdog_task = asyncio.ensure_future(
            self._query_stall_watchdog(
                session,
                warn_sec=warn_sec,
                hard_fail_sec=hard_fail_sec,
            )
        )

    async def _query_stall_watchdog(
        self,
        session: SessionState,
        *,
        warn_sec: float,
        hard_fail_sec: float | None,
    ) -> None:
        interval_basis = min(warn_sec, hard_fail_sec) if hard_fail_sec is not None else warn_sec
        interval = max(0.25, min(2.0, float(interval_basis) / 4.0))
        try:
            while session.query_active:
                await asyncio.sleep(interval)
                if not session.query_active:
                    return
                now = time.monotonic()
                last_session_event = max(session.last_session_event_mono, session.last_query_start_mono)
                stalled_for = max(0.0, now - last_session_event)
                provider_hint = (session.provider_hint or "unknown").strip().lower() or "unknown"

                if stalled_for >= warn_sec and (now - session.last_query_stall_warn_mono) >= warn_sec:
                    session.last_query_stall_warn_mono = now
                    await self._broadcast_session_event(
                        session,
                        {
                            "event": "warning",
                            "text": (
                                f"No daemon events for {stalled_for:.1f}s while query is active "
                                f"(provider={provider_hint})."
                            ),
                            "query_stall_elapsed_sec": round(stalled_for, 2),
                            "query_stall_warn_sec": warn_sec,
                            "provider": provider_hint,
                            "force_restart": False,
                            "forced_restart_triggered": False,
                        },
                    )

                if hard_fail_sec is None or stalled_for < hard_fail_sec:
                    continue

                session.pending_stall_restart = True
                await self._broadcast_session_event(
                    session,
                    {
                        "event": "warning",
                        "text": (
                            f"Query stall hard-fail threshold reached ({hard_fail_sec:.1f}s); "
                            "daemon restart deferred until turn completion."
                        ),
                        "query_stall_elapsed_sec": round(stalled_for, 2),
                        "query_stall_hard_fail_sec": hard_fail_sec,
                        "provider": provider_hint,
                        "force_restart": False,
                        "forced_restart_triggered": False,
                    },
                )
                return
        except asyncio.CancelledError:
            return

    def _schedule_pending_stall_restart(self, session: SessionState) -> None:
        if not session.pending_stall_restart or session.query_active:
            return
        existing = session.stall_restart_task
        if existing is not None and not existing.done():
            return
        session.pending_stall_restart = False
        session.stall_restart_task = asyncio.ensure_future(self._restart_session_after_stall(session))

    async def _restart_session_after_stall(self, session: SessionState) -> None:
        try:
            await self._stop_session_process(session)
            await self._ensure_session_process(session)
            await self._broadcast_session_event(
                session,
                {
                    "event": "warning",
                    "text": "Session daemon restarted after query stall hard-fail threshold.",
                    "forced_restart_triggered": True,
                    "force_restart": False,
                },
            )
        except Exception as exc:
            await self._broadcast_session_event(
                session,
                {
                    "event": "warning",
                    "text": f"Failed to restart session daemon after stall detection: {exc}",
                    "forced_restart_triggered": False,
                    "force_restart": False,
                },
            )
        finally:
            session.stall_restart_task = None

    async def _idle_session_cleanup(self, session: SessionState) -> None:
        try:
            await asyncio.sleep(_SESSION_IDLE_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        # Re-check: another client may have attached during the wait.
        if session.client_ids or session.query_active:
            return
        await self._stop_session_process(session)
        self._sessions.pop(session.session_id, None)
        self._maybe_schedule_broker_idle()

    def _cancel_broker_idle_timer(self) -> None:
        task = self._broker_idle_task
        if task is not None and not task.done():
            task.cancel()
        self._broker_idle_task = None

    def _maybe_schedule_broker_idle(self) -> None:
        """Schedule broker self-shutdown if no clients and no sessions remain."""
        if self._clients or self._sessions:
            self._cancel_broker_idle_timer()
            return
        if self._broker_idle_task is not None and not self._broker_idle_task.done():
            return  # already scheduled
        if _BROKER_IDLE_TIMEOUT_S <= 0:
            return
        self._broker_idle_task = asyncio.ensure_future(self._idle_broker_shutdown())

    async def _idle_broker_shutdown(self) -> None:
        try:
            await asyncio.sleep(_BROKER_IDLE_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        if self._clients or self._sessions:
            return
        self._stopped.set()

    async def _wait_for_process_exit(self, process: subprocess.Popen[str], *, timeout_s: float) -> bool:
        def _wait() -> bool:
            try:
                process.wait(timeout=timeout_s)
                return True
            except subprocess.TimeoutExpired:
                return False
            except Exception:
                return True

        return await asyncio.to_thread(_wait)

    async def _stop_session_process(self, session: SessionState) -> None:
        self._cancel_interrupt_watchdog(session)
        self._cancel_query_stall_watchdog(session)
        stall_restart_task = session.stall_restart_task
        current_task = asyncio.current_task()
        if stall_restart_task is not None and not stall_restart_task.done() and stall_restart_task is not current_task:
            stall_restart_task.cancel()
        if stall_restart_task is not current_task:
            session.stall_restart_task = None
        process = session.process
        if process is None:
            session.stdout_task = None
            session.query_active = False
            session.consent_pending = False
            session.controller_client_id = None
            session.started_new_session = False
            session.pending_stall_restart = False
            return

        if process.poll() is None and process.stdin is not None:
            shutdown_line = json.dumps({"cmd": "shutdown"}, ensure_ascii=False) + "\n"
            with contextlib.suppress(Exception):
                await asyncio.to_thread(process.stdin.write, shutdown_line)
                await asyncio.to_thread(process.stdin.flush)
            exited = await self._wait_for_process_exit(process, timeout_s=1.5)
        else:
            exited = process.poll() is not None

        if not exited and process.poll() is None:
            if os.name == "posix" and session.started_new_session and hasattr(os, "killpg"):
                with contextlib.suppress(Exception):
                    os.killpg(process.pid, signal.SIGTERM)
            else:
                with contextlib.suppress(Exception):
                    process.terminate()
            exited = await self._wait_for_process_exit(process, timeout_s=1.5)

        if not exited and process.poll() is None:
            if os.name == "posix" and session.started_new_session and hasattr(os, "killpg"):
                with contextlib.suppress(Exception):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                with contextlib.suppress(Exception):
                    process.kill()
            await self._wait_for_process_exit(process, timeout_s=1.0)

        with contextlib.suppress(Exception):
            if process.stdout is not None:
                process.stdout.close()
        with contextlib.suppress(Exception):
            if process.stdin is not None:
                process.stdin.close()

        task = session.stdout_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        session.process = None
        session.stdout_task = None
        session.query_active = False
        session.consent_pending = False
        session.controller_client_id = None
        session.started_new_session = False
        session.pending_stall_restart = False
        session.stdout_first_line_seen = False
        session.ready_received = False
        session.startup_failure_summary = None
        startup_future = session.startup_future
        session.startup_future = None
        if startup_future is not None and not startup_future.done():
            startup_future.cancel()

    async def _start_session_process(self, session: SessionState) -> None:
        command = [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--tui-daemon"]
        env = dict(os.environ)
        if session.env_overrides:
            env.update(session.env_overrides)
        env["SWARMEE_TUI_EVENTS"] = "1"
        env["SWARMEE_SESSION_ID"] = session.session_id
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "cwd": session.cwd,
            "env": env,
        }
        started_new_session = False
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
            started_new_session = True

        process = subprocess.Popen(command, **popen_kwargs)
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Failed to open daemon stdio pipes")

        session.process = process
        session.stdout_first_line_seen = False
        session.started_new_session = started_new_session
        session.ready_received = False
        session.startup_failure_summary = None
        session.startup_future = asyncio.get_running_loop().create_future()
        self._trace_session_transition(
            session_id=session.session_id,
            message="session_daemon_spawned",
            persist_session_warning=True,
            pid=process.pid,
        )
        session.stdout_task = asyncio.create_task(
            self._session_stdout_reader(session=session, process=process),
            name=f"runtime-session-{session.session_id}-stdout",
        )

    async def _ensure_session_process(self, session: SessionState) -> None:
        process = session.process
        if process is not None and process.poll() is None:
            task = session.stdout_task
            if task is None or task.done():
                session.stdout_task = asyncio.create_task(
                    self._session_stdout_reader(session=session, process=process),
                    name=f"runtime-session-{session.session_id}-stdout",
                )
            return
        await self._stop_session_process(session)
        await self._start_session_process(session)
        await self._observe_session_startup(session, timeout_s=_SESSION_STARTUP_OBSERVE_TIMEOUT_S)

    async def _observe_session_startup(self, session: SessionState, *, timeout_s: float) -> None:
        future = session.startup_future
        if future is None or future.done() or session.ready_received:
            return
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=max(0.1, float(timeout_s)))
        except asyncio.TimeoutError:
            return

    def _parse_session_output_line(self, raw_line: str) -> dict[str, Any] | None:
        text = raw_line.rstrip("\n").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"event": "warning", "text": self._truncate_text(text)}

        if isinstance(parsed, dict):
            return parsed
        return {"event": "warning", "text": f"Non-object daemon output: {self._truncate_text(text)}"}

    async def _broadcast_session_event(self, session: SessionState, payload: dict[str, Any]) -> None:
        controller_before = session.controller_client_id
        event_type = str(payload.get("event", "")).strip().lower()

        if event_type == "consent_prompt":
            session.consent_pending = True
        elif event_type == "turn_complete":
            self._cancel_interrupt_watchdog(session)
            self._cancel_query_stall_watchdog(session)
            session.query_active = False
            session.consent_pending = False
            session.controller_client_id = None
        elif event_type == "ready":
            session.ready_received = True
            startup_future = session.startup_future
            if startup_future is not None and not startup_future.done():
                startup_future.set_result(True)
        elif event_type == "model_info":
            provider = str(payload.get("provider", "")).strip().lower()
            if provider:
                session.provider_hint = provider
        elif event_type in {"error", "warning"} and not session.ready_received:
            text = str(payload.get("message") or payload.get("text") or "").strip()
            if text and not text.startswith("[broker]"):
                session.startup_failure_summary = text
                if event_type == "error":
                    startup_future = session.startup_future
                    if startup_future is not None and not startup_future.done():
                        startup_future.set_exception(RuntimeError(text))

        outgoing = dict(payload)
        outgoing.setdefault("session_id", session.session_id)
        if controller_before:
            outgoing.setdefault("controller_client_id", controller_before)
        with contextlib.suppress(Exception):
            self._persist_session_event(session.session_id, outgoing)

        for client_id in list(session.client_ids):
            client = self._clients.get(client_id)
            if client is None:
                session.client_ids.discard(client_id)
                continue
            await self._send_event(client, outgoing)

        if event_type == "turn_complete":
            self._schedule_pending_stall_restart(session)

    async def _session_stdout_reader(self, *, session: SessionState, process: subprocess.Popen[str]) -> None:
        stdout = process.stdout
        if stdout is None:
            await self._broadcast_session_event(
                session,
                {"event": "warning", "text": "Session daemon stdout unavailable"},
            )
            return

        try:
            while True:
                line = await asyncio.to_thread(stdout.readline)
                if line == "":
                    break
                if not session.stdout_first_line_seen and line.strip():
                    session.stdout_first_line_seen = True
                    self._trace_session_transition(
                        session_id=session.session_id,
                        message="session_stdout_first_line",
                        persist_session_warning=True,
                        preview=self._truncate_text(line, max_chars=180),
                    )
                parsed = self._parse_session_output_line(line)
                if parsed is None:
                    continue
                session.last_session_event_mono = time.monotonic()
                await self._broadcast_session_event(session, parsed)
        except asyncio.CancelledError:
            raise
        finally:
            with contextlib.suppress(Exception):
                stdout.close()
            with contextlib.suppress(Exception):
                if process.poll() is None:
                    await asyncio.to_thread(process.wait)
            if session.process is process:
                self._cancel_interrupt_watchdog(session)
                self._cancel_query_stall_watchdog(session)
                session.process = None
                session.stdout_task = None
                session.query_active = False
                session.consent_pending = False
                session.controller_client_id = None
                session.started_new_session = False
                session.pending_stall_restart = False
                session.stdout_first_line_seen = False
                if not session.ready_received:
                    startup_summary = (
                        session.startup_failure_summary
                        or f"Session daemon exited before becoming ready (code {process.poll()})."
                    )
                    startup_future = session.startup_future
                    if startup_future is not None and not startup_future.done():
                        startup_future.set_exception(RuntimeError(startup_summary))
                session.ready_received = False
                session.startup_future = None
            code = process.poll()
            if not session.ready_received and session.startup_failure_summary:
                await self._broadcast_session_event(
                    session,
                    {
                        "event": "error",
                        "code": "session_start_failed",
                        "message": session.startup_failure_summary,
                        "text": session.startup_failure_summary,
                    },
                )
            await self._broadcast_session_event(
                session,
                {"event": "warning", "text": f"Session daemon exited (code {code if code is not None else 'unknown'})"},
            )

    async def _forward_to_session(self, session: SessionState, payload: dict[str, Any]) -> None:
        process = session.process
        if process is None or process.poll() is not None:
            raise RuntimeError("Session runtime is not running")
        stdin = process.stdin
        if stdin is None:
            raise RuntimeError("Session runtime stdin unavailable")

        line = json.dumps(payload, ensure_ascii=False) + "\n"

        async with session.stdin_lock:

            def _write() -> None:
                stdin.write(line)
                stdin.flush()

            await asyncio.to_thread(_write)

    def _require_session_for_client(self, client: ClientState) -> SessionState | None:
        if not client.authenticated:
            return None
        session_id = client.session_id
        if not session_id:
            return None
        return self._sessions.get(session_id)

    async def _handle_hello(self, client: ClientState, payload: dict[str, Any]) -> bool:
        token = payload.get("token")
        if not isinstance(token, str) or token != self.token:
            await self._send_error(client, "auth_failed", "Invalid runtime token")
            await self._detach_and_close_client(client)
            return False

        client.authenticated = True
        client.client_name = str(payload.get("client_name", "")).strip()
        client.surface = str(payload.get("surface", "")).strip()
        await self._send_event(
            client,
            {
                "event": "hello_ack",
                "status": "ok",
                "client_id": client.client_id,
                "pid": self.pid,
                "host": self.host,
                "port": self.port,
            },
        )
        return True

    async def _handle_attach(self, client: ClientState, payload: dict[str, Any]) -> None:
        if not client.authenticated:
            await self._send_error(client, "unauthorized", "Send hello with a valid token first")
            return

        session_id_value = payload.get("session_id")
        if not isinstance(session_id_value, str) or not session_id_value.strip():
            await self._send_error(client, "invalid_session_id", "attach.session_id is required")
            return
        session_id = session_id_value.strip()
        resolved_cwd = self._resolve_attach_cwd(payload.get("cwd"))
        if resolved_cwd is None:
            await self._send_error(client, "invalid_cwd", "attach.cwd must point to an existing directory")
            return
        cwd = str(resolved_cwd)
        self._trace_session_transition(
            session_id=session_id,
            message="attach_requested",
            persist_session_warning=True,
            cwd_hash=self._cwd_short_hash(cwd),
        )

        self._detach_client_from_session(client)
        session = self._sessions.get(session_id)
        has_env_overrides = "env_overrides" in payload
        requested_env_overrides = _normalize_env_overrides(payload.get("env_overrides"))
        startup_outcome = "ok"
        defer_env_refresh = False
        if session is None:
            session = SessionState(
                session_id=session_id,
                cwd=cwd,
                env_overrides=(requested_env_overrides if has_env_overrides else {}),
            )
            self._sessions[session_id] = session
        elif session.cwd != cwd:
            await self._send_error(
                client,
                "cwd_mismatch",
                "Session already exists with a different cwd",
                detail=f"existing={session.cwd} requested={cwd}",
            )
            return
        elif has_env_overrides and requested_env_overrides != session.env_overrides:
            if session.query_active:
                defer_env_refresh = True
                startup_outcome = "env_refresh_deferred"
            else:
                session.env_overrides = requested_env_overrides
                startup_outcome = "restarted_for_env"
                await self._stop_session_process(session)

        try:
            await self._ensure_session_process(session)
        except Exception as exc:
            await self._send_error(client, "session_start_failed", "Failed to start session daemon", detail=str(exc))
            return

        self._cancel_session_idle_timer(session)
        session.client_ids.add(client.client_id)
        client.session_id = session_id
        self._cancel_broker_idle_timer()

        await self._send_event(
            client,
            {
                "event": "attached",
                "session_id": session_id,
                "cwd": session.cwd,
                "clients": len(session.client_ids),
                "startup_outcome": startup_outcome,
                "session_pid": session.process.pid if session.process is not None else None,
            },
        )
        if defer_env_refresh:
            await self._send_event(
                client,
                {
                    "event": "warning",
                    "text": (
                        "Session env overrides changed while a query is active; "
                        "env refresh deferred until the session is idle."
                    ),
                },
            )

    async def _handle_ping(self, client: ClientState) -> None:
        if not client.authenticated:
            await self._send_error(client, "unauthorized", "Send hello with a valid token first")
            return
        await self._send_event(client, {"event": "pong", "ts": utc_now_iso()})

    async def _handle_shutdown_service(self, client: ClientState) -> None:
        if not client.authenticated:
            await self._send_error(client, "unauthorized", "Send hello with a valid token first")
            return
        self._write_broker_log_line(
            "shutdown_service_received"
            f" | client={client.client_id}"
            f" | name={client.client_name or 'unknown'}"
            f" | surface={client.surface or 'unknown'}"
        )
        await self._send_event(client, {"event": "shutdown_ack", "scope": "service"})
        asyncio.create_task(self.stop(), name="runtime-service-stop")

    async def _handle_session_proxy_command(self, client: ClientState, payload: dict[str, Any], *, cmd: str) -> None:
        session = self._require_session_for_client(client)
        if session is None:
            if not client.authenticated:
                await self._send_error(client, "unauthorized", "Send hello with a valid token first")
            else:
                await self._send_error(client, "not_attached", "Attach to a session first")
            return

        if cmd == "query":
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                await self._send_error(client, "invalid_query", "query.text is required")
                return
            forwarded: dict[str, Any] = {"cmd": "query", "text": text.strip()}
            mode = payload.get("mode")
            if isinstance(mode, str) and mode.strip():
                forwarded["mode"] = mode.strip()
            auto_approve = payload.get("auto_approve")
            if isinstance(auto_approve, bool):
                forwarded["auto_approve"] = auto_approve
            tier = payload.get("tier")
            if isinstance(tier, str) and tier.strip():
                forwarded["tier"] = tier.strip()
            session.controller_client_id = client.client_id
            session.query_active = True
            session.consent_pending = False
            session.pending_stall_restart = False
            session.last_query_start_mono = time.monotonic()
            session.last_session_event_mono = session.last_query_start_mono
            session.last_query_stall_warn_mono = 0.0
            self._trace_session_transition(
                session_id=session.session_id,
                message="query_forwarded",
                persist_session_warning=True,
                provider=session.provider_hint or "unknown",
                text_len=len(text.strip()),
            )
            self._cancel_interrupt_watchdog(session)
            self._cancel_query_stall_watchdog(session)
        elif cmd == "consent_response":
            choice = payload.get("choice")
            if not isinstance(choice, str) or not choice.strip():
                await self._send_error(client, "invalid_choice", "consent_response.choice is required")
                return
            if not session.consent_pending:
                await self._send_error(client, "no_pending_consent", "No active consent prompt for this session")
                return
            if session.controller_client_id != client.client_id:
                await self._send_error(
                    client,
                    "not_controller",
                    "Only the active controller can answer consent prompts",
                )
                return
            forwarded = {"cmd": "consent_response", "choice": choice.strip().lower()}
        elif cmd == "set_context_sources":
            sources = payload.get("sources")
            if not isinstance(sources, list):
                await self._send_error(client, "invalid_sources", "set_context_sources.sources must be a list")
                return
            forwarded = {"cmd": "set_context_sources", "sources": sources}
        elif cmd == "set_sop":
            name = payload.get("name")
            content = payload.get("content")
            if not isinstance(name, str) or not name.strip():
                await self._send_error(client, "invalid_sop_name", "set_sop.name is required")
                return
            if content is not None and not isinstance(content, str):
                await self._send_error(client, "invalid_sop_content", "set_sop.content must be a string or null")
                return
            forwarded = {"cmd": "set_sop", "name": name.strip(), "content": content}
        elif cmd == "set_tier":
            tier = payload.get("tier")
            if not isinstance(tier, str) or not tier.strip():
                await self._send_error(client, "invalid_tier", "set_tier.tier is required")
                return
            forwarded = {"cmd": "set_tier", "tier": tier.strip()}
        elif cmd == "set_profile":
            if session.query_active:
                await self._send_error(client, "query_active", "Cannot set profile while a query is running")
                return
            profile = payload.get("profile")
            if not isinstance(profile, dict):
                await self._send_error(client, "invalid_profile", "set_profile.profile must be an object")
                return
            forwarded = {"cmd": "set_profile", "profile": profile}
        elif cmd == "get_bundles":
            forwarded = {"cmd": "get_bundles"}
        elif cmd == "set_bundle":
            bundle = payload.get("bundle")
            if not isinstance(bundle, dict):
                await self._send_error(client, "invalid_bundle", "set_bundle.bundle must be an object")
                return
            forwarded = {"cmd": "set_bundle", "bundle": bundle}
        elif cmd == "delete_bundle":
            bundle_id = payload.get("id")
            if not isinstance(bundle_id, str) or not bundle_id.strip():
                await self._send_error(client, "invalid_bundle_id", "delete_bundle.id is required")
                return
            forwarded = {"cmd": "delete_bundle", "id": bundle_id.strip()}
        elif cmd == "apply_bundle":
            if session.query_active:
                await self._send_error(client, "query_active", "Cannot apply bundle while a query is running")
                return
            bundle_id = payload.get("id")
            if not isinstance(bundle_id, str) or not bundle_id.strip():
                await self._send_error(client, "invalid_bundle_id", "apply_bundle.id is required")
                return
            forwarded = {"cmd": "apply_bundle", "id": bundle_id.strip()}
        elif cmd == "set_safety_overrides":
            if session.query_active:
                await self._send_error(client, "query_active", "Cannot set safety overrides while a query is running")
                return
            overrides_payload: dict[str, Any] = {}
            raw_nested = payload.get("overrides")
            if isinstance(raw_nested, dict):
                overrides_payload.update(raw_nested)
            for key in ("tool_consent", "tool_allowlist", "tool_blocklist"):
                if key in payload:
                    overrides_payload[key] = payload.get(key)
            if not overrides_payload:
                await self._send_error(
                    client,
                    "invalid_safety_overrides",
                    "set_safety_overrides requires tool_consent/tool_allowlist/tool_blocklist payload",
                )
                return
            forwarded = {"cmd": "set_safety_overrides", "overrides": overrides_payload}
        elif cmd == "connect":
            if session.query_active:
                await self._send_error(client, "query_active", "Cannot connect provider while a query is running")
                return
            forwarded = {"cmd": "connect"}
            provider = payload.get("provider")
            if isinstance(provider, str) and provider.strip():
                forwarded["provider"] = provider.strip()
            method = payload.get("method")
            if isinstance(method, str) and method.strip():
                forwarded["method"] = method.strip()
            open_browser = payload.get("open_browser")
            if isinstance(open_browser, bool):
                forwarded["open_browser"] = open_browser
            profile = payload.get("profile")
            if isinstance(profile, str) and profile.strip():
                forwarded["profile"] = profile.strip()
            api_key = payload.get("api_key")
            if isinstance(api_key, str) and api_key.strip():
                forwarded["api_key"] = api_key.strip()
        elif cmd == "auth":
            if session.query_active:
                await self._send_error(client, "query_active", "Cannot inspect auth while a query is running")
                return
            action = payload.get("action")
            forwarded = {"cmd": "auth", "action": str(action).strip() if isinstance(action, str) else "list"}
            provider = payload.get("provider")
            if isinstance(provider, str) and provider.strip():
                forwarded["provider"] = provider.strip()
        elif cmd == "interrupt":
            forwarded = {"cmd": "interrupt"}
        elif cmd == "restore_session":
            restore_session_id = payload.get("session_id")
            if not isinstance(restore_session_id, str) or not restore_session_id.strip():
                await self._send_error(client, "invalid_restore_session", "restore_session.session_id is required")
                return
            forwarded = {"cmd": "restore_session", "session_id": restore_session_id.strip()}
        elif cmd == "get_prompt_assets":
            forwarded = {"cmd": "get_prompt_assets"}
        elif cmd == "set_prompt_asset":
            asset = payload.get("asset")
            if not isinstance(asset, dict):
                await self._send_error(client, "invalid_prompt_asset", "set_prompt_asset.asset must be an object")
                return
            forwarded = {"cmd": "set_prompt_asset", "asset": asset}
        elif cmd == "delete_prompt_asset":
            prompt_id = payload.get("id")
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                await self._send_error(client, "invalid_prompt_asset_id", "delete_prompt_asset.id is required")
                return
            forwarded = {"cmd": "delete_prompt_asset", "id": prompt_id.strip()}
        else:
            await self._send_error(client, "unknown_cmd", f"Unknown command: {cmd}")
            return

        try:
            await self._ensure_session_process(session)
            await self._forward_to_session(session, forwarded)
            if cmd == "query":
                self._arm_query_stall_watchdog(session)
            if cmd == "interrupt":
                self._arm_interrupt_watchdog(session)
        except Exception as exc:
            if cmd == "query":
                session.query_active = False
                session.consent_pending = False
                session.controller_client_id = None
                self._cancel_query_stall_watchdog(session)
            await self._send_error(
                client,
                "session_proxy_failed",
                "Failed to proxy command to session daemon",
                detail=str(exc),
            )

    async def _handle_shutdown_session(self, client: ClientState) -> None:
        session = self._require_session_for_client(client)
        if session is None:
            if not client.authenticated:
                await self._send_error(client, "unauthorized", "Send hello with a valid token first")
            else:
                await self._send_error(client, "not_attached", "Attach to a session first")
            return
        await self._stop_session_process(session)
        await self._broadcast_session_event(session, {"event": "session_shutdown", "shutdown_outcome": "ok"})

    async def _dispatch_command(self, client: ClientState, payload: dict[str, Any]) -> None:
        cmd = str(payload.get("cmd", "")).strip().lower()
        if not cmd:
            await self._send_error(client, "missing_cmd", "Command requires a cmd field")
            return

        if cmd == "hello":
            await self._handle_hello(client, payload)
            return
        if cmd == "attach":
            await self._handle_attach(client, payload)
            return
        if cmd == "ping":
            await self._handle_ping(client)
            return
        if cmd == "shutdown_service":
            await self._handle_shutdown_service(client)
            return
        if cmd in {
            "query",
            "consent_response",
            "set_context_sources",
            "set_sop",
            "set_tier",
            "set_profile",
            "get_bundles",
            "set_bundle",
            "delete_bundle",
            "apply_bundle",
            "set_safety_overrides",
            "connect",
            "auth",
            "interrupt",
            "restore_session",
            "get_prompt_assets",
            "set_prompt_asset",
            "delete_prompt_asset",
        }:
            await self._handle_session_proxy_command(client, payload, cmd=cmd)
            return
        if cmd == "shutdown_session":
            await self._handle_shutdown_session(client)
            return

        await self._send_error(client, "unknown_cmd", f"Unknown command: {cmd}")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        client = ClientState(client_id=self._next_client_id(), writer=writer)
        self._clients[client.client_id] = client

        try:
            while True:
                raw_line = await reader.readline()
                if raw_line == b"":
                    break
                line = raw_line.decode("utf-8", errors="replace")
                command, error_event = parse_jsonl_command(line)
                if error_event is not None:
                    await self._send_event(client, error_event)
                    continue
                if command is None:
                    continue
                await self._dispatch_command(client, command)
                if client.client_id not in self._clients:
                    break
        except asyncio.CancelledError:
            raise
        finally:
            await self._detach_and_close_client(client)


async def run_runtime_service(*, port: int = 0, token: str | None = None) -> None:
    """
    Run the runtime broker until interrupted (SIGINT/SIGTERM/KeyboardInterrupt).
    """
    server = RuntimeServiceServer(port=port, token=token)
    await server.start()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        stop_event.set()

    handlers: list[signal.Signals] = [signal.SIGINT, signal.SIGTERM]
    for sig in handlers:
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, _request_stop)

    async def _wait_either() -> None:
        done, pending = await asyncio.wait(
            [asyncio.ensure_future(stop_event.wait()), asyncio.ensure_future(server._stopped.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    try:
        await _wait_either()
    except asyncio.CancelledError:
        raise
    finally:
        await server.stop()
