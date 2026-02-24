from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass
class SessionState:
    session_id: str
    cwd: str = ""
    client_ids: set[str] = field(default_factory=set)
    process: subprocess.Popen[str] | None = None
    stdout_task: asyncio.Task[None] | None = None
    stdin_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    query_active: bool = False
    consent_pending: bool = False
    controller_client_id: str | None = None
    started_new_session: bool = False


class RuntimeServiceServer:
    """Minimal shared runtime broker (MVP) with token-authenticated JSONL over TCP."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str | None = None,
        runtime_file: Path | None = None,
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

        self._server: asyncio.AbstractServer | None = None
        self._stopped = asyncio.Event()
        self._clients: dict[str, ClientState] = {}
        self._sessions: dict[str, SessionState] = {}
        self._client_seq = 0
        self._running = False

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

        server = self._server
        self._server = None
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()

        for session in list(self._sessions.values()):
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
            "host": self.host,
            "port": int(self.port),
            "token": self.token,
            "pid": int(self.pid),
            "started_at": self.started_at,
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

    async def _detach_and_close_client(self, client: ClientState) -> None:
        self._detach_client_from_session(client)
        self._clients.pop(client.client_id, None)
        if not client.writer.is_closing():
            client.writer.close()
            with contextlib.suppress(Exception):
                await client.writer.wait_closed()

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
        process = session.process
        if process is None:
            session.stdout_task = None
            session.query_active = False
            session.consent_pending = False
            session.controller_client_id = None
            session.started_new_session = False
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

    async def _start_session_process(self, session: SessionState) -> None:
        command = [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--tui-daemon"]
        env = dict(os.environ)
        env["SWARMEE_TUI_EVENTS"] = "1"
        env["SWARMEE_SPINNERS"] = "0"
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
        session.started_new_session = started_new_session
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
            session.query_active = False
            session.consent_pending = False
            session.controller_client_id = None

        outgoing = dict(payload)
        outgoing.setdefault("session_id", session.session_id)
        if controller_before:
            outgoing.setdefault("controller_client_id", controller_before)

        for client_id in list(session.client_ids):
            client = self._clients.get(client_id)
            if client is None:
                session.client_ids.discard(client_id)
                continue
            await self._send_event(client, outgoing)

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
                parsed = self._parse_session_output_line(line)
                if parsed is None:
                    continue
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
                session.process = None
                session.stdout_task = None
                session.query_active = False
                session.consent_pending = False
                session.controller_client_id = None
                session.started_new_session = False
            code = process.poll()
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

        self._detach_client_from_session(client)
        session = self._sessions.get(session_id)
        if session is None:
            session = SessionState(session_id=session_id, cwd=cwd)
            self._sessions[session_id] = session
        elif session.cwd != cwd:
            await self._send_error(
                client,
                "cwd_mismatch",
                "Session already exists with a different cwd",
                detail=f"existing={session.cwd} requested={cwd}",
            )
            return

        try:
            await self._ensure_session_process(session)
        except Exception as exc:
            await self._send_error(client, "session_start_failed", "Failed to start session daemon", detail=str(exc))
            return

        session.client_ids.add(client.client_id)
        client.session_id = session_id

        await self._send_event(
            client,
            {
                "event": "attached",
                "session_id": session_id,
                "cwd": session.cwd,
                "clients": len(session.client_ids),
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
        elif cmd == "interrupt":
            forwarded = {"cmd": "interrupt"}
        elif cmd == "restore_session":
            restore_session_id = payload.get("session_id")
            if not isinstance(restore_session_id, str) or not restore_session_id.strip():
                await self._send_error(client, "invalid_restore_session", "restore_session.session_id is required")
                return
            forwarded = {"cmd": "restore_session", "session_id": restore_session_id.strip()}
        else:
            await self._send_error(client, "unknown_cmd", f"Unknown command: {cmd}")
            return

        try:
            await self._ensure_session_process(session)
            await self._forward_to_session(session, forwarded)
        except Exception as exc:
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
        await self._broadcast_session_event(session, {"event": "session_shutdown"})

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
            "set_safety_overrides",
            "interrupt",
            "restore_session",
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

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        raise
    finally:
        await server.stop()
