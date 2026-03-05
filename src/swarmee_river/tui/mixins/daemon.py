from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from swarmee_river.runtime_service.client import ensure_runtime_broker, load_runtime_discovery, runtime_discovery_path
from swarmee_river.tui.transport import (
    _DaemonTransport,
    _SocketTransport,
    _SubprocessTransport,
    send_daemon_command,
)

_BROKER_STARTUP_TIMEOUT_S = 6.0
_BROKER_STARTUP_TIMEOUT_WINDOWS_S = 20.0
_BROKER_ATTACH_ATTEMPTS_WINDOWS = 6
_BROKER_ATTACH_RETRY_DELAY_S = 0.35


def _is_retryable_broker_connect_error(exc: Exception) -> bool:
    return isinstance(exc, (FileNotFoundError, ConnectionError, TimeoutError, OSError))


def _is_windows_platform() -> bool:
    return os.name == "nt"


def _broker_diagnostics_hint(cwd: Path) -> str:
    discovery = runtime_discovery_path(cwd=cwd)
    broker_log: str | None = None
    if discovery.exists():
        with contextlib.suppress(Exception):
            payload = load_runtime_discovery(discovery)
            broker_log = payload.broker_log_path
    if broker_log:
        return f"discovery={discovery} broker_log={broker_log}"
    return f"discovery={discovery}"


class DaemonMixin:
    # Override in tests to inject a mock transport instead of spawning a subprocess.
    # Set to a callable() -> _DaemonTransport before the app starts.
    _test_transport_factory: "Callable[[], _DaemonTransport] | None" = None

    def _on_auth_connect_popup_closed(self, screen: Any) -> None:
        if self._auth_connect_screen is screen:
            self._auth_connect_screen = None
        self._auth_connect_capture_warnings = False
        self._auth_connect_completion_announced = False
        self._auth_connect_provider = None

    def _show_auth_connect_popup(self, provider: str, *, profile: str | None = None) -> None:
        from swarmee_river.tui.widgets import AuthConnectScreen

        normalized = str(provider or "").strip().lower()
        if normalized == "github_copilot":
            title = "Connect Copilot"
            intro = [
                "Follow the device login instructions below.",
                "Keep this popup open while entering the code in the browser.",
            ]
        else:
            resolved_profile = str(profile or "").strip() or "default"
            title = "Connect AWS"
            intro = [
                f"Starting AWS auth flow for profile '{resolved_profile}'.",
                "Keep this popup open while completing browser/device steps.",
            ]
        if self._auth_connect_screen is not None:
            with contextlib.suppress(Exception):
                self._auth_connect_screen.dismiss(None)
            self._auth_connect_screen = None
        screen = AuthConnectScreen(title=title, lines=intro)
        self._auth_connect_screen = screen
        self._auth_connect_provider = normalized or None
        self._auth_connect_capture_warnings = True
        self._auth_connect_completion_announced = False
        self.push_screen(screen, lambda result, active=screen: self._on_auth_connect_popup_closed(active))

    def _append_auth_connect_popup_line(self, text: str) -> None:
        line = str(text or "").strip()
        if not line:
            return
        screen = self._auth_connect_screen
        if screen is None:
            return
        with contextlib.suppress(Exception):
            screen.append_line(line)

    def _handle_connect_status_warning(self, text: str) -> bool:
        if not bool(self._auth_connect_capture_warnings):
            return False
        screen = self._auth_connect_screen
        if screen is None:
            return False
        self._append_auth_connect_popup_line(text)
        return True

    def _handle_connect_model_info_event(self, _event: dict[str, Any]) -> None:
        if not bool(self._auth_connect_capture_warnings):
            return
        if bool(getattr(self, "_auth_connect_completion_announced", False)):
            return
        self._append_auth_connect_popup_line("Authentication status refreshed.")
        self._append_auth_connect_popup_line("You can close this popup.")
        self._auth_connect_completion_announced = True

    def _handle_daemon_exit(self, proc: _DaemonTransport, *, return_code: int) -> None:
        if self.state.daemon.proc is not proc:
            return
        was_query_active = self.state.daemon.query_active
        self.state.daemon.ready = False
        self.state.daemon.pending_model_select_value = None
        self.state.daemon.query_active = False
        self._context_ready_for_sync = bool(self._context_sources)
        self._sops_ready_for_sync = bool(self._active_sop_names)
        self._reset_consent_panel()
        self._reset_error_action_prompt()
        self._clear_pending_tool_starts()
        self.state.daemon.proc = None
        self.state.daemon.runner_thread = None

        if self.state.daemon.status_timer is not None:
            self.state.daemon.status_timer.stop()
            self.state.daemon.status_timer = None
        if self._status_bar is not None:
            self._status_bar.set_state("idle")

        if was_query_active:
            self._finalize_turn(exit_status="error")
        else:
            self._reset_thinking_state()
        if self.state.daemon.is_shutting_down:
            return
        self._write_transcript_line(f"[daemon] exited unexpectedly (code {return_code}).")
        self._write_transcript_line("[daemon] run /daemon restart to restart the background agent.")

    def _stream_daemon_output(self, proc: _DaemonTransport) -> None:
        try:
            while True:
                raw_line = proc.read_line()
                if raw_line == "":
                    break
                self._call_from_thread_safe(self._handle_output_line, raw_line.rstrip("\n"), raw_line)
        except Exception as exc:
            self._call_from_thread_safe(self._write_transcript_line, f"[daemon] output stream error: {exc}")
        finally:
            return_code = 0
            with contextlib.suppress(Exception):
                return_code = proc.wait()
            self._call_from_thread_safe(self._handle_daemon_exit, proc, return_code=return_code)

    def _shutdown_transport(self, proc: _DaemonTransport) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        if isinstance(proc, _SocketTransport):
            # Just disconnect; the broker keeps the session daemon alive
            # for other attached clients and cleans up after idle timeout.
            proc.close()
            return
        send_daemon_command(proc, {"cmd": "shutdown"})
        with contextlib.suppress(Exception):
            proc.wait(timeout=3.0)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.close()

    def _request_daemon_shutdown(self) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None:
            self.state.daemon.ready = False
            self._write_transcript_line("[daemon] already stopped.")
            return
        self.state.daemon.ready = False
        self.state.daemon.is_shutting_down = True
        payload = {"cmd": "shutdown_service"} if isinstance(proc, _SocketTransport) else {"cmd": "shutdown"}
        if send_daemon_command(proc, payload):
            self._write_transcript_line("[daemon] shutdown requested.")
            return
        self.state.daemon.is_shutting_down = False
        self._write_transcript_line("[daemon] failed to send shutdown command.")

    def _spawn_daemon(self, *, restart: bool = False) -> None:
        from swarmee_river.tui.transport import spawn_swarmee_daemon

        self.state.daemon.is_shutting_down = False
        proc = self.state.daemon.proc
        if proc is not None and proc.poll() is None:
            if restart:
                self.state.daemon.pending_model_select_value = None
                self._shutdown_transport(proc)
                self.state.daemon.proc = None
            else:
                return

        _stored_sid = (self.state.daemon.session_id or "").strip()
        requested_session_id = (_stored_sid if _stored_sid.lower() != "none" else "") or uuid.uuid4().hex
        self.state.daemon.session_id = requested_session_id
        daemon: _DaemonTransport | None = None
        broker_error: Exception | None = None

        if self._test_transport_factory is not None:
            try:
                daemon = self._test_transport_factory()
            except Exception as exc:
                self.state.daemon.ready = False
                self._write_transcript_line(f"[daemon] test transport factory failed: {exc}")
                return
        else:
            env_overrides = self._model_env_overrides()
            cwd = Path.cwd()
            is_windows = _is_windows_platform()
            broker_timeout_s = _BROKER_STARTUP_TIMEOUT_WINDOWS_S if is_windows else _BROKER_STARTUP_TIMEOUT_S
            try:
                ensure_runtime_broker(cwd=cwd, timeout_s=broker_timeout_s)
            except Exception as exc:
                broker_error = exc

            connect_attempts = _BROKER_ATTACH_ATTEMPTS_WINDOWS if is_windows else 1
            wrote_retry_notice = False
            for attempt in range(connect_attempts):
                try:
                    daemon = _SocketTransport.connect(
                        session_id=requested_session_id,
                        cwd=cwd,
                        client_name="swarmee-tui",
                        surface="tui",
                        env_overrides=env_overrides,
                    )
                    break
                except Exception as exc:
                    if broker_error is None:
                        broker_error = exc
                    is_retryable = _is_retryable_broker_connect_error(exc)
                    has_retries_remaining = (attempt + 1) < connect_attempts
                    if not is_retryable or not has_retries_remaining:
                        break
                    if not wrote_retry_notice:
                        wrote_retry_notice = True
                        self._write_transcript_line("[daemon] runtime broker still starting; retrying connection...")
                    time.sleep(_BROKER_ATTACH_RETRY_DELAY_S)

            if daemon is None:
                try:
                    daemon_proc = spawn_swarmee_daemon(
                        session_id=requested_session_id,
                        env_overrides=env_overrides,
                    )
                    daemon = _SubprocessTransport(daemon_proc)
                except Exception as exc:
                    self.state.daemon.ready = False
                    self._write_transcript_line(f"[daemon] failed to start: {exc}")
                    return

        self.state.daemon.proc = daemon
        self.state.daemon.ready = False
        self._context_ready_for_sync = bool(self._context_sources)
        self._sops_ready_for_sync = bool(self._active_sop_names)
        self.state.daemon.runner_thread = threading.Thread(
            target=self._stream_daemon_output,
            args=(daemon,),
            daemon=True,
            name="swarmee-tui-daemon-stream",
        )
        self.state.daemon.runner_thread.start()
        diagnostics_hint = _broker_diagnostics_hint(Path.cwd())
        if isinstance(daemon, _SocketTransport):
            self._write_transcript_line(
                "[daemon] connected to runtime broker, waiting for ready event."
                f" ({diagnostics_hint})"
            )
        else:
            if broker_error is not None and not isinstance(broker_error, FileNotFoundError):
                self._write_transcript_line(
                    f"[daemon] runtime broker unavailable ({broker_error}); using local daemon. ({diagnostics_hint})"
                )
            self._write_transcript_line("[daemon] started, waiting for ready event.")
        self._save_session()

    def _tick_status(self) -> None:
        if self.state.daemon.run_start_time is not None and self._status_bar is not None:
            self._status_bar.set_elapsed(time.time() - self.state.daemon.run_start_time)

    def _start_run(self, prompt: str, *, auto_approve: bool, mode: str | None = None) -> None:
        from textual.widgets import Select

        from swarmee_river.tui.transport import send_daemon_command

        if not self.state.daemon.ready:
            self._write_transcript_line("[run] daemon is not ready. Use /daemon restart.")
            return
        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None:
            self._write_transcript_line("[run] daemon is not running. Use /daemon restart.")
            self.state.daemon.ready = False
            return
        if self.state.daemon.query_active:
            self._write_transcript_line("[run] already running; use /stop.")
            return
        self._dismiss_action_sheet(restore_focus=False)
        self._sync_selected_model_before_run()

        self.state.plan.pending_prompt = None
        self._refresh_plan_actions_visibility()
        self._reset_artifacts_panel()
        self._reset_consent_panel()
        self._reset_error_action_prompt()
        self._reset_issues_panel()
        self._current_assistant_chunks = []
        self._streaming_buffer = []
        self._cancel_streaming_flush_timer()
        self._streaming_last_flush_mono = 0.0
        self._current_assistant_model = None
        self._current_assistant_timestamp = None
        self._assistant_completion_seen_turn = False
        self._assistant_placeholder_written = False
        self._stream_render_warning_emitted_turn = False
        self._structured_assistant_seen_turn = False
        self._raw_assistant_lines_suppressed_turn = 0
        self._last_structured_assistant_text_turn = ""
        self._callback_event_trace_turn = []
        self._active_assistant_message = None
        self._active_reasoning_block = None
        self._reset_thinking_state()
        self._last_thinking_text = ""
        self._thinking_seen_turn = False
        self._thinking_unavailable_notice_emitted_turn = False
        self._tool_blocks = {}
        self._clear_pending_tool_starts()
        self._tool_progress_pending_ids = set()
        self._cancel_tool_progress_flush_timer()
        self.state.daemon.run_tool_count = 0
        self.state.daemon.run_start_time = time.time()
        self.state.daemon.run_active_tier_warning_emitted = False
        self.state.plan.step_counter = 0
        self.state.plan.completion_announced = False
        mode_normalized = (mode or "").strip().lower()
        if mode_normalized == "execute" and self.state.plan.current_steps_total > 0:
            self.state.plan.current_step_statuses = ["pending"] * self.state.plan.current_steps_total
            self.state.plan.current_active_step = None
            self.state.plan.updates_seen = False
            self._render_plan_panel_from_status()
        else:
            self.state.plan.current_steps_total = 0
            self.state.plan.current_summary = ""
            self.state.plan.current_steps = []
            self.state.plan.current_step_statuses = []
            self.state.plan.current_active_step = None
            self.state.plan.updates_seen = False
        self.state.plan.received_structured_plan = False
        self.state.daemon.turn_output_chunks = []
        self.state.daemon.last_usage = None
        self.state.daemon.last_cost_usd = None
        if self._status_bar is not None:
            self._status_bar.set_state("running")
            self._status_bar.set_tool_count(0)
            self._status_bar.set_elapsed(0.0)
            self._status_bar.set_model(self._current_model_summary())
            self._status_bar.set_usage(None, cost_usd=None)
            self._status_bar.set_context(
                prompt_tokens_est=self.state.daemon.last_prompt_tokens_est,
                budget_tokens=self.state.daemon.last_budget_tokens,
            )
            if mode_normalized == "execute" and self.state.plan.current_steps_total > 0:
                self._refresh_plan_status_bar()
            else:
                self._status_bar.set_plan_step(current=None, total=None)
        self._refresh_prompt_metrics()
        if self.state.daemon.status_timer is not None:
            self.state.daemon.status_timer.stop()
        self.state.daemon.status_timer = self.set_interval(1.0, self._tick_status)
        self._last_prompt = prompt
        self._last_run_auto_approve = auto_approve
        self.state.daemon.query_active = True
        # Show immediate "LLM in progress" feedback even before first stream/tool events arrive.
        self._record_thinking_event("")
        self._current_assistant_model = self.state.daemon.current_model
        self._current_assistant_timestamp = self._turn_timestamp()
        self._assistant_placeholder_written = False
        desired_tier = ""
        pending_value = (self.state.daemon.pending_model_select_value or "").strip().lower()
        if "|" in pending_value:
            _pending_provider, pending_tier = pending_value.split("|", 1)
            desired_tier = pending_tier.strip().lower()
        if not desired_tier:
            desired_tier = (self.state.daemon.model_tier_override or "").strip().lower()
        if not desired_tier:
            with contextlib.suppress(Exception):
                selector = self.query_one("#model_select", Select)
                selected_value = str(getattr(selector, "value", "")).strip()
                from swarmee_river.tui.model_select import parse_model_select_value

                parsed = parse_model_select_value(selected_value)
                if parsed is not None:
                    _provider_name, parsed_tier = parsed
                    desired_tier = parsed_tier.strip().lower()
        command: dict[str, Any] = {
            "cmd": "query",
            "text": prompt,
            "auto_approve": bool(auto_approve),
        }
        if desired_tier:
            command["tier"] = desired_tier
        if mode:
            command["mode"] = mode
        if not send_daemon_command(proc, command):
            self.state.daemon.query_active = False
            self._dismiss_thinking(emit_summary=False)
            if self.state.daemon.status_timer is not None:
                self.state.daemon.status_timer.stop()
                self.state.daemon.status_timer = None
            if self._status_bar is not None:
                self._status_bar.set_state("idle")
            self._write_transcript_line("[run] failed to send query to daemon.")

    def _stop_run(self) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None:
            self._write_transcript_line("[run] no active run.")
            self.state.daemon.ready = False
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            return
        if not self.state.daemon.query_active:
            self._write_transcript_line("[run] no active run.")
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            return
        self._flush_all_streaming_buffers()
        self._clear_pending_tool_starts()
        self._dismiss_thinking(emit_summary=True)
        if send_daemon_command(proc, {"cmd": "interrupt"}):
            self._write_transcript_line("[run] interrupt requested.")
        else:
            self._write_transcript_line("[run] failed to send interrupt.")
        self._reset_consent_panel()
        self._reset_error_action_prompt()

    def _request_provider_connect(self, provider: str, *, profile: str | None = None) -> bool:

        from swarmee_river.tui.commands import _CONNECT_USAGE_TEXT
        from swarmee_river.utils.provider_utils import normalize_provider_name

        raw = (provider or "").strip() or "github_copilot"
        normalized = normalize_provider_name(raw)
        if normalized in {"aws", "bedrock"}:
            normalized = "bedrock"
        if normalized not in {"github_copilot", "bedrock"}:
            self._write_transcript_line(_CONNECT_USAGE_TEXT)
            return False

        proc = self.state.daemon.proc
        if not self.state.daemon.ready or proc is None or proc.poll() is not None:
            self._write_transcript_line("[connect] daemon is not ready.")
            return False
        if self.state.daemon.query_active:
            self._write_transcript_line("[connect] cannot connect while a run is active.")
            return False

        payload: dict[str, Any] = {"cmd": "connect", "provider": normalized}
        if normalized == "github_copilot":
            payload.update({"method": "device", "open_browser": True})
            self._write_transcript_line("[connect] starting provider auth for github_copilot...")
            self._show_auth_connect_popup("github_copilot")
        else:
            try:
                from swarmee_river.settings import load_settings

                settings = load_settings()
                bedrock = settings.models.providers.get("bedrock")
                extra = dict(getattr(bedrock, "extra", {}) or {})
                configured_profile = str(extra.get("aws_profile") or "").strip()
            except Exception:
                configured_profile = ""
            resolved_profile = (profile or "").strip() or configured_profile or "default"
            payload.update({"method": "sso", "profile": resolved_profile})
            self._write_transcript_line(f"[connect] starting provider auth for bedrock (profile={resolved_profile})...")
            self._show_auth_connect_popup("bedrock", profile=resolved_profile)
        self._pending_connect_payload = dict(payload)
        if not send_daemon_command(proc, payload):
            self._write_transcript_line("[connect] failed to send command.")
            self._append_auth_connect_popup_line("Failed to send connect command to daemon.")
            return False
        return True

    def _recover_runtime_unknown_proxy_command(self, command: str) -> bool:
        from pathlib import Path

        from swarmee_river.runtime_service.client import shutdown_runtime_broker

        normalized = str(command or "").strip().lower()
        if normalized not in {"connect", "auth"}:
            return False
        if normalized in self._runtime_proxy_recovery_attempted:
            return False
        proc = self.state.daemon.proc
        if not isinstance(proc, _SocketTransport):
            return False
        self._runtime_proxy_recovery_attempted.add(normalized)
        self._write_transcript_line(
            f"[daemon] runtime broker does not support '{normalized}'. Restarting broker/session transport..."
        )
        with contextlib.suppress(Exception):
            shutdown_runtime_broker(cwd=Path.cwd())
        if normalized == "connect" and isinstance(self._pending_connect_payload, dict):
            self._pending_connect_retry_payload = dict(self._pending_connect_payload)
            provider_label = str(self._pending_connect_retry_payload.get("provider", "provider")).strip()
            self._write_transcript_line(f"[connect] will retry auth for {provider_label} after reconnect.")
        self._spawn_daemon(restart=True)
        return True

    def _flush_pending_connect_retry(self) -> None:
        payload = self._pending_connect_retry_payload
        if not isinstance(payload, dict):
            return
        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None or not self.state.daemon.ready or self.state.daemon.query_active:
            return
        if send_daemon_command(proc, payload):
            provider_label = str(payload.get("provider", "provider")).strip()
            self._write_transcript_line(f"[connect] retrying provider auth for {provider_label}...")
            self._append_auth_connect_popup_line(f"Retrying auth for {provider_label}...")
            self._pending_connect_retry_payload = None

    def _handle_pre_run_command(self, text: str) -> bool:
        from swarmee_river.tui.commands import (
            _AUTH_USAGE_TEXT,
            _COMPACT_USAGE_TEXT,
            _CONSENT_USAGE_TEXT,
            _CONTEXT_USAGE_TEXT,
            _DIAGNOSTICS_USAGE_TEXT,
            _EXPAND_USAGE_TEXT,
            _OPEN_USAGE_TEXT,
            _SEARCH_USAGE_TEXT,
            _SOP_USAGE_TEXT,
            _TEXT_USAGE_TEXT,
            _THINKING_USAGE_TEXT,
            classify_pre_run_command,
        )
        from swarmee_river.tui.widgets import CommandPalette

        classified = classify_pre_run_command(text)
        if classified is None:
            return False

        action, argument = classified
        if action == "open":
            self._open_artifact(argument or "")
            return True
        if action == "help":
            lines = [f"  {cmd:<16} {desc}" for cmd, desc in CommandPalette.TUI_COMMANDS]
            self._write_transcript_line("Available commands:\n" + "\n".join(lines))
            return True
        if action == "open_usage":
            self._write_transcript_line(_OPEN_USAGE_TEXT)
            return True
        if action == "restore":
            self._restore_available_session()
            return True
        if action == "new":
            self._start_fresh_session()
            return True
        if action == "context":
            return self._handle_context_command(argument or "")
        if action == "context_usage":
            self._write_transcript_line(_CONTEXT_USAGE_TEXT)
            return True
        if action == "sop":
            return self._handle_sop_command(argument or "")
        if action == "sop_usage":
            self._write_transcript_line(_SOP_USAGE_TEXT)
            return True
        if action == "expand":
            self._expand_tool_call(argument or "")
            return True
        if action == "expand_usage":
            self._write_transcript_line(_EXPAND_USAGE_TEXT)
            return True
        if action == "search":
            self._search_transcript(argument or "")
            return True
        if action == "search_usage":
            self._write_transcript_line(_SEARCH_USAGE_TEXT)
            return True
        if action == "text":
            self._toggle_transcript_mode()
            return True
        if action == "text_usage":
            self._write_transcript_line(_TEXT_USAGE_TEXT)
            return True
        if action == "thinking":
            self._show_thinking_text()
            return True
        if action == "thinking_usage":
            self._write_transcript_line(_THINKING_USAGE_TEXT)
            return True
        if action == "compact":
            self._request_context_compact()
            return True
        if action == "compact_usage":
            self._write_transcript_line(_COMPACT_USAGE_TEXT)
            return True
        if action == "stop":
            self._stop_run()
            return True
        if action == "exit":
            self.action_quit()
            return True
        if action == "daemon_restart":
            self._spawn_daemon(restart=True)
            return True
        if action == "daemon_stop":
            self._request_daemon_shutdown()
            return True
        if action == "consent_usage":
            self._write_transcript_line(_CONSENT_USAGE_TEXT)
            return True
        if action == "consent":
            self._submit_consent_choice((argument or "").strip())
            return True
        if action == "connect":
            raw = (argument or "").strip()
            if not raw:
                self._request_provider_connect("github_copilot")
                return True
            parts = raw.split(maxsplit=1)
            provider = parts[0].strip()
            profile = parts[1].strip() if len(parts) > 1 else None
            self._request_provider_connect(provider, profile=profile)
            return True
        if action == "auth_usage":
            self._write_transcript_line(_AUTH_USAGE_TEXT)
            return True
        if action == "auth":
            raw = (argument or "").strip()
            normalized = raw.lower()
            proc = self.state.daemon.proc
            if not self.state.daemon.ready or proc is None or proc.poll() is not None:
                self._write_transcript_line("[auth] daemon is not ready.")
                return True
            if self.state.daemon.query_active:
                self._write_transcript_line("[auth] cannot run while a run is active.")
                return True
            if not raw:
                self._write_transcript_line(_AUTH_USAGE_TEXT)
                return True
            if normalized in {"list", "ls"}:
                send_daemon_command(proc, {"cmd": "auth", "action": "list"})
                return True
            if normalized.startswith("logout"):
                parts = raw.split()
                provider = parts[1].strip() if len(parts) >= 2 else "github_copilot"
                send_daemon_command(proc, {"cmd": "auth", "action": "logout", "provider": provider})
                return True
            self._write_transcript_line(_AUTH_USAGE_TEXT)
            return True
        if action == "diagnostics_usage":
            self._write_transcript_line(_DIAGNOSTICS_USAGE_TEXT)
            return True
        if action == "diagnostics_bundle":
            from swarmee_river.diagnostics import create_support_bundle

            try:
                bundle_path = create_support_bundle(cwd=Path.cwd())
            except Exception as exc:
                self._write_transcript_line(f"[diagnostics] failed to create support bundle: {exc}")
                return True
            self._add_artifact_paths([str(bundle_path)])
            self._write_transcript_line(f"[diagnostics] support bundle created: {bundle_path}")
            return True
        if action.startswith("model:"):
            normalized = text.lower()
            return self._handle_model_command(normalized)
        return False

    def _handle_post_run_command(self, text: str) -> bool:
        from swarmee_river.tui.commands import classify_post_run_command

        classified = classify_post_run_command(text)
        if classified is None:
            return False

        action, argument = classified

        if action == "approve":
            self._dispatch_plan_action("approve")
            return True

        if action == "replan":
            self._dispatch_plan_action("replan")
            return True

        if action == "clearplan":
            self._dispatch_plan_action("clearplan")
            return True

        if action == "plan_mode":
            self._default_auto_approve = False
            self._update_prompt_placeholder()
            self._write_transcript_line("[mode] auto-approve disabled for default prompts.")
            return True

        if action == "plan_prompt":
            prompt = (argument or "").strip()
            if not prompt:
                self._write_transcript_line("Usage: /plan <prompt>")
                return True
            self._start_run(prompt, auto_approve=False, mode="plan")
            return True

        if action == "run_mode":
            self._default_auto_approve = True
            self._update_prompt_placeholder()
            self._write_transcript_line("[mode] auto-approve enabled for default prompts.")
            return True

        if action == "run_prompt":
            prompt = (argument or "").strip()
            if not prompt:
                self._write_transcript_line("Usage: /run <prompt>")
                return True
            self._start_run(prompt, auto_approve=True, mode="execute")
            return True

        return False
