from __future__ import annotations

import contextlib
import logging
from typing import Any

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.state_paths import logs_dir
from swarmee_river.tui.text_sanitize import sanitize_output_text

_CONSENT_CHOICES = {"y", "n", "a", "v"}
_THINKING_EXPORT_MAX_CHARS = 5000
_TURN_TRACE_MAX = 16

_LOGGER = logging.getLogger(__name__)


def _artifact_paths_from_event(event: Any) -> list[str]:
    kind = str(getattr(event, "kind", "")).strip().lower()
    if kind != "artifact":
        return []
    meta = getattr(event, "meta", None)
    if not isinstance(meta, dict):
        return []
    path = str(meta.get("path", "")).strip()
    return [path] if path else []


class OutputMixin:
    def _record_consent_history(self, line: str) -> None:
        entry = line.strip()
        if not entry:
            return
        self._consent_history_lines.append(entry)
        if len(self._consent_history_lines) > 200:
            self._consent_history_lines = self._consent_history_lines[-200:]

    def _trace_turn_event(self, label: str) -> None:
        token = str(label or "").strip()
        if not token:
            return
        trace = list(getattr(self, "_callback_event_trace_turn", []))
        trace.append(token)
        if len(trace) > _TURN_TRACE_MAX:
            trace = trace[-_TURN_TRACE_MAX:]
        self._callback_event_trace_turn = trace

    @staticmethod
    def _normalize_assistant_compare_text(text: str) -> str:
        return " ".join(sanitize_output_text(str(text or "")).split())

    def _structured_assistant_compare_text(self) -> str:
        parts: list[str] = []
        snapshot = str(getattr(self, "_last_structured_assistant_text_turn", "") or "")
        if snapshot:
            parts.append(snapshot)
        current_chunks = getattr(self, "_current_assistant_chunks", None)
        if isinstance(current_chunks, list) and current_chunks:
            parts.append("".join(str(chunk) for chunk in current_chunks))
        streaming_buffer = getattr(self, "_streaming_buffer", None)
        if isinstance(streaming_buffer, list) and streaming_buffer:
            parts.append("".join(str(chunk) for chunk in streaming_buffer))
        active = getattr(self, "_active_assistant_message", None)
        full_text = getattr(active, "full_text", None)
        if isinstance(full_text, str) and full_text:
            parts.append(full_text)
        return self._normalize_assistant_compare_text("".join(parts))

    def _has_live_assistant_segment(self) -> bool:
        current_chunks = getattr(self, "_current_assistant_chunks", None)
        if isinstance(current_chunks, list) and current_chunks:
            return True
        streaming_buffer = getattr(self, "_streaming_buffer", None)
        if isinstance(streaming_buffer, list) and streaming_buffer:
            return True
        active = getattr(self, "_active_assistant_message", None)
        if active is None:
            return False
        full_text = getattr(active, "full_text", None)
        return isinstance(full_text, str) and bool(full_text)

    def _finalize_assistant_segment_before_intermediate_block(self) -> None:
        if not self._has_live_assistant_segment():
            return
        self._finalize_assistant_message()

    def _should_suppress_raw_assistant_output(self, text: str) -> bool:
        if not self.state.daemon.query_active:
            return False
        if not bool(getattr(self, "_structured_assistant_seen_turn", False)):
            return False
        candidate = self._normalize_assistant_compare_text(text)
        if not candidate:
            return False
        structured = self._structured_assistant_compare_text()
        if not structured:
            return False
        return candidate in structured

    def _record_raw_assistant_suppression(self, text: str) -> None:
        current = int(getattr(self, "_raw_assistant_lines_suppressed_turn", 0) or 0)
        self._raw_assistant_lines_suppressed_turn = current + 1
        self._trace_turn_event("raw_suppressed")
        _LOGGER.debug(
            "Suppressed raw assistant echo line during structured turn output (line=%r trace=%s).",
            sanitize_output_text(text)[:200],
            " -> ".join(getattr(self, "_callback_event_trace_turn", [])),
        )

    def _show_thinking_text(self) -> None:
        from swarmee_river.tui.widgets import render_system_message

        current_text = "".join(self._thinking_buffer).strip()
        text = current_text or (self._last_thinking_text or "").strip()
        if not text:
            self._write_transcript_line("No thinking content from this turn.")
            return

        total_chars = len(text)
        if total_chars > _THINKING_EXPORT_MAX_CHARS:
            shown = text[-_THINKING_EXPORT_MAX_CHARS:]
            self._write_transcript_line(
                f"[thinking] showing last {_THINKING_EXPORT_MAX_CHARS:,} of {total_chars:,} chars."
            )
            text = shown
        else:
            self._write_transcript_line(f"[thinking] showing {total_chars:,} chars.")

        self._mount_transcript_widget(render_system_message(text), plain_text=text)

    def _cancel_consent_hide_timer(self) -> None:
        timer = self._consent_hide_timer
        self._consent_hide_timer = None
        if timer is not None:
            with contextlib.suppress(RuntimeError):
                timer.stop()

    def _complete_consent_prompt_hide(self, expected_nonce: int) -> None:
        from textual.widgets import TextArea

        self._consent_hide_timer = None
        if expected_nonce != self._consent_prompt_nonce:
            return
        widget = self._consent_prompt_widget
        if widget is not None:
            with contextlib.suppress(Exception):
                widget.hide_prompt()
        with contextlib.suppress(Exception):
            self.query_one("#prompt", TextArea).focus()

    def _schedule_consent_prompt_hide(self, *, delay: float = 1.0) -> None:
        self._cancel_consent_hide_timer()
        nonce = self._consent_prompt_nonce
        self._consent_hide_timer = self.set_timer(
            delay,
            lambda: self._complete_consent_prompt_hide(nonce),
        )

    def _show_consent_prompt(
        self,
        *,
        context: str,
        options: list[str] | None = None,
        alert: bool = True,
        changed_paths: list[str] | None = None,
        diff_preview: str | None = None,
        diff_hidden_lines: int = 0,
        non_text_change_summary: str | None = None,
        diff_stats: dict[str, Any] | None = None,
    ) -> None:
        from swarmee_river.tui.widgets import ConsentPrompt, extract_consent_tool_name

        widget = self._consent_prompt_widget
        if widget is None:
            with contextlib.suppress(Exception):
                widget = self.query_one("#consent_prompt", ConsentPrompt)
                self._consent_prompt_widget = widget
        if widget is None:
            return
        self._cancel_consent_hide_timer()
        self._consent_prompt_nonce += 1
        self._consent_active = True
        self._consent_tool_name = extract_consent_tool_name(context)
        normalized_options = options or ["y", "n", "a", "v"]
        widget.set_prompt(
            context=context,
            options=normalized_options,
            alert=alert,
            changed_paths=changed_paths,
            diff_preview=diff_preview,
            diff_hidden_lines=diff_hidden_lines,
            non_text_change_summary=non_text_change_summary,
            diff_stats=diff_stats,
        )

    def _reset_consent_panel(self) -> None:
        self._cancel_consent_hide_timer()
        self._consent_prompt_nonce += 1
        self._consent_active = False
        self._consent_buffer = []
        self._consent_tool_name = "tool"
        widget = self._consent_prompt_widget
        if widget is not None:
            with contextlib.suppress(Exception):
                widget.hide_prompt()

    def _reset_error_action_prompt(self) -> None:
        self._pending_error_action = None
        widget = self._error_action_prompt_widget
        if widget is not None:
            with contextlib.suppress(Exception):
                widget.hide_prompt()

    def _reset_run_local_ui_state(self, *, clear_prompt_context: bool) -> None:
        self._reset_consent_panel()
        self._reset_error_action_prompt()
        self._flush_all_streaming_buffers()
        self._finalize_assistant_message()
        self._dismiss_thinking(emit_summary=False)
        self._clear_pending_tool_starts()
        self._cancel_tool_progress_flush_timer()
        self._tool_progress_pending_ids = set()
        self.state.daemon.run_tool_count = 0
        self.state.daemon.run_start_time = None
        self.state.daemon.query_active = False
        self._current_assistant_chunks = []
        self._streaming_buffer = []
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
        self._last_thinking_text = ""
        self._thinking_seen_turn = False
        self._thinking_unavailable_notice_emitted_turn = False
        self._tool_blocks = {}
        self.state.daemon.turn_output_chunks = []
        self.state.daemon.last_usage = None
        self.state.daemon.last_cost_usd = None
        if clear_prompt_context:
            self.state.daemon.last_prompt_tokens_est = 0
            self._prompt_input_tokens_est = 0
        if self.state.daemon.status_timer is not None:
            self.state.daemon.status_timer.stop()
            self.state.daemon.status_timer = None
        if self._status_bar is not None:
            self._status_bar.set_state("idle")
            self._status_bar.set_tool_count(0)
            self._status_bar.set_elapsed(0.0)
            self._status_bar.set_usage(None, cost_usd=None)
            self._status_bar.set_context(
                prompt_tokens_est=self.state.daemon.last_prompt_tokens_est,
                budget_tokens=self.state.daemon.last_budget_tokens,
            )
        self._refresh_prompt_metrics()

    def _next_available_tier_name(self) -> str | None:
        current_tier = (self.state.daemon.tier or "").strip().lower()
        available = [
            str(item.get("name", "")).strip().lower()
            for item in self.state.daemon.tiers
            if isinstance(item, dict) and bool(item.get("available"))
        ]
        if not available:
            return None
        if current_tier in available:
            idx = available.index(current_tier)
            for candidate in available[idx + 1 :]:
                if candidate:
                    return candidate
        for candidate in available:
            if candidate and candidate != current_tier:
                return candidate
        return None

    def _show_tool_error_actions(self, *, tool_use_id: str, tool_name: str) -> None:
        widget = self._error_action_prompt_widget
        if widget is None:
            return
        self._pending_error_action = {
            "kind": "tool",
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
        }
        with contextlib.suppress(Exception):
            widget.show_tool_error(tool_name=tool_name, tool_use_id=tool_use_id)

    def _show_escalation_actions(self, *, next_tier: str | None = None) -> None:
        widget = self._error_action_prompt_widget
        if widget is None:
            return
        resolved_next = (next_tier or "").strip().lower() or self._next_available_tier_name()
        self._pending_error_action = {
            "kind": "escalation",
            "next_tier": resolved_next or None,
        }
        with contextlib.suppress(Exception):
            widget.show_escalation(next_tier=resolved_next or None)

    def _resume_after_error(self, *, escalate: bool) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        if self.state.daemon.query_active:
            self._write_transcript_line("[run] already running; use /stop.")
            return
        prompt = (self._last_prompt or "").strip()
        if not prompt:
            self._write_transcript_line("[run] no previous prompt to continue.")
            self._reset_error_action_prompt()
            return
        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None or not self.state.daemon.ready:
            self._write_transcript_line("[run] daemon is not ready. Use /daemon restart.")
            self._reset_error_action_prompt()
            return
        pending = self._pending_error_action or {}
        if escalate:
            next_tier = str(pending.get("next_tier", "")).strip().lower()
            if next_tier:
                if not send_daemon_command(proc, {"cmd": "set_tier", "tier": next_tier}):
                    self._write_transcript_line("[model] failed to request tier escalation.")
                    return
                self._write_transcript_line(f"[model] escalated to {next_tier}.")
            else:
                self._write_transcript_line("[model] no higher tier available; continuing on current tier.")
        self._reset_error_action_prompt()
        self._start_run(prompt, auto_approve=self._last_run_auto_approve, mode="execute")

    def _retry_failed_tool(self) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        action = self._pending_error_action or {}
        tool_use_id = str(action.get("tool_use_id", "")).strip()
        if not tool_use_id:
            self._write_transcript_line("[recovery] no failed tool selected.")
            self._reset_error_action_prompt()
            return
        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None or not self.state.daemon.ready:
            self._write_transcript_line("[recovery] daemon is not ready.")
            self._reset_error_action_prompt()
            return
        if send_daemon_command(proc, {"cmd": "retry_tool", "tool_use_id": tool_use_id}):
            self._write_transcript_line(f"[recovery] retry requested for tool {tool_use_id}.")
            self._reset_error_action_prompt()
        else:
            self._write_transcript_line("[recovery] failed to send retry request.")

    def _skip_failed_tool(self) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        action = self._pending_error_action or {}
        tool_use_id = str(action.get("tool_use_id", "")).strip()
        if not tool_use_id:
            self._write_transcript_line("[recovery] no failed tool selected.")
            self._reset_error_action_prompt()
            return
        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None or not self.state.daemon.ready:
            self._write_transcript_line("[recovery] daemon is not ready.")
            self._reset_error_action_prompt()
            return
        if send_daemon_command(proc, {"cmd": "skip_tool", "tool_use_id": tool_use_id}):
            self._write_transcript_line(f"[recovery] skip requested for tool {tool_use_id}.")
            self._reset_error_action_prompt()
        else:
            self._write_transcript_line("[recovery] failed to send skip request.")

    def _apply_consent_capture(self, line: str) -> None:
        from swarmee_river.tui.event_types import update_consent_capture as _update_consent_capture

        previously_active = self._consent_active
        next_active, next_buffer = _update_consent_capture(
            self._consent_active,
            self._consent_buffer,
            line,
            max_lines=20,
        )
        if next_active != self._consent_active or next_buffer != self._consent_buffer:
            self._consent_active = next_active
            self._consent_buffer = next_buffer
            if self._consent_active:
                context = "\n".join(self._consent_buffer[-4:])
                self._show_consent_prompt(
                    context=context,
                    options=["y", "n", "a", "v"],
                    alert=not previously_active and next_active,
                )

    def _consent_decision_line(self, choice: str) -> str:
        tool_name = self._consent_tool_name or "tool"
        if choice == "y":
            return f"✓ {tool_name} allowed"
        if choice == "n":
            return f"✗ {tool_name} denied"
        if choice == "a":
            return f"✓ {tool_name} always allowed (session)"
        if choice == "v":
            return f"✗ {tool_name} never allowed (session)"
        return f"[consent] response: {choice}"

    def _submit_consent_choice(self, choice: str) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        normalized_choice = choice.strip().lower()
        if normalized_choice not in _CONSENT_CHOICES:
            self._write_transcript_line("Usage: /consent <y|n|a|v>")
            return
        if not self._consent_active:
            self._write_transcript_line("[consent] no active prompt.")
            return
        if self.state.daemon.proc is None or self.state.daemon.proc.poll() is not None:
            self._write_transcript_line("[consent] daemon is not running.")
            self._reset_consent_panel()
            return
        decision_line = self._consent_decision_line(normalized_choice)
        self._write_transcript(decision_line)
        self._record_consent_history(decision_line)
        self._consent_active = False
        self._consent_buffer = []
        approved = normalized_choice in {"y", "a"}
        widget = self._consent_prompt_widget
        if widget is not None:
            with contextlib.suppress(Exception):
                widget.show_confirmation(decision_line, approved=approved)
        self._schedule_consent_prompt_hide(delay=1.0)
        if not send_daemon_command(self.state.daemon.proc, {"cmd": "consent_response", "choice": normalized_choice}):
            self._write_transcript_line("[consent] failed to send response (stdin unavailable).")
            return

    def _finalize_assistant_message(self) -> None:
        from swarmee_river.tui.widgets import render_assistant_message

        self._cancel_streaming_flush_timer()
        self._flush_streaming_buffer()
        if not self._current_assistant_chunks:
            if self._active_assistant_message is not None:
                with contextlib.suppress(Exception):
                    self._active_assistant_message.finalize(
                        model=self._current_assistant_model,
                        timestamp=self._current_assistant_timestamp or self._turn_timestamp(),
                    )
            self._active_assistant_message = None
            self._current_assistant_model = None
            self._current_assistant_timestamp = None
            self._assistant_placeholder_written = False
            return

        full_text = "".join(self._current_assistant_chunks)
        self._last_assistant_text = full_text
        model = self._current_assistant_model
        timestamp = self._current_assistant_timestamp or self._turn_timestamp()
        plain_lines = [full_text]
        meta_parts = [part for part in [model, timestamp] if isinstance(part, str) and part.strip()]
        if meta_parts:
            plain_lines.append(" · ".join(meta_parts))
        if self._active_assistant_message is not None:
            with contextlib.suppress(Exception):
                self._active_assistant_message.finalize(model=model, timestamp=timestamp)
            if meta_parts:
                self._record_transcript_fallback(" · ".join(meta_parts))
        elif not self._assistant_placeholder_written:
            self._mount_transcript_widget(
                render_assistant_message(full_text, model=model, timestamp=timestamp),
                plain_text="\n".join(plain_lines),
            )
        elif meta_parts:
            meta_line = " · ".join(meta_parts)
            self._record_transcript_fallback(meta_line)

        self._current_assistant_chunks = []
        self._streaming_buffer = []
        self._current_assistant_model = None
        self._current_assistant_timestamp = None
        self._assistant_placeholder_written = False
        self._active_assistant_message = None

    def _handle_output_line(self, line: str, raw_line: str | None = None) -> None:
        from swarmee_river.tui.event_router import handle_daemon_event as _handle_daemon_event_router
        from swarmee_river.tui.event_types import parse_output_line, parse_tui_event

        if self.state.daemon.query_active:
            chunk = raw_line if raw_line is not None else (line + "\n")
            self.state.daemon.turn_output_chunks.append(sanitize_output_text(chunk))
        sanitized = sanitize_output_text(line)
        # Try structured JSONL first (emitted by TuiCallbackHandler).
        tui_event = parse_tui_event(sanitized)
        if tui_event is not None:
            _handle_daemon_event_router(self, tui_event)
            return

        # Legacy fallback for non-JSON lines (stderr leakage, library warnings, etc.).
        event = parse_output_line(sanitized)
        if event is None:
            if sanitized.strip() == "return meta(":
                return
            if self._should_suppress_raw_assistant_output(sanitized):
                self._record_raw_assistant_suppression(sanitized)
                self._apply_consent_capture(sanitized)
                return
            self._append_plain_text(sanitized)
            self._apply_consent_capture(sanitized)
            return

        if event.kind == "noise":
            return

        if event.kind == "error":
            text = event.text
            if not text.startswith("ERROR:"):
                text = f"ERROR: {text}"
            self.state.session.error_count += 1
            self._write_issue(text)
            self._update_header_status()
        elif event.kind == "warning":
            text = event.text
            if not text.startswith("WARN:"):
                text = f"WARN: {text}"
            self.state.session.warning_count += 1
            self._write_issue(text)
            self._update_header_status()
        else:
            self._append_plain_text(event.text)

        artifact_paths = _artifact_paths_from_event(event)
        if artifact_paths:
            self._add_artifact_paths(artifact_paths)

        self._apply_consent_capture(sanitized)

    def _handle_tui_event(self, event: dict[str, Any]) -> None:
        """Process a structured JSONL event from the subprocess."""
        from swarmee_river.tui.event_router import handle_daemon_event as _handle_daemon_event_router

        _handle_daemon_event_router(self, event)

    def render_plan_panel(self, plan_json: dict[str, Any]) -> Any:
        from swarmee_river.tui.widgets import render_plan_panel

        return render_plan_panel(plan_json)

    def render_system_message(self, text: str) -> Any:
        from swarmee_river.tui.widgets import render_system_message

        return render_system_message(text)

    def render_tool_result_line(
        self,
        tool_name: str,
        *,
        status: str,
        duration_s: float,
        tool_input: dict | None = None,
        tool_use_id: str | None = None,
    ) -> Any:
        from swarmee_river.tui.widgets import render_tool_result_line

        return render_tool_result_line(
            tool_name,
            status=status,
            duration_s=duration_s,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
        )

    def _discover_session_log_path(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        try:
            matches = list(logs_dir().glob(f"*_{session_id}.jsonl"))
        except Exception:
            return None
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(matches[0])

    def _persist_run_transcript(
        self,
        *,
        pid: int | None,
        session_id: str | None,
        prompt: str,
        auto_approve: bool,
        exit_code: int,
        output_text: str,
    ) -> str | None:
        if not output_text:
            return None
        try:
            store = ArtifactStore()
            ref = store.write_text(
                kind="tui_transcript",
                text=output_text,
                metadata={
                    "pid": pid,
                    "session_id": session_id,
                    "prompt": prompt,
                    "auto_approve": auto_approve,
                    "exit_code": exit_code,
                },
            )
            return str(ref.path)
        except Exception:
            return None

    def _collapse_intermediate_activity_boxes(self) -> None:
        block = self._active_reasoning_block
        if block is not None:
            with contextlib.suppress(Exception):
                block.collapse()
        for record in list(self._tool_blocks.values()):
            if not isinstance(record, dict):
                continue
            widget = record.get("widget")
            if widget is None:
                continue
            with contextlib.suppress(Exception):
                widget.collapse()

    def _finalize_turn(self, *, exit_status: str) -> None:
        import time

        from swarmee_river.tui.text_sanitize import extract_plan_section_from_output, render_tui_hint_after_plan
        from swarmee_river.tui.text_sanitize import sanitize_output_text as _sanitize

        self.state.daemon.run_active_tier_warning_emitted = False
        if self.state.daemon.status_timer is not None:
            self.state.daemon.status_timer.stop()
            self.state.daemon.status_timer = None
        elapsed = (
            time.time() - self.state.daemon.run_start_time if self.state.daemon.run_start_time is not None else 0.0
        )
        if self._status_bar is not None:
            self._status_bar.set_state("idle")
            self._status_bar.set_elapsed(elapsed)
            self._status_bar.set_plan_step(current=None, total=None)
        self.state.daemon.run_start_time = None
        self.state.daemon.query_active = False
        self._clear_pending_tool_starts()
        self._cancel_tool_progress_flush_timer()
        self._tool_progress_pending_ids = set()
        for tool_use_id in list(self._tool_blocks.keys()):
            self._flush_tool_progress_render(tool_use_id, force=True)

        run_tool_count = self.state.daemon.run_tool_count
        completion_line = f"[run] completed in {elapsed:.1f}s ({run_tool_count} tool calls, status={exit_status})"
        self._write_transcript(completion_line)
        suppressed_raw_lines = int(getattr(self, "_raw_assistant_lines_suppressed_turn", 0) or 0)
        if suppressed_raw_lines > 0:
            _LOGGER.info(
                "Suppressed %s raw assistant echo line(s) for turn (trace=%s).",
                suppressed_raw_lines,
                " -> ".join(getattr(self, "_callback_event_trace_turn", [])),
            )

        self._finalize_assistant_message()
        self._maybe_emit_reasoning_unavailable_notice()
        self._dismiss_thinking(emit_summary=True)
        self._collapse_intermediate_activity_boxes()

        output_text = "".join(self.state.daemon.turn_output_chunks)
        self.state.daemon.turn_output_chunks = []
        transcript_path = self._persist_run_transcript(
            pid=(self.state.daemon.proc.pid if self.state.daemon.proc is not None else None),
            session_id=self.state.daemon.session_id,
            prompt=self._last_prompt or "",
            auto_approve=self._last_run_auto_approve,
            exit_code=0 if exit_status == "ok" else 1,
            output_text=output_text,
        )
        if transcript_path:
            self._add_artifact_paths([transcript_path])

        log_path = self._discover_session_log_path(self.state.daemon.session_id)
        if log_path:
            self._add_artifact_paths([log_path])

        if not self.state.plan.received_structured_plan:
            extracted_plan = extract_plan_section_from_output(_sanitize(output_text))
            if extracted_plan:
                self.state.plan.pending_prompt = self._last_prompt
                self.state.plan.text = extracted_plan
                self._refresh_plan_actions_visibility()
                self._write_transcript_line(render_tui_hint_after_plan())

        self._reset_consent_panel()
        self.state.plan.received_structured_plan = False
        self._save_session()
