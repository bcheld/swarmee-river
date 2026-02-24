"""Routing and domain handlers for structured daemon/TUI events."""

from __future__ import annotations

import contextlib
import json as _json
from typing import Any

from swarmee_river.error_classification import (
    ERROR_CATEGORY_AUTH_ERROR,
    ERROR_CATEGORY_ESCALATABLE,
    ERROR_CATEGORY_FATAL,
    ERROR_CATEGORY_TOOL_ERROR,
    ERROR_CATEGORY_TRANSIENT,
)
from swarmee_river.profiles import AgentProfile
from swarmee_river.tui.agent_studio import normalize_session_safety_overrides, normalize_team_presets
from swarmee_river.tui.event_types import extract_tui_text_chunk
from swarmee_river.tui.text_sanitize import sanitize_output_text

_TOOL_FAST_COMPLETE_SUPPRESS_START_S = 0.5


def _handle_connection_and_session_events(app: Any, etype: str, event: dict[str, Any]) -> bool:
    if etype in {"ready", "attached"}:
        app.state.daemon.ready = True
        session_id = str(event.get("session_id", "")).strip()
        if session_id:
            app.state.daemon.session_id = session_id
            app._save_session()
        if etype == "attached":
            clients_raw = event.get("clients")
            clients = int(clients_raw) if isinstance(clients_raw, int) else None
            if clients is not None and clients > 1:
                app._write_transcript_line(
                    f"[daemon] attached to shared runtime session ({clients} clients connected)."
                )
            else:
                app._write_transcript_line("[daemon] attached to shared runtime session.")
        else:
            app._write_transcript("Swarmee daemon ready. Enter a prompt to run Swarmee.")
        if app._context_sources or app._context_ready_for_sync:
            app._sync_context_sources_with_daemon(notify_on_failure=True)
        if app._active_sop_names or app._sops_ready_for_sync:
            app._sync_active_sops_with_daemon(notify_on_failure=True)
        app._refresh_agent_summary()
        app._schedule_session_timeline_refresh()
        return True

    if etype == "session_available":
        session_id = str(event.get("session_id", "")).strip()
        turn_count_raw = event.get("turn_count", 0)
        try:
            turn_count = int(turn_count_raw or 0)
        except (TypeError, ValueError):
            turn_count = 0
        app.state.daemon.available_restore_session_id = session_id or None
        app.state.daemon.available_restore_turn_count = max(0, turn_count)
        if session_id:
            app._write_transcript_line(
                f"Previous session found ({app.state.daemon.available_restore_turn_count} turns). "
                "Type /restore to resume or /new to start fresh."
            )
        return True

    if etype == "session_restored":
        session_id = str(event.get("session_id", "")).strip()
        if session_id:
            app.state.daemon.session_id = session_id
        turn_count_raw = event.get("turn_count", 0)
        try:
            app.state.daemon.last_restored_turn_count = max(0, int(turn_count_raw or 0))
        except (TypeError, ValueError):
            app.state.daemon.last_restored_turn_count = 0
        app.state.daemon.available_restore_session_id = None
        app.state.daemon.available_restore_turn_count = 0
        app._save_session()
        app._schedule_session_timeline_refresh()
        return True

    if etype == "replay_turn":
        role = str(event.get("role", "")).strip().lower()
        text = sanitize_output_text(str(event.get("text", "")))
        if not text.strip():
            return True
        timestamp = str(event.get("timestamp", "")).strip() or None
        if role == "user":
            app._write_user_message(text, timestamp=timestamp)
        elif role == "assistant":
            model = str(event.get("model", "")).strip() or None
            app._write_assistant_message(text, model=model, timestamp=timestamp)
        return True

    if etype == "replay_complete":
        turn_count_raw = event.get("turn_count", app.state.daemon.last_restored_turn_count)
        try:
            turns = max(0, int(turn_count_raw or 0))
        except (TypeError, ValueError):
            turns = max(0, app.state.daemon.last_restored_turn_count)
        app._write_transcript_line(f"Session restored ({turns} turns).")
        return True

    if etype == "turn_complete":
        exit_status = str(event.get("exit_status", "ok"))
        app._finalize_turn(exit_status=exit_status)
        if exit_status in {"ok", "interrupted"}:
            app._reset_error_action_prompt()
        app._schedule_session_timeline_refresh()
        return True

    if etype == "model_info":
        app._handle_model_info(event)
        return True

    return False


def _handle_agent_events(app: Any, etype: str, event: dict[str, Any]) -> bool:
    if etype == "profile_applied":
        raw_profile = event.get("profile")
        try:
            applied_profile = AgentProfile.from_dict(raw_profile)
        except Exception:
            app._write_transcript_line("[agent] received invalid profile_applied payload.")
            return True
        app.state.agent_studio.effective_profile = applied_profile
        app._refresh_agent_summary()
        app._reload_saved_profiles(selected_id=applied_profile.id)
        app.state.agent_studio.team_presets = normalize_team_presets(applied_profile.team_presets)
        app.state.agent_studio.team_selected_item_id = None
        app._render_agent_team_panel()
        app._set_agent_form_values(profile_id=applied_profile.id, profile_name=applied_profile.name)
        app._set_agent_draft_dirty(False, note=f"Applied profile '{applied_profile.name}'.")
        return True

    if etype == "safety_overrides":
        app.state.agent_studio.session_safety_overrides = normalize_session_safety_overrides(event.get("overrides"))
        app._render_agent_tools_panel()
        if app.state.agent_studio.session_safety_overrides:
            app._set_agent_tools_status("Session overrides active.")
        else:
            app._set_agent_tools_status("Session overrides cleared.")
        return True

    return False


def _handle_usage_and_compaction_events(app: Any, etype: str, event: dict[str, Any]) -> bool:
    if etype == "context":
        prompt_tokens_est = event.get("prompt_tokens_est")
        budget_tokens = event.get("budget_tokens")
        app.state.daemon.last_prompt_tokens_est = int(prompt_tokens_est) if isinstance(prompt_tokens_est, int) else None
        app.state.daemon.last_budget_tokens = int(budget_tokens) if isinstance(budget_tokens, int) else None
        if app._status_bar is not None:
            app._status_bar.set_context(
                prompt_tokens_est=app.state.daemon.last_prompt_tokens_est,
                budget_tokens=app.state.daemon.last_budget_tokens,
            )
        app._refresh_prompt_metrics()
        return True

    if etype == "usage":
        usage = event.get("usage")
        app.state.daemon.last_usage = usage if isinstance(usage, dict) else None
        cost = event.get("cost_usd")
        app.state.daemon.last_cost_usd = float(cost) if isinstance(cost, (int, float)) else None
        if app._status_bar is not None:
            app._status_bar.set_usage(app.state.daemon.last_usage, cost_usd=app.state.daemon.last_cost_usd)
        app._refresh_prompt_metrics()
        return True

    if etype == "compact_complete":
        compacted = bool(event.get("compacted", False))
        warning_text = str(event.get("warning", "")).strip()
        before_tokens = event.get("before_tokens_est")
        after_tokens = event.get("after_tokens_est")
        if isinstance(before_tokens, int) and isinstance(after_tokens, int):
            app.state.daemon.last_prompt_tokens_est = after_tokens
            if app._status_bar is not None:
                app._status_bar.set_context(
                    prompt_tokens_est=app.state.daemon.last_prompt_tokens_est,
                    budget_tokens=app.state.daemon.last_budget_tokens,
                )
            app._refresh_prompt_metrics()
        if compacted:
            app._notify("Context compacted.", severity="information", timeout=4.0)
        elif warning_text:
            app._notify(warning_text, severity="warning", timeout=6.0)
        else:
            app._notify("Context compaction made no changes.", severity="information", timeout=4.0)
        return True

    return False


def _handle_streaming_events(app: Any, etype: str, event: dict[str, Any]) -> bool:
    if etype in {"text_delta", "message_delta", "output_text_delta", "delta"}:
        chunk = sanitize_output_text(extract_tui_text_chunk(event))
        if not chunk:
            return True
        app._dismiss_thinking(emit_summary=True)
        if not app._current_assistant_chunks and not app._streaming_buffer:
            app._current_assistant_model = app.state.daemon.current_model
            app._current_assistant_timestamp = app._turn_timestamp()
        app._streaming_buffer.append(chunk)
        app._schedule_streaming_flush()
        return True

    if etype in {"text_complete", "message_complete", "output_text_complete", "complete"}:
        app._cancel_streaming_flush_timer()
        app._flush_streaming_buffer()
        app._finalize_assistant_message()
        return True

    if etype == "thinking":
        app._record_thinking_event(str(event.get("text", "")))
        return True

    return False


def _handle_tool_events(app: Any, etype: str, event: dict[str, Any]) -> bool:
    if etype == "tool_start":
        app._dismiss_thinking(emit_summary=True)
        tid = str(event.get("tool_use_id", "")).strip() or f"tool-{app.state.daemon.run_tool_count + 1}"
        tool_name = str(event.get("tool", "unknown"))
        app._tool_blocks[tid] = {
            "tool_use_id": tid,
            "tool": tool_name,
            "status": "running",
            "duration_s": 0.0,
            "input": None,
            "output": "",
            "pending_output": "",
            "pending_stream": "stdout",
            "elapsed_s": 0.0,
            "last_progress_render_mono": 0.0,
            "last_heartbeat_rendered_s": 0.0,
            "start_rendered": False,
        }
        app._schedule_tool_start_line(tid)
        app.state.daemon.run_tool_count += 1
        if app._status_bar is not None:
            app._status_bar.set_tool_count(app.state.daemon.run_tool_count)
        return True

    if etype == "tool_progress":
        tid = str(event.get("tool_use_id", "")).strip()
        record = app._tool_blocks.get(tid)
        if record is None and tid:
            fallback_tool_name = str(event.get("tool", "unknown"))
            record = {
                "tool_use_id": tid,
                "tool": fallback_tool_name,
                "status": "running",
                "duration_s": 0.0,
                "input": None,
                "output": "",
                "pending_output": "",
                "pending_stream": "stdout",
                "elapsed_s": 0.0,
                "last_progress_render_mono": 0.0,
                "last_heartbeat_rendered_s": 0.0,
                "start_rendered": False,
            }
            app._tool_blocks[tid] = record
            app._schedule_tool_start_line(tid)
        if record is not None:
            chars = event.get("chars")
            if isinstance(chars, int):
                record["chars"] = chars
            elapsed_raw = event.get("elapsed_s")
            if isinstance(elapsed_raw, (int, float)):
                record["elapsed_s"] = float(elapsed_raw)
            content = event.get("content")
            if isinstance(content, str) and content:
                stream = str(event.get("stream", "stdout")).strip().lower() or "stdout"
                app._queue_tool_progress_content(record, content=content, stream=stream)
            app._schedule_tool_progress_flush(tid)
        return True

    if etype == "tool_input":
        tid = str(event.get("tool_use_id", "")).strip()
        record = app._tool_blocks.get(tid)
        if record is not None:
            record["input"] = event.get("input", {})
            if tid in app._tool_pending_start:
                app._emit_tool_start_line(tid)
        return True

    if etype == "tool_result":
        tid = str(event.get("tool_use_id", "")).strip()
        status = str(event.get("status", "unknown"))
        duration_raw = event.get("duration_s", 0.0)
        try:
            duration_s = float(duration_raw or 0.0)
        except (TypeError, ValueError):
            duration_s = 0.0
        record = app._tool_blocks.get(tid)
        tool_name = str(event.get("tool", "unknown"))
        if record is not None:
            record["status"] = status
            record["duration_s"] = duration_s
            record["elapsed_s"] = duration_s
            tool_name = str(record.get("tool", tool_name))
            pending_since = app._tool_pending_start.get(tid)
            if pending_since is not None:
                app._tool_pending_start.pop(tid, None)
                app._cancel_tool_start_timer(tid)
                if duration_s >= _TOOL_FAST_COMPLETE_SUPPRESS_START_S:
                    app._emit_tool_start_line(tid)
            app._tool_progress_pending_ids.discard(tid)
            app._flush_tool_progress_render(tid, force=True)
        tool_input = record.get("input") if isinstance(record, dict) else None
        plain = app._tool_result_plain_text(tool_name, status, duration_s, tool_input)
        app._mount_transcript_widget(
            app.render_tool_result_line(  # type: ignore[attr-defined]
                tool_name,
                status=status,
                duration_s=duration_s,
                tool_input=tool_input if isinstance(tool_input, dict) else None,
                tool_use_id=tid,
            ),
            plain_text=plain,
        )
        if status != "success":
            app.state.session.error_count += 1
            app._write_issue(f"ERROR: tool {tool_name} failed ({status}) [{tid}]")
            app._update_header_status()
            app._notify(f"{tool_name} tool failed", severity="error", timeout=6.0)
            if tid:
                app._mount_transcript_widget(
                    app.render_system_message(  # type: ignore[attr-defined]
                        "Tool failed. Retry or skip using buttons above the prompt."
                    ),
                    plain_text="Tool failed. Retry or skip using buttons above the prompt.",
                )
                app._show_tool_error_actions(tool_use_id=tid, tool_name=tool_name)
        return True

    if etype == "consent_prompt":
        context = str(event.get("context", ""))
        raw_options = event.get("options", ["y", "n", "a", "v"])
        options = (
            [str(item).strip() for item in raw_options if str(item).strip()]
            if isinstance(raw_options, (list, tuple))
            else ["y", "n", "a", "v"]
        )
        if not options:
            options = ["y", "n", "a", "v"]
        app._consent_buffer = [context]
        app._show_consent_prompt(context=context, options=options, alert=True)
        return True

    return False


def _handle_plan_events(app: Any, etype: str, event: dict[str, Any]) -> bool:
    if etype == "plan":
        rendered = event.get("rendered", "")
        plan_json = event.get("plan_json")
        if plan_json and not rendered:
            rendered = _json.dumps(plan_json, indent=2)
        app._set_plan_panel(rendered)
        app.state.plan.received_structured_plan = True
        app.state.plan.completion_announced = False
        app.state.plan.step_counter = 0
        app.state.plan.current_steps_total = 0
        app.state.plan.current_summary = ""
        app.state.plan.current_steps = []
        app.state.plan.current_step_statuses = []
        app.state.plan.current_active_step = None
        app.state.plan.updates_seen = False
        if not app._last_run_auto_approve and app._last_prompt:
            app.state.plan.pending_prompt = app._last_prompt
        if plan_json and isinstance(plan_json, dict):
            app.state.plan.current_summary = str(plan_json.get("summary", plan_json.get("title", ""))).strip()
            app.state.plan.current_steps = app._extract_plan_step_descriptions(plan_json)
            app.state.plan.current_steps_total = len(app.state.plan.current_steps)
            app.state.plan.current_step_statuses = ["pending"] * app.state.plan.current_steps_total
            app._render_plan_panel_from_status()
            app._mount_transcript_widget(
                app.render_plan_panel(plan_json),  # type: ignore[attr-defined]
                plain_text=rendered if isinstance(rendered, str) else _json.dumps(plan_json, indent=2),
            )
        else:
            app._refresh_plan_status_bar()
        return True

    if etype == "plan_step_update":
        step_index_raw = event.get("step_index")
        status = str(event.get("status", "")).strip().lower()
        if not isinstance(step_index_raw, int):
            with contextlib.suppress(Exception):
                step_index_raw = int(step_index_raw)
        if not isinstance(step_index_raw, int):
            return True
        step_index = step_index_raw
        if step_index < 0:
            return True
        if not app.state.plan.current_step_statuses:
            return True
        if step_index >= len(app.state.plan.current_step_statuses):
            app._write_transcript_line(f"[plan] ignoring out-of-range step index: {step_index + 1}")
            return True
        if status not in {"in_progress", "completed"}:
            return True
        app.state.plan.updates_seen = True
        if status == "in_progress":
            app.state.plan.current_active_step = step_index
            if app.state.plan.current_step_statuses[step_index] != "completed":
                app.state.plan.current_step_statuses[step_index] = "in_progress"
        elif status == "completed":
            app.state.plan.current_step_statuses[step_index] = "completed"
            if app.state.plan.current_active_step == step_index:
                app.state.plan.current_active_step = None
        app.state.plan.step_counter = sum(1 for item in app.state.plan.current_step_statuses if item == "completed")
        app._render_plan_panel_from_status()
        if (
            app.state.plan.current_steps_total > 0
            and app.state.plan.step_counter >= app.state.plan.current_steps_total
            and not app.state.plan.completion_announced
        ):
            app.state.plan.completion_announced = True
            app._write_transcript_line("Plan complete. Clear?")
        return True

    if etype == "plan_complete":
        app.state.plan.step_counter = app.state.plan.current_steps_total
        if app.state.plan.current_step_statuses:
            app.state.plan.current_step_statuses = ["completed"] * len(app.state.plan.current_step_statuses)
        app.state.plan.current_active_step = None
        app._render_plan_panel_from_status()
        if not app.state.plan.completion_announced:
            app.state.plan.completion_announced = True
            app._write_transcript_line("Plan complete. Clear?")
        return True

    return False


def _handle_artifact_events(app: Any, etype: str, event: dict[str, Any]) -> bool:
    if etype != "artifact":
        return False
    paths = event.get("paths", [])
    if paths:
        app._add_artifact_paths(paths)
    return True


def _handle_error_warning_events(app: Any, etype: str, event: dict[str, Any]) -> bool:
    if etype == "error":
        error_info = app._classify_tui_error_event(event)
        error_message = str(error_info.get("message", "")).strip()
        error_text = error_message
        if not error_text.startswith("ERROR:"):
            error_text = f"ERROR: {error_text}"
        normalized_error = error_text.lower()
        if app.state.daemon.pending_model_select_value and (
            "set_tier" in normalized_error or "cannot set tier" in normalized_error or "tier" in normalized_error
        ):
            app.state.daemon.pending_model_select_value = None
            app._refresh_model_select()
        app.state.session.error_count += 1
        app._write_issue(error_text)
        app._update_header_status()
        toast_message, severity, timeout = app._summarize_error_for_toast(error_info)
        app._notify(toast_message, severity=severity, timeout=timeout)

        category = str(error_info.get("category", ERROR_CATEGORY_FATAL))
        if category == ERROR_CATEGORY_TRANSIENT:
            app._reset_error_action_prompt()
        elif category == ERROR_CATEGORY_TOOL_ERROR:
            tool_use_id = str(error_info.get("tool_use_id", "")).strip()
            if tool_use_id:
                tool_record = app._tool_blocks.get(tool_use_id)
                tool_name = str(tool_record.get("tool", "tool")) if isinstance(tool_record, dict) else "tool"
                app._show_tool_error_actions(tool_use_id=tool_use_id, tool_name=tool_name)
        elif category == ERROR_CATEGORY_ESCALATABLE:
            next_tier = str(error_info.get("next_tier", "")).strip() or app._next_available_tier_name()
            app._show_escalation_actions(next_tier=next_tier or None)
        elif category == ERROR_CATEGORY_AUTH_ERROR:
            app._mount_transcript_widget(
                app.render_system_message(  # type: ignore[attr-defined]
                    "Authentication failed. Verify credentials/permissions for the active provider."
                ),
                plain_text="Authentication failed. Verify credentials/permissions for the active provider.",
            )
            app._reset_error_action_prompt()
        elif category == ERROR_CATEGORY_FATAL:
            app._reset_error_action_prompt()
        return True

    if etype == "warning":
        warn_text = event.get("text", "")
        if not warn_text.startswith("WARN:"):
            warn_text = f"WARN: {warn_text}"
        app.state.session.warning_count += 1
        app._write_issue(warn_text)
        app._update_header_status()
        return True

    return False


def handle_daemon_event(app: Any, event_dict: dict[str, Any]) -> None:
    """Dispatch a structured daemon/TUI event into domain handlers."""
    etype = str(event_dict.get("event", "")).strip().lower()
    if not etype:
        return

    handlers = (
        _handle_connection_and_session_events,
        _handle_agent_events,
        _handle_usage_and_compaction_events,
        _handle_streaming_events,
        _handle_tool_events,
        _handle_plan_events,
        _handle_artifact_events,
        _handle_error_warning_events,
    )
    for handler in handlers:
        if handler(app, etype, event_dict):
            return
