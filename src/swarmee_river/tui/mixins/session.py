from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from swarmee_river.session.graph_index import (
    build_session_graph_index,
    load_session_graph_index,
    write_session_graph_index,
)
from swarmee_river.state_paths import sessions_dir
from swarmee_river.tui.agent_studio import normalize_agent_studio_view_mode
from swarmee_river.tui.sidebar_session import (
    build_session_issue_sidebar_items,
    build_session_timeline_sidebar_items,
    classify_session_timeline_event_kind,
    normalize_session_view_mode,
    render_session_issue_detail_text,
    render_session_timeline_detail_text,
    session_issue_actions,
    session_timeline_actions,
)


def _load_session_id(raw: Any) -> str | None:
    """Safely extract a session_id from persisted JSON, rejecting falsy / sentinel values.

    ``str(None)`` produces ``"None"`` which is truthy — previous code accidentally
    persisted this sentinel into tui_session.json, causing every subsequent daemon
    to inherit ``SWARMEE_SESSION_ID="None"`` and write to ``*_None.jsonl`` files
    instead of per-session log files.
    """
    if not isinstance(raw, str):
        return None
    sid = raw.strip()
    # Reject stale sentinel produced by str(None) bug and similar artifacts.
    if not sid or sid.lower() == "none":
        return None
    return sid


class SessionMixin:
    def _reset_plan_panel(self) -> None:
        self._set_plan_panel("")
        self.state.plan.current_steps_total = 0
        self.state.plan.current_summary = ""
        self.state.plan.current_steps = []
        self.state.plan.current_step_statuses = []
        self.state.plan.current_active_step = None
        self.state.plan.updates_seen = False
        self.state.plan.step_counter = 0
        self.state.plan.completion_announced = False
        self.state.plan.plan_json = None
        self._refresh_plan_status_bar()
        self._refresh_plan_actions_visibility()
        self._clear_planning_view()

    def _reset_issues_panel(self) -> None:
        self.state.session.issue_lines = []
        self.state.session.issues = []
        self.state.session.selected_issue_id = None
        self.state.session.issues_repeat_line = None
        self.state.session.issues_repeat_count = 0
        self.state.session.warning_count = 0
        self.state.session.error_count = 0
        self._render_session_panel()
        self._update_header_status()

    def _reset_session_timeline_panel(self) -> None:
        self.state.session.timeline_index = None
        self.state.session.timeline_events = []
        self.state.session.timeline_selected_event_id = None
        self._render_session_timeline_panel()

    def _session_issue_by_id(self, issue_id: str | None) -> dict[str, Any] | None:
        target = str(issue_id or "").strip()
        if not target:
            return None
        for issue in self.state.session.issues:
            if str(issue.get("id", "")).strip() == target:
                return issue
        return None

    def _append_session_issue(
        self,
        *,
        severity: str,
        title: str,
        text: str,
        category: str = "issue",
        tool_use_id: str | None = None,
        tool_name: str | None = None,
        next_tier: str | None = None,
    ) -> None:
        normalized_severity = severity.strip().lower()
        if normalized_severity not in {"warning", "error"}:
            normalized_severity = "warning"
        issue = {
            "id": uuid.uuid4().hex[:12],
            "severity": normalized_severity,
            "title": title.strip() or "Issue",
            "text": text.strip(),
            "category": category.strip().lower() or "issue",
            "tool_use_id": (tool_use_id or "").strip(),
            "tool_name": (tool_name or "").strip(),
            "next_tier": (next_tier or "").strip().lower(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.state.session.issues.append(issue)
        if len(self.state.session.issues) > 500:
            self.state.session.issues = self.state.session.issues[-500:]
        self._render_session_panel()

    def _session_timeline_event_by_id(self, event_id: str | None) -> dict[str, Any] | None:
        target = str(event_id or "").strip()
        if not target:
            return None
        for event in self.state.session.timeline_events:
            if str(event.get("id", "")).strip() == target:
                return event
        return None

    def _session_issue_from_line(self, line: str) -> dict[str, Any]:
        text = line.strip()
        lowered = text.lower()
        severity = "error" if lowered.startswith("error:") else "warning"
        title = "Error" if severity == "error" else "Warning"
        category = "issue"
        tool_use_id = ""
        tool_name = ""
        next_tier = ""

        match = re.search(
            r"^error:\s*tool (?P<tool>.+?) failed \((?P<status>.+?)\)\s*\[(?P<tool_use_id>[^\]]+)\]",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            category = "tool_failure"
            title = f"Tool Failed: {match.group('tool').strip()}"
            tool_use_id = match.group("tool_use_id").strip()
            tool_name = match.group("tool").strip()
            next_tier = self._next_available_tier_name() or ""
        return {
            "severity": severity,
            "title": title,
            "text": text,
            "category": category,
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "next_tier": next_tier,
        }

    def _set_session_issue_selection(self, issue: dict[str, Any] | None) -> None:
        detail = self._session_issue_detail
        if detail is None:
            return
        if issue is None:
            self.state.session.selected_issue_id = None
            detail.set_preview("(no issues yet)")
            detail.set_actions([])
            return
        self.state.session.selected_issue_id = str(issue.get("id", "")).strip() or None
        detail.set_preview(render_session_issue_detail_text(issue))
        detail.set_actions(session_issue_actions(issue))

    def _render_session_panel(self) -> None:
        issues = list(self.state.session.issues)
        items = build_session_issue_sidebar_items(issues)
        list_widget = self._session_issue_list
        if list_widget is not None:
            selected_id = self.state.session.selected_issue_id
            if not selected_id and issues:
                selected_id = str(issues[-1].get("id", "")).strip()
            list_widget.set_items(items, selected_id=selected_id, emit=False)
            selected_id = list_widget.selected_id()
            selected_issue = self._session_issue_by_id(selected_id)
            if selected_issue is None and issues:
                selected_issue = issues[-1]
                with contextlib.suppress(Exception):
                    list_widget.select_by_id(str(selected_issue.get("id", "")), emit=False)
            self._set_session_issue_selection(selected_issue)
        else:
            self._set_session_issue_selection(issues[-1] if issues else None)
        self._refresh_session_header()

    def _set_session_timeline_selection(self, event: dict[str, Any] | None) -> None:
        detail = self._session_timeline_detail
        if detail is None:
            return
        if event is None:
            self.state.session.timeline_selected_event_id = None
            detail.set_preview("(no timeline events yet)")
            detail.set_actions([])
            return
        self.state.session.timeline_selected_event_id = str(event.get("id", "")).strip() or None
        detail.set_preview(render_session_timeline_detail_text(event))
        detail.set_actions(session_timeline_actions(event))

    def _render_session_timeline_panel(self) -> None:
        events = [item for item in self.state.session.timeline_events if isinstance(item, dict)]
        items = build_session_timeline_sidebar_items(events)
        list_widget = self._session_timeline_list
        if list_widget is not None:
            selected_id = self.state.session.timeline_selected_event_id
            if not selected_id and events:
                selected_id = str(events[-1].get("id", "")).strip()
            list_widget.set_items(items, selected_id=selected_id, emit=False)
            selected_id = list_widget.selected_id()
            selected_event = self._session_timeline_event_by_id(selected_id)
            if selected_event is None and events:
                selected_event = events[-1]
                with contextlib.suppress(Exception):
                    list_widget.select_by_id(str(selected_event.get("id", "")), emit=False)
            self._set_session_timeline_selection(selected_event)
        else:
            self._set_session_timeline_selection(events[-1] if events else None)
        self._refresh_session_timeline_header()

    def _refresh_session_header(self) -> None:
        header = self._session_header
        if header is None:
            return
        badges = [
            f"warn {self.state.session.warning_count}",
            f"err {self.state.session.error_count}",
            f"issues {len(self.state.session.issues)}",
        ]
        header.set_badges(badges)
        self._refresh_session_timeline_header()

    def _refresh_session_timeline_header(self) -> None:
        header = self._session_timeline_header
        if header is None:
            return
        events = list(self.state.session.timeline_events)
        error_count = 0
        for event in events:
            if classify_session_timeline_event_kind(event) == "error":
                error_count += 1
        badges = [f"events {len(events)}", f"errors {error_count}"]
        header.set_badges(badges)

    def _set_session_view_mode(self, mode: str) -> None:
        normalized = normalize_session_view_mode(mode)
        if normalized == "issues":
            normalized = "artifacts"
        self.state.session.view_mode = normalized

        timeline_view = self._session_timeline_view
        artifacts_view = self._session_artifacts_view
        if timeline_view is not None:
            timeline_view.styles.display = "block" if normalized == "timeline" else "none"
        if artifacts_view is not None:
            artifacts_view.styles.display = "block" if normalized == "artifacts" else "none"

        timeline_button = self._session_view_timeline_button
        artifacts_button = self._session_view_artifacts_button
        if timeline_button is not None:
            timeline_button.variant = "primary" if normalized == "timeline" else "default"
        if artifacts_button is not None:
            artifacts_button.variant = "primary" if normalized == "artifacts" else "default"

    def _schedule_session_timeline_refresh(self, *, delay: float = 0.35) -> None:
        timer = self.state.session.timeline_refresh_timer
        self.state.session.timeline_refresh_timer = None
        if timer is not None:
            with contextlib.suppress(RuntimeError):
                timer.stop()
        self.state.session.timeline_refresh_timer = self.set_timer(delay, self._launch_session_timeline_refresh)

    def _launch_session_timeline_refresh(self) -> None:
        self.state.session.timeline_refresh_timer = None
        with contextlib.suppress(RuntimeError):
            asyncio.create_task(self._refresh_session_timeline_async())

    async def _refresh_session_timeline_async(self) -> None:
        session_id = str(self.state.daemon.session_id or "").strip()
        if not session_id:
            self._reset_session_timeline_panel()
            return
        if self.state.session.timeline_refresh_inflight:
            self.state.session.timeline_refresh_pending = True
            return
        self.state.session.timeline_refresh_inflight = True
        next_pending = False
        try:
            existing_index: dict[str, Any] | None = None
            try:
                loaded = await asyncio.to_thread(load_session_graph_index, session_id)
                existing_index = loaded if isinstance(loaded, dict) else None
            except Exception as _load_err:
                existing_index = None
                self._append_session_issue(
                    severity="warning",
                    title="Timeline cache load failed",
                    text=repr(_load_err),
                    category="issue",
                )
            built_index = None
            try:
                built_index = await asyncio.to_thread(build_session_graph_index, session_id, cwd=Path.cwd())
                await asyncio.to_thread(write_session_graph_index, session_id, built_index)
            except Exception as _build_err:
                built_index = None
                self._append_session_issue(
                    severity="warning",
                    title="Timeline index build failed",
                    text=repr(_build_err),
                    category="issue",
                )
            index = built_index
            if isinstance(index, dict):
                built_events = index.get("events")
                existing_events = existing_index.get("events") if isinstance(existing_index, dict) else None
                if (
                    isinstance(existing_events, list)
                    and existing_events
                    and (not isinstance(built_events, list) or not built_events)
                ):
                    index = existing_index
            elif isinstance(existing_index, dict):
                index = existing_index
            if isinstance(index, dict):
                indexed_session_id = str(index.get("session_id", "")).strip()
                if indexed_session_id and indexed_session_id != session_id:
                    return
                events_raw = index.get("events")
                normalized_events: list[dict[str, Any]] = []
                if isinstance(events_raw, list):
                    for offset, raw in enumerate(events_raw, start=1):
                        if not isinstance(raw, dict):
                            continue
                        event = dict(raw)
                        event.setdefault("id", f"timeline-{offset}")
                        normalized_events.append(event)
                # Drop stale async updates if active session changed while refreshing.
                if str(self.state.daemon.session_id or "").strip() != session_id:
                    return
                self.state.session.timeline_index = index
                self.state.session.timeline_events = normalized_events
                self._render_session_timeline_panel()
        finally:
            self.state.session.timeline_refresh_inflight = False
            next_pending = self.state.session.timeline_refresh_pending
            self.state.session.timeline_refresh_pending = False
        if next_pending:
            self._schedule_session_timeline_refresh(delay=0.1)

    def _write_issue(self, line: str) -> None:
        if self.state.session.issues_repeat_line == line:
            self.state.session.issues_repeat_count += 1
            return
        if self.state.session.issues_repeat_line is not None and self.state.session.issues_repeat_count > 0:
            repeated = (
                f"… repeated {self.state.session.issues_repeat_count} more time(s): "
                f"{self.state.session.issues_repeat_line}"
            )
            self.state.session.issue_lines.append(repeated)
        self.state.session.issues_repeat_line = line
        self.state.session.issues_repeat_count = 0
        self.state.session.issue_lines.append(line)
        if len(self.state.session.issue_lines) > 2000:
            self.state.session.issue_lines = self.state.session.issue_lines[-2000:]
        issue_meta = self._session_issue_from_line(line)
        self._append_session_issue(
            severity=str(issue_meta.get("severity", "warning")),
            title=str(issue_meta.get("title", "Issue")),
            text=str(issue_meta.get("text", line)),
            category=str(issue_meta.get("category", "issue")),
            tool_use_id=str(issue_meta.get("tool_use_id", "")) or None,
            tool_name=str(issue_meta.get("tool_name", "")) or None,
            next_tier=str(issue_meta.get("next_tier", "")) or None,
        )

    def _flush_issue_repeats(self) -> None:
        if self.state.session.issues_repeat_line is None or self.state.session.issues_repeat_count <= 0:
            self.state.session.issues_repeat_line = None
            self.state.session.issues_repeat_count = 0
            return
        repeated = (
            f"… repeated {self.state.session.issues_repeat_count} more time(s): "
            f"{self.state.session.issues_repeat_line}"
        )
        self.state.session.issue_lines.append(repeated)
        self.state.session.issues_repeat_line = None
        self.state.session.issues_repeat_count = 0
        self._append_session_issue(
            severity="warning",
            title="Repeated Issue",
            text=repeated,
            category="issue",
        )

    def action_copy_issues(self) -> None:
        self._flush_issue_repeats()
        payload = (
            ("\n".join(self.state.session.issue_lines).rstrip() + "\n") if self.state.session.issue_lines else ""
        )
        self._copy_text(payload, label="issues")

    def _session_retry_tool(self, tool_use_id: str) -> None:
        from swarmee_river.tui.transport import send_daemon_command
        tool_id = str(tool_use_id or "").strip()
        if not tool_id:
            self._notify("Issue has no tool_use_id.", severity="warning")
            return
        proc = self.state.daemon.proc
        if not self.state.daemon.ready or proc is None or proc.poll() is not None:
            self._write_transcript_line("[session] daemon is not ready.")
            return
        if send_daemon_command(proc, {"cmd": "retry_tool", "tool_use_id": tool_id}):
            self._write_transcript_line(f"[session] retry requested for tool {tool_id}.")
        else:
            self._write_transcript_line("[session] failed to send retry request.")

    def _session_skip_tool(self, tool_use_id: str) -> None:
        from swarmee_river.tui.transport import send_daemon_command
        tool_id = str(tool_use_id or "").strip()
        if not tool_id:
            self._notify("Issue has no tool_use_id.", severity="warning")
            return
        proc = self.state.daemon.proc
        if not self.state.daemon.ready or proc is None or proc.poll() is not None:
            self._write_transcript_line("[session] daemon is not ready.")
            return
        if send_daemon_command(proc, {"cmd": "skip_tool", "tool_use_id": tool_id}):
            self._write_transcript_line(f"[session] skip requested for tool {tool_id}.")
        else:
            self._write_transcript_line("[session] failed to send skip request.")

    def _session_escalate_tier(self, tier_name: str | None = None) -> None:
        from swarmee_river.tui.transport import send_daemon_command
        next_tier = str(tier_name or "").strip().lower() or (self._next_available_tier_name() or "")
        if not next_tier:
            self._write_transcript_line("[session] no higher tier available.")
            return
        proc = self.state.daemon.proc
        if not self.state.daemon.ready or proc is None or proc.poll() is not None:
            self._write_transcript_line("[session] daemon is not ready.")
            return
        if send_daemon_command(proc, {"cmd": "set_tier", "tier": next_tier}):
            self._write_transcript_line(f"[session] tier change requested: {next_tier}")
        else:
            self._write_transcript_line("[session] failed to send tier change request.")

    def _session_interrupt(self) -> None:
        from swarmee_river.tui.transport import send_daemon_command
        proc = self.state.daemon.proc
        if not self.state.daemon.ready or proc is None or proc.poll() is not None:
            self._write_transcript_line("[session] daemon is not ready.")
            return
        if send_daemon_command(proc, {"cmd": "interrupt"}):
            self._write_transcript_line("[session] interrupt requested.")
        else:
            self._write_transcript_line("[session] failed to send interrupt.")

    def _save_session(self) -> None:
        import json as _json
        try:
            session_path = sessions_dir() / "tui_session.json"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "prompt_history": self._prompt_history[-self._MAX_PROMPT_HISTORY :],
                "last_prompt": self._last_prompt,
                "plan_text": self.state.plan.text,
                "plan_json": self.state.plan.plan_json,
                "plan_pending_prompt": self.state.plan.pending_prompt,
                "plan_current_steps": self.state.plan.current_steps,
                "plan_current_step_statuses": self.state.plan.current_step_statuses,
                "plan_current_summary": self.state.plan.current_summary,
                "plan_current_steps_total": self.state.plan.current_steps_total,
                "plan_step_counter": self.state.plan.step_counter,
                "context_sources": self._context_sources,
                "active_sop_names": sorted(self._active_sop_names),
                "daemon_session_id": self.state.daemon.session_id,
                "available_restore_session_id": self.state.daemon.available_restore_session_id,
                "available_restore_turn_count": self.state.daemon.available_restore_turn_count,
                "model_provider_override": self.state.daemon.model_provider_override,
                "model_tier_override": self.state.daemon.model_tier_override,
                "default_auto_approve": self._default_auto_approve,
                "split_ratio": self._split_ratio,
                "session_view_mode": self.state.session.view_mode,
                "agent_studio_view_mode": self.state.agent_studio.view_mode,
            }
            session_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_session(self) -> None:
        import json as _json
        import os
        from swarmee_river.tui.mixins.context_sources import _normalize_context_sources
        try:
            session_path = sessions_dir() / "tui_session.json"
            if not session_path.exists():
                return
            data = _json.loads(session_path.read_text(encoding="utf-8", errors="replace"))
            self._prompt_history = data.get("prompt_history", [])[-self._MAX_PROMPT_HISTORY :]
            self._last_prompt = data.get("last_prompt")
            plan_text = data.get("plan_text", "")
            if plan_text and plan_text != "(no plan)":
                self._set_plan_panel(plan_text)
            plan_json = data.get("plan_json")
            if plan_json and isinstance(plan_json, dict):
                self.state.plan.plan_json = plan_json
                self.state.plan.pending_prompt = data.get("plan_pending_prompt") or None
                self.state.plan.current_summary = str(data.get("plan_current_summary", "")).strip()
                self.state.plan.current_steps = data.get("plan_current_steps") or []
                self.state.plan.current_steps_total = int(data.get("plan_current_steps_total", 0) or 0)
                self.state.plan.step_counter = int(data.get("plan_step_counter", 0) or 0)
                raw_statuses = data.get("plan_current_step_statuses") or []
                self.state.plan.current_step_statuses = raw_statuses if isinstance(raw_statuses, list) else []
                self.state.plan.received_structured_plan = True
                self._render_plan_panel_from_status()
                self._refresh_plan_actions_visibility()
                self._populate_planning_view(plan_json)
            else:
                self._set_plan_input_mode(editable=True)
            self._context_sources = _normalize_context_sources(data.get("context_sources", []))
            self._render_context_sources_panel()
            self._context_ready_for_sync = bool(self._context_sources)
            loaded_active_sops = data.get("active_sop_names", [])
            if isinstance(loaded_active_sops, list):
                self._active_sop_names = {str(item).strip() for item in loaded_active_sops if str(item).strip()}
            self._refresh_sop_catalog()
            self._render_sop_panel()
            self._sops_ready_for_sync = bool(self._active_sop_names)
            self.state.daemon.session_id = _load_session_id(data.get("daemon_session_id"))
            self.state.daemon.available_restore_session_id = _load_session_id(
                data.get("available_restore_session_id")
            )
            restore_turn_count_raw = data.get("available_restore_turn_count", 0)
            try:
                self.state.daemon.available_restore_turn_count = max(0, int(restore_turn_count_raw or 0))
            except (TypeError, ValueError):
                self.state.daemon.available_restore_turn_count = 0
            if self.state.daemon.available_restore_session_id:
                self._write_transcript_line(
                    f"Previous session found ({self.state.daemon.available_restore_turn_count} turns). "
                    "Type /restore to resume or /new to start fresh."
                )
            # Do not restore model overrides from prior sessions.
            # The daemon-reported model_info is the source of truth for startup model state.
            self.state.daemon.model_provider_override = None
            self.state.daemon.model_tier_override = None
            auto_env = (os.getenv("SWARMEE_AUTO_APPROVE") or "").strip().lower()
            if auto_env in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}:
                self._default_auto_approve = True
            elif auto_env in {"0", "false", "f", "no", "n", "off", "disabled", "disable"}:
                self._default_auto_approve = False
            else:
                self._default_auto_approve = data.get("default_auto_approve", False)
            self._split_ratio = data.get("split_ratio", 2)
            self.state.session.view_mode = normalize_session_view_mode(data.get("session_view_mode"))
            self.state.agent_studio.view_mode = normalize_agent_studio_view_mode(data.get("agent_studio_view_mode"))
            self._apply_split_ratio()
            self._set_session_view_mode(self.state.session.view_mode)
            self._set_agent_studio_view_mode(self.state.agent_studio.view_mode)
            self._refresh_model_select()
            self._update_header_status()
            self._update_prompt_placeholder()
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())
            if self._prompt_history:
                self._write_transcript(f"[session] restored ({len(self._prompt_history)} history entries).")
        except Exception:
            pass

    def _restore_available_session(self) -> None:
        from swarmee_river.tui.transport import send_daemon_command
        if self.state.daemon.query_active:
            self._write_transcript_line("[restore] cannot restore while a run is active.")
            return
        session_id = (self.state.daemon.available_restore_session_id or "").strip()
        if not session_id:
            self._write_transcript_line("[restore] no previous session available.")
            return
        proc = self.state.daemon.proc
        if not self.state.daemon.ready or proc is None or proc.poll() is not None:
            self._write_transcript_line("[restore] daemon is not ready.")
            return
        if send_daemon_command(proc, {"cmd": "restore_session", "session_id": session_id}):
            self._write_transcript_line(f"[restore] requesting session restore: {session_id}")
        else:
            self._write_transcript_line("[restore] failed to send restore command.")

    def _start_fresh_session(self) -> None:
        self.state.daemon.available_restore_session_id = None
        self.state.daemon.available_restore_turn_count = 0
        self.state.daemon.last_restored_turn_count = 0
        self._reset_session_timeline_panel()
        self._write_transcript_line("[session] starting fresh.")
        self._save_session()

    def _on_active_session_changed(self, old_session_id: str | None, new_session_id: str | None) -> None:
        previous = str(old_session_id or "").strip() or None
        current = str(new_session_id or "").strip() or None
        if previous == current:
            return
        self.state.daemon.session_id = current
        self._reset_issues_panel()
        self._reset_session_timeline_panel()
        self._reset_artifacts_panel()
        if current:
            self._schedule_session_timeline_refresh(delay=0.1)
