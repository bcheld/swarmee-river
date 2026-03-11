from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from typing import Any


class PromptUIMixin:
    def _update_prompt_placeholder(self) -> None:
        from textual.widgets import TextArea

        input_widget = self.query_one("#prompt", TextArea)
        approval = "on" if self._default_auto_approve else "off"
        input_widget.placeholder = f"Auto-approve: {approval}. Enter submits. Shift+Enter/Ctrl+J adds newline."

    def _update_command_palette(self, text: str) -> None:
        if self._command_palette is None:
            return
        stripped = text.strip()
        if stripped.startswith("/") and "\n" not in stripped:
            self._command_palette.filter(stripped)
        else:
            self._command_palette.hide()

    def _switch_side_tab(self, tab_id: str) -> None:
        from textual.widgets import TabbedContent

        with contextlib.suppress(Exception):
            tabs = self.query_one("#side_tabs", TabbedContent)
            tabs.active = tab_id
        self._sync_settings_sidebar_autosize(tab_id)

    def _seed_prompt_with_command(self, command: str) -> None:
        from textual.widgets import TextArea

        prompt_widget = self.query_one("#prompt", TextArea)
        existing = (prompt_widget.text or "").strip()
        prompt_widget.clear()
        command_text = command.strip()
        if existing and not existing.startswith("/"):
            seeded = f"{command_text} {existing}".strip() + " "
        else:
            seeded = f"{command_text} "
        for method_name in ("insert", "insert_text_at_cursor"):
            method = getattr(prompt_widget, method_name, None)
            if callable(method):
                with contextlib.suppress(Exception):
                    method(seeded)
                    break
        prompt_widget.focus()

    def _estimate_prompt_tokens(self, text: str) -> int:
        # Lightweight heuristic for pre-send token estimate.
        return max(0, (len(text or "") + 3) // 4)

    def _apply_prompt_estimate(self) -> None:
        text = self._pending_prompt_estimate_text
        self._prompt_input_tokens_est = self._estimate_prompt_tokens(text)
        self._refresh_prompt_metrics()

    def _schedule_prompt_estimate_update(self, text: str) -> None:
        timer = self._prompt_estimate_timer
        self._prompt_estimate_timer = None
        if timer is not None:
            with contextlib.suppress(RuntimeError):
                timer.stop()
        self._pending_prompt_estimate_text = text or ""
        self._prompt_estimate_timer = self.set_timer(0.2, self._apply_prompt_estimate)

    def _on_prompt_text_changed(self, text: str) -> None:
        self._schedule_prompt_estimate_update(text)

    def _refresh_prompt_metrics(self) -> None:
        if self._prompt_metrics is None:
            return
        set_context = getattr(self._prompt_metrics, "set_context", None)
        if callable(set_context):
            set_context(
                prompt_tokens_est=self.state.daemon.last_prompt_tokens_est,
                budget_tokens=self.state.daemon.last_budget_tokens,
                animate=True,
            )
        set_prompt_estimate = getattr(self._prompt_metrics, "set_prompt_input_estimate", None)
        if callable(set_prompt_estimate):
            set_prompt_estimate(self._prompt_input_tokens_est)
        set_provider_usage = getattr(self._prompt_metrics, "set_provider_usage", None)
        if callable(set_provider_usage):
            set_provider_usage(
                input_tokens=self.state.daemon.last_provider_input_tokens,
                cached_input_tokens=self.state.daemon.last_provider_cached_input_tokens,
                output_tokens=self.state.daemon.last_provider_output_tokens,
                cost_usd=self.state.daemon.last_cost_usd,
            )

    def action_focus_prompt(self) -> None:
        from textual.widgets import TextArea

        self.query_one("#prompt", TextArea).focus()

    def action_submit_prompt(self) -> None:
        from textual.widgets import TextArea

        prompt_widget = self.query_one("#prompt", TextArea)
        text = (prompt_widget.text or "").strip()
        prompt_widget.clear()
        if text:
            self._prompt_history.append(text)
            if len(self._prompt_history) > self._MAX_PROMPT_HISTORY:
                self._prompt_history = self._prompt_history[-self._MAX_PROMPT_HISTORY :]
            self._history_index = -1
            self._history_draft_text = None
            self._handle_user_input(text)

    def action_interrupt_run(self) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None or not self.state.daemon.query_active:
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            return
        self._flush_all_streaming_buffers()
        self._clear_pending_tool_starts()
        self._dismiss_thinking(emit_summary=True)
        send_daemon_command(proc, {"cmd": "interrupt"})
        self._write_transcript_line("[run] interrupted.")
        self._reset_consent_panel()
        self._reset_error_action_prompt()

    def action_copy_selection(self) -> None:
        from textual.containers import VerticalScroll
        from textual.widgets import TextArea

        focused = getattr(self, "focused", None)
        if isinstance(focused, TextArea):
            selected_text = focused.selected_text or ""
            if selected_text.strip():
                self._copy_text(selected_text, label="selection")
                return
            if focused.id == "transcript_text":
                transcript_text = self._get_transcript_text()
                if transcript_text.strip():
                    self._copy_text(transcript_text, label="transcript")
                    return
                self._notify("transcript: nothing to copy.", severity="warning")
                return
            focused_text = (getattr(focused, "text", "") or "").strip()
            if focused.id in {"issues", "plan", "artifacts", "agent_summary"} and focused_text:
                self._copy_text(focused_text + "\n", label=f"{focused.id} pane")
                return
            self._notify("Select text first.", severity="warning")
            return

        transcript_widget: Any
        if self._transcript_mode == "text":
            transcript_widget = self.query_one("#transcript_text", TextArea)
        else:
            transcript_widget = self.query_one("#transcript", VerticalScroll)

        if isinstance(transcript_widget, TextArea):
            selected_text = self._get_richlog_selection_text(transcript_widget)
            if selected_text.strip():
                self._copy_text(selected_text, label="selection")
                return
            transcript_text = self._get_transcript_text()
            if transcript_text.strip():
                self._copy_text(transcript_text, label="transcript")
                return
            self._notify("Select text first.", severity="warning")
            return

        node = focused
        while node is not None:
            if node is transcript_widget:
                self._copy_text(self._get_transcript_text(), label="transcript")
                return
            node = getattr(node, "parent", None)

        self._notify("No text area focused.", severity="warning")

    def _copy_text(self, text: str, *, label: str) -> None:
        payload = text or ""
        if not payload.strip():
            self._notify(f"{label}: nothing to copy.", severity="warning")
            return

        # Prefer native clipboard commands in terminal contexts (more reliable than Textual clipboard bridges).
        try:
            if sys.platform == "darwin" and shutil.which("pbcopy"):
                subprocess.run(["pbcopy"], input=payload, text=True, encoding="utf-8", check=True)
                self._notify(f"{label}: copied to clipboard.")
                return
            if os.name == "nt" and shutil.which("clip"):
                subprocess.run(["clip"], input=payload, text=True, encoding="utf-8", check=True)
                self._notify(f"{label}: copied to clipboard.")
                return
            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=payload, text=True, encoding="utf-8", check=True)
                self._notify(f"{label}: copied to clipboard.")
                return
            if shutil.which("xclip"):
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=payload,
                    text=True,
                    encoding="utf-8",
                    check=True,
                )
                self._notify(f"{label}: copied to clipboard.")
                return
        except Exception:
            pass

        try:
            self.copy_to_clipboard(payload)
            self._notify(f"{label}: copied to clipboard.")
            return
        except Exception:
            pass

        artifact_path = self._persist_run_transcript(
            pid=(self.state.daemon.proc.pid if self.state.daemon.proc is not None else None),
            session_id=self.state.daemon.session_id,
            prompt=f"(copy) {label}",
            auto_approve=False,
            exit_code=0,
            output_text=payload,
        )
        if artifact_path:
            self._add_artifact_paths([artifact_path])
            self._write_transcript_line(f"[copy] {label}: clipboard unavailable. Saved to artifact: {artifact_path}")
        else:
            self._write_transcript_line(f"[copy] {label}: clipboard unavailable.")
