from __future__ import annotations

import base64
import contextlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClipboardStatus:
    backend: str
    confidence: str
    diagnostics: tuple[str, ...] = ()


class PromptUIMixin:
    @staticmethod
    def _osc52_sequence(payload: str) -> str:
        encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        return f"\x1b]52;c;{encoded}\x07"

    @classmethod
    def _terminal_clipboard_sequence(cls, payload: str) -> str:
        osc52 = cls._osc52_sequence(payload)
        if os.getenv("TMUX"):
            escaped = osc52.replace("\x1b", "\x1b\x1b")
            return f"\x1bPtmux;{escaped}\x1b\\"
        term = str(os.getenv("TERM", "") or "").lower()
        if term.startswith("screen"):
            escaped = osc52.replace("\x1b", "\x1b\x1b")
            return f"\x1bP{escaped}\x1b\\"
        return osc52

    def _write_terminal_clipboard_sequence(self, sequence: str) -> bool:
        targets: list[Any] = []
        try:
            tty = open("/dev/tty", "w", encoding="utf-8", errors="ignore")
            targets.append(tty)
        except Exception:
            pass
        stdout_stream = getattr(sys, "__stdout__", None)
        if stdout_stream is not None and stdout_stream not in targets:
            targets.append(stdout_stream)
        stream = getattr(self, "_driver", None)
        if stream is not None and stream not in targets:
            targets.append(stream)

        for target in targets:
            try:
                write = getattr(target, "write", None)
                if callable(write):
                    write(sequence)
                    flush = getattr(target, "flush", None)
                    if callable(flush):
                        flush()
                    return True
            except Exception:
                continue
            finally:
                if getattr(target, "name", None) == "/dev/tty":
                    with contextlib.suppress(Exception):
                        target.close()
        return False

    def _copy_text_via_terminal_clipboard(self, payload: str) -> bool:
        sequence = self._terminal_clipboard_sequence(payload)
        return self._write_terminal_clipboard_sequence(sequence)

    def _copy_text_via_native_clipboard(self, payload: str) -> bool:
        try:
            if sys.platform == "darwin" and shutil.which("pbcopy"):
                subprocess.run(["pbcopy"], input=payload, text=True, encoding="utf-8", check=True)
                return True
            if os.name == "nt" and shutil.which("clip"):
                subprocess.run(["clip"], input=payload, text=True, encoding="utf-8", check=True)
                return True
            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=payload, text=True, encoding="utf-8", check=True)
                return True
            if shutil.which("xclip"):
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=payload,
                    text=True,
                    encoding="utf-8",
                    check=True,
                )
                return True
        except Exception:
            return False
        return False

    def _tmux_option_value(self, option: str) -> str | None:
        if not os.getenv("TMUX") or not shutil.which("tmux"):
            return None
        try:
            result = subprocess.run(
                ["tmux", "show-options", "-gqv", option],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return value or None

    def _tmux_clipboard_diagnostics(self) -> tuple[str, ...]:
        if not os.getenv("TMUX"):
            return ()
        if not shutil.which("tmux"):
            return (
                "tmux session detected, but `tmux` is unavailable for clipboard checks; OSC 52 requests may stay "
                "inside tmux.",
            )

        diagnostics: list[str] = []
        set_clipboard = str(self._tmux_option_value("set-clipboard") or "").strip().lower()
        if set_clipboard not in {"on", "external"}:
            diagnostics.append(
                "tmux `set-clipboard` is not enabled; enable it with `set -g set-clipboard on` so OSC 52 can reach "
                "the terminal clipboard."
            )
        allow_passthrough = str(self._tmux_option_value("allow-passthrough") or "").strip().lower()
        if allow_passthrough not in {"all", "on"}:
            diagnostics.append(
                "tmux `allow-passthrough` is not enabled; set `set -g allow-passthrough on` if your terminal "
                "supports OSC 52 passthrough."
            )
        return tuple(diagnostics)

    def _copy_text_to_clipboard_status(self, payload: str) -> ClipboardStatus:
        if self._copy_text_via_native_clipboard(payload):
            return ClipboardStatus(backend="native_clipboard", confidence="confirmed")
        if self._copy_text_via_terminal_clipboard(payload):
            return ClipboardStatus(
                backend="terminal_clipboard",
                confidence="best_effort",
                diagnostics=self._tmux_clipboard_diagnostics(),
            )
        try:
            self.copy_to_clipboard(payload)
            return ClipboardStatus(
                backend="terminal_bridge",
                confidence="best_effort",
                diagnostics=self._tmux_clipboard_diagnostics(),
            )
        except Exception:
            return ClipboardStatus(backend="unavailable", confidence="none")

    def _refresh_shortcut_map(self, settings: Any | None = None) -> None:
        from swarmee_river.settings import load_settings

        loaded = settings or load_settings()
        shortcuts = getattr(getattr(loaded, "tui", None), "shortcuts", None)
        self._shortcut_map = {
            "toggle_transcript_mode": tuple(getattr(shortcuts, "toggle_transcript_mode", ["f8"])),
            "copy_selection": tuple(
                getattr(shortcuts, "copy_selection", ["ctrl+shift+c", "ctrl+c", "meta+c", "super+c"])
            ),
        }

    def _shortcut_matches(self, action: str, key: str) -> bool:
        normalized_action = str(action or "").strip()
        normalized_key = str(key or "").strip().lower()
        if not normalized_action or not normalized_key:
            return False
        alias_map = {
            "cmd+c": "meta+c",
            "command+c": "meta+c",
        }
        normalized_key = alias_map.get(normalized_key, normalized_key)
        mapping = getattr(self, "_shortcut_map", {}) or {}
        tokens = mapping.get(normalized_action)
        if not isinstance(tokens, tuple):
            return False
        return normalized_key in tokens

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
        if self._status_bar is not None:
            self._status_bar.set_state("cancelling")
        self._write_transcript_line("[run] cancelling...")
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
            if focused.id != "prompt" and focused_text:
                label = f"{focused.id} pane" if getattr(focused, "id", None) else "text area"
                self._copy_text(focused_text + "\n", label=label)
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

        status = self._copy_text_to_clipboard_status(payload)
        if status.backend == "native_clipboard" and status.confidence == "confirmed":
            self._notify(f"{label}: copied to clipboard.")
            return
        if status.backend in {"terminal_clipboard", "terminal_bridge"}:
            message = f"{label}: copy requested via terminal clipboard."
            if status.diagnostics:
                message += " " + " ".join(status.diagnostics)
            self._notify(message, severity="information")
            return

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
