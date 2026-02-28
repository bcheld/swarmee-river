from __future__ import annotations

import contextlib
from typing import Any

from swarmee_river.tui.text_sanitize import sanitize_output_text


class TranscriptMixin:
    def _record_transcript_fallback(self, text: str) -> None:
        clean = sanitize_output_text(text).rstrip("\n")
        if not clean:
            return
        self._transcript_fallback_lines.extend(clean.splitlines())
        if len(self._transcript_fallback_lines) > self._TRANSCRIPT_MAX_LINES:
            self._transcript_fallback_lines = self._transcript_fallback_lines[-self._TRANSCRIPT_MAX_LINES :]

    def _sync_transcript_text_widget(self, *, scroll_to_end: bool = True) -> None:
        from textual.widgets import TextArea

        text_widget = self.query_one("#transcript_text", TextArea)
        text = "\n".join(self._transcript_fallback_lines).rstrip()
        if text:
            text += "\n"
        text_widget.load_text(text)
        if scroll_to_end:
            self._scroll_transcript_text_to_end()

    def _scroll_transcript_text_to_end(self) -> None:
        from textual.widgets import TextArea

        text_widget = self.query_one("#transcript_text", TextArea)
        with contextlib.suppress(Exception):
            text_widget.scroll_end(animate=False)
        for method_name in ("action_cursor_document_end", "action_end"):
            method = getattr(text_widget, method_name, None)
            if callable(method):
                with contextlib.suppress(Exception):
                    method()
                    break

    def _get_scroll_proportion(self, widget: Any) -> float:
        """Get 0.0-1.0 proportion of current scroll position."""
        try:
            scroll_y = float(getattr(getattr(widget, "scroll_offset", None), "y", 0.0) or 0.0)
            virtual_h = float(getattr(getattr(widget, "virtual_size", None), "height", 0.0) or 0.0)
            viewport_h = float(getattr(getattr(widget, "size", None), "height", 0.0) or 0.0)
            max_scroll = virtual_h - viewport_h
            if max_scroll <= 0:
                return 1.0
            return min(1.0, max(0.0, scroll_y / max_scroll))
        except Exception:
            return 1.0

    def _set_scroll_proportion(self, widget: Any, proportion: float) -> None:
        """Set scroll position from 0.0-1.0 proportion."""
        try:
            normalized = min(1.0, max(0.0, float(proportion)))
            virtual_h = float(getattr(getattr(widget, "virtual_size", None), "height", 0.0) or 0.0)
            viewport_h = float(getattr(getattr(widget, "size", None), "height", 0.0) or 0.0)
            max_scroll = virtual_h - viewport_h
            if max_scroll <= 0:
                return
            target = int(normalized * max_scroll)
            scroll_to = getattr(widget, "scroll_to", None)
            if callable(scroll_to):
                scroll_to(0, target, animate=False)
                return
            scroll_relative = getattr(widget, "scroll_relative", None)
            if callable(scroll_relative):
                current_y = float(getattr(getattr(widget, "scroll_offset", None), "y", 0.0) or 0.0)
                scroll_relative(y=target - current_y, animate=False)
        except Exception:
            pass

    def _set_transcript_mode(self, mode: str, *, notify: bool = True) -> None:
        from textual.containers import VerticalScroll
        from textual.widgets import TextArea

        normalized = mode.strip().lower()
        if normalized not in {"rich", "text"}:
            return
        rich_widget = self.query_one("#transcript", VerticalScroll)
        text_widget = self.query_one("#transcript_text", TextArea)
        if normalized == "text":
            proportion = self._get_scroll_proportion(rich_widget)
            at_bottom = proportion > 0.95
            self._sync_transcript_text_widget(scroll_to_end=at_bottom)
            rich_widget.styles.display = "none"
            text_widget.styles.display = "block"
            if at_bottom:
                self._scroll_transcript_text_to_end()
            else:
                self.set_timer(0.05, lambda p=proportion: self._set_scroll_proportion(text_widget, p))
            self._transcript_mode = "text"
            if notify:
                self._notify("Text mode: select text with mouse. /text to return.", severity="information")
            return

        proportion = self._get_scroll_proportion(text_widget)
        at_bottom = proportion > 0.95
        text_widget.styles.display = "none"
        rich_widget.styles.display = "block"
        if at_bottom:
            with contextlib.suppress(Exception):
                rich_widget.scroll_end(animate=False)
        else:
            self.set_timer(0.05, lambda p=proportion: self._set_scroll_proportion(rich_widget, p))
        self._transcript_mode = "rich"
        if notify:
            self._notify("Rich mode restored.", severity="information")

    def _toggle_transcript_mode(self) -> None:
        target = "text" if self._transcript_mode != "text" else "rich"
        self._set_transcript_mode(target, notify=True)

    def _mount_transcript_widget(self, renderable: Any, *, plain_text: str | None = None) -> None:
        """Write a renderable widget/content into the transcript view."""
        from textual.containers import VerticalScroll
        from textual.widget import Widget as WidgetBase
        from textual.widgets import Static

        transcript = self.query_one("#transcript", VerticalScroll)
        if isinstance(renderable, WidgetBase):
            node = renderable
        else:
            node = Static(renderable)
        with contextlib.suppress(Exception):
            transcript.mount(node)
        children = list(getattr(transcript, "children", []))
        if len(children) > self._TRANSCRIPT_MAX_LINES:
            overflow = len(children) - self._TRANSCRIPT_MAX_LINES
            for child in children[:overflow]:
                with contextlib.suppress(Exception):
                    child.remove()
        if isinstance(plain_text, str):
            self._record_transcript_fallback(plain_text)
        elif isinstance(renderable, str):
            self._record_transcript_fallback(renderable)
        if self._transcript_mode == "text":
            self._sync_transcript_text_widget()
        with contextlib.suppress(Exception):
            transcript.scroll_end(animate=False)

    def _write_transcript(self, line: str) -> None:
        """Write a system/info message to the transcript."""
        from swarmee_river.tui.widgets import render_system_message

        self._mount_transcript_widget(render_system_message(line), plain_text=line)

    def _call_from_thread_safe(self, callback: Any, *args: Any, **kwargs: Any) -> None:
        import contextlib as _ctx
        if self.state.daemon.is_shutting_down:
            return
        with _ctx.suppress(Exception):
            self.call_from_thread(callback, *args, **kwargs)

    def _write_user_input(self, text: str) -> None:
        from swarmee_river.tui.widgets import render_user_message
        timestamp = self._turn_timestamp()
        plain = f"YOU> {text}\n{timestamp}"
        self._mount_transcript_widget(render_user_message(text, timestamp=timestamp), plain_text=plain)

    def _write_user_message(self, text: str, *, timestamp: str | None = None) -> None:
        from swarmee_river.tui.widgets import render_user_message
        resolved_timestamp = (timestamp or "").strip() or self._turn_timestamp()
        plain = f"YOU> {text}\n{resolved_timestamp}"
        self._mount_transcript_widget(render_user_message(text, timestamp=resolved_timestamp), plain_text=plain)

    def _write_assistant_message(
        self,
        text: str,
        *,
        model: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        from swarmee_river.tui.widgets import render_assistant_message
        resolved_timestamp = (timestamp or "").strip() or self._turn_timestamp()
        self._last_assistant_text = text
        plain_lines = [text]
        meta_parts = [part for part in [model, resolved_timestamp] if isinstance(part, str) and part.strip()]
        if meta_parts:
            plain_lines.append(" · ".join(meta_parts))
        self._mount_transcript_widget(
            render_assistant_message(text, model=model, timestamp=resolved_timestamp),
            plain_text="\n".join(plain_lines),
        )

    def _turn_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%I:%M %p").lstrip("0")

    def _append_plain_text(self, text: str) -> None:
        """Write a plain text line for non-event fallback output."""
        if not text.strip():
            return
        self._mount_transcript_widget(text, plain_text=text)

    def _get_transcript_text(self) -> str:
        if self._transcript_fallback_lines:
            return "\n".join(self._transcript_fallback_lines).rstrip() + "\n"
        return ""

    def _get_richlog_selection_text(self, transcript: Any) -> str:
        from textual.widgets import TextArea
        if isinstance(transcript, TextArea):
            selected = transcript.selected_text or ""
            return selected if isinstance(selected, str) else ""
        return ""

    def _get_all_text(self) -> str:
        from swarmee_river.tui.widgets import render_agent_profile_summary_text
        parts = [
            "# Transcript",
            self._get_transcript_text().rstrip(),
            "",
            "# Plan",
            (self.state.plan.text or "").rstrip() or "(no plan)",
            "",
            "# Session Issues",
            "\n".join(self.state.session.issue_lines).rstrip() or "(no issues)",
            "",
            "# Artifacts",
            self._get_artifacts_text().rstrip() or "(no artifacts)",
            "",
            "# Agent Profile",
            render_agent_profile_summary_text(self._session_effective_profile().to_dict()),
            "",
            "# Context Sources",
            "\n".join(self._context_list_lines()).rstrip() or "(no context sources)",
            "",
            "# Consent History",
            "\n".join(self._consent_history_lines).rstrip() or "(no consent decisions)",
            "",
        ]
        return "\n".join(parts).rstrip() + "\n"

    def action_copy_transcript(self) -> None:
        self._copy_text(self._get_transcript_text(), label="transcript")

    def action_widen_side(self) -> None:
        if self._split_ratio > 1:
            self._split_ratio -= 1
            self._apply_split_ratio()

    def action_widen_transcript(self) -> None:
        if self._split_ratio < 4:
            self._split_ratio += 1
            self._apply_split_ratio()

    def _apply_split_ratio(self) -> None:
        from textual.containers import VerticalScroll, Vertical
        from textual.widgets import TextArea
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript_text = self.query_one("#transcript_text", TextArea)
        side = self.query_one("#side", Vertical)
        transcript.styles.width = f"{self._split_ratio}fr"
        transcript_text.styles.width = f"{self._split_ratio}fr"
        side.styles.width = "1fr"
        self.refresh(layout=True)

    def action_toggle_transcript_mode(self) -> None:
        self._toggle_transcript_mode()

    def action_search_transcript(self) -> None:
        from textual.widgets import TextArea
        prompt_widget = self.query_one("#prompt", TextArea)
        prompt_widget.clear()
        # Hide palette first to avoid rendering issues
        if self._command_palette is not None:
            self._command_palette.hide()
        for method_name in ("insert", "insert_text_at_cursor"):
            method = getattr(prompt_widget, method_name, None)
            if callable(method):
                with contextlib.suppress(Exception):
                    method("/search ")
                    break
        # Hide palette again after insert (on_text_area_changed may re-show it)
        if self._command_palette is not None:
            self._command_palette.hide()
        prompt_widget.focus()

    def _search_transcript(self, term: str) -> None:
        from textual.containers import VerticalScroll
        if not term.strip():
            self._write_transcript_line("Usage: /search <term>")
            return
        term_lower = term.lower()
        transcript_text = self._get_transcript_text()
        if term_lower in transcript_text.lower():
            with contextlib.suppress(Exception):
                self.query_one("#transcript", VerticalScroll).scroll_end(animate=True)
            self._write_transcript_line("[search] found match in transcript.")
            return
        self._write_transcript_line(f"[search] no match for '{term}'.")

    def _handle_copy_command(self, normalized: str) -> bool:
        from swarmee_river.tui.commands import classify_copy_command
        command = classify_copy_command(normalized)
        if command == "transcript":
            self.action_copy_transcript()
            return True

        if command == "plan":
            self.action_copy_plan()
            return True

        if command == "issues":
            self.action_copy_issues()
            return True

        if command == "artifacts":
            self.action_copy_artifacts()
            return True

        if command == "last":
            self._copy_text(self._last_assistant_text, label="last response")
            return True

        if command == "all":
            self._copy_text(self._get_all_text(), label="all")
            return True

        return False
