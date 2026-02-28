from __future__ import annotations

import contextlib
import time
from typing import Any

from swarmee_river.tui.text_sanitize import sanitize_output_text

_STREAMING_FLUSH_INTERVAL_S = 0.15
_TOOL_PROGRESS_RENDER_INTERVAL_S = 0.15
_TOOL_START_COALESCE_INTERVAL_S = 0.1
_TOOL_HEARTBEAT_RENDER_MIN_STEP_S = 0.5
_TOOL_OUTPUT_RETENTION_MAX_CHARS = 4096


class ToolsMixin:
    def _cancel_streaming_flush_timer(self) -> None:
        timer = self._streaming_flush_timer
        self._streaming_flush_timer = None
        if timer is not None:
            with contextlib.suppress(RuntimeError):
                timer.stop()

    def _schedule_streaming_flush(self) -> None:
        if self._streaming_flush_timer is None:
            self._streaming_flush_timer = self.set_timer(
                _STREAMING_FLUSH_INTERVAL_S,
                self._on_streaming_flush_timer,
            )

    def _on_streaming_flush_timer(self) -> None:
        self._streaming_flush_timer = None
        self._flush_streaming_buffer()

    def _flush_streaming_buffer(self) -> None:
        from swarmee_river.tui.widgets import AssistantMessage

        if not self._streaming_buffer:
            return
        text = "".join(self._streaming_buffer)
        self._streaming_buffer = []
        if not text:
            return
        self._current_assistant_chunks.append(text)
        if self._active_assistant_message is None:
            self._active_assistant_message = AssistantMessage(
                model=self._current_assistant_model,
                timestamp=self._current_assistant_timestamp,
            )
            self._mount_transcript_widget(self._active_assistant_message)
        with contextlib.suppress(Exception):
            self._active_assistant_message.append_delta(text)
        self._record_transcript_fallback(text)
        self._assistant_placeholder_written = True

    def _cancel_tool_progress_flush_timer(self) -> None:
        timer = self._tool_progress_flush_timer
        self._tool_progress_flush_timer = None
        if timer is not None:
            with contextlib.suppress(RuntimeError):
                timer.stop()

    def _schedule_tool_progress_flush(self, tool_use_id: str | None = None) -> None:
        if isinstance(tool_use_id, str) and tool_use_id.strip():
            self._tool_progress_pending_ids.add(tool_use_id.strip())
        if self._tool_progress_flush_timer is None:
            self._tool_progress_flush_timer = self.set_timer(
                _STREAMING_FLUSH_INTERVAL_S,
                self._on_tool_progress_flush_timer,
            )

    def _on_tool_progress_flush_timer(self) -> None:
        self._tool_progress_flush_timer = None
        pending_ids = list(self._tool_progress_pending_ids)
        self._tool_progress_pending_ids.clear()
        for tool_use_id in pending_ids:
            rendered = self._flush_tool_progress_render(tool_use_id)
            record = self._tool_blocks.get(tool_use_id)
            has_pending = bool(record and str(record.get("pending_output", "")))
            if has_pending and not rendered:
                self._tool_progress_pending_ids.add(tool_use_id)
        if self._tool_progress_pending_ids:
            self._schedule_tool_progress_flush()

    def _flush_all_streaming_buffers(self) -> None:
        self._cancel_streaming_flush_timer()
        self._flush_streaming_buffer()
        self._cancel_tool_progress_flush_timer()
        self._tool_progress_pending_ids.clear()
        for tool_use_id in list(self._tool_blocks.keys()):
            self._flush_tool_progress_render(tool_use_id, force=True)

    def _flush_transcript_dedup(self) -> None:
        """Flush any pending deduplicated transcript line."""
        if self._last_transcript_dedup_count > 1:
            msg = f"{self._last_transcript_dedup_line} (×{self._last_transcript_dedup_count})"
            self._write_transcript(msg)
        # count == 1 lines were already written immediately in _write_transcript_line;
        # no need to re-emit them here.
        self._last_transcript_dedup_line = ""
        self._last_transcript_dedup_count = 0

    def _write_transcript_line(self, line: str) -> None:
        """Write a plain text line to the transcript (used for TUI-internal messages)."""
        if self.state.daemon.query_active:
            self.state.daemon.turn_output_chunks.append(sanitize_output_text(f"[tui] {line}\n"))
        # Deduplicate consecutive identical lines.
        if line == self._last_transcript_dedup_line:
            self._last_transcript_dedup_count += 1
            return
        self._flush_transcript_dedup()
        self._last_transcript_dedup_line = line
        self._last_transcript_dedup_count = 1
        self._write_transcript(line)

    def _tool_input_summary(self, tool_name: str, tool_input: Any) -> str:
        from swarmee_river.tui.widgets import format_tool_input_oneliner

        if not isinstance(tool_input, dict):
            return ""
        return format_tool_input_oneliner(tool_name, tool_input)

    def _tool_start_plain_text(self, tool_name: str, tool_input: Any) -> str:
        summary = self._tool_input_summary(tool_name, tool_input)
        if summary:
            return f"⚙ {tool_name} — {summary} ..."
        return f"⚙ {tool_name} ..."

    def _tool_result_plain_text(self, tool_name: str, status: str, duration_s: float, tool_input: Any) -> str:
        succeeded = status == "success"
        glyph = "✓" if succeeded else "✗"
        summary = self._tool_input_summary(tool_name, tool_input)
        base = f"{glyph} {tool_name} ({duration_s:.1f}s)"
        if summary:
            base = f"{base} — {summary}"
        if not succeeded:
            label = (status or "error").strip()
            base = f"{base} ({label})"
        return base

    def _cancel_tool_start_timer(self, tool_use_id: str) -> None:
        timer = self._tool_pending_start_timers.pop(tool_use_id, None)
        if timer is not None:
            with contextlib.suppress(RuntimeError):
                timer.stop()

    def _clear_pending_tool_starts(self) -> None:
        for tool_use_id in list(self._tool_pending_start_timers.keys()):
            self._cancel_tool_start_timer(tool_use_id)
        self._tool_pending_start_timers = {}
        self._tool_pending_start = {}

    def _emit_tool_start_line(self, tool_use_id: str) -> bool:
        from swarmee_river.tui.widgets import ToolCallBlock

        record = self._tool_blocks.get(tool_use_id)
        if record is None:
            self._tool_pending_start.pop(tool_use_id, None)
            self._cancel_tool_start_timer(tool_use_id)
            return False
        if bool(record.get("start_rendered")):
            self._tool_pending_start.pop(tool_use_id, None)
            self._cancel_tool_start_timer(tool_use_id)
            return False
        tool_name = str(record.get("tool", "unknown"))
        tool_input = record.get("input")
        block = ToolCallBlock(tool_name, tool_use_id)
        if isinstance(tool_input, dict):
            with contextlib.suppress(Exception):
                block.set_input(tool_input)
        record["widget"] = block
        self._mount_transcript_widget(block, plain_text=self._tool_start_plain_text(tool_name, tool_input))
        record["start_rendered"] = True
        self._tool_pending_start.pop(tool_use_id, None)
        self._cancel_tool_start_timer(tool_use_id)
        return True

    def _on_tool_start_coalesce_timer(self, tool_use_id: str) -> None:
        self._tool_pending_start_timers.pop(tool_use_id, None)
        self._emit_tool_start_line(tool_use_id)

    def _schedule_tool_start_line(self, tool_use_id: str) -> None:
        if not tool_use_id:
            return
        self._tool_pending_start[tool_use_id] = time.monotonic()
        self._cancel_tool_start_timer(tool_use_id)
        self._tool_pending_start_timers[tool_use_id] = self.set_timer(
            _TOOL_START_COALESCE_INTERVAL_S,
            lambda tid=tool_use_id: self._on_tool_start_coalesce_timer(tid),
        )

    def _append_tool_output(self, record: dict[str, Any], chunk: str) -> None:
        text = sanitize_output_text(str(chunk or ""))
        if not text:
            return
        output = str(record.get("output", "")) + text
        if len(output) > _TOOL_OUTPUT_RETENTION_MAX_CHARS:
            output = output[-_TOOL_OUTPUT_RETENTION_MAX_CHARS:]
        record["output"] = output

    def _queue_tool_progress_content(self, record: dict[str, Any], *, content: str, stream: str) -> None:
        text = sanitize_output_text(str(content or ""))
        if not text:
            return
        self._append_tool_output(record, text)
        pending = str(record.get("pending_output", ""))
        pending_stream = str(record.get("pending_stream", "stdout") or "stdout")
        normalized_stream = stream if stream in {"stdout", "stderr", "mixed"} else "stdout"
        chunk = text
        if pending:
            if pending_stream != normalized_stream:
                pending_stream = "mixed"
                chunk = f"[{normalized_stream}] {text}"
        else:
            pending_stream = normalized_stream
        pending += chunk
        if len(pending) > _TOOL_OUTPUT_RETENTION_MAX_CHARS:
            pending = pending[-_TOOL_OUTPUT_RETENTION_MAX_CHARS:]
        record["pending_output"] = pending
        record["pending_stream"] = pending_stream

    def _flush_tool_progress_render(self, tool_use_id: str, *, force: bool = False) -> bool:
        from swarmee_river.tui.widgets import render_tool_heartbeat_line, render_tool_progress_chunk

        record = self._tool_blocks.get(tool_use_id)
        if record is None:
            return False
        if not bool(record.get("start_rendered")):
            self._emit_tool_start_line(tool_use_id)
        widget = record.get("widget")
        now = time.monotonic()
        last = float(record.get("last_progress_render_mono", 0.0))
        pending = str(record.get("pending_output", ""))
        if pending:
            if force or (now - last) >= _TOOL_PROGRESS_RENDER_INTERVAL_S:
                stream = str(record.get("pending_stream", "stdout") or "stdout")
                if widget is not None:
                    with contextlib.suppress(Exception):
                        widget.append_output(pending, stream=stream)
                else:
                    self._mount_transcript_widget(
                        render_tool_progress_chunk(pending, stream=stream),
                        plain_text=pending,
                    )
                self._record_transcript_fallback(pending)
                record["pending_output"] = ""
                record["pending_stream"] = "stdout"
                record["last_progress_render_mono"] = now
                return True
            return False

        elapsed = record.get("elapsed_s")
        if force or not isinstance(elapsed, (int, float)):
            return False
        elapsed_s = float(elapsed)
        previous = float(record.get("last_heartbeat_rendered_s", 0.0))
        if (elapsed_s - previous) < _TOOL_HEARTBEAT_RENDER_MIN_STEP_S:
            return False
        if (now - last) < _TOOL_PROGRESS_RENDER_INTERVAL_S:
            return False
        tool_name = str(record.get("tool", "unknown"))
        if widget is not None:
            with contextlib.suppress(Exception):
                widget.set_elapsed(elapsed_s)
        else:
            self._mount_transcript_widget(
                render_tool_heartbeat_line(tool_name, elapsed_s=elapsed_s, tool_use_id=tool_use_id),
                plain_text=f"⚙ {tool_name} running... ({elapsed_s:.1f}s)",
            )
        record["last_progress_render_mono"] = now
        record["last_heartbeat_rendered_s"] = elapsed_s
        return True

    def _expand_tool_call(self, tool_use_id: str) -> None:
        import json as _json
        from swarmee_river.tui.widgets import render_tool_details_panel
        from swarmee_river.tui.commands import _EXPAND_USAGE_TEXT
        tid = tool_use_id.strip()
        if not tid:
            self._write_transcript_line(_EXPAND_USAGE_TEXT)
            return
        record = self._tool_blocks.get(tid)
        if record is None:
            self._write_transcript_line(f"[expand] unknown tool id: {tid}")
            return
        self._mount_transcript_widget(
            render_tool_details_panel(record),
            plain_text=_json.dumps(record, indent=2, ensure_ascii=False),
        )
