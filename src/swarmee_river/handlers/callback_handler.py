import atexit
import contextlib
import re
import sys
import time
import warnings
import weakref
from threading import Event
from typing import Any

from colorama import Fore, Style, init
from halo import Halo
from rich.status import Status

from swarmee_river.utils.env_utils import truthy_env
from swarmee_river.utils.stdio_utils import configure_stdio_for_utf8, write_stdout_jsonl


def _safe_print(*parts: Any, end: str = "\n") -> None:
    try:
        print(*parts, end=end)
        return
    except UnicodeEncodeError:
        pass

    payload = f"{' '.join(str(part) for part in parts)}{end}"
    stream = sys.stdout
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        with contextlib.suppress(Exception):
            buffer.write(payload.encode("utf-8", errors="replace"))
            buffer.flush()
            return

    with contextlib.suppress(Exception):
        stream.write(payload.encode("ascii", errors="replace").decode("ascii"))
        stream.flush()


# Initialize Colorama
configure_stdio_for_utf8()
init(autoreset=True)

# Configure spinner templates
SPINNERS = {
    "dots": {
        "interval": 80,
        "frames": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    }
}

# Tool state colors
TOOL_COLORS = {
    "running": Fore.GREEN,
    "success": Fore.GREEN,
    "error": Fore.RED,
    "info": Fore.CYAN,
}

_ACTIVE_TOOL_SPINNERS: weakref.WeakSet["ToolSpinner"] = weakref.WeakSet()
_TUI_TOOL_PROGRESS_EMIT_INTERVAL_S = 0.2
_TUI_TOOL_HEARTBEAT_INTERVAL_S = 2.0
_TUI_TOOL_OUTPUT_MAX_CHARS = 4096
_PLAN_STEP_START_RE = re.compile(r"starting\s+step\s+(?P<num>\d+)\s*:\s*(?P<desc>.*)", re.IGNORECASE)
_PLAN_STEP_DONE_RE = re.compile(r"completed\s+step\s+(?P<num>\d+)\b", re.IGNORECASE)


def format_message(message: str, color: str | None = None, max_length: int = 50) -> str:
    """Format message with color and length control."""
    if len(message) > max_length:
        message = message[:max_length] + "..."
    return f"{color if color else ''}{message}{Style.RESET_ALL}"


class ToolSpinner:
    def __init__(self, text: str = "", color: str | int = TOOL_COLORS["running"]) -> None:
        self.enabled = truthy_env("SWARMEE_SPINNERS", True)
        self.spinner = Halo(
            text=text,
            spinner=SPINNERS["dots"],
            color="green",
            text_color="green",
            interval=80,
            enabled=self.enabled,
        )
        self.color = color
        self.current_text = text
        if self.enabled:
            _ACTIVE_TOOL_SPINNERS.add(self)

    def start(self, text: str | None = None) -> None:
        if not self.enabled:
            return
        if text:
            self.current_text = text
        _safe_print()  # Move to new line before starting spinner, prevents spinner from eating the previous line
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"setDaemon\(\) is deprecated, set the daemon attribute instead",
                category=DeprecationWarning,
            )
            self.spinner.start(f"{self.color}{self.current_text}{Style.RESET_ALL}")

    def update(self, text: str) -> None:
        if not self.enabled:
            return
        self.current_text = text
        self.spinner.text = f"{self.color}{text}{Style.RESET_ALL}"

    def succeed(self, text: str | None = None) -> None:
        if not self.enabled:
            return
        if text:
            self.current_text = text
        self.spinner.succeed(f"{TOOL_COLORS['success']}{self.current_text}{Style.RESET_ALL}")

    def fail(self, text: str | None = None) -> None:
        if not self.enabled:
            return
        if text:
            self.current_text = text
        self.spinner.fail(f"{TOOL_COLORS['error']}{self.current_text}{Style.RESET_ALL}")

    def info(self, text: str | None = None) -> None:
        if not self.enabled:
            return
        if text:
            self.current_text = text
        self.spinner.info(f"{TOOL_COLORS['info']}{self.current_text}{Style.RESET_ALL}")

    def stop(self) -> None:
        if not self.enabled:
            return
        with contextlib.suppress(Exception):
            self.spinner.stop()
        thread = getattr(self.spinner, "_spinner_thread", None)
        if thread is not None and hasattr(thread, "join"):
            with contextlib.suppress(Exception):
                thread.join(timeout=0.2)
        with contextlib.suppress(Exception):
            _ACTIVE_TOOL_SPINNERS.discard(self)


class CallbackHandler:
    def __init__(self) -> None:
        self.thinking_spinner: Status | None = None

        # Tool tracking
        self.current_spinner: ToolSpinner | None = None
        self.current_tool: str | None = None
        self.tool_histories: dict[str, dict[str, Any]] = {}
        self.interrupt_event: Event | None = None

    def notify(self, title: str, message: str, sound: bool = True) -> None:
        """Send a native notification using mac_automation tool."""
        _safe_print(f"Notification: {title} - {message}")

    def callback_handler(
        self,
        reasoningText: str | bool = False,
        data: str = "",
        complete: bool = False,
        force_stop: bool = False,
        message: dict[str, Any] | None = None,
        current_tool_use: dict[str, Any] | None = None,
        init_event_loop: bool = False,
        start_event_loop: bool = False,
        event_loop_throttled_delay: int | None = None,
        console: Any = None,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[Any] | None = None,
        structured_output_prompt: str | None = None,
        result: Any = None,
        **extra_event_fields: Any,
    ) -> None:
        # Future-compatible callback shape support:
        # Strands may add event fields (or merge invocation_state keys into callback kwargs).
        # Keep these accepted and ignored unless needed by this handler.
        del invocation_state
        del structured_output_model
        del structured_output_prompt
        del result
        del extra_event_fields
        message = message or {}
        current_tool_use = current_tool_use or {}

        # Cleanup calls (e.g., from top-level exception handlers) should never re-raise an interrupt.
        if force_stop:
            if self.thinking_spinner:
                self.thinking_spinner.stop()
            if self.current_spinner:
                self.current_spinner.stop()
            return

        if self.interrupt_event is not None and self.interrupt_event.is_set():
            # Don't raise from inside the callback path. Interrupt cancellation is handled by the caller.
            # Raising here can cause noisy generator shutdown errors in upstream libraries.
            if self.thinking_spinner:
                self.thinking_spinner.stop()
            if self.current_spinner:
                self.current_spinner.stop()
            return

        try:
            # Concurrent thinking spinners are usual, which leads to:
            # "Only one live display may be active at once" error thrown,
            # This try except block ignore overlap of thinking spinners.
            if self.thinking_spinner and (data or current_tool_use):
                self.thinking_spinner.stop()

            if init_event_loop and truthy_env("SWARMEE_SPINNERS", True):
                self.thinking_spinner = Status(
                    "[blue] retrieving memories...[/blue]",
                    spinner="dots",
                    console=console,
                )
                self.thinking_spinner.start()

            if reasoningText:
                _safe_print(reasoningText, end="")

            if start_event_loop and self.thinking_spinner is not None and truthy_env("SWARMEE_SPINNERS", True):
                self.thinking_spinner.update("[blue] thinking...[/blue]")
        except BaseException:
            pass

        if event_loop_throttled_delay and console:
            if self.current_spinner:
                self.current_spinner.stop()
            console.print(
                f"[red]Throttled! Waiting [bold]{event_loop_throttled_delay} seconds[/bold] before retrying...[/red]"
            )

        # Handle regular output
        if data:
            # Print to stdout
            if complete:
                _safe_print(f"{Fore.WHITE}{data}{Style.RESET_ALL}")
            else:
                _safe_print(f"{Fore.WHITE}{data}{Style.RESET_ALL}", end="")

        # Handle tool input streaming
        if current_tool_use and current_tool_use.get("input"):
            raw_tool_id = current_tool_use.get("toolUseId")
            tool_id = raw_tool_id if isinstance(raw_tool_id, str) else None
            tool_name = current_tool_use.get("name")
            tool_input = current_tool_use.get("input", "")

            if tool_id is not None:
                # Check if this is a new tool execution
                if tool_id != self.current_tool:
                    # Stop previous spinner if exists
                    if self.current_spinner:
                        self.current_spinner.stop()

                    self.current_tool = tool_id

                    self.current_spinner = ToolSpinner(f"🛠️  {tool_name}: Preparing...", TOOL_COLORS["running"])
                    self.current_spinner.start()

                    # Record tool start
                    self.tool_histories[tool_id] = {
                        "name": tool_name,
                        "start_time": time.time(),
                        "input_size": 0,
                    }

                # Update tool progress
                if tool_id in self.tool_histories:
                    current_size = len(tool_input)
                    if current_size > self.tool_histories[tool_id]["input_size"]:
                        self.tool_histories[tool_id]["input_size"] = current_size
                        if self.current_spinner:
                            self.current_spinner.update(f"🛠️  {tool_name}: {current_size} chars")

        # Process messages
        if isinstance(message, dict):
            # Handle assistant messages (tool starts)
            if message.get("role") == "assistant":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_use = content.get("toolUse")
                        if tool_use:
                            tool_name = tool_use.get("name")
                            if self.current_spinner:
                                self.current_spinner.info(f"🔧 Starting {tool_name}...")

            # Handle user messages (tool results)
            elif message.get("role") == "user":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_result = content.get("toolResult")
                        if tool_result:
                            tool_id = tool_result.get("toolUseId")
                            status = tool_result.get("status")

                            if isinstance(tool_id, str) and tool_id in self.tool_histories:
                                tool_info = self.tool_histories[tool_id]
                                duration = round(time.time() - tool_info["start_time"], 2)

                                # Prepare notification message
                                if status == "success":
                                    status_message = f"{tool_info['name']} completed in {duration}s"
                                else:
                                    status_message = f"{tool_info['name']} failed after {duration}s"

                                # Update spinner only if not in Lambda
                                if self.current_spinner:
                                    if status == "success":
                                        self.current_spinner.succeed(status_message)
                                    else:
                                        self.current_spinner.fail(status_message)

                                # Send notification
                                # Uncomment for enabling notifications.
                                # self.notify(title, message, sound=(status != "success"))

                                # Cleanup
                                del self.tool_histories[tool_id]
                                self.current_spinner = None
                                self.current_tool = None


class TuiCallbackHandler:
    """Callback handler that emits structured JSONL events to stdout for TUI consumption."""

    def __init__(self) -> None:
        self.current_tool: str | None = None
        self.tool_histories: dict[str, dict[str, Any]] = {}
        self.interrupt_event: Event | None = None
        self._saw_text_delta: bool = False
        self._assistant_text_snapshot: str = ""
        self._emitted_tool_results: set[str] = set()
        self._plan_step_status: dict[int, str] = {}
        self._plan_marker_buffer: str = ""
        self._plan_total_steps: int = 0
        self._plan_complete_emitted: bool = False

    def _emit(self, event: dict[str, Any]) -> None:
        """Write a single JSONL event to stdout."""
        write_stdout_jsonl(event)

    def _reset_turn_state(self) -> None:
        self._saw_text_delta = False
        self._assistant_text_snapshot = ""
        self.current_tool = None
        self.tool_histories.clear()
        self._emitted_tool_results.clear()
        self._plan_step_status = {}
        self._plan_marker_buffer = ""
        self._plan_total_steps = 0
        self._plan_complete_emitted = False

    def _update_plan_metadata_from_invocation_state(self, invocation_state: Any) -> None:
        if not isinstance(invocation_state, dict):
            return
        sw = invocation_state.get("swarmee")
        if not isinstance(sw, dict):
            return
        total = sw.get("plan_step_count")
        if isinstance(total, int) and total >= 0:
            self._plan_total_steps = total
            if total == 0:
                self._plan_complete_emitted = False

    def _normalize_plan_step_index(self, *, step: Any, step_index: Any) -> int | None:
        if isinstance(step_index, int):
            return step_index if step_index >= 0 else None
        if isinstance(step_index, str) and step_index.strip().lstrip("-").isdigit():
            parsed = int(step_index.strip())
            return parsed if parsed >= 0 else None
        if isinstance(step, int):
            if step < 0:
                return None
            if step == 0:
                return 0
            return step - 1
        if isinstance(step, str) and step.strip().lstrip("-").isdigit():
            parsed = int(step.strip())
            if parsed < 0:
                return None
            if parsed == 0:
                return 0
            return parsed - 1
        return None

    def _emit_plan_step_update(self, *, step_index: int, status: str, note: str | None = None) -> None:
        if step_index < 0:
            return
        normalized_status = status.strip().lower()
        if normalized_status not in {"in_progress", "completed"}:
            return
        if self._plan_step_status.get(step_index) == normalized_status:
            return
        self._plan_step_status[step_index] = normalized_status
        payload: dict[str, Any] = {
            "event": "plan_step_update",
            "step_index": step_index,
            "status": normalized_status,
        }
        if isinstance(note, str) and note.strip():
            payload["note"] = note.strip()
        self._emit(payload)
        self._emit_plan_complete_if_ready()

    def _emit_plan_complete_if_ready(self) -> None:
        if self._plan_complete_emitted:
            return
        total = self._plan_total_steps
        if total <= 0:
            return
        completed = sum(1 for idx in range(total) if self._plan_step_status.get(idx) == "completed")
        if completed < total:
            return
        self._plan_complete_emitted = True
        self._emit({"event": "plan_complete", "completed_steps": completed, "total_steps": total})

    def _consume_text_for_plan_markers(self, text: str) -> None:
        chunk = str(text or "")
        if not chunk:
            return
        merged = self._plan_marker_buffer + chunk
        if len(merged) > 4096:
            merged = merged[-4096:]
        lines = merged.splitlines(keepends=True)
        next_buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            next_buffer = lines.pop()
        self._plan_marker_buffer = next_buffer
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            match_start = _PLAN_STEP_START_RE.search(line)
            if match_start:
                step_number = int(match_start.group("num"))
                desc = match_start.group("desc").strip()
                if step_number > 0:
                    self._emit_plan_step_update(
                        step_index=step_number - 1,
                        status="in_progress",
                        note=desc or None,
                    )
                continue
            match_done = _PLAN_STEP_DONE_RE.search(line)
            if match_done:
                step_number = int(match_done.group("num"))
                if step_number > 0:
                    self._emit_plan_step_update(step_index=step_number - 1, status="completed")

    def _flush_plan_marker_buffer(self) -> None:
        trailing = self._plan_marker_buffer
        if not trailing:
            return
        self._plan_marker_buffer = ""
        marker_text = trailing.strip()
        if marker_text:
            self._consume_text_for_plan_markers(f"{marker_text}\n")

    def _emit_plan_progress_from_tool(self, *, tool_name: Any, tool_input: Any) -> None:
        if str(tool_name or "").strip().lower() != "plan_progress":
            return
        if not isinstance(tool_input, dict):
            return
        status = str(tool_input.get("status", "")).strip().lower()
        step_index = self._normalize_plan_step_index(
            step=tool_input.get("step"),
            step_index=tool_input.get("step_index"),
        )
        if step_index is None:
            return
        note_value = tool_input.get("note")
        note = str(note_value).strip() if isinstance(note_value, str) else None
        self._emit_plan_step_update(step_index=step_index, status=status, note=note)

    def _now_mono(self) -> float:
        return time.monotonic()

    def _resolve_tool_use_id(self, candidate: Any) -> str | None:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if self.current_tool:
            return self.current_tool
        if len(self.tool_histories) == 1:
            return next(iter(self.tool_histories.keys()))
        return None

    def _ensure_tool_history(
        self,
        tool_use_id: str,
        *,
        tool_name: str | None = None,
        tool_input: Any = None,
        emit_start: bool = False,
    ) -> dict[str, Any]:
        info = self.tool_histories.get(tool_use_id)
        if info is not None:
            if tool_name and not info.get("name"):
                info["name"] = tool_name
            return info

        initial_size = len(str(tool_input)) if tool_input is not None else 0
        info = {
            "name": tool_name or "unknown",
            "start_time": time.time(),
            "input_size": initial_size,
            "pending_progress": "",
            "pending_stream": "stdout",
            "last_progress_emit_mono": -_TUI_TOOL_PROGRESS_EMIT_INTERVAL_S,
            "last_heartbeat_emit_mono": self._now_mono(),
            "output_preview": "",
        }
        self.current_tool = tool_use_id
        self.tool_histories[tool_use_id] = info
        if emit_start:
            self._emit(
                {
                    "event": "tool_start",
                    "tool_use_id": tool_use_id,
                    "tool": tool_name or "unknown",
                    "input": tool_input if isinstance(tool_input, dict) else {},
                }
            )
        return info

    def _append_tool_progress(self, tool_use_id: str, content: str, *, stream: str = "stdout") -> None:
        text = str(content or "")
        if not text:
            return
        normalized_stream = stream if stream in {"stdout", "stderr", "mixed"} else "stdout"
        info = self._ensure_tool_history(tool_use_id, emit_start=False)

        preview = str(info.get("output_preview", "")) + text
        if len(preview) > _TUI_TOOL_OUTPUT_MAX_CHARS:
            preview = preview[-_TUI_TOOL_OUTPUT_MAX_CHARS:]
        info["output_preview"] = preview

        pending = str(info.get("pending_progress", ""))
        pending_stream = str(info.get("pending_stream", "stdout") or "stdout")
        chunk = text
        if pending:
            if pending_stream != normalized_stream:
                pending_stream = "mixed"
                chunk = f"[{normalized_stream}] {text}"
        else:
            pending_stream = normalized_stream
        pending += chunk
        if len(pending) > _TUI_TOOL_OUTPUT_MAX_CHARS:
            pending = pending[-_TUI_TOOL_OUTPUT_MAX_CHARS:]
        info["pending_progress"] = pending
        info["pending_stream"] = pending_stream
        self._emit_tool_progress_if_due(tool_use_id)

    def _emit_tool_progress_if_due(self, tool_use_id: str, *, force: bool = False, heartbeat_only: bool = False) -> bool:
        info = self.tool_histories.get(tool_use_id)
        if info is None:
            return False
        now = self._now_mono()
        last_emit = float(info.get("last_progress_emit_mono", 0.0))
        elapsed = max(0.0, time.time() - float(info.get("start_time", time.time())))
        pending = str(info.get("pending_progress", ""))

        if pending and not heartbeat_only:
            if force or (now - last_emit) >= _TUI_TOOL_PROGRESS_EMIT_INTERVAL_S:
                stream = str(info.get("pending_stream", "stdout") or "stdout")
                self._emit(
                    {
                        "event": "tool_progress",
                        "tool_use_id": tool_use_id,
                        "content": pending,
                        "stream": stream,
                        "elapsed_s": round(elapsed, 2),
                    }
                )
                info["pending_progress"] = ""
                info["pending_stream"] = "stdout"
                info["last_progress_emit_mono"] = now
                info["last_heartbeat_emit_mono"] = now
                return True
            return False

        last_heartbeat = float(info.get("last_heartbeat_emit_mono", 0.0))
        if force:
            return False
        if (now - last_heartbeat) < _TUI_TOOL_HEARTBEAT_INTERVAL_S:
            return False
        if (now - last_emit) < _TUI_TOOL_PROGRESS_EMIT_INTERVAL_S:
            return False

        self._emit(
            {
                "event": "tool_progress",
                "tool_use_id": tool_use_id,
                "elapsed_s": round(elapsed, 2),
            }
        )
        info["last_progress_emit_mono"] = now
        info["last_heartbeat_emit_mono"] = now
        return True

    def _flush_tool_progress(self, tool_use_id: str) -> None:
        self._emit_tool_progress_if_due(tool_use_id, force=True)

    def _emit_tool_heartbeats(self) -> None:
        for tool_use_id in list(self.tool_histories.keys()):
            self._emit_tool_progress_if_due(tool_use_id, heartbeat_only=True)

    def _extract_tool_progress_chunks(
        self,
        *,
        current_tool_use: dict[str, Any],
        result: Any,
        extra_event_fields: dict[str, Any],
    ) -> list[tuple[str, str, str]]:
        chunks: list[tuple[str, str, str]] = []

        def _from_payload(payload: Any, *, default_tool_id: str | None = None) -> None:
            if not isinstance(payload, dict):
                return
            tool_use_id = self._resolve_tool_use_id(
                payload.get("toolUseId") or payload.get("tool_use_id") or payload.get("id") or default_tool_id
            )
            if not tool_use_id:
                return
            stdout = payload.get("stdout")
            stderr = payload.get("stderr")
            if isinstance(stdout, str) and stdout:
                chunks.append((tool_use_id, "stdout", stdout))
            if isinstance(stderr, str) and stderr:
                chunks.append((tool_use_id, "stderr", stderr))
            stream_name = str(payload.get("stream", "stdout")).strip().lower() or "stdout"
            for key in ("content", "output", "chunk"):
                value = payload.get(key)
                if not isinstance(value, str) or not value:
                    continue
                chunks.append((tool_use_id, stream_name, value))

        default_tool_id = self._resolve_tool_use_id(None)
        _from_payload(current_tool_use, default_tool_id=default_tool_id)
        _from_payload(result, default_tool_id=default_tool_id)

        tool_use_id = self._resolve_tool_use_id(
            extra_event_fields.get("toolUseId")
            or extra_event_fields.get("tool_use_id")
            or extra_event_fields.get("tool_id")
            or default_tool_id
        )
        if tool_use_id:
            stdout = extra_event_fields.get("tool_stdout")
            stderr = extra_event_fields.get("tool_stderr")
            if isinstance(stdout, str) and stdout:
                chunks.append((tool_use_id, "stdout", stdout))
            if isinstance(stderr, str) and stderr:
                chunks.append((tool_use_id, "stderr", stderr))
            if isinstance(extra_event_fields.get("stdout"), str):
                value = str(extra_event_fields.get("stdout"))
                if value:
                    chunks.append((tool_use_id, "stdout", value))
            if isinstance(extra_event_fields.get("stderr"), str):
                value = str(extra_event_fields.get("stderr"))
                if value:
                    chunks.append((tool_use_id, "stderr", value))

            raw_progress = extra_event_fields.get("tool_progress")
            if isinstance(raw_progress, dict):
                _from_payload(raw_progress, default_tool_id=tool_use_id)
            raw_output = extra_event_fields.get("tool_output")
            if isinstance(raw_output, dict):
                _from_payload(raw_output, default_tool_id=tool_use_id)
            elif isinstance(raw_output, str) and raw_output:
                stream_name = str(extra_event_fields.get("stream", "stdout")).strip().lower() or "stdout"
                chunks.append((tool_use_id, stream_name, raw_output))

        return chunks

    def _emit_tool_result(self, tool_use_id: str, status: str, *, tool_name: str | None = None) -> None:
        if tool_use_id in self._emitted_tool_results:
            return
        self._flush_tool_progress(tool_use_id)
        info = self.tool_histories.get(tool_use_id)
        duration = round(time.time() - info["start_time"], 2) if info is not None else 0.0
        label = tool_name or (info["name"] if info is not None else "unknown")
        self._emit({
            "event": "tool_result",
            "tool_use_id": tool_use_id,
            "tool": label,
            "status": status,
            "duration_s": duration,
        })
        self._emitted_tool_results.add(tool_use_id)
        if info is not None:
            del self.tool_histories[tool_use_id]
        if self.current_tool == tool_use_id:
            self.current_tool = None

    def _emit_text_fallback_if_needed(self, text: str | None) -> bool:
        if self._saw_text_delta:
            return False
        if not isinstance(text, str):
            return False
        if not text:
            return False
        self._emit({"event": "text_delta", "data": text})
        self._emit({"event": "text_complete"})
        self._saw_text_delta = True
        return True

    def _extract_text_from_result(self, result: Any) -> str | None:
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            chunks: list[str] = []
            for item in result:
                text = self._extract_text_from_result(item)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                return "".join(chunks)
            return None
        if isinstance(result, dict):
            for key in ("text", "data", "delta", "output_text", "outputText", "textDelta"):
                text_value = result.get(key)
                if isinstance(text_value, str) and text_value:
                    return text_value
            message = result.get("message")
            if isinstance(message, dict):
                return self._extract_text_from_result(message)
            if isinstance(message, list):
                return self._extract_text_from_result(message)
            content = result.get("content")
            if content is not None:
                extracted = self._extract_text_from_result(content)
                if isinstance(extracted, str) and extracted:
                    return extracted
            for key in ("chunk", "event", "response"):
                extracted = self._extract_text_from_result(result.get(key))
                if isinstance(extracted, str) and extracted:
                    return extracted
        return None

    def _emit_assistant_message_text(self, text: str | None) -> bool:
        if not isinstance(text, str):
            return False
        if not text:
            return False

        delta = text
        snapshot = self._assistant_text_snapshot
        if snapshot:
            if text.startswith(snapshot):
                delta = text[len(snapshot):]
                self._assistant_text_snapshot = text
            elif snapshot.endswith(text):
                delta = ""
            else:
                self._assistant_text_snapshot = snapshot + text
        else:
            self._assistant_text_snapshot = text

        if not delta:
            return False
        self._emit({"event": "text_delta", "data": delta})
        self._consume_text_for_plan_markers(delta)
        self._saw_text_delta = True
        return True

    def _extract_text_from_assistant_message(self, message: dict[str, Any]) -> str | None:
        if message.get("role") != "assistant":
            return None
        raw_content = message.get("content")
        if not isinstance(raw_content, list):
            return None
        chunks: list[str] = []
        for item in raw_content:
            if isinstance(item, dict) and (item.get("toolUse") or item.get("toolResult")):
                continue
            text = self._extract_text_from_result(item)
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return "".join(chunks)
        return None

    def _extract_text_from_extra_fields(self, extra_fields: dict[str, Any]) -> str | None:
        for key in ("data", "text", "delta", "content_delta", "text_delta", "output_text", "outputText"):
            value = extra_fields.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("message", "event", "chunk", "content", "response", "payload"):
            text = self._extract_text_from_result(extra_fields.get(key))
            if isinstance(text, str) and text:
                return text
        return None

    def callback_handler(
        self,
        reasoningText: str | bool = False,
        data: str = "",
        complete: bool = False,
        force_stop: bool = False,
        message: dict[str, Any] | None = None,
        current_tool_use: dict[str, Any] | None = None,
        init_event_loop: bool = False,
        start_event_loop: bool = False,
        event_loop_throttled_delay: int | None = None,
        console: Any = None,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[Any] | None = None,
        structured_output_prompt: str | None = None,
        result: Any = None,
        **extra_event_fields: Any,
    ) -> None:
        del structured_output_model, structured_output_prompt
        del console, start_event_loop
        message = message or {}
        current_tool_use = current_tool_use or {}

        self._update_plan_metadata_from_invocation_state(invocation_state)

        if init_event_loop:
            self._reset_turn_state()

        if force_stop:
            self._reset_turn_state()
            return

        if self.interrupt_event is not None and self.interrupt_event.is_set():
            self._reset_turn_state()
            return

        if reasoningText:
            self._emit({"event": "thinking", "text": str(reasoningText)})

        emitted_stream_text = False
        if data:
            self._emit({"event": "text_delta", "data": data})
            self._consume_text_for_plan_markers(data)
            self._saw_text_delta = True
            self._assistant_text_snapshot += data
            emitted_stream_text = True

        if not emitted_stream_text and isinstance(message, dict):
            emitted_stream_text = self._emit_assistant_message_text(self._extract_text_from_assistant_message(message))

        if not emitted_stream_text and extra_event_fields:
            self._emit_assistant_message_text(self._extract_text_from_extra_fields(extra_event_fields))

        if complete:
            self._flush_plan_marker_buffer()
            self._emit({"event": "text_complete"})

        if current_tool_use:
            raw_tool_id = current_tool_use.get("toolUseId")
            tool_id = raw_tool_id if isinstance(raw_tool_id, str) else None
            tool_name = current_tool_use.get("name")
            tool_input = current_tool_use.get("input")

            if tool_id is not None:
                self._ensure_tool_history(
                    tool_id,
                    tool_name=str(tool_name) if tool_name is not None else None,
                    tool_input=tool_input,
                    emit_start=(tool_id not in self.tool_histories),
                )

                if tool_input is not None and tool_id in self.tool_histories:
                    current_size = len(str(tool_input))
                    if current_size > self.tool_histories[tool_id]["input_size"]:
                        info = self.tool_histories[tool_id]
                        info["input_size"] = current_size
                        now = self._now_mono()
                        last_emit = float(info.get("last_progress_emit_mono", 0.0))
                        if (now - last_emit) >= _TUI_TOOL_PROGRESS_EMIT_INTERVAL_S:
                            self._emit(
                                {
                                    "event": "tool_progress",
                                    "tool_use_id": tool_id,
                                    "chars": current_size,
                                }
                            )
                            info["last_progress_emit_mono"] = now
                self._emit_plan_progress_from_tool(tool_name=tool_name, tool_input=tool_input)

        for tool_use_id, stream, content in self._extract_tool_progress_chunks(
            current_tool_use=current_tool_use,
            result=result,
            extra_event_fields=extra_event_fields,
        ):
            self._append_tool_progress(tool_use_id, content, stream=stream)

        if isinstance(message, dict):
            if message.get("role") == "assistant":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_use = content.get("toolUse")
                        if tool_use:
                            tid = tool_use.get("toolUseId")
                            tool_name = tool_use.get("name")
                            tool_input = tool_use.get("input")
                            self._emit_plan_progress_from_tool(tool_name=tool_name, tool_input=tool_input)
                            if isinstance(tid, str) and isinstance(tool_input, dict):
                                self._emit({
                                    "event": "tool_input",
                                    "tool_use_id": tid,
                                    "input": tool_input,
                                })

            elif message.get("role") == "user":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_result = content.get("toolResult")
                        if tool_result:
                            tid = tool_result.get("toolUseId")
                            status = tool_result.get("status", "unknown")
                            if isinstance(tid, str):
                                tool_label = tool_result.get("name")
                                self._emit_tool_result(tid, str(status), tool_name=str(tool_label) if tool_label else None)

        if event_loop_throttled_delay:
            self._emit({
                "event": "warning",
                "text": f"Throttled! Waiting {event_loop_throttled_delay}s before retrying...",
            })

        emitted_text_fallback = False
        if isinstance(result, dict):
            tid = result.get("toolUseId")
            status = result.get("status")
            if isinstance(tid, str) and status is not None:
                tool_label = result.get("name")
                self._emit_tool_result(tid, str(status), tool_name=str(tool_label) if tool_label else None)
            else:
                emitted_text_fallback = self._emit_text_fallback_if_needed(self._extract_text_from_result(result))
        elif result is not None:
            emitted_text_fallback = self._emit_text_fallback_if_needed(self._extract_text_from_result(result))
            if isinstance(result, str):
                self._consume_text_for_plan_markers(result)

        self._emit_tool_heartbeats()

        if emitted_text_fallback:
            self._flush_plan_marker_buffer()
            self._reset_turn_state()


# ---------------------------------------------------------------------------
# Module-level handler selection: TuiCallbackHandler when SWARMEE_TUI_EVENTS
# is set (subprocess launched by the TUI), otherwise the standard CLI handler.
# ---------------------------------------------------------------------------

if truthy_env("SWARMEE_TUI_EVENTS", False):
    callback_handler_instance: CallbackHandler | TuiCallbackHandler = TuiCallbackHandler()
else:
    callback_handler_instance = CallbackHandler()

callback_handler = callback_handler_instance.callback_handler


def set_interrupt_event(event: Event | None) -> None:
    callback_handler_instance.interrupt_event = event


def _cleanup_spinners_at_exit() -> None:
    if isinstance(callback_handler_instance, CallbackHandler):
        with contextlib.suppress(Exception):
            callback_handler_instance.callback_handler(force_stop=True)
        for spinner in list(_ACTIVE_TOOL_SPINNERS):
            with contextlib.suppress(Exception):
                spinner.stop()


atexit.register(_cleanup_spinners_at_exit)
