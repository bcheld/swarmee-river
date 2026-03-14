import atexit
import contextlib
import logging
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
_BEDROCK_STREAM_MARKER_KEYS = {
    "messageStart",
    "messageStop",
    "contentBlockStart",
    "contentBlockDelta",
    "contentBlockStop",
    "metadata",
}
_NON_TEXT_EVENT_TOKENS = {
    "after_invocation",
    "after_model_call",
    "after_tool_call",
    "before_model_call",
    "complete",
    "delta",
    "llm_start",
    "message_complete",
    "message_delta",
    "model_start",
    "output_text_complete",
    "output_text_delta",
    "text_complete",
    "text_delta",
    "thinking",
}
_PLAN_PROTOCOL_MAX_TEXT_CHARS = 160

_LOGGER = logging.getLogger(__name__)


def _is_plan_turn(invocation_state: dict[str, Any] | None) -> bool:
    if not isinstance(invocation_state, dict):
        return False
    sw = invocation_state.get("swarmee")
    if not isinstance(sw, dict):
        return False
    return str(sw.get("mode", "")).strip().lower() == "plan"


def _normalize_protocol_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _message_content_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    raw_content = message.get("content")
    if not isinstance(raw_content, list):
        return []
    return [item for item in raw_content if isinstance(item, dict)]


def _ensure_invoke_diag(invocation_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(invocation_state, dict):
        return {}
    sw = invocation_state.get("swarmee")
    if not isinstance(sw, dict):
        sw = {}
        invocation_state["swarmee"] = sw
    diag = sw.get("invoke_diag")
    if not isinstance(diag, dict):
        diag = {}
        sw["invoke_diag"] = diag
    now = time.monotonic()
    diag.setdefault("invoke_start_mono", now)
    diag.setdefault("last_callback_mono", now)
    return diag


def _mark_invoke_diag(
    invocation_state: dict[str, Any] | None,
    *,
    stage: str | None = None,
    saw_callback: bool = False,
    saw_progress: bool = False,
    saw_text_delta: bool = False,
    saw_complete: bool = False,
    saw_tool_activity: bool = False,
) -> None:
    diag = _ensure_invoke_diag(invocation_state)
    if not diag:
        return
    now = time.monotonic()
    if saw_callback:
        diag["last_callback_mono"] = now
        callback_count = int(diag.get("callback_count", 0) or 0)
        diag["callback_count"] = callback_count + 1
    if saw_progress:
        diag["last_progress_mono"] = now
        diag.setdefault("first_progress_mono", now)
        progress_count = int(diag.get("progress_count", 0) or 0)
        diag["progress_count"] = progress_count + 1
    if stage:
        normalized_stage = str(stage).strip()
        if normalized_stage:
            diag["stage"] = normalized_stage
            diag["stage_mono"] = now
    if saw_text_delta and "first_text_delta_mono" not in diag:
        diag["first_text_delta_mono"] = now
    if saw_complete:
        diag["complete_mono"] = now
    if saw_tool_activity:
        diag["saw_tool_activity"] = True


def format_message(message: str, color: str | None = None, max_length: int = 50) -> str:
    """Format message with color and length control."""
    if len(message) > max_length:
        message = message[:max_length] + "..."
    return f"{color if color else ''}{message}{Style.RESET_ALL}"


class ToolSpinner:
    def __init__(self, text: str = "", color: str | int = TOOL_COLORS["running"]) -> None:
        # Spinners are disabled when running in TUI-events mode (JSONL daemon).
        # This is internal behavior; end-user env toggles are intentionally not supported.
        self.enabled = not truthy_env("SWARMEE_TUI_EVENTS", False)
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
        self._assistant_stream_buffer: str = ""
        self._plan_protocol_text_buffer: str = ""

    def _reset_text_turn_state(self) -> None:
        self._assistant_stream_buffer = ""
        self._plan_protocol_text_buffer = ""

    def _record_plan_protocol_text(self, text: str | None) -> None:
        normalized = _normalize_protocol_text(text)
        if not normalized:
            return
        candidate = f"{self._plan_protocol_text_buffer} {normalized}".strip()
        self._plan_protocol_text_buffer = candidate
        if len(candidate) > _PLAN_PROTOCOL_MAX_TEXT_CHARS:
            raise RuntimeError(
                "Plan generation drifted into assistant text before WorkPlan. "
                "Stop solving the task and return a WorkPlan directly."
            )

    @staticmethod
    def _unprinted_suffix(full_text: str, printed_text: str) -> str:
        if not printed_text:
            return full_text
        if full_text.startswith(printed_text):
            return full_text[len(printed_text) :]
        max_overlap = min(len(full_text), len(printed_text))
        for size in range(max_overlap, 0, -1):
            if printed_text.endswith(full_text[:size]):
                return full_text[size:]
        return full_text

    def _emit_terminal_text(self, text: str, *, complete: bool) -> None:
        if not text:
            if complete:
                _safe_print("")
                self._reset_text_turn_state()
            return
        if complete:
            suffix = self._unprinted_suffix(text, self._assistant_stream_buffer)
            if self._assistant_stream_buffer and suffix != text:
                _LOGGER.debug("Suppressing duplicate terminal completion snapshot after streamed deltas.")
            if suffix:
                _safe_print(f"{Fore.WHITE}{suffix}{Style.RESET_ALL}")
            else:
                _safe_print("")
            self._reset_text_turn_state()
            return

        self._assistant_stream_buffer += text
        _safe_print(f"{Fore.WHITE}{text}{Style.RESET_ALL}", end="")

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
        del structured_output_model
        del structured_output_prompt
        del result
        message = message if isinstance(message, dict) else {}
        current_tool_use = current_tool_use if isinstance(current_tool_use, dict) else {}
        warning_text = extra_event_fields.pop("warning_text", None)
        _mark_invoke_diag(invocation_state, saw_callback=True)

        # Cleanup calls (e.g., from top-level exception handlers) should never re-raise an interrupt.
        if force_stop:
            if self.thinking_spinner:
                self.thinking_spinner.stop()
            if self.current_spinner:
                self.current_spinner.stop()
            self._reset_text_turn_state()
            return

        if self.interrupt_event is not None and self.interrupt_event.is_set():
            # Don't raise from inside the callback path. Interrupt cancellation is handled by the caller.
            # Raising here can cause noisy generator shutdown errors in upstream libraries.
            if self.thinking_spinner:
                self.thinking_spinner.stop()
            if self.current_spinner:
                self.current_spinner.stop()
            self._reset_text_turn_state()
            return

        try:
            # Concurrent thinking spinners are usual, which leads to:
            # "Only one live display may be active at once" error thrown,
            # This try except block ignore overlap of thinking spinners.
            if self.thinking_spinner and (data or current_tool_use):
                self.thinking_spinner.stop()

            if init_event_loop and not truthy_env("SWARMEE_TUI_EVENTS", False):
                self._reset_text_turn_state()
                self.thinking_spinner = Status(
                    "[blue] retrieving memories...[/blue]",
                    spinner="dots",
                    console=console,
                )
                self.thinking_spinner.start()
                _mark_invoke_diag(invocation_state, stage="init_event_loop")

            suppress_reasoning_ui = False
            if isinstance(invocation_state, dict):
                sw = invocation_state.get("swarmee")
                if isinstance(sw, dict):
                    suppress_reasoning_ui = bool(sw.get("suppress_reasoning_ui"))

            if reasoningText and not suppress_reasoning_ui:
                _safe_print(reasoningText, end="")
            if reasoningText:
                _mark_invoke_diag(invocation_state, stage="reasoning", saw_progress=True)

            if start_event_loop and self.thinking_spinner is not None and not truthy_env("SWARMEE_TUI_EVENTS", False):
                self.thinking_spinner.update("[blue] thinking...[/blue]")
                _mark_invoke_diag(invocation_state, stage="start_event_loop")
        except BaseException:
            pass

        if event_loop_throttled_delay and console:
            if self.current_spinner:
                self.current_spinner.stop()
            console.print(
                f"[red]Throttled! Waiting [bold]{event_loop_throttled_delay} seconds[/bold] before retrying...[/red]"
            )

        if isinstance(warning_text, str) and warning_text.strip():
            _safe_print(f"[warn] {warning_text.strip()}")

        # Handle regular output
        plan_turn = _is_plan_turn(invocation_state)
        if plan_turn and data:
            self._record_plan_protocol_text(data)
            data = ""
        if data:
            _mark_invoke_diag(
                invocation_state,
                stage="first_text_delta",
                saw_progress=True,
                saw_text_delta=True,
            )
            self._emit_terminal_text(data, complete=complete)
        elif complete:
            self._reset_text_turn_state()

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
                    _mark_invoke_diag(
                        invocation_state,
                        stage="tool_start",
                        saw_progress=True,
                        saw_tool_activity=True,
                    )

                # Update tool progress
                if tool_id in self.tool_histories:
                    current_size = len(tool_input)
                    if current_size > self.tool_histories[tool_id]["input_size"]:
                        self.tool_histories[tool_id]["input_size"] = current_size
                        _mark_invoke_diag(
                            invocation_state,
                            stage="tool_progress",
                            saw_progress=True,
                            saw_tool_activity=True,
                        )
                        if self.current_spinner:
                            self.current_spinner.update(f"🛠️  {tool_name}: {current_size} chars")

        # Process messages
        if message.get("role") == "assistant":
            for content in _message_content_blocks(message):
                tool_use = content.get("toolUse")
                if tool_use:
                    tool_name = tool_use.get("name")
                    _mark_invoke_diag(
                        invocation_state,
                        stage="tool_start",
                        saw_progress=True,
                        saw_tool_activity=True,
                    )
                    if self.current_spinner:
                        self.current_spinner.info(f"🔧 Starting {tool_name}...")

        elif message.get("role") == "user":
            for content in _message_content_blocks(message):
                tool_result = content.get("toolResult")
                if tool_result:
                    tool_id = tool_result.get("toolUseId")
                    status = tool_result.get("status")
                    _mark_invoke_diag(
                        invocation_state,
                        stage="tool_result",
                        saw_progress=True,
                        saw_tool_activity=True,
                    )

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

        if complete:
            _mark_invoke_diag(invocation_state, stage="complete", saw_complete=True)


class TuiCallbackHandler:
    """Callback handler that emits structured JSONL events to stdout for TUI consumption."""

    def __init__(self) -> None:
        self.current_tool: str | None = None
        self.tool_histories: dict[str, dict[str, Any]] = {}
        self.interrupt_event: Event | None = None
        self._saw_text_delta: bool = False
        self._assistant_text_snapshot: str = ""
        self._assistant_text_sources_turn: set[str] = set()
        self._text_complete_emitted_turn: bool = False
        self._emitted_tool_results: set[str] = set()
        self._plan_step_status: dict[int, str] = {}
        self._plan_marker_buffer: str = ""
        self._plan_total_steps: int = 0
        self._plan_complete_emitted: bool = False
        self._plan_run_id: str | None = None
        self._suppress_reasoning_ui: bool = False
        self._llm_start_emitted: bool = False
        self._saw_bedrock_stream_chunk: bool = False
        self._bedrock_stream_markers_seen: set[str] = set()
        self._bedrock_missing_text_warning_emitted: bool = False
        self._emitted_reasoning_texts_turn: set[str] = set()
        self._plan_protocol_text_buffer: str = ""

    def _emit(self, event: dict[str, Any]) -> None:
        """Write a single JSONL event to stdout."""
        write_stdout_jsonl(event)

    def _reset_turn_state(self) -> None:
        self._saw_text_delta = False
        self._assistant_text_snapshot = ""
        self._assistant_text_sources_turn.clear()
        self._text_complete_emitted_turn = False
        self.current_tool = None
        self.tool_histories.clear()
        self._emitted_tool_results.clear()
        self._plan_step_status = {}
        self._plan_marker_buffer = ""
        self._plan_total_steps = 0
        self._plan_complete_emitted = False
        self._plan_run_id = None
        self._suppress_reasoning_ui = False
        self._llm_start_emitted = False
        self._emitted_reasoning_texts_turn.clear()
        self._plan_protocol_text_buffer = ""
        self._reset_bedrock_stream_state()

    def _reset_bedrock_stream_state(self) -> None:
        self._saw_bedrock_stream_chunk = False
        self._bedrock_stream_markers_seen.clear()
        self._bedrock_missing_text_warning_emitted = False

    def _emit_llm_start_if_needed(self) -> None:
        if self._llm_start_emitted:
            return
        self._emit({"event": "llm_start"})
        self._llm_start_emitted = True

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
        plan_run_id = str(sw.get("plan_run_id", "") or "").strip()
        self._plan_run_id = plan_run_id or None
        self._suppress_reasoning_ui = bool(sw.get("suppress_reasoning_ui"))

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
        if self._plan_run_id:
            payload["plan_run_id"] = self._plan_run_id
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
        payload: dict[str, Any] = {"event": "plan_complete", "completed_steps": completed, "total_steps": total}
        if self._plan_run_id:
            payload["plan_run_id"] = self._plan_run_id
        self._emit(payload)

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
        emit_start = tool_use_id not in self.tool_histories
        info = self._ensure_tool_history(tool_use_id, emit_start=emit_start)

        preview = str(info.get("output_preview", "")) + text
        if len(preview) > _TUI_TOOL_OUTPUT_MAX_CHARS:
            preview = preview[-_TUI_TOOL_OUTPUT_MAX_CHARS:]
        info["output_preview"] = preview

        elapsed = max(0.0, time.time() - float(info.get("start_time", time.time())))
        self._emit(
            {
                "event": "tool_progress",
                "tool_use_id": tool_use_id,
                "content": text,
                "stream": normalized_stream,
                "elapsed_s": round(elapsed, 2),
            }
        )
        now = self._now_mono()
        info["last_progress_emit_mono"] = now
        info["last_heartbeat_emit_mono"] = now

    def _emit_tool_progress_if_due(
        self,
        tool_use_id: str,
        *,
        force: bool = False,
        heartbeat_only: bool = False,
    ) -> bool:
        info = self.tool_histories.get(tool_use_id)
        if info is None:
            return False
        del heartbeat_only

        last_heartbeat = float(info.get("last_heartbeat_emit_mono", 0.0))
        if force:
            return False
        now = self._now_mono()
        if (now - last_heartbeat) < _TUI_TOOL_HEARTBEAT_INTERVAL_S:
            return False
        elapsed = max(0.0, time.time() - float(info.get("start_time", time.time())))

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
        del tool_use_id

    def _emit_tool_heartbeats(self) -> bool:
        emitted = False
        for tool_use_id in list(self.tool_histories.keys()):
            emitted = self._emit_tool_progress_if_due(tool_use_id, heartbeat_only=True) or emitted
        return emitted

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
        for key in ("tool_progress", "tool_output", "on_tool_progress"):
            raw_payload = extra_event_fields.get(key)
            if isinstance(raw_payload, dict):
                _from_payload(raw_payload, default_tool_id=tool_use_id or default_tool_id)

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
            raw_output = extra_event_fields.get("tool_output")
            if isinstance(raw_output, str) and raw_output:
                stream_name = str(extra_event_fields.get("stream", "stdout")).strip().lower() or "stdout"
                chunks.append((tool_use_id, stream_name, raw_output))

        return chunks

    def _extract_tool_start_from_extra_fields(self, extra_fields: dict[str, Any]) -> tuple[str, str | None, Any] | None:
        for key in ("tool_start", "on_tool_start"):
            payload = extra_fields.get(key)
            if not isinstance(payload, dict):
                continue
            tool_use_id = self._resolve_tool_use_id(
                payload.get("toolUseId") or payload.get("tool_use_id") or payload.get("id")
            )
            if not tool_use_id:
                continue
            tool_name_raw = payload.get("tool") or payload.get("tool_name") or payload.get("name")
            tool_name = str(tool_name_raw) if tool_name_raw is not None else None
            return tool_use_id, tool_name, payload.get("input")
        return None

    def _extract_tool_result_from_extra_fields(
        self, extra_fields: dict[str, Any]
    ) -> tuple[str, str, str | None] | None:
        for key in ("tool_end", "on_tool_end", "tool_result"):
            payload = extra_fields.get(key)
            if not isinstance(payload, dict):
                continue
            tool_use_id = self._resolve_tool_use_id(
                payload.get("toolUseId") or payload.get("tool_use_id") or payload.get("id")
            )
            if not tool_use_id:
                continue
            status_value = payload.get("status") or payload.get("tool_status") or payload.get("result")
            if status_value is None:
                continue
            tool_name_raw = payload.get("tool") or payload.get("tool_name") or payload.get("name")
            tool_name = str(tool_name_raw) if tool_name_raw is not None else None
            return tool_use_id, str(status_value), tool_name
        return None

    def _emit_tool_result(self, tool_use_id: str, status: str, *, tool_name: str | None = None) -> None:
        if tool_use_id in self._emitted_tool_results:
            return
        self._flush_tool_progress(tool_use_id)
        info = self.tool_histories.get(tool_use_id)
        duration = round(time.time() - info["start_time"], 2) if info is not None else 0.0
        label = tool_name or (info["name"] if info is not None else "unknown")
        self._emit(
            {
                "event": "tool_result",
                "tool_use_id": tool_use_id,
                "tool": label,
                "status": status,
                "duration_s": duration,
            }
        )
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
        self._assistant_text_sources_turn.add("result")
        self._assistant_text_snapshot = text
        self._emit({"event": "text_delta", "data": text})
        if not self._text_complete_emitted_turn:
            self._emit({"event": "text_complete"})
            self._text_complete_emitted_turn = True
        self._saw_text_delta = True
        return True

    @staticmethod
    def _normalize_reasoning_chunk(text: str | None) -> str:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines: list[str] = []
        previous_blank = False
        for line in raw.split("\n"):
            cleaned = re.sub(r"[ \t]+", " ", line).strip()
            if not cleaned:
                if previous_blank:
                    continue
                lines.append("")
                previous_blank = True
                continue
            lines.append(cleaned)
            previous_blank = False
        return "\n".join(lines).strip()

    def _emit_reasoning_text(self, text: str | None) -> None:
        chunk = self._normalize_reasoning_chunk(text)
        if not chunk or chunk in self._emitted_reasoning_texts_turn:
            return
        self._emitted_reasoning_texts_turn.add(chunk)
        if self._suppress_reasoning_ui:
            return
        self._emit({"event": "thinking", "text": chunk})

    def _record_plan_protocol_text(self, text: str | None) -> None:
        normalized = _normalize_protocol_text(text)
        if not normalized:
            return
        candidate = f"{self._plan_protocol_text_buffer} {normalized}".strip()
        self._plan_protocol_text_buffer = candidate
        if len(candidate) > _PLAN_PROTOCOL_MAX_TEXT_CHARS:
            raise RuntimeError(
                "Plan generation drifted into assistant text before WorkPlan. "
                "Stop solving the task and return a WorkPlan directly."
            )

    def _extract_text_from_result(self, result: Any, *, exclude_reasoning_wrappers: bool = False) -> str | None:
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            chunks: list[str] = []
            for item in result:
                text = self._extract_text_from_result(item, exclude_reasoning_wrappers=exclude_reasoning_wrappers)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                return "".join(chunks)
            return None
        if isinstance(result, dict):
            bedrock_text = self._extract_text_from_bedrock_payload(result)
            if isinstance(bedrock_text, str) and bedrock_text:
                return bedrock_text

            for key in ("text", "data", "delta", "output_text", "outputText", "textDelta"):
                text_value = result.get(key)
                if isinstance(text_value, str) and text_value:
                    return text_value
            for key in ("delta", "text_delta", "content_delta"):
                nested = result.get(key)
                if isinstance(nested, dict):
                    nested_text = self._extract_text_from_delta_payload(nested)
                    if isinstance(nested_text, str) and nested_text:
                        return nested_text
            message = result.get("message")
            if isinstance(message, dict):
                return self._extract_text_from_result(message, exclude_reasoning_wrappers=exclude_reasoning_wrappers)
            if isinstance(message, list):
                return self._extract_text_from_result(message, exclude_reasoning_wrappers=exclude_reasoning_wrappers)
            content = result.get("content")
            if content is not None:
                extracted = self._extract_text_from_result(
                    content,
                    exclude_reasoning_wrappers=exclude_reasoning_wrappers,
                )
                if isinstance(extracted, str) and extracted:
                    return extracted
            for key in ("chunk", "event", "response"):
                nested_value = result.get(key)
                if key in {"event", "type", "kind"} and isinstance(nested_value, str):
                    if self._is_non_text_event_token(nested_value):
                        continue
                extracted = self._extract_text_from_result(
                    nested_value,
                    exclude_reasoning_wrappers=exclude_reasoning_wrappers,
                )
                if isinstance(extracted, str) and extracted:
                    return extracted
        return None

    @staticmethod
    def _extract_text_from_delta_payload(delta_payload: dict[str, Any]) -> str | None:
        for key in ("text", "data", "output_text", "outputText", "textDelta"):
            value = delta_payload.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _extract_text_from_bedrock_payload(self, payload: Any) -> str | None:
        if isinstance(payload, list):
            for item in payload:
                extracted = self._extract_text_from_bedrock_payload(item)
                if isinstance(extracted, str) and extracted:
                    return extracted
            return None
        if not isinstance(payload, dict):
            return None

        for key in ("contentBlockDelta", "content_block_delta"):
            block = payload.get(key)
            if not isinstance(block, dict):
                continue
            delta_value = block.get("delta")
            if isinstance(delta_value, dict):
                extracted = self._extract_text_from_delta_payload(delta_value)
                if isinstance(extracted, str) and extracted:
                    return extracted
            if isinstance(delta_value, str) and delta_value:
                return delta_value
            extracted = self._extract_text_from_delta_payload(block)
            if isinstance(extracted, str) and extracted:
                return extracted

        for key in ("delta", "text_delta", "content_delta"):
            delta_payload = payload.get(key)
            if isinstance(delta_payload, dict):
                extracted = self._extract_text_from_delta_payload(delta_payload)
                if isinstance(extracted, str) and extracted:
                    return extracted
        return None

    def _record_bedrock_stream_markers(self, payload: Any, *, depth: int = 5) -> bool:
        if depth < 0:
            return False
        found = False
        if isinstance(payload, dict):
            for key in payload.keys():
                if key in _BEDROCK_STREAM_MARKER_KEYS:
                    self._bedrock_stream_markers_seen.add(key)
                    found = True
            for value in payload.values():
                if self._record_bedrock_stream_markers(value, depth=depth - 1):
                    found = True
            return found
        if isinstance(payload, list):
            for item in payload:
                if self._record_bedrock_stream_markers(item, depth=depth - 1):
                    found = True
        return found

    def _extract_text_from_bedrock_extra_fields(self, extra_fields: dict[str, Any]) -> str | None:
        candidates: list[Any] = []
        for key in ("payload", "raw_event", "response"):
            if key in extra_fields:
                candidates.append(extra_fields.get(key))

        for payload in candidates:
            extracted = self._extract_text_from_bedrock_payload(payload)
            if isinstance(extracted, str) and extracted:
                return extracted
        return None

    @staticmethod
    def _is_non_text_event_token(value: str) -> bool:
        token = value.strip().lower()
        return token in _NON_TEXT_EVENT_TOKENS

    def _emit_assistant_text_delta(self, text: str | None, *, source: str) -> bool:
        if not isinstance(text, str):
            return False
        if not text:
            return False
        self._assistant_text_sources_turn.add(source)
        self._assistant_text_snapshot += text
        self._emit({"event": "text_delta", "data": text})
        self._consume_text_for_plan_markers(text)
        self._saw_text_delta = True
        return True

    def _emit_assistant_message_text(self, text: str | None, *, source: str) -> bool:
        if not isinstance(text, str):
            return False
        if not text:
            return False
        self._assistant_text_sources_turn.add(source)

        delta = self._merge_assistant_text_snapshot(text)

        if not delta:
            return False
        self._emit({"event": "text_delta", "data": delta})
        self._consume_text_for_plan_markers(delta)
        self._saw_text_delta = True
        return True

    @staticmethod
    def _longest_suffix_prefix_overlap(left: str, right: str) -> int:
        max_len = min(len(left), len(right))
        for size in range(max_len, 0, -1):
            if left.endswith(right[:size]):
                return size
        return 0

    def _merge_assistant_text_snapshot(self, incoming: str) -> str:
        text = str(incoming or "")
        if not text:
            return ""
        snapshot = self._assistant_text_snapshot
        if not snapshot:
            self._assistant_text_snapshot = text
            return text
        if text == snapshot:
            return ""
        if text.startswith(snapshot):
            delta = text[len(snapshot) :]
            self._assistant_text_snapshot = text
            return delta
        if snapshot.startswith(text):
            return ""

        overlap = self._longest_suffix_prefix_overlap(snapshot, text)
        if overlap > 0:
            delta = text[overlap:]
            self._assistant_text_snapshot = snapshot + delta
            return delta

        self._assistant_text_snapshot = snapshot + text
        return text

    def _extract_text_from_assistant_message(
        self,
        message: dict[str, Any],
        *,
        exclude_reasoning_wrappers: bool = False,
    ) -> str | None:
        if message.get("role") != "assistant":
            return None
        raw_content = message.get("content")
        if not isinstance(raw_content, list):
            return None
        chunks: list[str] = []
        for item in raw_content:
            if isinstance(item, dict) and (item.get("toolUse") or item.get("toolResult")):
                continue
            text = self._extract_text_from_result(item, exclude_reasoning_wrappers=exclude_reasoning_wrappers)
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return "".join(chunks)
        return None

    def _extract_text_from_extra_fields(self, extra_fields: dict[str, Any]) -> str | None:
        for key in (
            "data",
            "text",
            "delta",
            "content_delta",
            "text_delta",
            "output_text",
            "outputText",
            "token",
            "new_token",
            "llm_token",
        ):
            value = extra_fields.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("delta", "text_delta", "content_delta"):
            nested = extra_fields.get(key)
            if isinstance(nested, dict):
                nested_text = self._extract_text_from_delta_payload(nested)
                if isinstance(nested_text, str) and nested_text:
                    return nested_text
        return None

    def _extract_transport_text_from_extra_fields(
        self,
        *,
        extra_fields: dict[str, Any],
        invocation_state: Any,
    ) -> str | None:
        provider, _tier, _model_id = self._invocation_model_tokens(invocation_state)
        if provider != "bedrock":
            return None
        return self._extract_text_from_bedrock_extra_fields(extra_fields)

    def _extract_reasoning_text_from_value(
        self,
        payload: Any,
        *,
        depth: int = 6,
        in_reasoning: bool = False,
    ) -> list[str]:
        if depth < 0 or payload is None:
            return []
        if isinstance(payload, str):
            text = payload.strip()
            return [text] if in_reasoning and text else []
        if isinstance(payload, list):
            chunks: list[str] = []
            for item in payload:
                chunks.extend(self._extract_reasoning_text_from_value(item, depth=depth - 1, in_reasoning=in_reasoning))
            return chunks
        if not isinstance(payload, dict):
            return []

        chunks: list[str] = []
        reasoning_keys = {
            "reasoning",
            "reasoningtext",
            "reasoning_text",
            "reasoningcontent",
            "reasoning_content",
            "thinking",
            "thinking_content",
        }
        wrapper_keys = {
            "message",
            "event",
            "chunk",
            "content",
            "response",
            "payload",
            "raw_event",
            "model_extra",
            "extra",
            "delta",
            "result",
        }

        for key, value in payload.items():
            normalized = str(key).strip().lower()
            if normalized in reasoning_keys:
                chunks.extend(self._extract_reasoning_text_from_value(value, depth=depth - 1, in_reasoning=True))
                continue
            if in_reasoning and normalized in {"text", "content", "summary", "value"} and isinstance(value, str):
                text = value.strip()
                if text:
                    chunks.append(text)
                continue
            if normalized in wrapper_keys:
                chunks.extend(
                    self._extract_reasoning_text_from_value(
                        value,
                        depth=depth - 1,
                        in_reasoning=in_reasoning,
                    )
                )
                continue
            if in_reasoning and isinstance(value, (dict, list)):
                chunks.extend(self._extract_reasoning_text_from_value(value, depth=depth - 1, in_reasoning=True))
        return chunks

    def _extract_reasoning_text_candidates(
        self,
        *,
        message: dict[str, Any],
        result: Any,
        extra_event_fields: dict[str, Any],
    ) -> list[str]:
        structured_chunks = self._extract_structured_reasoning_text_candidates(
            message=message,
            result=result,
            extra_event_fields=extra_event_fields,
        )
        if structured_chunks:
            return structured_chunks
        chunks: list[str] = []
        chunks.extend(self._extract_reasoning_text_from_value(message))
        chunks.extend(self._extract_reasoning_text_from_value(result))
        chunks.extend(self._extract_reasoning_text_from_value(extra_event_fields))
        deduped: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            text = self._normalize_reasoning_chunk(chunk)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    def _extract_structured_reasoning_text_candidates(
        self,
        *,
        message: dict[str, Any],
        result: Any,
        extra_event_fields: dict[str, Any],
    ) -> list[str]:
        def _collect(value: Any, out: list[str]) -> None:
            if isinstance(value, list):
                for item in value:
                    _collect(item, out)
                return
            if not isinstance(value, dict):
                return
            reasoning_content = value.get("reasoningContent")
            if isinstance(reasoning_content, dict):
                reasoning_text = reasoning_content.get("reasoningText")
                if isinstance(reasoning_text, dict):
                    normalized = self._normalize_reasoning_chunk(reasoning_text.get("text"))
                    if normalized:
                        out.append(normalized)
                normalized = self._normalize_reasoning_chunk(reasoning_content.get("text"))
                if normalized:
                    out.append(normalized)
            reasoning_text = value.get("reasoningText")
            if isinstance(reasoning_text, dict):
                normalized = self._normalize_reasoning_chunk(reasoning_text.get("text"))
                if normalized:
                    out.append(normalized)
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    _collect(nested, out)

        chunks: list[str] = []
        for source in (message, result, extra_event_fields):
            _collect(source, chunks)
        deduped: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            if not chunk or chunk in seen:
                continue
            seen.add(chunk)
            deduped.append(chunk)
        return deduped

    @staticmethod
    def _invocation_model_tokens(invocation_state: Any) -> tuple[str, str, str]:
        if not isinstance(invocation_state, dict):
            return "", "", ""
        sw = invocation_state.get("swarmee")
        if not isinstance(sw, dict):
            return "", "", ""
        provider = str(sw.get("provider", "") or "").strip().lower()
        tier = str(sw.get("tier", "") or "").strip().lower()
        model_id = str(sw.get("model_id", "") or "").strip().lower()
        return provider, tier, model_id

    @classmethod
    def _reasoning_expected_for_turn(cls, invocation_state: Any) -> bool:
        provider, _tier, _model_id = cls._invocation_model_tokens(invocation_state)
        if not isinstance(invocation_state, dict):
            return False
        sw = invocation_state.get("swarmee")
        if not isinstance(sw, dict):
            return False
        reasoning_effort = str(sw.get("reasoning_effort", "") or "").strip().lower()
        reasoning_mode = str(sw.get("reasoning_mode", "") or "").strip().lower()
        transport = str(sw.get("transport", "") or "").strip().lower()
        if provider == "bedrock":
            return reasoning_mode in {"extended", "adaptive"} and reasoning_effort in {"low", "medium", "high"}
        if provider == "openai" and transport == "responses":
            return reasoning_effort in {"low", "medium", "high"}
        if reasoning_effort in {"low", "medium", "high"}:
            return True
        return False

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
        del console
        message = message if isinstance(message, dict) else {}
        current_tool_use = current_tool_use if isinstance(current_tool_use, dict) else {}
        warning_text = extra_event_fields.pop("warning_text", None)
        warning_metadata = extra_event_fields.pop("warning_metadata", None)

        _mark_invoke_diag(invocation_state, saw_callback=True)

        self._update_plan_metadata_from_invocation_state(invocation_state)

        if init_event_loop:
            self._reset_turn_state()
            self._emit_llm_start_if_needed()
            _mark_invoke_diag(invocation_state, stage="init_event_loop")

        if start_event_loop:
            self._emit_llm_start_if_needed()
            _mark_invoke_diag(invocation_state, stage="start_event_loop")

        if force_stop:
            self._reset_turn_state()
            return

        if self.interrupt_event is not None and self.interrupt_event.is_set():
            self._reset_turn_state()
            return

        if isinstance(warning_text, str) and warning_text.strip():
            payload: dict[str, Any] = {"event": "warning", "text": warning_text.strip()}
            if isinstance(warning_metadata, dict):
                for key, value in warning_metadata.items():
                    token = str(key).strip()
                    if token and token not in {"event", "text"}:
                        payload[token] = value
            self._emit(payload)

        should_emit_complete = bool(complete)
        reasoning_chunks = self._extract_reasoning_text_candidates(
            message=message,
            result=result,
            extra_event_fields=extra_event_fields,
        )
        emitted_reasoning = False
        if reasoningText:
            self._emit_reasoning_text(str(reasoningText))
            emitted_reasoning = True
        for chunk in reasoning_chunks:
            self._emit_reasoning_text(chunk)
            emitted_reasoning = True
        if emitted_reasoning:
            _mark_invoke_diag(invocation_state, stage="reasoning", saw_progress=True)

        suppress_reasoning_snapshot_text = bool(self._emitted_reasoning_texts_turn)
        assistant_snapshot_text = self._extract_text_from_assistant_message(
            message,
            exclude_reasoning_wrappers=suppress_reasoning_snapshot_text,
        )
        extra_snapshot_text = self._extract_text_from_extra_fields(extra_event_fields) if extra_event_fields else None
        transport_snapshot_text = (
            self._extract_transport_text_from_extra_fields(
                extra_fields=extra_event_fields,
                invocation_state=invocation_state,
            )
            if extra_event_fields and not data and not assistant_snapshot_text and not extra_snapshot_text
            else None
        )
        result_text = (
            self._extract_text_from_result(
                result,
                exclude_reasoning_wrappers=suppress_reasoning_snapshot_text,
            )
            if result is not None
            else None
        )
        plan_turn = _is_plan_turn(invocation_state)
        if plan_turn:
            for payload_text in (
                data,
                assistant_snapshot_text,
                extra_snapshot_text,
                transport_snapshot_text,
                result_text,
            ):
                self._record_plan_protocol_text(payload_text)
            data = ""
            assistant_snapshot_text = None
            extra_snapshot_text = None
            transport_snapshot_text = None
            result_text = None

        saw_text_before = self._saw_text_delta
        suppress_snapshot_reemit = should_emit_complete and saw_text_before
        if suppress_snapshot_reemit:
            snapshot_sources = [
                source
                for source, payload in (
                    ("message", assistant_snapshot_text),
                    ("extra", extra_snapshot_text),
                    ("transport", transport_snapshot_text),
                    ("result", result_text),
                )
                if isinstance(payload, str) and payload
            ]
            if snapshot_sources:
                _LOGGER.debug(
                    "Suppressing duplicate assistant snapshot on completion after streamed deltas (sources=%s).",
                    ",".join(snapshot_sources),
                )
        saw_bedrock_chunk = self._record_bedrock_stream_markers(extra_event_fields)
        saw_bedrock_chunk = self._record_bedrock_stream_markers(result) or saw_bedrock_chunk
        if saw_bedrock_chunk:
            self._saw_bedrock_stream_chunk = True

        if should_emit_complete:
            _mark_invoke_diag(invocation_state, stage="complete", saw_complete=True)

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
                _mark_invoke_diag(
                    invocation_state,
                    stage="tool_start",
                    saw_progress=True,
                    saw_tool_activity=True,
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
                            _mark_invoke_diag(
                                invocation_state,
                                stage="tool_progress",
                                saw_progress=True,
                                saw_tool_activity=True,
                            )
                self._emit_plan_progress_from_tool(tool_name=tool_name, tool_input=tool_input)

        extra_tool_start = self._extract_tool_start_from_extra_fields(extra_event_fields)
        if extra_tool_start is not None:
            tool_use_id, tool_name, tool_input = extra_tool_start
            self._ensure_tool_history(
                tool_use_id,
                tool_name=tool_name,
                tool_input=tool_input,
                emit_start=(tool_use_id not in self.tool_histories),
            )
            _mark_invoke_diag(
                invocation_state,
                stage="tool_start",
                saw_progress=True,
                saw_tool_activity=True,
            )
            self._emit_plan_progress_from_tool(tool_name=tool_name, tool_input=tool_input)

        for tool_use_id, stream, content in self._extract_tool_progress_chunks(
            current_tool_use=current_tool_use,
            result=result,
            extra_event_fields=extra_event_fields,
        ):
            self._append_tool_progress(tool_use_id, content, stream=stream)
            _mark_invoke_diag(
                invocation_state,
                stage="tool_progress",
                saw_progress=True,
                saw_tool_activity=True,
            )

        extra_tool_result = self._extract_tool_result_from_extra_fields(extra_event_fields)
        if extra_tool_result is not None:
            tool_use_id, status, tool_name = extra_tool_result
            self._emit_tool_result(tool_use_id, status, tool_name=tool_name)
            _mark_invoke_diag(
                invocation_state,
                stage="tool_result",
                saw_progress=True,
                saw_tool_activity=True,
            )

        if message.get("role") == "assistant":
            for content in _message_content_blocks(message):
                tool_use = content.get("toolUse")
                if tool_use:
                    tid = self._resolve_tool_use_id(
                        tool_use.get("toolUseId") or tool_use.get("tool_use_id") or tool_use.get("id")
                    )
                    tool_name = tool_use.get("name")
                    tool_input = tool_use.get("input")
                    self._emit_plan_progress_from_tool(tool_name=tool_name, tool_input=tool_input)
                    if isinstance(tid, str):
                        self._ensure_tool_history(
                            tid,
                            tool_name=str(tool_name) if tool_name is not None else None,
                            tool_input=tool_input,
                            emit_start=(tid not in self.tool_histories),
                        )
                        _mark_invoke_diag(
                            invocation_state,
                            stage="tool_start",
                            saw_progress=True,
                            saw_tool_activity=True,
                        )
                        if isinstance(tool_input, dict):
                            self._emit(
                                {
                                    "event": "tool_input",
                                    "tool_use_id": tid,
                                    "input": tool_input,
                                }
                            )

        elif message.get("role") == "user":
            for content in _message_content_blocks(message):
                tool_result = content.get("toolResult")
                if tool_result:
                    tid = tool_result.get("toolUseId")
                    status = tool_result.get("status", "unknown")
                    if isinstance(tid, str):
                        tool_label = tool_result.get("name")
                        self._emit_tool_result(
                            tid,
                            str(status),
                            tool_name=str(tool_label) if tool_label else None,
                        )
                        _mark_invoke_diag(
                            invocation_state,
                            stage="tool_result",
                            saw_progress=True,
                            saw_tool_activity=True,
                        )

        result_tool_emitted = False
        if isinstance(result, dict):
            tid = result.get("toolUseId")
            status = result.get("status")
            if isinstance(tid, str) and status is not None:
                tool_label = result.get("name")
                self._emit_tool_result(tid, str(status), tool_name=str(tool_label) if tool_label else None)
                result_tool_emitted = True
                _mark_invoke_diag(
                    invocation_state,
                    stage="tool_result",
                    saw_progress=True,
                    saw_tool_activity=True,
                )

        if event_loop_throttled_delay:
            self._emit(
                {
                    "event": "warning",
                    "text": f"Throttled! Waiting {event_loop_throttled_delay}s before retrying...",
                }
            )

        emitted_stream_text = False
        if data:
            emitted_stream_text = self._emit_assistant_text_delta(data, source="data") or emitted_stream_text
        if not suppress_snapshot_reemit:
            emitted_stream_text = (
                self._emit_assistant_message_text(assistant_snapshot_text, source="message")
                or emitted_stream_text
            )
            if extra_event_fields:
                emitted_stream_text = (
                    self._emit_assistant_message_text(extra_snapshot_text, source="extra")
                    or emitted_stream_text
                )
                emitted_stream_text = (
                    self._emit_assistant_message_text(transport_snapshot_text, source="transport")
                    or emitted_stream_text
                )
        if self._saw_text_delta and not saw_text_before:
            _mark_invoke_diag(
                invocation_state,
                stage="first_text_delta",
                saw_progress=True,
                saw_text_delta=True,
            )
        elif emitted_stream_text:
            _mark_invoke_diag(invocation_state, stage="text_delta", saw_progress=True)

        emitted_text_fallback = False
        allow_result_text_fallback = not bool(
            self._assistant_text_sources_turn.intersection({"data", "message", "extra", "transport"})
        )
        if isinstance(result, dict):
            if not result_tool_emitted and not suppress_snapshot_reemit and allow_result_text_fallback:
                emitted_text_fallback = self._emit_text_fallback_if_needed(result_text)
        elif result is not None and not suppress_snapshot_reemit and allow_result_text_fallback:
            emitted_text_fallback = self._emit_text_fallback_if_needed(result_text)
            if isinstance(result, str):
                self._consume_text_for_plan_markers(result)

        if should_emit_complete:
            self._flush_plan_marker_buffer()

        if self._emit_tool_heartbeats():
            _mark_invoke_diag(
                invocation_state,
                stage="tool_progress",
                saw_progress=True,
                saw_tool_activity=True,
            )

        if (
            should_emit_complete
            and self._saw_bedrock_stream_chunk
            and not self._saw_text_delta
            and not self._bedrock_missing_text_warning_emitted
        ):
            markers = ", ".join(sorted(self._bedrock_stream_markers_seen)) or "unknown"
            _LOGGER.warning(
                "Bedrock stream completed without extractable text delta (markers=%s).",
                markers,
            )
            self._bedrock_missing_text_warning_emitted = True

        if should_emit_complete and self._reasoning_expected_for_turn(invocation_state):
            saw_reasoning = (isinstance(reasoningText, str) and bool(reasoningText.strip())) or bool(reasoning_chunks)
            if not saw_reasoning:
                provider, tier, model_id = self._invocation_model_tokens(invocation_state)
                _LOGGER.info(
                    "Reasoning payload missing for expected reasoning-capable turn (provider=%s tier=%s model_id=%s).",
                    provider or "?",
                    tier or "?",
                    model_id or "?",
                )

        if (
            should_emit_complete
            and not emitted_text_fallback
            and self._saw_text_delta
            and not self._text_complete_emitted_turn
        ):
            self._emit({"event": "text_complete"})
            self._text_complete_emitted_turn = True

        if should_emit_complete:
            self._reset_bedrock_stream_state()

        if emitted_text_fallback:
            _mark_invoke_diag(invocation_state, stage="result_text", saw_progress=True)
            self._flush_plan_marker_buffer()
            self._reset_turn_state()


def _resolve_handler_mode() -> str:
    return "tui" if truthy_env("SWARMEE_TUI_EVENTS", False) else "cli"


def _build_handler_for_mode(mode: str) -> CallbackHandler | TuiCallbackHandler:
    if mode == "tui":
        return TuiCallbackHandler()
    return CallbackHandler()


class CallbackHandlerDispatcher:
    """Stable callback proxy that can rebind handler mode safely at runtime."""

    def __init__(self) -> None:
        self._forced_mode: str | None = None
        self._active_mode: str | None = None
        self._delegate: CallbackHandler | TuiCallbackHandler | None = None
        self._ensure_delegate()

    def _target_mode(self) -> str:
        if self._forced_mode in {"cli", "tui"}:
            return self._forced_mode
        return _resolve_handler_mode()

    def _ensure_delegate(self) -> CallbackHandler | TuiCallbackHandler:
        target_mode = self._target_mode()
        if self._delegate is not None and self._active_mode == target_mode:
            return self._delegate

        previous_interrupt = self._delegate.interrupt_event if self._delegate is not None else None
        if isinstance(self._delegate, CallbackHandler):
            with contextlib.suppress(Exception):
                self._delegate.callback_handler(force_stop=True)

        self._delegate = _build_handler_for_mode(target_mode)
        self._active_mode = target_mode
        self._delegate.interrupt_event = previous_interrupt
        return self._delegate

    def configure_mode(self, *, tui_events: bool | None = None) -> None:
        self._forced_mode = None if tui_events is None else ("tui" if tui_events else "cli")
        self._ensure_delegate()

    @property
    def interrupt_event(self) -> Event | None:
        return self._ensure_delegate().interrupt_event

    @interrupt_event.setter
    def interrupt_event(self, event: Event | None) -> None:
        self._ensure_delegate().interrupt_event = event

    def callback_handler(self, *args: Any, **kwargs: Any) -> None:
        self._ensure_delegate().callback_handler(*args, **kwargs)

    def cleanup(self) -> None:
        delegate = self._ensure_delegate()
        if isinstance(delegate, CallbackHandler):
            with contextlib.suppress(Exception):
                delegate.callback_handler(force_stop=True)
            for spinner in list(_ACTIVE_TOOL_SPINNERS):
                with contextlib.suppress(Exception):
                    spinner.stop()


callback_handler_instance = CallbackHandlerDispatcher()


def configure_callback_handler_mode(*, tui_events: bool | None = None) -> None:
    callback_handler_instance.configure_mode(tui_events=tui_events)


def callback_handler(*args: Any, **kwargs: Any) -> None:
    callback_handler_instance.callback_handler(*args, **kwargs)


def set_interrupt_event(event: Event | None) -> None:
    callback_handler_instance.interrupt_event = event


def _cleanup_spinners_at_exit() -> None:
    callback_handler_instance.cleanup()


atexit.register(_cleanup_spinners_at_exit)
