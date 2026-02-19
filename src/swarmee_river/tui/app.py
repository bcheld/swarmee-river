"""Optional Textual app scaffold for `swarmee tui`."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json as _json
import os
import re
import shutil
import signal
import subprocess
import time
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.state_paths import logs_dir, sessions_dir

_CONSENT_CHOICES = {"y", "n", "a", "v"}
_MODEL_AUTO_VALUE = "__auto__"
_TRUNCATED_ARTIFACT_RE = re.compile(r"full output saved to (?P<path>[^\]]+)")
_PATH_TOKEN_RE = re.compile(r"[A-Za-z]:\\[^\s,;]+|/(?:[^\s,;]+)|\./[^\s,;]+|\.\./[^\s,;]+")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROVIDER_NOISE_PREFIX = "[provider]"
_PROVIDER_FALLBACK_PHRASE = "falling back to"


@dataclass(frozen=True)
class ParsedEvent:
    kind: str
    text: str
    meta: dict[str, str] | None = None


def build_swarmee_cmd(prompt: str, *, auto_approve: bool) -> list[str]:
    """Build a subprocess command for a one-shot Swarmee run."""
    command = [sys.executable, "-u", "-m", "swarmee_river.swarmee"]
    if auto_approve:
        command.append("--yes")
    command.append(prompt)
    return command


def extract_plan_section(output: str) -> str | None:
    """Extract the one-shot plan block beginning at 'Proposed plan:' if present."""
    marker = "Proposed plan:"
    marker_index = output.find(marker)
    if marker_index < 0:
        return None

    trailing_hint_prefix = "Plan generated. Re-run with --yes"
    candidate = output[marker_index:]
    extracted_lines: list[str] = []
    for line in candidate.splitlines():
        if not extracted_lines:
            local_index = line.find(marker)
            if local_index >= 0:
                line = line[local_index:]
        if line.strip().startswith(trailing_hint_prefix):
            break
        extracted_lines.append(line.rstrip())

    while extracted_lines and not extracted_lines[-1].strip():
        extracted_lines.pop()

    if not extracted_lines:
        return None

    extracted = "\n".join(extracted_lines).strip()
    return extracted or None


def looks_like_plan_output(text: str) -> bool:
    """Detect whether one-shot output likely contains a generated plan."""
    return extract_plan_section(text) is not None


def render_tui_hint_after_plan() -> str:
    """Hint shown when a plan-only run is detected."""
    return "Plan detected. Type /approve to execute, /replan to regenerate, /clearplan to clear."


def is_multiline_newline_key(event: Any) -> bool:
    """Detect Shift+Enter, Alt+Enter, or Ctrl+J — NOT plain Enter."""
    key = str(getattr(event, "key", "")).lower()
    aliases = [str(a).lower() for a in getattr(event, "aliases", [])]
    event_name = str(getattr(event, "name", "")).lower()

    # Explicit modifier+enter combinations only.
    # Plain Enter must NOT match — it submits the prompt.
    modifier_enter_keys = {
        "shift+enter", "shift+return", "shift+ctrl+m",
        "alt+enter", "alt+return",
        "ctrl+j",
    }
    if key in modifier_enter_keys:
        return True
    if event_name in {"shift_enter", "shift_return"}:
        return True
    if any(alias in modifier_enter_keys for alias in aliases):
        return True
    return False


def build_plan_mode_prompt(prompt: str) -> str:
    """Wrap user input so `/plan` consistently routes through plan generation."""
    cleaned = prompt.strip()
    if not cleaned:
        return cleaned
    return (
        "Create a concrete work plan for the following request. "
        "Return only the plan details.\n\n"
        f"Request:\n{cleaned}"
    )


def _extract_paths_from_text(text: str) -> list[str]:
    return [match.strip() for match in _PATH_TOKEN_RE.findall(text) if match.strip()]


def sanitize_output_text(text: str) -> str:
    """Remove common control sequences that render poorly in a TUI transcript."""
    cleaned = text.replace("\r", "")
    return _ANSI_ESCAPE_RE.sub("", cleaned)


def resolve_model_config_summary(*, provider_override: str | None = None, tier_override: str | None = None) -> str:
    """
    Best-effort summary of the configured model selection (provider/tier/model_id) for display in the TUI.

    This is intentionally approximate: final provider/model can still vary at runtime based on CLI args,
    environment variables, and credential availability.
    """
    try:
        from swarmee_river.settings import load_settings
        from swarmee_river.utils.provider_utils import resolve_model_provider
    except Exception:
        return "Model: (unavailable)"

    try:
        settings = load_settings()
    except Exception:
        return "Model: (unavailable)"

    selected_provider, notice = resolve_model_provider(
        cli_provider=None,
        env_provider=provider_override if provider_override is not None else os.getenv("SWARMEE_MODEL_PROVIDER"),
        settings_provider=settings.models.provider,
    )
    tier = (
        tier_override
        if tier_override is not None
        else (os.getenv("SWARMEE_MODEL_TIER") or settings.models.default_tier)
    )
    tier = (tier or "balanced").strip().lower()

    model_id: str | None = None
    try:
        # Provider-specific tiers are the primary source.
        provider_cfg = settings.models.providers.get(selected_provider)
        if provider_cfg:
            tier_cfg = provider_cfg.tiers.get(tier)
            if tier_cfg and tier_cfg.model_id:
                model_id = tier_cfg.model_id
        # Global tier overrides.
        if model_id is None:
            global_tier_cfg = settings.models.tiers.get(tier)
            if global_tier_cfg and global_tier_cfg.model_id:
                model_id = global_tier_cfg.model_id
    except Exception:
        model_id = None

    suffix = f" ({model_id})" if model_id else ""
    fallback = " (fallback)" if notice else ""
    return f"Model: {selected_provider}/{tier}{suffix}{fallback}"


def _model_option_model_id(
    *,
    settings: Any,
    provider_name: str,
    tier_name: str,
) -> str | None:
    model_id: str | None = None
    provider_cfg = settings.models.providers.get(provider_name)
    if provider_cfg is not None:
        provider_tier_cfg = provider_cfg.tiers.get(tier_name)
        if provider_tier_cfg is not None and provider_tier_cfg.model_id:
            model_id = provider_tier_cfg.model_id
    if model_id is None:
        global_tier_cfg = settings.models.tiers.get(tier_name)
        if global_tier_cfg is not None and global_tier_cfg.model_id:
            model_id = global_tier_cfg.model_id
    return model_id


def model_select_options(
    *,
    provider_override: str | None = None,
    tier_override: str | None = None,
) -> tuple[list[tuple[str, str]], str]:
    """
    Build model selector dropdown options and the currently selected option value.

    Returns:
        (options, selected_value)
    """
    auto_summary = resolve_model_config_summary().removeprefix("Model: ").strip()
    options: list[tuple[str, str]] = [(f"Auto ({auto_summary})", _MODEL_AUTO_VALUE)]
    selected_value = _MODEL_AUTO_VALUE

    try:
        from swarmee_river.settings import load_settings
        from swarmee_river.utils.provider_utils import resolve_model_provider
    except Exception:
        return options, selected_value

    try:
        settings = load_settings()
    except Exception:
        return options, selected_value

    selected_provider, _ = resolve_model_provider(
        cli_provider=None,
        env_provider=provider_override if provider_override is not None else os.getenv("SWARMEE_MODEL_PROVIDER"),
        settings_provider=settings.models.provider,
    )
    selected_tier = (
        tier_override
        if tier_override is not None
        else (os.getenv("SWARMEE_MODEL_TIER") or settings.models.default_tier)
    )
    selected_provider = (selected_provider or "").strip().lower()
    selected_tier = (selected_tier or "").strip().lower()

    provider_cfg = settings.models.providers.get(selected_provider)
    tier_names = sorted(provider_cfg.tiers.keys()) if provider_cfg and provider_cfg.tiers else []

    for tier_name in tier_names:
        value = f"{selected_provider}|{tier_name}"
        model_id = _model_option_model_id(settings=settings, provider_name=selected_provider, tier_name=tier_name)
        suffix = f" ({model_id})" if model_id else ""
        options.append((f"{selected_provider}/{tier_name}{suffix}", value))
        if tier_name == selected_tier:
            selected_value = value

    if provider_override is None and tier_override is None:
        selected_value = _MODEL_AUTO_VALUE

    available_values = {value for _, value in options}
    if selected_value not in available_values:
        selected_value = _MODEL_AUTO_VALUE
    return options, selected_value


def parse_output_line(line: str) -> ParsedEvent | None:
    """Best-effort parser for notable subprocess output events."""
    text = line.rstrip("\n")
    stripped = text.strip()
    lower = stripped.lower()

    if stripped.startswith(_PROVIDER_NOISE_PREFIX) and _PROVIDER_FALLBACK_PHRASE in lower:
        return ParsedEvent(kind="noise", text=text)

    if "~ consent>" in lower:
        return ParsedEvent(kind="consent_prompt", text=text)

    if stripped.startswith("Proposed plan:"):
        return ParsedEvent(kind="plan_header", text=text)

    if "[tool result truncated:" in lower:
        match = _TRUNCATED_ARTIFACT_RE.search(text)
        if match:
            path = match.group("path").strip()
            return ParsedEvent(kind="artifact", text=text, meta={"path": path})
        return ParsedEvent(kind="tool_truncated", text=text)

    if lower.startswith("patch:"):
        path = stripped.split(":", 1)[1].strip()
        if path:
            return ParsedEvent(kind="artifact", text=text, meta={"path": path})
        return ParsedEvent(kind="patch", text=text)

    if lower.startswith("backups:"):
        rest = stripped.split(":", 1)[1].strip()
        paths = _extract_paths_from_text(rest)
        if paths:
            return ParsedEvent(kind="artifact", text=text, meta={"paths": ",".join(paths)})
        return ParsedEvent(kind="backups", text=text)

    if stripped.startswith("Error:") or stripped.startswith("ERROR:"):
        return ParsedEvent(kind="error", text=text)

    if "operation not permitted" in lower and "/bin/ps" in lower:
        return ParsedEvent(kind="warning", text=text)

    if "warning" in lower or "deprecationwarning" in lower or "runtimewarning" in lower or "userwarning" in lower:
        return ParsedEvent(kind="warning", text=text)

    if lower.startswith("[tool ") and any(token in lower for token in {" start", " started", " running"}):
        return ParsedEvent(kind="tool_start", text=text)

    if lower.startswith("[tool ") and any(token in lower for token in {" done", " end", " finished", " stopped"}):
        return ParsedEvent(kind="tool_stop", text=text)

    return None


def parse_tui_event(line: str) -> dict[str, Any] | None:
    """Parse a JSONL event line emitted by TuiCallbackHandler. Returns None for non-JSON lines."""
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = _json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, _json.JSONDecodeError):
        return None


def extract_plan_section_from_output(output: str) -> str | None:
    """Extract plan text from mixed output by ignoring structured JSONL event lines."""
    plain_lines = [line for line in output.splitlines() if parse_tui_event(line) is None]
    if not plain_lines:
        return None
    return extract_plan_section("\n".join(plain_lines))


def artifact_paths_from_event(event: ParsedEvent) -> list[str]:
    """Extract artifact paths from a parsed event."""
    if event.meta is None:
        return []
    result: list[str] = []
    path = event.meta.get("path")
    if path:
        result.append(path)
    paths = event.meta.get("paths")
    if paths:
        for item in paths.split(","):
            token = item.strip()
            if token:
                result.append(token)
    return result


def add_recent_artifacts(existing: list[str], new_paths: list[str], *, max_items: int = 20) -> list[str]:
    """Dedupe and cap artifact paths while preserving recency."""
    updated = list(existing)
    for item in new_paths:
        path = item.strip()
        if not path:
            continue
        if path in updated:
            updated.remove(path)
        updated.append(path)
    if len(updated) > max_items:
        updated = updated[-max_items:]
    return updated


def detect_consent_prompt(line: str) -> str | None:
    """Detect consent-related subprocess output lines."""
    normalized = line.strip().lower()
    if "~ consent>" in normalized:
        return "prompt"
    if "allow tool '" in normalized:
        return "header"
    return None


def update_consent_capture(
    consent_active: bool,
    consent_buffer: list[str],
    line: str,
    *,
    max_lines: int = 20,
) -> tuple[bool, list[str]]:
    """Update consent capture state from a single output line."""
    kind = detect_consent_prompt(line)
    if kind is None and not consent_active:
        return consent_active, consent_buffer

    updated = list(consent_buffer)
    updated.append(line.rstrip("\n"))
    if len(updated) > max_lines:
        updated = updated[-max_lines:]
    return True, updated


def write_to_proc(proc: subprocess.Popen[str], text: str) -> bool:
    """Write a response line to a subprocess stdin."""
    if proc.stdin is None:
        return False

    payload = text if text.endswith("\n") else f"{text}\n"
    try:
        proc.stdin.write(payload)
        proc.stdin.flush()
    except Exception:
        return False
    return True


def spawn_swarmee(
    prompt: str,
    *,
    auto_approve: bool,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Spawn Swarmee as a subprocess with line-buffered merged output."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # The CLI callback handler prints spinners using ANSI + carriage returns; disable for TUI subprocess capture.
    env["SWARMEE_SPINNERS"] = "0"
    # Enable structured JSONL event output for TUI consumption.
    env["SWARMEE_TUI_EVENTS"] = "1"
    existing_warning_filters = env.get("PYTHONWARNINGS", "").strip()
    tui_warning_filters = [
        # `PYTHONWARNINGS` is parsed via `warnings._setoption`, which `re.escape`s message+module.
        # Use exact (literal) values, not regex patterns.
        'ignore:Field name "json" in "Http_requestTool" shadows an attribute in parent "BaseModel"'
        ":UserWarning:pydantic.main",
    ]
    env["PYTHONWARNINGS"] = ",".join(
        [item for item in [*tui_warning_filters, existing_warning_filters] if isinstance(item, str) and item.strip()]
    )
    if env_overrides:
        env.update(env_overrides)
    if session_id:
        env["SWARMEE_SESSION_ID"] = session_id
    return subprocess.Popen(
        build_swarmee_cmd(prompt, auto_approve=auto_approve),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )


def stop_process(proc: subprocess.Popen[str], *, timeout_s: float = 2.0) -> None:
    """Stop a running subprocess, escalating from interrupt to terminate to kill."""
    if proc.poll() is not None:
        return

    if os.name == "posix" and hasattr(signal, "SIGINT"):
        with contextlib.suppress(Exception):
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass

    with contextlib.suppress(Exception):
        proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return

    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout_s)


def run_tui() -> int:
    """Run the full-screen TUI if Textual is installed."""
    try:
        textual_app = importlib.import_module("textual.app")
        textual_binding = importlib.import_module("textual.binding")
        textual_containers = importlib.import_module("textual.containers")
        textual_widgets = importlib.import_module("textual.widgets")
    except ImportError:
        print(
            "Textual is required for `swarmee tui`.\n"
            'Install with: pip install "swarmee-river[tui]"\n'
            'For editable installs in this repo: pip install -e ".[tui]"',
            file=sys.stderr,
        )
        return 1

    AppBase = textual_app.App
    Binding = textual_binding.Binding
    Horizontal = textual_containers.Horizontal
    Vertical = textual_containers.Vertical
    VerticalScroll = textual_containers.VerticalScroll
    Header = textual_widgets.Header
    Footer = textual_widgets.Footer
    Select = textual_widgets.Select
    Static = textual_widgets.Static
    TabbedContent = textual_widgets.TabbedContent
    TabPane = textual_widgets.TabPane
    TextArea = textual_widgets.TextArea

    from swarmee_river.tui.widgets import (
        AssistantMessage,
        CommandPalette,
        ConsentCard,
        PlanCard,
        StatusBar,
        SystemMessage,
        ThinkingIndicator,
        ToolCallBlock,
        UserMessage,
    )

    class PromptTextArea(TextArea):
        """Prompt editor that submits on Enter and inserts newline on Shift+Enter/Ctrl+J."""

        _ENTER_KEYS = {"enter", "return", "ctrl+m"}

        def _insert_newline(self) -> None:
            for method_name, args in (
                ("insert", ("\n",)),
                ("insert_text_at_cursor", ("\n",)),
                ("action_newline", ()),
                ("action_insert_newline", ()),
            ):
                method = getattr(self, method_name, None)
                if not callable(method):
                    continue
                with contextlib.suppress(Exception):
                    method(*args)
                    return

        def _adjust_height(self) -> None:
            line_count = self.text.count("\n") + 1
            # Height controls the input area only. The prompt container provides the
            # border and a one-line footer row (model selector).
            target = max(4, min(11, line_count))
            if getattr(self, "_last_height", None) != target:
                self._last_height = target
                self.styles.height = target

        def on_text_area_changed(self, event: Any) -> None:
            self._adjust_height()
            app = getattr(self, "app", None)
            if app is not None and hasattr(app, "_update_command_palette"):
                app._update_command_palette(self.text)

        async def _on_key(self, event: Any) -> None:
            """Override TextArea._on_key to control Enter behaviour and route
            special keys (palette, history) before falling back to TextArea's
            default character-insertion handler.

            Textual 8's dispatch uses ``cls.__dict__.get("_on_key") or
            cls.__dict__.get("on_key")`` per MRO class, so defining *both*
            ``_on_key`` and ``on_key`` on the same class causes ``on_key`` to
            be silently skipped.  Therefore ALL key logic lives here.
            """
            key = str(getattr(event, "key", "")).lower()
            app = getattr(self, "app", None)

            # ── Shift+Enter / Alt+Enter / Ctrl+J → insert newline ──
            if is_multiline_newline_key(event):
                event.stop()
                event.prevent_default()
                self._insert_newline()
                return

            # ── Arrow keys: palette navigation (when visible) ──
            if key in {"up", "down"} and app is not None and hasattr(app, "_command_palette"):
                palette = app._command_palette
                if palette is not None and palette.is_visible:
                    event.stop()
                    event.prevent_default()
                    palette.move_selection(-1 if key == "up" else 1)
                    return

            # ── Arrow keys: prompt history (when palette hidden) ──
            if key == "up" and app is not None and hasattr(app, "_prompt_history"):
                history = app._prompt_history
                if history and app._history_index < len(history) - 1:
                    event.stop()
                    event.prevent_default()
                    app._history_index += 1
                    self.clear()
                    entry = history[-(app._history_index + 1)]
                    for method_name in ("insert", "insert_text_at_cursor"):
                        method = getattr(self, method_name, None)
                        if callable(method):
                            with contextlib.suppress(Exception):
                                method(entry)
                                break
                    return
            if key == "down" and app is not None and hasattr(app, "_prompt_history"):
                if app._history_index > 0:
                    event.stop()
                    event.prevent_default()
                    app._history_index -= 1
                    self.clear()
                    entry = app._prompt_history[-(app._history_index + 1)]
                    for method_name in ("insert", "insert_text_at_cursor"):
                        method = getattr(self, method_name, None)
                        if callable(method):
                            with contextlib.suppress(Exception):
                                method(entry)
                                break
                    return
                elif app._history_index == 0:
                    event.stop()
                    event.prevent_default()
                    app._history_index = -1
                    self.clear()
                    return

            # ── Enter with palette visible → submit ──
            # The palette is suggestions/autocomplete; Enter should still execute the
            # current prompt text (otherwise slash commands never run because the
            # palette reopens on every change).
            if key in self._ENTER_KEYS and app is not None and hasattr(app, "_command_palette"):
                palette = app._command_palette
                if palette is not None and palette.is_visible:
                    event.stop()
                    event.prevent_default()
                    palette.hide()
                    app.action_submit_prompt()
                    return

            # ── Tab with palette visible → select command ──
            if key == "tab" and app is not None and hasattr(app, "_command_palette"):
                palette = app._command_palette
                if palette is not None and palette.is_visible:
                    event.stop()
                    event.prevent_default()
                    selected = palette.get_selected()
                    if selected:
                        self.clear()
                        for method_name in ("insert", "insert_text_at_cursor"):
                            method = getattr(self, method_name, None)
                            if callable(method):
                                with contextlib.suppress(Exception):
                                    method(selected + " ")
                                    break
                    palette.hide()
                    return

            # ── Escape → dismiss palette ──
            if key == "escape" and app is not None and hasattr(app, "_command_palette"):
                palette = app._command_palette
                if palette is not None and palette.is_visible:
                    event.stop()
                    event.prevent_default()
                    palette.hide()
                    return

            # ── Plain Enter → submit prompt ──
            # Do NOT call super() — TextArea would insert a newline.
            if key in self._ENTER_KEYS:
                event.stop()
                event.prevent_default()
                if app is not None:
                    app.action_submit_prompt()
                return

            # ── Space: explicit insert ──
            # Some terminal / Textual 8.0.0 combinations appear to drop space insertion
            # when overriding `_on_key`. Handle it directly to keep typing reliable.
            if key == "space":
                event.stop()
                event.prevent_default()
                with contextlib.suppress(Exception):
                    self.insert(" ")
                    return
                with contextlib.suppress(Exception):
                    self.insert_text_at_cursor(" ")
                    return

            # ── Everything else: delegate to TextArea (space, printable, etc.) ──
            # Textual has changed `TextArea._on_key` between releases (sync vs async),
            # and some versions route default behavior via `on_key` instead.
            # Call what exists, and only await if it returns an awaitable.
            handler = getattr(super(), "_on_key", None) or getattr(super(), "on_key", None)
            if handler is None:
                return
            result = handler(event)
            if inspect.isawaitable(result):
                await result

    class SwarmeeTUI(AppBase):
        CSS = """
        Screen {
            layout: vertical;
        }

        #panes {
            layout: horizontal;
            height: 1fr;
            width: 100%;
        }

        #transcript {
            width: 2fr;
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            overflow-y: auto;
        }

        #side {
            width: 1fr;
            height: 1fr;
            layout: vertical;
        }

        #side_tabs {
            height: 1fr;
        }

        #plan, #issues, #artifacts {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
        }

        #consent {
            height: 8;
            border: round $warning;
            padding: 0 1;
        }

        #prompt_box {
            min-height: 5;
            max-height: 12;
            height: auto;
            width: 1fr;
            border: round $accent;
            padding: 0 1;
        }

        #prompt {
            border: none;
            padding: 0;
            height: auto;
        }

        #prompt_bottom {
            height: 1;
            layout: horizontal;
            align: right middle;
        }

        #prompt_spacer {
            width: 1fr;
        }

        #model_select {
            width: 34;
            min-width: 24;
        }
        """
        BINDINGS = [
            ("f5", "submit_prompt", "Send prompt"),
            ("escape", "interrupt_run", "Interrupt run"),
            ("ctrl+c", "copy_selection", "Copy selection"),
            ("meta+c", "copy_selection", "Copy selection"),
            ("super+c", "copy_selection", "Copy selection"),
            ("ctrl+d", "quit", "Quit"),
            ("tab", "focus_prompt", "Focus prompt"),
            Binding("ctrl+left", "widen_side", "Widen side", priority=True),
            Binding("ctrl+right", "widen_transcript", "Widen transcript", priority=True),
            Binding("ctrl+shift+left", "widen_side", "Widen side", priority=True),
            Binding("ctrl+shift+right", "widen_transcript", "Widen transcript", priority=True),
            ("f6", "widen_side", "Widen side"),
            ("f7", "widen_transcript", "Widen transcript"),
            ("ctrl+f", "search_transcript", "Search"),
        ]

        _proc: subprocess.Popen[str] | None = None
        _runner_thread: threading.Thread | None = None
        _last_prompt: str | None = None
        _pending_plan_prompt: str | None = None
        _last_run_auto_approve: bool = False
        _default_auto_approve: bool = False
        _run_session_id: str | None = None
        _consent_active: bool = False
        _consent_buffer: list[str] = []
        _artifacts: list[str] = []
        _plan_text: str = ""
        _issues_lines: list[str] = []
        _issues_repeat_line: str | None = None
        _issues_repeat_count: int = 0
        _warning_count: int = 0
        _error_count: int = 0
        _model_provider_override: str | None = None
        _model_tier_override: str | None = None
        _model_select_syncing: bool = False
        # Conversation view state
        _current_assistant_msg: Any = None  # AssistantMessage | None
        _current_thinking: Any = None  # ThinkingIndicator | None
        _tool_blocks: dict[str, Any] = {}  # tool_use_id → ToolCallBlock
        _current_plan_card: Any = None  # PlanCard | None
        _command_palette: Any = None  # CommandPalette | None
        _status_bar: Any = None  # StatusBar | None
        _run_tool_count: int = 0
        _run_start_time: float | None = None
        _status_timer: Any = None
        _last_assistant_text: str = ""
        _prompt_history: list[str] = []
        _history_index: int = -1
        _MAX_PROMPT_HISTORY: int = 50
        _split_ratio: int = 2
        _search_active: bool = False
        _plan_step_counter: int = 0
        _transcript_widget_count: int = 0
        _MAX_TRANSCRIPT_WIDGETS: int = 500
        _received_structured_plan: bool = False

        def compose(self) -> Any:
            yield Header()
            with Horizontal(id="panes"):
                yield VerticalScroll(id="transcript")
                with Vertical(id="side"):
                    with TabbedContent(id="side_tabs"):
                        with TabPane("Plan", id="tab_plan"):
                            yield TextArea(
                                text="",
                                language="markdown",
                                read_only=True,
                                show_cursor=False,
                                id="plan",
                                soft_wrap=True,
                            )
                        with TabPane("Issues", id="tab_issues"):
                            yield TextArea(
                                text="",
                                read_only=True,
                                show_cursor=False,
                                id="issues",
                                soft_wrap=True,
                            )
                        with TabPane("Artifacts", id="tab_artifacts"):
                            yield TextArea(
                                text="",
                                read_only=True,
                                show_cursor=False,
                                id="artifacts",
                                soft_wrap=True,
                            )
                    yield TextArea(text="", read_only=True, show_cursor=False, id="consent", soft_wrap=True)
            yield CommandPalette(id="command_palette")
            yield StatusBar(id="status_bar")
            with Vertical(id="prompt_box"):
                yield PromptTextArea(
                    text="",
                    placeholder="Type prompt. Enter submits, Shift+Enter inserts newline.",
                    id="prompt",
                    soft_wrap=True,
                )
                with Horizontal(id="prompt_bottom"):
                    yield Static("", id="prompt_spacer")
                    yield Select(
                        options=[("Auto", _MODEL_AUTO_VALUE)],
                        allow_blank=False,
                        id="model_select",
                        compact=True,
                    )
            yield Footer()

        def on_mount(self) -> None:
            self._command_palette = self.query_one("#command_palette", CommandPalette)
            self._status_bar = self.query_one("#status_bar", StatusBar)
            self._status_bar.set_model(self._current_model_summary())
            self.query_one("#prompt", PromptTextArea).focus()
            self._reset_plan_panel()
            self._reset_issues_panel()
            self._reset_artifacts_panel()
            self._reset_consent_panel()
            self._refresh_model_select()
            self.title = "Swarmee"
            self.sub_title = self._current_model_summary()
            self._update_prompt_placeholder()
            # Show ASCII art banner at the top of the transcript.
            # Use plain Static widgets to avoid Rich markup interpretation
            # (the banner contains / characters that break [dim]...[/dim] wrapping).
            from swarmee_river.utils.welcome_utils import SWARMEE_BANNER
            for banner_line in SWARMEE_BANNER.strip().splitlines():
                self._mount_transcript_widget(Static(banner_line))
            self._write_transcript("Swarmee TUI ready. Enter a prompt to run Swarmee.")
            self._write_transcript(self.sub_title)
            self._write_transcript(
                "Commands: /plan <prompt>, /run <prompt>, /approve, /replan, /clearplan, /model, /stop, /exit."
            )
            self._write_transcript("Consent: /consent <y|n|a|v> (or press y/n/a/v when prompted).")
            self._write_transcript(
                "Keys: Enter submit, Shift+Enter newline (Ctrl+J/Alt+Enter fallback), "
                "Ctrl+Left/Right (or F6/F7) resize panes, F5 submit, Esc interrupt run, Ctrl+C/Cmd+C copy selection."
            )
            self._write_transcript("Model selector: prompt box footer dropdown (or /model commands).")
            self._write_transcript("Text is selectable in all panes.")
            self._write_transcript("Optional export: /copy, /copy plan, /copy issues, /copy all.")
            self._load_session()

        def _mount_transcript_widget(self, widget: Any) -> None:
            """Mount a widget into the transcript VerticalScroll and prune if needed."""
            transcript = self.query_one("#transcript", VerticalScroll)
            transcript.mount(widget)
            self._transcript_widget_count += 1
            if self._transcript_widget_count > self._MAX_TRANSCRIPT_WIDGETS:
                children = list(transcript.children)
                prune = len(children) - self._MAX_TRANSCRIPT_WIDGETS
                for child in children[:prune]:
                    child.remove()
                self._transcript_widget_count = len(list(transcript.children))
            transcript.scroll_end(animate=False)

        def _write_transcript(self, line: str) -> None:
            """Write a system/info message to the transcript."""
            self._mount_transcript_widget(SystemMessage(line))

        def _write_transcript_line(self, line: str) -> None:
            """Write a plain text line to the transcript (used for TUI-internal messages)."""
            self._write_transcript(line)

        def _write_user_input(self, text: str) -> None:
            self._mount_transcript_widget(UserMessage(text))

        def _append_plain_text(self, text: str) -> None:
            """Mount a plain Static widget for non-event output lines (legacy fallback)."""
            if not text.strip():
                return
            self._mount_transcript_widget(Static(text))

        def _set_plan_panel(self, content: str) -> None:
            self._plan_text = content
            plan_panel = self.query_one("#plan", TextArea)
            text = content if content.strip() else "(no plan)"
            plan_panel.load_text(text)
            plan_panel.scroll_end(animate=False)

        def _reset_plan_panel(self) -> None:
            self._set_plan_panel("(no plan)")

        def _reset_issues_panel(self) -> None:
            self._issues_lines = []
            self._issues_repeat_line = None
            self._issues_repeat_count = 0
            self._warning_count = 0
            self._error_count = 0
            panel = self.query_one("#issues", TextArea)
            panel.load_text("(no issues yet)")
            self._update_header_status()

        def _write_issue(self, line: str) -> None:
            if self._issues_repeat_line == line:
                self._issues_repeat_count += 1
                return
            if self._issues_repeat_line is not None and self._issues_repeat_count > 0:
                repeated = f"… repeated {self._issues_repeat_count} more time(s): {self._issues_repeat_line}"
                self._issues_lines.append(repeated)
            self._issues_repeat_line = line
            self._issues_repeat_count = 0
            self._issues_lines.append(line)
            if len(self._issues_lines) > 2000:
                self._issues_lines = self._issues_lines[-2000:]
            self._render_issues_panel()

        def _render_issues_panel(self) -> None:
            issues_panel = self.query_one("#issues", TextArea)
            text = "\n".join(self._issues_lines) if self._issues_lines else "(no issues yet)"
            issues_panel.load_text(text)
            issues_panel.scroll_end(animate=False)

        def _flush_issue_repeats(self) -> None:
            if self._issues_repeat_line is None or self._issues_repeat_count <= 0:
                self._issues_repeat_line = None
                self._issues_repeat_count = 0
                return
            repeated = f"… repeated {self._issues_repeat_count} more time(s): {self._issues_repeat_line}"
            self._issues_lines.append(repeated)
            self._issues_repeat_line = None
            self._issues_repeat_count = 0
            self._render_issues_panel()

        def _update_header_status(self) -> None:
            counts = []
            if self._warning_count:
                counts.append(f"warn={self._warning_count}")
            if self._error_count:
                counts.append(f"err={self._error_count}")
            suffix = (" | " + " ".join(counts)) if counts else ""
            self.sub_title = f"{self._current_model_summary()}{suffix}"
            if self._status_bar is not None:
                self._status_bar.set_counts(warnings=self._warning_count, errors=self._error_count)

        def _current_model_summary(self) -> str:
            return resolve_model_config_summary(
                provider_override=self._model_provider_override,
                tier_override=self._model_tier_override,
            )

        def _model_env_overrides(self) -> dict[str, str]:
            overrides: dict[str, str] = {}
            if self._model_provider_override:
                overrides["SWARMEE_MODEL_PROVIDER"] = self._model_provider_override
            if self._model_tier_override:
                overrides["SWARMEE_MODEL_TIER"] = self._model_tier_override
            return overrides

        def _refresh_model_select(self) -> None:
            selector = self.query_one("#model_select", Select)
            options, selected_value = model_select_options(
                provider_override=self._model_provider_override,
                tier_override=self._model_tier_override,
            )
            self._model_select_syncing = True
            try:
                selector.set_options(options)
                with contextlib.suppress(Exception):
                    selector.value = selected_value
            finally:
                self._model_select_syncing = False

        def _update_prompt_placeholder(self) -> None:
            input_widget = self.query_one("#prompt", PromptTextArea)
            mode = "execute" if self._default_auto_approve else "plan"
            input_widget.placeholder = f"Mode: {mode}. Enter submits. Shift+Enter/Ctrl+J adds newline."

        def _update_command_palette(self, text: str) -> None:
            if self._command_palette is None:
                return
            stripped = text.strip()
            if stripped.startswith("/") and "\n" not in stripped:
                self._command_palette.filter(stripped)
            else:
                self._command_palette.hide()

        def _notify(self, message: str, *, severity: str = "information") -> None:
            with contextlib.suppress(Exception):
                self.notify(message, severity=severity, timeout=2.5)

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
                pid=(self._proc.pid if self._proc is not None else None),
                session_id=self._run_session_id,
                prompt=f"(copy) {label}",
                auto_approve=False,
                exit_code=0,
                output_text=payload,
            )
            if artifact_path:
                self._add_artifact_paths([artifact_path])
                self._write_transcript_line(
                    f"[copy] {label}: clipboard unavailable. Saved to artifact: {artifact_path}"
                )
            else:
                self._write_transcript_line(f"[copy] {label}: clipboard unavailable.")

        def _get_transcript_text(self) -> str:
            transcript = self.query_one("#transcript", VerticalScroll)
            lines: list[str] = []
            for child in transcript.children:
                if isinstance(child, AssistantMessage):
                    lines.append(child.full_text)
                elif isinstance(child, UserMessage):
                    lines.append(str(child.renderable))
                else:
                    lines.append(str(child.renderable))
            return "\n".join(lines).rstrip() + "\n"

        def _get_all_text(self) -> str:
            parts = [
                "# Transcript",
                self._get_transcript_text().rstrip(),
                "",
                "# Plan",
                (self._plan_text or "").rstrip() or "(no plan)",
                "",
                "# Issues",
                "\n".join(self._issues_lines).rstrip() or "(no issues)",
                "",
                "# Artifacts",
                "\n".join(self._artifacts).rstrip() or "(no artifacts)",
                "",
                "# Consent",
                "\n".join(self._consent_buffer).rstrip() or "(no consent buffer)",
                "",
            ]
            return "\n".join(parts).rstrip() + "\n"

        def _render_artifacts_panel(self) -> None:
            panel = self.query_one("#artifacts", TextArea)
            if not self._artifacts:
                panel.load_text("(no artifacts yet)")
            else:
                lines = [f"{i + 1}. {path}" for i, path in enumerate(self._artifacts)]
                panel.load_text("\n".join(lines))
            panel.scroll_end(animate=False)

        def _reset_artifacts_panel(self) -> None:
            self._artifacts = []
            self._render_artifacts_panel()

        def _add_artifact_paths(self, paths: list[str]) -> None:
            updated = add_recent_artifacts(self._artifacts, paths, max_items=20)
            if updated != self._artifacts:
                self._artifacts = updated
                self._render_artifacts_panel()

        def _render_consent_panel(self) -> None:
            consent_panel = self.query_one("#consent", TextArea)
            if not self._consent_active:
                consent_panel.load_text("(no active consent prompt)")
                return
            lines = self._consent_buffer[-10:] + [
                "",
                "[y] yes  [n] no  [a] always(session)  [v] never(session)",
            ]
            consent_panel.load_text("\n".join(lines))
            consent_panel.scroll_end(animate=False)

        def _reset_consent_panel(self) -> None:
            self._consent_active = False
            self._consent_buffer = []
            self._render_consent_panel()

        def _apply_consent_capture(self, line: str) -> None:
            next_active, next_buffer = update_consent_capture(
                self._consent_active,
                self._consent_buffer,
                line,
                max_lines=20,
            )
            if next_active != self._consent_active or next_buffer != self._consent_buffer:
                self._consent_active = next_active
                self._consent_buffer = next_buffer
                self._render_consent_panel()

        def _submit_consent_choice(self, choice: str) -> None:
            normalized_choice = choice.strip().lower()
            if normalized_choice not in _CONSENT_CHOICES:
                self._write_transcript_line("Usage: /consent <y|n|a|v>")
                return
            if not self._consent_active:
                self._write_transcript_line("[consent] no active prompt.")
                return
            if self._proc is None or self._proc.poll() is not None:
                self._write_transcript_line("[consent] process is not running.")
                self._reset_consent_panel()
                return
            if not write_to_proc(self._proc, normalized_choice):
                self._write_transcript_line("[consent] failed to send response (stdin unavailable).")
                self._reset_consent_panel()
                return
            self._reset_consent_panel()
            self.query_one("#prompt", TextArea).focus()

        def _handle_output_line(self, line: str) -> None:
            # Try structured JSONL first (emitted by TuiCallbackHandler).
            tui_event = parse_tui_event(line)
            if tui_event is not None:
                self._handle_tui_event(tui_event)
                return

            # Legacy fallback for non-JSON lines (stderr leakage, library warnings, etc.).
            sanitized = sanitize_output_text(line)
            event = parse_output_line(sanitized)
            if event is None:
                if sanitized.strip() == "return meta(":
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
                self._error_count += 1
                self._write_issue(text)
                self._update_header_status()
            elif event.kind == "warning":
                text = event.text
                if not text.startswith("WARN:"):
                    text = f"WARN: {text}"
                self._warning_count += 1
                self._write_issue(text)
                self._update_header_status()
            else:
                self._append_plain_text(event.text)

            artifact_paths = artifact_paths_from_event(event)
            if artifact_paths:
                self._add_artifact_paths(artifact_paths)

            self._apply_consent_capture(sanitized)

        def _handle_tui_event(self, event: dict[str, Any]) -> None:
            """Process a structured JSONL event from the subprocess."""
            etype = event.get("event", "")

            if etype == "text_delta":
                if self._current_thinking is not None:
                    self._current_thinking.remove()
                    self._current_thinking = None
                if self._current_assistant_msg is None:
                    self._current_assistant_msg = AssistantMessage()
                    self._mount_transcript_widget(self._current_assistant_msg)
                self._current_assistant_msg.append_delta(event.get("data", ""))
                self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

            elif etype == "text_complete":
                if self._current_assistant_msg is not None:
                    self._last_assistant_text = self._current_assistant_msg.finalize()
                    self._current_assistant_msg = None

            elif etype == "thinking":
                if self._current_thinking is None:
                    self._current_thinking = ThinkingIndicator()
                    self._mount_transcript_widget(self._current_thinking)

            elif etype == "tool_start":
                tid = event.get("tool_use_id", "")
                tool_name = event.get("tool", "unknown")
                block = ToolCallBlock(tool_name=tool_name, tool_use_id=tid)
                self._tool_blocks[tid] = block
                self._mount_transcript_widget(block)
                self._run_tool_count += 1
                if self._status_bar is not None:
                    self._status_bar.set_tool_count(self._run_tool_count)

            elif etype == "tool_progress":
                tid = event.get("tool_use_id", "")
                block = self._tool_blocks.get(tid)
                if block is not None:
                    block.update_progress(event.get("chars", 0))

            elif etype == "tool_input":
                tid = event.get("tool_use_id", "")
                block = self._tool_blocks.get(tid)
                if block is not None:
                    block.set_input(event.get("input", {}))

            elif etype == "tool_result":
                tid = event.get("tool_use_id", "")
                block = self._tool_blocks.get(tid)
                if block is not None:
                    block.set_result(event.get("status", "unknown"), event.get("duration_s", 0))
                # Auto-check plan steps sequentially
                if self._current_plan_card is not None:
                    self._current_plan_card.mark_step_complete(self._plan_step_counter)
                    self._plan_step_counter += 1

            elif etype == "consent_prompt":
                self._consent_active = True
                self._consent_buffer = [event.get("context", "")]
                card = ConsentCard(
                    context=event.get("context", ""),
                    options=event.get("options", ["y", "n", "a", "v"]),
                )
                self._mount_transcript_widget(card)
                self._render_consent_panel()

            elif etype == "plan":
                rendered = event.get("rendered", "")
                plan_json = event.get("plan_json")
                if plan_json and not rendered:
                    rendered = _json.dumps(plan_json, indent=2)
                self._set_plan_panel(rendered)
                self._received_structured_plan = True
                if not self._last_run_auto_approve and self._last_prompt:
                    self._pending_plan_prompt = self._last_prompt
                if plan_json and isinstance(plan_json, dict):
                    card = PlanCard(plan_json=plan_json)
                    self._current_plan_card = card
                    self._mount_transcript_widget(card)

            elif etype == "artifact":
                paths = event.get("paths", [])
                if paths:
                    self._add_artifact_paths(paths)

            elif etype == "error":
                error_text = event.get("text", "")
                if not error_text.startswith("ERROR:"):
                    error_text = f"ERROR: {error_text}"
                self._error_count += 1
                self._write_issue(error_text)
                self._update_header_status()

            elif etype == "warning":
                warn_text = event.get("text", "")
                if not warn_text.startswith("WARN:"):
                    warn_text = f"WARN: {warn_text}"
                self._warning_count += 1
                self._write_issue(warn_text)
                self._update_header_status()

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

        def _finalize_run(
            self,
            proc: subprocess.Popen[str],
            *,
            return_code: int,
            prompt: str,
            output_text: str,
            auto_approve: bool,
            session_id: str | None,
        ) -> None:
            if self._proc is not proc:
                return

            # Stop status timer and update status bar.
            if self._status_timer is not None:
                self._status_timer.stop()
                self._status_timer = None
            elapsed = time.time() - self._run_start_time if self._run_start_time is not None else 0.0
            if self._status_bar is not None:
                self._status_bar.set_state("idle")
                self._status_bar.set_elapsed(elapsed)
            self._run_start_time = None

            # Write run completion summary.
            self._write_transcript(
                f"[run] completed in {elapsed:.1f}s ({self._run_tool_count} tool calls, exit code {return_code})"
            )

            # Finalize any in-progress assistant message.
            if self._current_assistant_msg is not None:
                self._last_assistant_text = self._current_assistant_msg.finalize()
                self._current_assistant_msg = None
            if self._current_thinking is not None:
                self._current_thinking.remove()
                self._current_thinking = None

            transcript_path = self._persist_run_transcript(
                pid=proc.pid,
                session_id=session_id,
                prompt=prompt,
                auto_approve=auto_approve,
                exit_code=return_code,
                output_text=output_text,
            )
            if transcript_path:
                self._add_artifact_paths([transcript_path])

            log_path = self._discover_session_log_path(session_id)
            if log_path:
                self._add_artifact_paths([log_path])

            if not self._received_structured_plan:
                extracted_plan = extract_plan_section_from_output(sanitize_output_text(output_text))
                if extracted_plan:
                    self._pending_plan_prompt = prompt
                    self._set_plan_panel(extracted_plan)
                    self._write_transcript_line(render_tui_hint_after_plan())
            self._reset_consent_panel()
            self._proc = None
            self._runner_thread = None
            self._run_session_id = None
            self._received_structured_plan = False
            self._save_session()

        def _stream_output(self, proc: subprocess.Popen[str], prompt: str) -> None:
            output_chunks: list[str] = []
            if proc.stdout is None:
                self.call_from_thread(self._write_transcript_line, "[run] error: subprocess stdout unavailable.")
                return_code = proc.poll()
                self.call_from_thread(
                    self._finalize_run,
                    proc,
                    return_code=(return_code if return_code is not None else 1),
                    prompt=prompt,
                    output_text="",
                    auto_approve=self._last_run_auto_approve,
                    session_id=self._run_session_id,
                )
                return

            try:
                for raw_line in proc.stdout:
                    output_chunks.append(raw_line)
                    self.call_from_thread(self._handle_output_line, raw_line.rstrip("\n"))
            except Exception as exc:
                self.call_from_thread(self._write_transcript_line, f"[run] output stream error: {exc}")
            finally:
                with contextlib.suppress(Exception):
                    proc.stdout.close()
                return_code = proc.wait()
                self.call_from_thread(
                    self._finalize_run,
                    proc,
                    return_code=return_code,
                    prompt=prompt,
                    output_text="".join(output_chunks),
                    auto_approve=self._last_run_auto_approve,
                    session_id=self._run_session_id,
                )

        def _tick_status(self) -> None:
            if self._run_start_time is not None and self._status_bar is not None:
                self._status_bar.set_elapsed(time.time() - self._run_start_time)

        def _start_run(self, prompt: str, *, auto_approve: bool) -> None:
            self._pending_plan_prompt = None
            self._reset_artifacts_panel()
            self._reset_consent_panel()
            self._reset_issues_panel()
            self._current_assistant_msg = None
            self._current_thinking = None
            self._tool_blocks = {}
            self._run_tool_count = 0
            self._run_start_time = time.time()
            self._plan_step_counter = 0
            self._received_structured_plan = False
            if self._status_bar is not None:
                self._status_bar.set_state("running")
                self._status_bar.set_tool_count(0)
                self._status_bar.set_elapsed(0.0)
                self._status_bar.set_model(self._current_model_summary())
            if self._status_timer is not None:
                self._status_timer.stop()
            self._status_timer = self.set_interval(1.0, self._tick_status)
            run_prompt = prompt if auto_approve else build_plan_mode_prompt(prompt)
            try:
                self._run_session_id = uuid.uuid4().hex
                proc = spawn_swarmee(
                    run_prompt,
                    auto_approve=auto_approve,
                    session_id=self._run_session_id,
                    env_overrides=self._model_env_overrides(),
                )
            except Exception as exc:
                if self._status_timer is not None:
                    self._status_timer.stop()
                    self._status_timer = None
                self._run_start_time = None
                self._run_session_id = None
                if self._status_bar is not None:
                    self._status_bar.set_state("idle")
                    self._status_bar.set_elapsed(0.0)
                self._write_transcript_line(f"[run] failed to start: {exc}")
                return

            self._proc = proc
            self._last_prompt = prompt
            self._last_run_auto_approve = auto_approve
            self._runner_thread = threading.Thread(
                target=self._stream_output,
                args=(proc, prompt),
                daemon=True,
                name="swarmee-tui-runner",
            )
            self._runner_thread.start()

        def _stop_run(self) -> None:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._write_transcript_line("[run] no active run.")
                if proc is not None and proc.poll() is not None:
                    self._proc = None
                return
            stop_process(proc)
            self._write_transcript_line("[run] stopped.")

        def action_quit(self) -> None:
            if self._proc is not None and self._proc.poll() is None:
                stop_process(self._proc)
                self._write_transcript_line("[run] stopped.")
            self._save_session()
            self.exit(return_code=0)

        def action_copy_transcript(self) -> None:
            self._copy_text(self._get_transcript_text(), label="transcript")

        def action_copy_plan(self) -> None:
            self._copy_text((self._plan_text or "").rstrip() + "\n", label="plan")

        def action_copy_issues(self) -> None:
            self._flush_issue_repeats()
            payload = ("\n".join(self._issues_lines).rstrip() + "\n") if self._issues_lines else ""
            self._copy_text(payload, label="issues")

        def action_focus_prompt(self) -> None:
            self.query_one("#prompt", PromptTextArea).focus()

        def action_submit_prompt(self) -> None:
            prompt_widget = self.query_one("#prompt", PromptTextArea)
            text = (prompt_widget.text or "").strip()
            prompt_widget.clear()
            if text:
                self._prompt_history.append(text)
                if len(self._prompt_history) > self._MAX_PROMPT_HISTORY:
                    self._prompt_history = self._prompt_history[-self._MAX_PROMPT_HISTORY:]
                self._history_index = -1
                self._handle_user_input(text)

        def action_interrupt_run(self) -> None:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return
            stop_process(proc)
            self._write_transcript_line("[run] interrupted.")

        def action_copy_selection(self) -> None:
            focused = getattr(self, "focused", None)
            if not isinstance(focused, TextArea):
                self._notify("No text area focused.", severity="warning")
                return
            selected_text = focused.selected_text or ""
            if not selected_text.strip():
                self._notify("Select text first.", severity="warning")
                return
            self._copy_text(selected_text, label="selection")

        def action_widen_side(self) -> None:
            if self._split_ratio > 1:
                self._split_ratio -= 1
                self._apply_split_ratio()

        def action_widen_transcript(self) -> None:
            if self._split_ratio < 4:
                self._split_ratio += 1
                self._apply_split_ratio()

        def _apply_split_ratio(self) -> None:
            transcript = self.query_one("#transcript", VerticalScroll)
            side = self.query_one("#side", Vertical)
            transcript.styles.width = f"{self._split_ratio}fr"
            side.styles.width = "1fr"
            self.refresh(layout=True)

        def action_search_transcript(self) -> None:
            prompt_widget = self.query_one("#prompt", PromptTextArea)
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
            if not term.strip():
                self._write_transcript_line("Usage: /search <term>")
                return
            transcript = self.query_one("#transcript", VerticalScroll)
            term_lower = term.lower()
            for child in transcript.children:
                text = ""
                if isinstance(child, AssistantMessage):
                    text = child.full_text
                elif hasattr(child, "renderable"):
                    text = str(child.renderable)
                if term_lower in text.lower():
                    child.scroll_visible(animate=True)
                    self._write_transcript_line(f"[search] found match in transcript.")
                    return
            self._write_transcript_line(f"[search] no match for '{term}'.")

        def _open_artifact(self, index_str: str) -> None:
            try:
                index = int(index_str.strip()) - 1
            except ValueError:
                self._write_transcript_line("Usage: /open <number>")
                return
            if index < 0 or index >= len(self._artifacts):
                self._write_transcript_line(f"[open] invalid index. {len(self._artifacts)} artifacts available.")
                return
            path = self._artifacts[index]
            editor = os.environ.get("EDITOR", "")
            try:
                if editor:
                    subprocess.Popen([editor, path])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", path])
                elif shutil.which("xdg-open"):
                    subprocess.Popen(["xdg-open", path])
                else:
                    self._write_transcript_line(f"[open] set $EDITOR. Path: {path}")
                    return
                self._write_transcript_line(f"[open] opened: {path}")
            except Exception as exc:
                self._write_transcript_line(f"[open] failed: {exc}")

        def _save_session(self) -> None:
            try:
                session_path = sessions_dir() / "tui_session.json"
                session_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "prompt_history": self._prompt_history[-self._MAX_PROMPT_HISTORY:],
                    "last_prompt": self._last_prompt,
                    "plan_text": self._plan_text,
                    "artifacts": self._artifacts,
                    "model_provider_override": self._model_provider_override,
                    "model_tier_override": self._model_tier_override,
                    "default_auto_approve": self._default_auto_approve,
                    "split_ratio": self._split_ratio,
                }
                session_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass

        def _load_session(self) -> None:
            try:
                session_path = sessions_dir() / "tui_session.json"
                if not session_path.exists():
                    return
                data = _json.loads(session_path.read_text(encoding="utf-8", errors="replace"))
                self._prompt_history = data.get("prompt_history", [])[-self._MAX_PROMPT_HISTORY:]
                self._last_prompt = data.get("last_prompt")
                plan_text = data.get("plan_text", "")
                if plan_text and plan_text != "(no plan)":
                    self._set_plan_panel(plan_text)
                artifacts = data.get("artifacts", [])
                if artifacts:
                    self._artifacts = artifacts
                    self._render_artifacts_panel()
                self._model_provider_override = data.get("model_provider_override")
                self._model_tier_override = data.get("model_tier_override")
                self._default_auto_approve = data.get("default_auto_approve", False)
                self._split_ratio = data.get("split_ratio", 2)
                self._apply_split_ratio()
                self._refresh_model_select()
                self._update_header_status()
                self._update_prompt_placeholder()
                if self._status_bar is not None:
                    self._status_bar.set_model(self._current_model_summary())
                if self._prompt_history:
                    self._write_transcript(f"[session] restored ({len(self._prompt_history)} history entries).")
            except Exception:
                pass

        def on_key(self, event: Any) -> None:
            key = str(getattr(event, "key", "")).lower()

            if self._consent_active and key in _CONSENT_CHOICES:
                event.stop()
                event.prevent_default()
                self._submit_consent_choice(key)
                return

            if key in {"ctrl+c", "meta+c", "super+c", "cmd+c", "command+c"}:
                event.stop()
                event.prevent_default()
                self.action_copy_selection()
                return

        def on_select_changed(self, event: Any) -> None:
            if self._model_select_syncing:
                return
            select_widget = getattr(event, "select", None)
            if getattr(select_widget, "id", None) != "model_select":
                return

            value = str(getattr(event, "value", "")).strip()
            if not value:
                return
            if value == _MODEL_AUTO_VALUE:
                self._model_provider_override = None
                self._model_tier_override = None
            elif "|" in value:
                provider, tier = value.split("|", 1)
                self._model_provider_override = provider.strip().lower() or None
                self._model_tier_override = tier.strip().lower() or None
            self._update_header_status()
            self._update_prompt_placeholder()
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())
            # The model selector is always visible; avoid transient notifications.

        def _handle_user_input(self, text: str) -> None:
            self._write_user_input(text)

            normalized = text.lower()

            if normalized in {"/copy", ":copy"}:
                self.action_copy_transcript()
                return

            if normalized in {"/copy plan", ":copy plan"}:
                self.action_copy_plan()
                return

            if normalized in {"/copy issues", ":copy issues"}:
                self.action_copy_issues()
                return

            if normalized in {"/copy last", ":copy last"}:
                self._copy_text(self._last_assistant_text, label="last response")
                return

            if normalized in {"/copy all", ":copy all"}:
                self._copy_text(self._get_all_text(), label="all")
                return

            if normalized.startswith("/open "):
                self._open_artifact(text[len("/open "):])
                return

            if normalized == "/open":
                self._write_transcript_line("Usage: /open <number>")
                return

            if normalized.startswith("/search "):
                self._search_transcript(text[len("/search "):])
                return

            if normalized == "/search":
                self._write_transcript_line("Usage: /search <term>")
                return

            if normalized in {"/stop", ":stop"}:
                self._stop_run()
                return

            if normalized in {"/exit", ":exit"}:
                if self._proc is not None and self._proc.poll() is None:
                    stop_process(self._proc)
                    self._write_transcript_line("[run] stopped.")
                self._save_session()
                self.exit(return_code=0)
                return

            if normalized == "/consent":
                self._write_transcript_line("Usage: /consent <y|n|a|v>")
                return

            if normalized.startswith("/consent "):
                choice = normalized.split(maxsplit=1)[1].strip()
                self._submit_consent_choice(choice)
                return

            if normalized == "/model":
                self._write_transcript_line(self._current_model_summary())
                self._write_transcript_line(
                    "Usage: /model show | /model list | /model provider <name> | /model tier <name> | /model reset"
                )
                return

            if normalized == "/model show":
                self._write_transcript_line(self._current_model_summary())
                return

            if normalized == "/model list":
                options, _ = model_select_options(
                    provider_override=self._model_provider_override,
                    tier_override=self._model_tier_override,
                )
                for label, _value in options:
                    self._write_transcript_line(f"- {label}")
                return

            if normalized == "/model reset":
                self._model_provider_override = None
                self._model_tier_override = None
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] reset. {self._current_model_summary()}")
                return

            if normalized.startswith("/model provider "):
                provider = normalized.split(maxsplit=2)[2].strip()
                if not provider:
                    self._write_transcript_line("Usage: /model provider <name>")
                    return
                self._model_provider_override = provider
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] provider set to {provider}.")
                self._write_transcript_line(self._current_model_summary())
                return

            if normalized.startswith("/model tier "):
                tier = normalized.split(maxsplit=2)[2].strip()
                if not tier:
                    self._write_transcript_line("Usage: /model tier <name>")
                    return
                self._model_tier_override = tier
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] tier set to {tier}.")
                self._write_transcript_line(self._current_model_summary())
                return

            if self._proc is not None and self._proc.poll() is None:
                self._write_transcript_line("[run] already running; use /stop.")
                return

            if normalized == "/approve":
                if not self._pending_plan_prompt:
                    self._write_transcript_line("[run] no pending plan.")
                    return
                self._start_run(self._pending_plan_prompt, auto_approve=True)
                return

            if normalized == "/replan":
                if not self._last_prompt:
                    self._write_transcript_line("[run] no previous prompt to replan.")
                    return
                self._start_run(self._last_prompt, auto_approve=False)
                return

            if normalized == "/clearplan":
                self._pending_plan_prompt = None
                self._reset_plan_panel()
                self._write_transcript_line("[run] plan cleared.")
                return

            if normalized == "/plan":
                self._default_auto_approve = False
                self._update_prompt_placeholder()
                self._write_transcript_line("[mode] default set to plan. Type a prompt to generate a plan.")
                return

            if text.startswith("/plan "):
                prompt = text[len("/plan ") :].strip()
                if not prompt:
                    self._write_transcript_line("Usage: /plan <prompt>")
                    return
                self._start_run(prompt, auto_approve=False)
                return

            if normalized == "/run":
                self._default_auto_approve = True
                self._update_prompt_placeholder()
                self._write_transcript_line("[mode] default set to execute. Type a prompt to run immediately.")
                return

            if text.startswith("/run "):
                prompt = text[len("/run ") :].strip()
                if not prompt:
                    self._write_transcript_line("Usage: /run <prompt>")
                    return
                self._start_run(prompt, auto_approve=True)
                return

            if text.startswith("/") or text.startswith(":"):
                self._write_transcript_line(f"[run] unknown command: {text}")
                return

            self._start_run(text, auto_approve=self._default_auto_approve)

    try:
        SwarmeeTUI().run()
    except KeyboardInterrupt:
        return 130
    return 0
