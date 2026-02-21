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
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.state_paths import logs_dir, sessions_dir

_CONSENT_CHOICES = {"y", "n", "a", "v"}
_MODEL_AUTO_VALUE = "__auto__"
_MODEL_LOADING_VALUE = "__loading__"
_TRUNCATED_ARTIFACT_RE = re.compile(r"full output saved to (?P<path>[^\]]+)")
_PATH_TOKEN_RE = re.compile(r"[A-Za-z]:\\[^\s,;]+|/(?:[^\s,;]+)|\./[^\s,;]+|\.\./[^\s,;]+")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC_ESCAPE_RE = re.compile(r"\x1b\][^\x1b\x07]*(?:\x07|\x1b\\)")
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


def build_swarmee_daemon_cmd() -> list[str]:
    """Build a subprocess command for a long-running Swarmee daemon."""
    return [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--tui-daemon"]


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
    cleaned = _OSC_ESCAPE_RE.sub("", cleaned)
    cleaned = _ANSI_ESCAPE_RE.sub("", cleaned)
    # Remove any stray ESC bytes left by malformed or partial sequences.
    return cleaned.replace("\x1b", "")


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


def choose_daemon_model_select_value(
    *,
    provider: str,
    tier: str,
    option_values: list[str],
    pending_value: str | None = None,
    override_provider: str | None = None,
    override_tier: str | None = None,
) -> str | None:
    """Choose which model-select value should be shown for daemon-backed options."""
    available = [str(v).strip().lower() for v in option_values if isinstance(v, str) and str(v).strip()]
    if not available:
        return None

    provider_name = (provider or "").strip().lower()
    tier_name = (tier or "").strip().lower()

    pending = (pending_value or "").strip().lower()
    if pending and pending in available:
        return pending

    override_provider_name = (override_provider or "").strip().lower()
    override_tier_name = (override_tier or "").strip().lower()
    if provider_name and override_provider_name == provider_name and override_tier_name:
        override_value = f"{provider_name}|{override_tier_name}"
        if override_value in available:
            return override_value

    daemon_value = f"{provider_name}|{tier_name}" if provider_name and tier_name else ""
    if daemon_value and daemon_value in available:
        return daemon_value
    return available[0]


def choose_model_summary_parts(
    *,
    daemon_provider: str | None,
    daemon_tier: str | None,
    daemon_model_id: str | None,
    daemon_tiers: list[dict[str, Any]] | None = None,
    pending_value: str | None = None,
    override_provider: str | None = None,
    override_tier: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Choose provider/tier/model_id for top-level model summary display."""
    def _lookup_model_id(provider_name: str, tier_name: str) -> str | None:
        tiers = daemon_tiers if isinstance(daemon_tiers, list) else []
        for item in tiers:
            item_provider = str(item.get("provider", "")).strip().lower()
            item_tier = str(item.get("name", "")).strip().lower()
            if item_provider != provider_name or item_tier != tier_name:
                continue
            model_id_value = str(item.get("model_id", "")).strip()
            if model_id_value:
                return model_id_value
        return None

    pending = (pending_value or "").strip().lower()
    if pending and "|" in pending:
        pending_provider, pending_tier = pending.split("|", 1)
        pending_provider = pending_provider.strip().lower()
        pending_tier = pending_tier.strip().lower()
        if pending_provider and pending_tier:
            # Keep selected tier stable in header while daemon confirmation is in-flight.
            if (
                daemon_provider
                and daemon_tier
                and daemon_model_id
                and pending_provider == daemon_provider.strip().lower()
                and pending_tier == daemon_tier.strip().lower()
            ):
                return pending_provider, pending_tier, daemon_model_id
            return pending_provider, pending_tier, _lookup_model_id(pending_provider, pending_tier)

    override_provider_name = (override_provider or "").strip().lower()
    override_tier_name = (override_tier or "").strip().lower()
    if override_provider_name and override_tier_name:
        if (
            daemon_provider
            and daemon_tier
            and daemon_model_id
            and override_provider_name == daemon_provider.strip().lower()
            and override_tier_name == daemon_tier.strip().lower()
        ):
            return override_provider_name, override_tier_name, daemon_model_id
        return override_provider_name, override_tier_name, _lookup_model_id(override_provider_name, override_tier_name)

    provider_name = (daemon_provider or "").strip().lower()
    tier_name = (daemon_tier or "").strip().lower()
    if provider_name and tier_name:
        model_id = str(daemon_model_id).strip() if daemon_model_id is not None else None
        if not model_id:
            model_id = _lookup_model_id(provider_name, tier_name)
        return provider_name, tier_name, (model_id or None)
    return None, None, None


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
    stripped = sanitize_output_text(line).strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = _json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, _json.JSONDecodeError):
        return None


def extract_tui_text_chunk(event: dict[str, Any]) -> str:
    """Extract a text chunk from a structured TUI event payload."""
    for key in ("data", "text", "delta", "content", "output_text", "outputText", "textDelta"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return ""


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


def send_daemon_command(proc: subprocess.Popen[str], cmd_dict: dict[str, Any]) -> bool:
    """Serialize and send a daemon command as JSONL."""
    if proc.stdin is None:
        return False
    try:
        payload = _json.dumps(cmd_dict, ensure_ascii=False) + "\n"
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
        # Isolate child subprocesses from the interactive terminal session to
        # prevent terminal-title churn while tools (git/rg/etc.) execute.
        start_new_session=True,
    )


def spawn_swarmee_daemon(
    *,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Spawn Swarmee daemon with line-buffered merged output."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["SWARMEE_SPINNERS"] = "0"
    env["SWARMEE_TUI_EVENTS"] = "1"
    existing_warning_filters = env.get("PYTHONWARNINGS", "").strip()
    tui_warning_filters = [
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
        build_swarmee_daemon_cmd(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        # Keep daemon and all descendants in a separate session/process-group so
        # macOS Terminal does not treat transient tool commands as foreground title context.
        start_new_session=True,
    )


def stop_process(proc: subprocess.Popen[str], *, timeout_s: float = 2.0) -> None:
    """Stop a running subprocess, escalating from interrupt to terminate to kill."""
    if proc.poll() is not None:
        return

    if os.name == "posix" and hasattr(signal, "SIGINT"):
        signaled = False
        if hasattr(os, "killpg"):
            with contextlib.suppress(Exception):
                os.killpg(proc.pid, signal.SIGINT)
                signaled = True
        if not signaled:
            with contextlib.suppress(Exception):
                proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass

    terminated = False
    if os.name == "posix" and hasattr(os, "killpg") and hasattr(signal, "SIGTERM"):
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGTERM)
            terminated = True
    if not terminated:
        with contextlib.suppress(Exception):
            proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return

    killed = False
    if os.name == "posix" and hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
            killed = True
    if not killed:
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
        PlanActions,
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
        $accent: #6a9955;
        $primary: #6a9955;
        $footer-key-foreground: #6a9955;
        $footer-key-background: #2f2f2f;
        $footer-description-background: #1f1f1f;
        $footer-item-background: #1f1f1f;
        $footer-background: #1f1f1f;

        Screen {
            layout: vertical;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
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
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
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
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        #plan_actions {
            height: auto;
        }

        #consent {
            height: 8;
            border: round $accent;
            padding: 0 1;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
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
            align: left middle;
        }

        #prompt_metrics {
            width: auto;
            color: $text-muted;
        }

        #prompt_spacer {
            width: 1fr;
        }

        #model_select {
            width: 34;
            min-width: 24;
        }

        Footer {
            color: $text-muted;
            background: #1f1f1f;
        }

        Footer FooterKey {
            background: #1f1f1f;
        }

        Footer FooterKey .footer-key--key {
            color: $accent;
            background: #2f2f2f;
        }

        Footer FooterKey .footer-key--description {
            color: $text-muted;
            background: #1f1f1f;
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
        _pending_model_select_value: str | None = None
        # Conversation view state
        _current_assistant_msg: Any = None  # AssistantMessage | None
        _current_thinking: Any = None  # ThinkingIndicator | None
        _tool_blocks: dict[str, Any] = {}  # tool_use_id → ToolCallBlock
        _current_plan_card: Any = None  # PlanCard | None
        _command_palette: Any = None  # CommandPalette | None
        _status_bar: Any = None  # StatusBar | None
        _prompt_metrics: Any = None  # Static | None
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
        _plan_completion_announced: bool = False
        _transcript_widget_count: int = 0
        _MAX_TRANSCRIPT_WIDGETS: int = 500
        _received_structured_plan: bool = False
        _daemon_ready: bool = False
        _query_active: bool = False
        _daemon_tiers: list[dict[str, Any]] = []
        _daemon_provider: str | None = None
        _daemon_tier: str | None = None
        _daemon_model_id: str | None = None
        _current_daemon_model: str | None = None
        _turn_output_chunks: list[str] = []
        _daemon_session_id: str | None = None
        _is_shutting_down: bool = False
        _last_usage: dict[str, Any] | None = None
        _last_cost_usd: float | None = None
        _last_prompt_tokens_est: int | None = None
        _last_budget_tokens: int | None = None
        _help_text: str = ""

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
                            yield PlanActions(id="plan_actions")
                        with TabPane("Help", id="tab_help"):
                            yield TextArea(
                                text="",
                                read_only=True,
                                show_cursor=False,
                                id="help",
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
                    yield Static("", id="prompt_metrics")
                    yield Static("", id="prompt_spacer")
                    yield Select(
                        options=[("Loading model info...", _MODEL_LOADING_VALUE)],
                        allow_blank=False,
                        id="model_select",
                        compact=True,
                    )
            yield Footer()

        def on_mount(self) -> None:
            self._command_palette = self.query_one("#command_palette", CommandPalette)
            self._status_bar = self.query_one("#status_bar", StatusBar)
            self._prompt_metrics = self.query_one("#prompt_metrics", Static)
            self._status_bar.set_model(self._current_model_summary())
            self.query_one("#prompt", PromptTextArea).focus()
            self._reset_help_panel()
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
            self._write_transcript("Starting Swarmee daemon...")
            self._write_transcript(self.sub_title)
            self._write_transcript("Tips: see the Help tab for commands/keys/copy shortcuts.")
            self._set_help_panel(
                "\n".join(
                    [
                        "Commands:",
                        "- /plan <prompt>, /run <prompt>",
                        "- /approve, /replan, /clearplan",
                        "- /model, /stop, /daemon restart, /exit",
                        "",
                        "Consent:",
                        "- /consent <y|n|a|v> (or press y/n/a/v when prompted)",
                        "",
                        "Keys:",
                        "- Enter submit, Shift+Enter newline (Ctrl+J/Alt+Enter fallback)",
                        "- Ctrl+Left/Right (or F6/F7) resize panes, F5 submit",
                        "- Esc interrupt run, Ctrl+C/Cmd+C copy selection",
                        "",
                        "Copy/export:",
                        "- /copy, /copy plan, /copy issues, /copy all",
                        "",
                        "Notes:",
                        "- Model selector is in the prompt box footer dropdown.",
                        "- In transcript view, use Ctrl+C/Cmd+C (or /copy) to export text.",
                    ]
                )
            )
            self._load_session()
            self._spawn_daemon()

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

        def _dismiss_thinking(self) -> None:
            """Immediately hide and discard the current thinking indicator."""
            indicator = self._current_thinking
            if indicator is None:
                return
            self._current_thinking = None
            # Stop the animation timer so it doesn't fire after removal.
            timer = getattr(indicator, "_timer", None)
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()
            # Hide synchronously, then schedule async removal.
            indicator.display = False
            indicator.remove()

        def _write_transcript_line(self, line: str) -> None:
            """Write a plain text line to the transcript (used for TUI-internal messages)."""
            self._write_transcript(line)

        def _call_from_thread_safe(self, callback: Any, *args: Any, **kwargs: Any) -> None:
            if self._is_shutting_down:
                return
            with contextlib.suppress(Exception):
                self.call_from_thread(callback, *args, **kwargs)

        def _write_user_input(self, text: str) -> None:
            self._mount_transcript_widget(UserMessage(text, timestamp=self._turn_timestamp()))

        def _turn_timestamp(self) -> str:
            return datetime.now().strftime("%I:%M %p").lstrip("0")

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

        def _set_help_panel(self, content: str) -> None:
            self._help_text = content
            panel = self.query_one("#help", TextArea)
            text = content if content.strip() else "(no help yet)"
            panel.load_text(text)
            panel.scroll_home(animate=False)

        def _reset_help_panel(self) -> None:
            self._set_help_panel("(help)")

        def _reset_plan_panel(self) -> None:
            self._set_plan_panel("(no plan)")
            self._current_plan_card = None
            self._plan_step_counter = 0
            self._plan_completion_announced = False

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
            provider_name, tier_name, model_id = choose_model_summary_parts(
                daemon_provider=self._daemon_provider,
                daemon_tier=self._daemon_tier,
                daemon_model_id=self._daemon_model_id,
                daemon_tiers=self._daemon_tiers,
                pending_value=self._pending_model_select_value,
                override_provider=self._model_provider_override,
                override_tier=self._model_tier_override,
            )
            if provider_name and tier_name:
                suffix = f" ({model_id})" if model_id else ""
                return f"Model: {provider_name}/{tier_name}{suffix}"
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
            if self._daemon_provider and self._daemon_tier and self._daemon_tiers:
                self._refresh_model_select_from_daemon(
                    provider=self._daemon_provider,
                    tier=self._daemon_tier,
                    tiers=self._daemon_tiers,
                )
                return

            selector = self.query_one("#model_select", Select)
            options, selected_value = self._model_select_options()
            self._model_select_syncing = True
            try:
                selector.set_options(options)
                with contextlib.suppress(Exception):
                    selector.value = selected_value
            finally:
                self._model_select_syncing = False

        def _refresh_model_select_from_daemon(
            self,
            *,
            provider: str,
            tier: str,
            tiers: list[dict[str, Any]],
        ) -> None:
            selector = self.query_one("#model_select", Select)
            provider_name = (provider or "").strip().lower()

            options: list[tuple[str, str]] = []
            for item in tiers:
                item_provider = str(item.get("provider", "")).strip().lower()
                item_tier = str(item.get("name", "")).strip().lower()
                if not item_tier or item_provider != provider_name:
                    continue
                if not bool(item.get("available", False)):
                    continue
                model_id = str(item.get("model_id", "")).strip()
                suffix = f" ({model_id})" if model_id else ""
                value = f"{item_provider}|{item_tier}"
                options.append((f"{item_provider}/{item_tier}{suffix}", value))

            if not options:
                options = [("No available tiers", _MODEL_LOADING_VALUE)]
                selected_value = _MODEL_LOADING_VALUE
            else:
                selected_value = choose_daemon_model_select_value(
                    provider=provider_name,
                    tier=tier,
                    option_values=[value for _label, value in options],
                    pending_value=self._pending_model_select_value,
                    override_provider=self._model_provider_override,
                    override_tier=self._model_tier_override,
                )
                if selected_value is None:
                    selected_value = options[0][1]

            self._model_select_syncing = True
            try:
                selector.set_options(options)
                with contextlib.suppress(Exception):
                    selector.value = selected_value
            finally:
                self._model_select_syncing = False

        def _model_select_options(self) -> tuple[list[tuple[str, str]], str]:
            if self._daemon_tiers and self._daemon_provider:
                options: list[tuple[str, str]] = []
                selected_value: str | None = None
                for tier in self._daemon_tiers:
                    tier_name = str(tier.get("name", "")).strip().lower()
                    provider_name = str(tier.get("provider", "")).strip().lower()
                    if not tier_name or provider_name != self._daemon_provider:
                        continue
                    if not bool(tier.get("available", False)):
                        continue
                    model_id = str(tier.get("model_id", "")).strip()
                    suffix = f" ({model_id})" if model_id else ""
                    value = f"{provider_name}|{tier_name}"
                    options.append((f"{provider_name}/{tier_name}{suffix}", value))
                    if tier_name == (self._daemon_tier or ""):
                        selected_value = value
                if not options:
                    return [("No available tiers", _MODEL_LOADING_VALUE)], _MODEL_LOADING_VALUE
                if selected_value is None:
                    selected_value = options[0][1]
                return options, selected_value
            return model_select_options(
                provider_override=self._model_provider_override,
                tier_override=self._model_tier_override,
            )

        def _handle_model_info(self, event: dict[str, Any]) -> None:
            provider = str(event.get("provider", "")).strip().lower()
            tier = str(event.get("tier", "")).strip().lower()
            model_id = event.get("model_id")
            tiers = event.get("tiers")

            self._daemon_provider = provider or None
            self._daemon_tier = tier or None
            self._daemon_model_id = str(model_id).strip() if model_id is not None and str(model_id).strip() else None
            self._daemon_tiers = tiers if isinstance(tiers, list) else []
            self._current_daemon_model = self._daemon_model_id or (
                f"{self._daemon_provider}/{self._daemon_tier}" if self._daemon_provider and self._daemon_tier else None
            )
            pending_value = (self._pending_model_select_value or "").strip().lower()
            if pending_value and "|" in pending_value:
                pending_provider, pending_tier = pending_value.split("|", 1)
                if pending_provider == provider and pending_tier == tier:
                    self._pending_model_select_value = None

            if self._daemon_provider and self._daemon_tier:
                self._refresh_model_select_from_daemon(
                    provider=self._daemon_provider,
                    tier=self._daemon_tier,
                    tiers=self._daemon_tiers,
                )
            else:
                self._refresh_model_select()

            self._update_header_status()
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())

        def _update_prompt_placeholder(self) -> None:
            input_widget = self.query_one("#prompt", PromptTextArea)
            approval = "on" if self._default_auto_approve else "off"
            input_widget.placeholder = (
                f"Auto-approve: {approval}. Enter submits. Shift+Enter/Ctrl+J adds newline."
            )

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
                session_id=self._daemon_session_id,
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
                else:
                    # Textual Static implementations differ by version; prefer public
                    # attributes/methods and gracefully fall back when unavailable.
                    text = ""
                    with contextlib.suppress(Exception):
                        renderable = getattr(child, "renderable", None)
                        if renderable is not None:
                            text = str(renderable)
                    if not text:
                        with contextlib.suppress(Exception):
                            text = str(child.render())
                    if not text:
                        with contextlib.suppress(Exception):
                            text = str(getattr(child, "_content", ""))
                    if text:
                        lines.append(text)
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
                self._write_transcript_line("[consent] daemon is not running.")
                self._reset_consent_panel()
                return
            if not send_daemon_command(self._proc, {"cmd": "consent_response", "choice": normalized_choice}):
                self._write_transcript_line("[consent] failed to send response (stdin unavailable).")
                self._reset_consent_panel()
                return
            self._reset_consent_panel()
            self.query_one("#prompt", TextArea).focus()

        def _handle_output_line(self, line: str, raw_line: str | None = None) -> None:
            if self._query_active:
                chunk = raw_line if raw_line is not None else (line + "\n")
                self._turn_output_chunks.append(sanitize_output_text(chunk))
            sanitized = sanitize_output_text(line)
            # Try structured JSONL first (emitted by TuiCallbackHandler).
            tui_event = parse_tui_event(sanitized)
            if tui_event is not None:
                self._handle_tui_event(tui_event)
                return

            # Legacy fallback for non-JSON lines (stderr leakage, library warnings, etc.).
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

            if etype == "ready":
                self._daemon_ready = True
                session_id = str(event.get("session_id", "")).strip()
                if session_id:
                    self._daemon_session_id = session_id
                self._write_transcript("Swarmee daemon ready. Enter a prompt to run Swarmee.")

            elif etype == "turn_complete":
                self._finalize_turn(exit_status=str(event.get("exit_status", "ok")))

            elif etype == "model_info":
                self._handle_model_info(event)

            elif etype == "context":
                prompt_tokens_est = event.get("prompt_tokens_est")
                budget_tokens = event.get("budget_tokens")
                self._last_prompt_tokens_est = int(prompt_tokens_est) if isinstance(prompt_tokens_est, int) else None
                self._last_budget_tokens = int(budget_tokens) if isinstance(budget_tokens, int) else None
                if self._status_bar is not None:
                    self._status_bar.set_context(
                        prompt_tokens_est=self._last_prompt_tokens_est,
                        budget_tokens=self._last_budget_tokens,
                    )
                self._refresh_prompt_metrics()

            elif etype == "usage":
                usage = event.get("usage")
                self._last_usage = usage if isinstance(usage, dict) else None
                cost = event.get("cost_usd")
                self._last_cost_usd = float(cost) if isinstance(cost, (int, float)) else None
                if self._status_bar is not None:
                    self._status_bar.set_usage(self._last_usage, cost_usd=self._last_cost_usd)
                self._refresh_prompt_metrics()

            elif etype in {"text_delta", "message_delta", "output_text_delta", "delta"}:
                chunk = sanitize_output_text(extract_tui_text_chunk(event))
                if not chunk:
                    return
                self._dismiss_thinking()
                if self._current_assistant_msg is None:
                    self._current_assistant_msg = AssistantMessage(
                        model=self._current_daemon_model,
                        timestamp=self._turn_timestamp(),
                    )
                    self._mount_transcript_widget(self._current_assistant_msg)
                self._current_assistant_msg.append_delta(chunk)
                self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

            elif etype in {"text_complete", "message_complete", "output_text_complete", "complete"}:
                if self._current_assistant_msg is not None:
                    self._last_assistant_text = self._current_assistant_msg.finalize()
                    self._current_assistant_msg = None

            elif etype == "thinking":
                if self._current_thinking is None:
                    self._current_thinking = ThinkingIndicator()
                    self._mount_transcript_widget(self._current_thinking)

            elif etype == "tool_start":
                self._dismiss_thinking()
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
                    step_status = getattr(self._current_plan_card, "_step_status", [])
                    if (
                        isinstance(step_status, list)
                        and step_status
                        and all(bool(item) for item in step_status)
                        and not self._plan_completion_announced
                    ):
                        self._plan_completion_announced = True
                        self._write_transcript_line("[plan] all steps complete. Clear plan?")

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
                self._plan_completion_announced = False
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
                normalized_error = error_text.lower()
                if self._pending_model_select_value and (
                    "set_tier" in normalized_error
                    or "cannot set tier" in normalized_error
                    or "tier" in normalized_error
                ):
                    self._pending_model_select_value = None
                    self._refresh_model_select()
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

        def _finalize_turn(self, *, exit_status: str) -> None:
            if self._status_timer is not None:
                self._status_timer.stop()
                self._status_timer = None
            elapsed = time.time() - self._run_start_time if self._run_start_time is not None else 0.0
            if self._status_bar is not None:
                self._status_bar.set_state("idle")
                self._status_bar.set_elapsed(elapsed)
            self._run_start_time = None
            self._query_active = False

            self._write_transcript(
                f"[run] completed in {elapsed:.1f}s ({self._run_tool_count} tool calls, status={exit_status})"
            )

            if self._current_assistant_msg is not None:
                self._last_assistant_text = self._current_assistant_msg.finalize()
                self._current_assistant_msg = None
            self._dismiss_thinking()

            output_text = "".join(self._turn_output_chunks)
            self._turn_output_chunks = []
            transcript_path = self._persist_run_transcript(
                pid=(self._proc.pid if self._proc is not None else None),
                session_id=self._daemon_session_id,
                prompt=self._last_prompt or "",
                auto_approve=self._last_run_auto_approve,
                exit_code=0 if exit_status == "ok" else 1,
                output_text=output_text,
            )
            if transcript_path:
                self._add_artifact_paths([transcript_path])

            log_path = self._discover_session_log_path(self._daemon_session_id)
            if log_path:
                self._add_artifact_paths([log_path])

            if not self._received_structured_plan:
                extracted_plan = extract_plan_section_from_output(sanitize_output_text(output_text))
                if extracted_plan:
                    self._pending_plan_prompt = self._last_prompt
                    self._set_plan_panel(extracted_plan)
                    self._write_transcript_line(render_tui_hint_after_plan())

            self._reset_consent_panel()
            self._received_structured_plan = False
            self._save_session()

        def _handle_daemon_exit(self, proc: subprocess.Popen[str], *, return_code: int) -> None:
            if self._proc is not proc:
                return
            was_query_active = self._query_active
            self._daemon_ready = False
            self._pending_model_select_value = None
            self._query_active = False
            self._proc = None
            self._runner_thread = None

            if self._status_timer is not None:
                self._status_timer.stop()
                self._status_timer = None
            if self._status_bar is not None:
                self._status_bar.set_state("idle")

            if was_query_active:
                self._finalize_turn(exit_status="error")
            if self._is_shutting_down:
                return
            self._write_transcript_line(f"[daemon] exited unexpectedly (code {return_code}).")
            self._write_transcript_line("[daemon] run /daemon restart to restart the background agent.")

        def _stream_daemon_output(self, proc: subprocess.Popen[str]) -> None:
            if proc.stdout is None:
                self._call_from_thread_safe(
                    self._write_transcript_line,
                    "[daemon] error: subprocess stdout unavailable.",
                )
                return_code = proc.poll()
                self._call_from_thread_safe(
                    self._handle_daemon_exit,
                    proc,
                    return_code=(return_code if return_code is not None else 1),
                )
                return

            try:
                for raw_line in proc.stdout:
                    self._call_from_thread_safe(self._handle_output_line, raw_line.rstrip("\n"), raw_line)
            except Exception as exc:
                self._call_from_thread_safe(self._write_transcript_line, f"[daemon] output stream error: {exc}")
            finally:
                with contextlib.suppress(Exception):
                    proc.stdout.close()
                return_code = proc.wait()
                self._call_from_thread_safe(self._handle_daemon_exit, proc, return_code=return_code)

        def _spawn_daemon(self, *, restart: bool = False) -> None:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                if restart:
                    self._pending_model_select_value = None
                    send_daemon_command(proc, {"cmd": "shutdown"})
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=3.0)
                    if proc.poll() is None:
                        stop_process(proc)
                    self._proc = None
                else:
                    return

            try:
                self._daemon_session_id = uuid.uuid4().hex
                daemon = spawn_swarmee_daemon(
                    session_id=self._daemon_session_id,
                    env_overrides=self._model_env_overrides(),
                )
            except Exception as exc:
                self._daemon_ready = False
                self._write_transcript_line(f"[daemon] failed to start: {exc}")
                return

            self._proc = daemon
            self._daemon_ready = False
            self._runner_thread = threading.Thread(
                target=self._stream_daemon_output,
                args=(daemon,),
                daemon=True,
                name="swarmee-tui-daemon-stream",
            )
            self._runner_thread.start()
            self._write_transcript_line("[daemon] started, waiting for ready event.")

        def _tick_status(self) -> None:
            if self._run_start_time is not None and self._status_bar is not None:
                self._status_bar.set_elapsed(time.time() - self._run_start_time)

        def _start_run(self, prompt: str, *, auto_approve: bool, mode: str | None = None) -> None:
            if not self._daemon_ready:
                self._write_transcript_line("[run] daemon is not ready. Use /daemon restart.")
                return
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._write_transcript_line("[run] daemon is not running. Use /daemon restart.")
                self._daemon_ready = False
                return
            if self._query_active:
                self._write_transcript_line("[run] already running; use /stop.")
                return

            self._pending_plan_prompt = None
            self._reset_artifacts_panel()
            self._reset_consent_panel()
            self._reset_issues_panel()
            self._current_assistant_msg = None
            self._current_thinking = ThinkingIndicator()
            self._mount_transcript_widget(self._current_thinking)
            self._tool_blocks = {}
            self._run_tool_count = 0
            self._run_start_time = time.time()
            self._plan_step_counter = 0
            self._plan_completion_announced = False
            self._received_structured_plan = False
            self._turn_output_chunks = []
            self._last_usage = None
            self._last_cost_usd = None
            if self._status_bar is not None:
                self._status_bar.set_state("running")
                self._status_bar.set_tool_count(0)
                self._status_bar.set_elapsed(0.0)
                self._status_bar.set_model(self._current_model_summary())
                self._status_bar.set_usage(None, cost_usd=None)
                self._status_bar.set_context(
                    prompt_tokens_est=self._last_prompt_tokens_est,
                    budget_tokens=self._last_budget_tokens,
                )
            self._refresh_prompt_metrics()
            if self._status_timer is not None:
                self._status_timer.stop()
            self._status_timer = self.set_interval(1.0, self._tick_status)
            self._last_prompt = prompt
            self._last_run_auto_approve = auto_approve
            self._query_active = True
            command: dict[str, Any] = {
                "cmd": "query",
                "text": prompt,
                "auto_approve": bool(auto_approve),
            }
            if mode:
                command["mode"] = mode
            if not send_daemon_command(proc, command):
                self._query_active = False
                if self._status_timer is not None:
                    self._status_timer.stop()
                    self._status_timer = None
                if self._status_bar is not None:
                    self._status_bar.set_state("idle")
                self._write_transcript_line("[run] failed to send query to daemon.")

        def _stop_run(self) -> None:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._write_transcript_line("[run] no active run.")
                self._daemon_ready = False
                return
            if not self._query_active:
                self._write_transcript_line("[run] no active run.")
                return
            if send_daemon_command(proc, {"cmd": "interrupt"}):
                self._write_transcript_line("[run] interrupt requested.")
            else:
                self._write_transcript_line("[run] failed to send interrupt.")

        def _refresh_prompt_metrics(self) -> None:
            if self._prompt_metrics is None:
                return

            parts: list[str] = []
            if isinstance(self._last_prompt_tokens_est, int):
                if isinstance(self._last_budget_tokens, int) and self._last_budget_tokens > 0:
                    parts.append(f"ctx {self._last_prompt_tokens_est}/{self._last_budget_tokens}")
                else:
                    parts.append(f"ctx {self._last_prompt_tokens_est}")

            if self._last_usage:
                in_tokens = self._last_usage.get("input_tokens") or self._last_usage.get("prompt_tokens")
                out_tokens = self._last_usage.get("output_tokens") or self._last_usage.get("completion_tokens")
                if isinstance(in_tokens, int) or isinstance(out_tokens, int):
                    usage_parts: list[str] = []
                    if isinstance(in_tokens, int):
                        usage_parts.append(f"in {in_tokens}")
                    if isinstance(out_tokens, int):
                        usage_parts.append(f"out {out_tokens}")
                    parts.append(" ".join(usage_parts))
            if isinstance(self._last_cost_usd, (int, float)):
                parts.append(f"${self._last_cost_usd:.4f}")

            self._prompt_metrics.update(" | ".join(parts))

        def action_quit(self) -> None:
            self._is_shutting_down = True
            if self._proc is not None and self._proc.poll() is None:
                send_daemon_command(self._proc, {"cmd": "shutdown"})
                with contextlib.suppress(Exception):
                    self._proc.wait(timeout=3.0)
                if self._proc.poll() is None:
                    stop_process(self._proc)
            if self._runner_thread is not None and self._runner_thread.is_alive():
                with contextlib.suppress(Exception):
                    self._runner_thread.join(timeout=1.0)
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
            if proc is None or proc.poll() is not None or not self._query_active:
                return
            send_daemon_command(proc, {"cmd": "interrupt"})
            self._write_transcript_line("[run] interrupted.")

        def action_copy_selection(self) -> None:
            focused = getattr(self, "focused", None)
            if isinstance(focused, TextArea):
                selected_text = focused.selected_text or ""
                if selected_text.strip():
                    self._copy_text(selected_text, label="selection")
                    return
                focused_text = (getattr(focused, "text", "") or "").strip()
                if focused.id in {"issues", "plan", "artifacts", "consent", "help"} and focused_text:
                    self._copy_text(focused_text + "\n", label=f"{focused.id} pane")
                    return
                self._notify("Select text first.", severity="warning")
                return

            transcript = self.query_one("#transcript", VerticalScroll)
            node = focused
            while node is not None:
                if node is transcript:
                    fallback = self._get_all_text()
                    self._copy_text(fallback, label="transcript+panes")
                    return
                node = getattr(node, "parent", None)

            self._notify("No text area focused.", severity="warning")

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
                    self._write_transcript_line("[search] found match in transcript.")
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
                self._pending_model_select_value = None
                self._model_provider_override = None
                self._model_tier_override = None
            elif value == _MODEL_LOADING_VALUE:
                return
            elif "|" in value:
                provider, tier = value.split("|", 1)
                requested_provider = provider.strip().lower()
                requested_tier = tier.strip().lower()
                if not requested_provider or not requested_tier:
                    return
                if self._daemon_ready and self._proc is not None and self._proc.poll() is None:
                    current_tier = (self._daemon_tier or "").strip().lower()
                    current_provider = (self._daemon_provider or "").strip().lower()
                    if requested_tier == current_tier:
                        self._pending_model_select_value = None
                        self._model_provider_override = requested_provider or None
                        self._model_tier_override = requested_tier or None
                        self._update_header_status()
                        self._update_prompt_placeholder()
                        if self._status_bar is not None:
                            self._status_bar.set_model(self._current_model_summary())
                        return
                    if self._query_active:
                        self._write_transcript_line("[model] cannot change tier while a run is active.")
                        self._pending_model_select_value = None
                        self._refresh_model_select_from_daemon(
                            provider=current_provider or requested_provider,
                            tier=current_tier or requested_tier,
                            tiers=self._daemon_tiers,
                        )
                        return
                    else:
                        if not send_daemon_command(self._proc, {"cmd": "set_tier", "tier": requested_tier}):
                            self._write_transcript_line("[model] failed to send tier change to daemon.")
                            self._pending_model_select_value = None
                            self._refresh_model_select()
                            return
                        self._pending_model_select_value = f"{requested_provider}|{requested_tier}"
                else:
                    self._pending_model_select_value = None
                self._model_provider_override = requested_provider or None
                self._model_tier_override = requested_tier or None
            self._update_header_status()
            self._update_prompt_placeholder()
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())
            # The model selector is always visible; avoid transient notifications.

        def _dispatch_plan_action(self, action: str) -> None:
            normalized = action.strip().lower()
            if normalized == "approve":
                if not self._pending_plan_prompt:
                    self._write_transcript_line("[run] no pending plan.")
                    return
                self._start_run(self._pending_plan_prompt, auto_approve=True)
                return
            if normalized == "replan":
                if not self._last_prompt:
                    self._write_transcript_line("[run] no previous prompt to replan.")
                    return
                self._start_run(self._last_prompt, auto_approve=False, mode="plan")
                return
            if normalized == "clearplan":
                self._pending_plan_prompt = None
                self._reset_plan_panel()
                self._write_transcript_line("[run] plan cleared.")
                return

        def on_button_pressed(self, event: Any) -> None:
            button_id = str(getattr(getattr(event, "button", None), "id", "")).strip().lower()
            if button_id == "plan_action_approve":
                self._dispatch_plan_action("approve")
                return
            if button_id == "plan_action_replan":
                self._dispatch_plan_action("replan")
                return
            if button_id == "plan_action_clear":
                self._dispatch_plan_action("clearplan")
                return

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
                self.action_quit()
                return

            if normalized in {"/daemon restart", "/restart-daemon"}:
                self._spawn_daemon(restart=True)
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
                options, _ = self._model_select_options()
                for label, _value in options:
                    self._write_transcript_line(f"- {label}")
                return

            if normalized == "/model reset":
                self._pending_model_select_value = None
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
                self._pending_model_select_value = None
                self._model_provider_override = provider
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] provider set to {provider}.")
                if self._daemon_ready:
                    self._write_transcript_line("[model] restart daemon to apply provider changes.")
                self._write_transcript_line(self._current_model_summary())
                return

            if normalized.startswith("/model tier "):
                tier = normalized.split(maxsplit=2)[2].strip()
                if not tier:
                    self._write_transcript_line("Usage: /model tier <name>")
                    return
                self._pending_model_select_value = None
                self._model_tier_override = tier
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] tier set to {tier}.")
                if (
                    self._daemon_ready
                    and self._proc is not None
                    and self._proc.poll() is None
                    and not self._query_active
                ):
                    requested_provider = (self._model_provider_override or self._daemon_provider or "").strip().lower()
                    requested_tier = tier.strip().lower()
                    if not send_daemon_command(self._proc, {"cmd": "set_tier", "tier": tier}):
                        self._pending_model_select_value = None
                        self._write_transcript_line("[model] failed to send tier change to daemon.")
                    elif requested_provider and requested_tier:
                        self._pending_model_select_value = f"{requested_provider}|{requested_tier}"
                self._write_transcript_line(self._current_model_summary())
                return

            if self._query_active:
                self._write_transcript_line("[run] already running; use /stop.")
                return

            if normalized == "/approve":
                self._dispatch_plan_action("approve")
                return

            if normalized == "/replan":
                self._dispatch_plan_action("replan")
                return

            if normalized == "/clearplan":
                self._dispatch_plan_action("clearplan")
                return

            if normalized == "/plan":
                self._default_auto_approve = False
                self._update_prompt_placeholder()
                self._write_transcript_line("[mode] auto-approve disabled for default prompts.")
                return

            if text.startswith("/plan "):
                prompt = text[len("/plan ") :].strip()
                if not prompt:
                    self._write_transcript_line("Usage: /plan <prompt>")
                    return
                self._start_run(prompt, auto_approve=False, mode="plan")
                return

            if normalized == "/run":
                self._default_auto_approve = True
                self._update_prompt_placeholder()
                self._write_transcript_line("[mode] auto-approve enabled for default prompts.")
                return

            if text.startswith("/run "):
                prompt = text[len("/run ") :].strip()
                if not prompt:
                    self._write_transcript_line("Usage: /run <prompt>")
                    return
                self._start_run(prompt, auto_approve=True, mode="execute")
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
