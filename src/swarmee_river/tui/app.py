"""Optional Textual app scaffold for `swarmee tui`."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json as _json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from swarmee_river.state_paths import sessions_dir
from swarmee_river.tui.agent_studio import (
    _normalized_tool_name_list,
    _policy_tier_profile,
    build_activated_agent_sidebar_items,
    build_activated_agents_run_prompt,
    build_agent_policy_lens,
    build_agent_team_sidebar_items,
    build_agent_tools_safety_sidebar_items,
    build_team_preset_run_prompt,
    normalize_agent_studio_view_mode,
    normalize_session_safety_overrides,
    normalize_team_preset,
    normalize_team_presets,
    render_activated_agent_detail_text,
    render_agent_team_detail_text,
    render_agent_tools_safety_detail_text,
)
from swarmee_river.tui.commands import (
    _AUTH_USAGE_TEXT,
    _COMPACT_USAGE_TEXT,
    _CONNECT_USAGE_TEXT,
    _CONSENT_USAGE_TEXT,
    _CONTEXT_USAGE_TEXT,
    _EXPAND_USAGE_TEXT,
    _MODEL_USAGE_TEXT,
    _OPEN_USAGE_TEXT,
    _SEARCH_USAGE_TEXT,
    _SOP_USAGE_TEXT,
    _TEXT_USAGE_TEXT,
    _THINKING_USAGE_TEXT,
    classify_copy_command,
    classify_model_command,
    classify_post_run_command,
    classify_pre_run_command,
)
from swarmee_river.tui.event_router import classify_tui_error_event, summarize_error_for_toast
from swarmee_river.tui.event_types import (
    ParsedEvent,
    extract_tui_text_chunk,
    parse_output_line,
    parse_tui_event,
)
from swarmee_river.tui.event_types import (
    detect_consent_prompt as _event_detect_consent_prompt,
)
from swarmee_river.tui.event_types import (
    update_consent_capture as _event_update_consent_capture,
)
from swarmee_river.tui.mixins.agent_studio import AgentStudioMixin
from swarmee_river.tui.mixins.artifacts import ArtifactsMixin
from swarmee_river.tui.mixins.context_sources import ContextSourcesMixin
from swarmee_river.tui.mixins.daemon import DaemonMixin
from swarmee_river.tui.mixins.output import OutputMixin
from swarmee_river.tui.mixins.plan import PlanMixin
from swarmee_river.tui.mixins.prompt_ui import PromptUIMixin
from swarmee_river.tui.mixins.session import SessionMixin
from swarmee_river.tui.mixins.settings import SettingsMixin
from swarmee_river.tui.mixins.thinking import ThinkingMixin
from swarmee_river.tui.mixins.tools import ToolsMixin
from swarmee_river.tui.mixins.transcript import TranscriptMixin
from swarmee_river.tui.model_select import (
    _MODEL_AUTO_VALUE,
    _MODEL_LOADING_VALUE,
    choose_daemon_model_select_value,
    choose_model_summary_parts,
    daemon_model_select_options,
    model_select_options,
    parse_model_select_value,
    resolve_model_config_summary,
    resolve_model_fallback_notice,
)
from swarmee_river.tui.sidebar_artifacts import (
    add_recent_artifacts,
    artifact_context_source_payload,
    build_artifact_sidebar_items,
    normalize_artifact_index_entry,
)
from swarmee_river.tui.sidebar_session import (
    build_session_issue_sidebar_items,
    build_session_timeline_sidebar_items,
    classify_session_timeline_event_kind,
    normalize_session_view_mode,
    render_session_issue_detail_text,
    render_session_timeline_detail_text,
    session_issue_actions,
    session_timeline_actions,
    summarize_session_timeline_event,
)
from swarmee_river.tui.sops import (
    _SOP_FILE_SUFFIX,
    _first_sop_paragraph,
    _load_sop_file,
    _strip_sop_frontmatter,
    discover_available_sop_names,
    discover_available_sops,
)
from swarmee_river.tui.state import AppState
from swarmee_river.tui.text_sanitize import (
    extract_plan_section,
    extract_plan_section_from_output,
    looks_like_plan_output,
    render_tui_hint_after_plan,
    sanitize_output_text,
)
from swarmee_river.tui.transport import (
    _build_swarmee_subprocess_env as _transport_build_swarmee_subprocess_env,
)
from swarmee_river.tui.transport import (
    _SocketTransport as _transport_SocketTransport,
)
from swarmee_river.tui.transport import (
    send_daemon_command as _transport_send_daemon_command,
)
from swarmee_river.tui.transport import (
    spawn_swarmee as _transport_spawn_swarmee,
)
from swarmee_river.tui.transport import (
    spawn_swarmee_daemon as _transport_spawn_swarmee_daemon,
)
from swarmee_river.tui.transport import (
    stop_process as _transport_stop_process,
)
from swarmee_river.tui.transport import (
    write_to_proc as _transport_write_to_proc,
)

_COMPAT_AGENT_HELPERS = (
    build_activated_agent_sidebar_items,
    build_activated_agents_run_prompt,
    render_activated_agent_detail_text,
)

_CONSENT_CHOICES = {"y", "n", "a", "v"}
_RUN_ACTIVE_TIER_WARNING = "[model] cannot change tier while a run is active."
_AGENT_PROFILE_SELECT_NONE = "__agent_profile_none__"
_AGENT_TOOL_CONSENT_VALUES = {"ask", "allow", "deny"}
_CONTEXT_SOURCE_ICONS: dict[str, str] = {
    "file": "📄",
    "url": "🌐",
    "kb": "📚",
    "sop": "📋",
    "note": "📝",
}
_CONTEXT_SOURCE_TYPES = {"file", "note", "sop", "kb", "url"}
_CONTEXT_SOURCE_MAX_LABEL = 72
_CONTEXT_SELECT_PLACEHOLDER = "__context_select_none__"
_CONTEXT_INPUT_SOURCE_TYPES = {"file", "note", "kb"}
_CONTEXT_SOP_SOURCE_TYPE = "sop"
_STREAMING_FLUSH_INTERVAL_S = 0.15
_TOOL_PROGRESS_RENDER_INTERVAL_S = 0.15
_TOOL_HEARTBEAT_RENDER_MIN_STEP_S = 0.5
_TOOL_OUTPUT_RETENTION_MAX_CHARS = 4096
_TOOL_START_COALESCE_INTERVAL_S = 0.1
_THINKING_DISPLAY_DEBOUNCE_S = 0.2
_THINKING_ANIMATION_INTERVAL_S = 0.5
_THINKING_EXPORT_MAX_CHARS = 5000

# Compatibility re-exports for callers importing pure helpers from tui.app.
_COMPAT_REEXPORTS = (
    _AUTH_USAGE_TEXT,
    _COMPACT_USAGE_TEXT,
    _CONNECT_USAGE_TEXT,
    _CONSENT_USAGE_TEXT,
    _CONTEXT_USAGE_TEXT,
    _EXPAND_USAGE_TEXT,
    _MODEL_AUTO_VALUE,
    _MODEL_LOADING_VALUE,
    _MODEL_USAGE_TEXT,
    _OPEN_USAGE_TEXT,
    _SEARCH_USAGE_TEXT,
    _SOP_FILE_SUFFIX,
    _SOP_USAGE_TEXT,
    _TEXT_USAGE_TEXT,
    _THINKING_USAGE_TEXT,
    _first_sop_paragraph,
    _load_sop_file,
    _normalized_tool_name_list,
    _policy_tier_profile,
    _strip_sop_frontmatter,
    add_recent_artifacts,
    artifact_context_source_payload,
    build_agent_policy_lens,
    build_agent_team_sidebar_items,
    build_agent_tools_safety_sidebar_items,
    build_artifact_sidebar_items,
    build_session_issue_sidebar_items,
    build_session_timeline_sidebar_items,
    build_team_preset_run_prompt,
    choose_daemon_model_select_value,
    choose_model_summary_parts,
    classify_copy_command,
    classify_tui_error_event,
    classify_model_command,
    classify_post_run_command,
    classify_pre_run_command,
    classify_session_timeline_event_kind,
    daemon_model_select_options,
    discover_available_sop_names,
    discover_available_sops,
    extract_plan_section,
    extract_plan_section_from_output,
    looks_like_plan_output,
    model_select_options,
    normalize_agent_studio_view_mode,
    normalize_artifact_index_entry,
    normalize_session_safety_overrides,
    normalize_session_view_mode,
    normalize_team_preset,
    normalize_team_presets,
    ParsedEvent,
    parse_output_line,
    parse_tui_event,
    extract_tui_text_chunk,
    parse_model_select_value,
    render_tui_hint_after_plan,
    render_agent_team_detail_text,
    render_agent_tools_safety_detail_text,
    render_session_issue_detail_text,
    render_session_timeline_detail_text,
    resolve_model_config_summary,
    sanitize_output_text,
    sessions_dir,
    session_issue_actions,
    session_timeline_actions,
    summarize_session_timeline_event,
    summarize_error_for_toast,
)

# Backwards-compatible alias used by tests and older imports.
_SocketTransport = _transport_SocketTransport


def _sanitize_profile_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


def _default_profile_id(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"profile-{ts.strftime('%Y%m%d-%H%M%S')}"


def prompt_history_previous(
    history: list[str],
    *,
    current_index: int,
    draft_text: str | None,
    current_text: str,
) -> tuple[int, str | None, str | None]:
    if not history:
        return current_index, draft_text, None
    if current_index < 0:
        next_draft = current_text
        next_index = len(history) - 1
        return next_index, next_draft, history[next_index]
    if current_index > 0:
        next_index = current_index - 1
        return next_index, draft_text, history[next_index]
    return current_index, draft_text, history[0]


def prompt_history_next(
    history: list[str],
    *,
    current_index: int,
    draft_text: str | None,
) -> tuple[int, str | None, str | None]:
    if current_index < 0 or not history:
        return current_index, draft_text, None
    if current_index < len(history) - 1:
        next_index = current_index + 1
        return next_index, draft_text, history[next_index]
    return -1, None, draft_text or ""


def _default_profile_name(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"Profile {ts.strftime('%Y-%m-%d %H:%M')}"


def _default_team_preset_id(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"team-{ts.strftime('%Y%m%d-%H%M%S')}"


def _default_team_preset_name(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"Team Preset {ts.strftime('%Y-%m-%d %H:%M')}"


def _sanitize_context_source_id(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


def _normalize_context_source(source: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(source, dict):
        return None
    source_type = str(source.get("type", "")).strip().lower()
    if source_type not in _CONTEXT_SOURCE_TYPES:
        return None

    normalized: dict[str, str] = {"type": source_type}
    source_id = str(source.get("id", "")).strip()
    if source_id:
        normalized["id"] = _sanitize_context_source_id(source_id)
    else:
        source_seed = (
            source.get("path", "") or source.get("text", "") or source.get("name", "") or source.get("kb_id", "")
        )
        seed = str(source_seed).strip() or uuid.uuid4().hex
        normalized["id"] = _sanitize_context_source_id(f"{source_type}-{seed}")

    if source_type == "file":
        path = str(source.get("path", "")).strip()
        if not path:
            return None
        normalized["path"] = path
        return normalized
    if source_type == "note":
        text = str(source.get("text", "")).strip()
        if not text:
            return None
        normalized["text"] = text
        return normalized
    if source_type == "sop":
        name = str(source.get("name", "")).strip()
        if not name:
            return None
        normalized["name"] = name
        return normalized
    if source_type == "kb":
        kb_id = str(source.get("id", source.get("kb_id", ""))).strip()
        if not kb_id:
            return None
        normalized["id"] = kb_id
        return normalized
    if source_type == "url":
        url = str(source.get("url", source.get("path", ""))).strip()
        if not url:
            return None
        normalized["url"] = url
        return normalized
    return None


def _normalize_context_sources(raw_sources: Any) -> list[dict[str, str]]:
    if not isinstance(raw_sources, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_sources:
        source = _normalize_context_source(item if isinstance(item, dict) else {})
        if source is None:
            continue
        source_type = source.get("type", "")
        value_key = (
            source.get("path")
            or source.get("text")
            or source.get("name")
            or source.get("id")
            or source.get("url")
            or ""
        ).strip()
        dedupe_key = (source_type, value_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(source)
    return normalized


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


def is_multiline_newline_key(event: Any) -> bool:
    """Detect Shift+Enter, Alt+Enter, or Ctrl+J — NOT plain Enter."""
    key = str(getattr(event, "key", "")).lower()
    aliases = [str(a).lower() for a in getattr(event, "aliases", [])]
    event_name = str(getattr(event, "name", "")).lower()

    # Explicit modifier+enter combinations only.
    # Plain Enter must NOT match — it submits the prompt.
    modifier_enter_keys = {
        "shift+enter",
        "shift+return",
        "shift+ctrl+m",
        "alt+enter",
        "alt+return",
        "ctrl+j",
    }
    if key in modifier_enter_keys:
        return True
    if event_name in {"shift_enter", "shift_return"}:
        return True
    if any(alias in modifier_enter_keys for alias in aliases):
        return True
    return False


def _event_key_variants(event: Any) -> set[str]:
    """Collect normalized key spellings from key/name/aliases."""
    values: set[str] = set()
    raw_values = [getattr(event, "key", ""), getattr(event, "name", "")]
    raw_values.extend(getattr(event, "aliases", []) or [])
    for raw in raw_values:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        values.add(token)
        values.add(token.replace("left_arrow", "left").replace("right_arrow", "right"))
        values.add(token.replace("-", "+"))
        values.add(token.replace("-", "+").replace("left_arrow", "left").replace("right_arrow", "right"))
    return values


_WIDEN_SIDE_KEYS = {
    "f6",
    "ctrl+left",
    "ctrl+shift+left",
    "ctrl+[",
    "ctrl+left_square_bracket",
    "ctrl+left_bracket",
    "alt+left",
    "ctrl+h",
}

_WIDEN_TRANSCRIPT_KEYS = {
    "f7",
    "ctrl+right",
    "ctrl+shift+right",
    "ctrl+]",
    "ctrl+right_square_bracket",
    "ctrl+right_bracket",
    "alt+right",
    "ctrl+l",
}


def is_widen_side_key(event: Any) -> bool:
    return bool(_event_key_variants(event) & _WIDEN_SIDE_KEYS)


def is_widen_transcript_key(event: Any) -> bool:
    return bool(_event_key_variants(event) & _WIDEN_TRANSCRIPT_KEYS)


def should_ignore_programmatic_model_select_change(
    *,
    value: str,
    programmatic_value: str | None,
) -> bool:
    normalized_value = str(value or "").strip().lower()
    marker = str(programmatic_value or "").strip().lower()
    return bool(normalized_value and marker and normalized_value == marker)


def should_process_model_select_change(
    *,
    value: str,
    model_select_syncing: bool,
    has_focus: bool,
    programmatic_value: str | None,
) -> bool:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return False
    if model_select_syncing:
        return False
    if should_ignore_programmatic_model_select_change(
        value=normalized_value,
        programmatic_value=programmatic_value,
    ):
        return False
    if not has_focus:
        return False
    return True


def should_ignore_stale_model_info_update(
    *,
    incoming_value: str | None,
    target_value: str | None,
    target_until_mono: float | None,
    now_mono: float,
) -> bool:
    incoming = str(incoming_value or "").strip().lower()
    target = str(target_value or "").strip().lower()
    if not incoming or not target:
        return False
    if incoming == target:
        return False
    if target_until_mono is None:
        return False
    return now_mono < float(target_until_mono)


def should_ignore_model_select_reversion_during_target(
    *,
    requested_value: str | None,
    current_value: str | None,
    target_value: str | None,
    target_until_mono: float | None,
    now_mono: float,
) -> bool:
    requested = str(requested_value or "").strip().lower()
    current = str(current_value or "").strip().lower()
    target = str(target_value or "").strip().lower()
    if not requested or not current or not target:
        return False
    if target_until_mono is None or now_mono >= float(target_until_mono):
        return False
    return requested == current and requested != target


def should_skip_active_run_tier_warning(
    *,
    requested_provider: str,
    requested_tier: str,
    pending_value: str | None,
) -> bool:
    """Return True when a select change is a stale echo of an already-pending pre-run tier switch."""
    requested = f"{requested_provider.strip().lower()}|{requested_tier.strip().lower()}"
    pending = (pending_value or "").strip().lower()
    return bool(requested) and requested == pending


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


def detect_consent_prompt(line: str) -> str | None:
    """Detect consent-related subprocess output lines."""
    return _event_detect_consent_prompt(line)


def update_consent_capture(
    consent_active: bool,
    consent_buffer: list[str],
    line: str,
    *,
    max_lines: int = 20,
) -> tuple[bool, list[str]]:
    """Update consent capture state from a single output line."""
    return _event_update_consent_capture(
        consent_active,
        consent_buffer,
        line,
        max_lines=max_lines,
    )


def write_to_proc(proc: subprocess.Popen[str], text: str) -> bool:
    """Write a response line to a subprocess stdin."""
    return _transport_write_to_proc(proc, text)


def send_daemon_command(proc: Any, cmd_dict: dict[str, Any]) -> bool:
    """Serialize and send a daemon command as JSONL."""
    return _transport_send_daemon_command(
        proc,
        cmd_dict,
        json_module=_json,
        write_to_proc_fn=write_to_proc,
    )


def _build_swarmee_subprocess_env(
    *,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
    os_module: Any = os,
) -> dict[str, str]:
    return _transport_build_swarmee_subprocess_env(
        session_id=session_id,
        env_overrides=env_overrides,
        os_module=os_module,
    )


def spawn_swarmee(
    prompt: str,
    *,
    auto_approve: bool,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Spawn Swarmee as a subprocess with line-buffered merged output."""
    return _transport_spawn_swarmee(
        prompt,
        auto_approve=auto_approve,
        session_id=session_id,
        env_overrides=env_overrides,
        popen=subprocess.Popen,
        subprocess_module=subprocess,
        os_module=os,
        build_swarmee_cmd_fn=build_swarmee_cmd,
        env_builder=_build_swarmee_subprocess_env,
    )


def spawn_swarmee_daemon(
    *,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Spawn Swarmee daemon with line-buffered merged output."""
    return _transport_spawn_swarmee_daemon(
        session_id=session_id,
        env_overrides=env_overrides,
        popen=subprocess.Popen,
        subprocess_module=subprocess,
        os_module=os,
        build_swarmee_daemon_cmd_fn=build_swarmee_daemon_cmd,
        env_builder=_build_swarmee_subprocess_env,
    )


def stop_process(proc: subprocess.Popen[str], *, timeout_s: float = 2.0) -> None:
    """Stop a running subprocess, escalating from interrupt to terminate to kill."""
    _transport_stop_process(
        proc,
        timeout_s=timeout_s,
        os_module=os,
        signal_module=signal,
        subprocess_module=subprocess,
        contextlib_module=contextlib,
    )


def get_swarmee_tui_class() -> type:
    """Return the SwarmeeTUI class, building it if necessary.

    ``run_tui`` sets ``run_tui._app_class`` before calling ``.run()``.
    We trigger class construction by calling ``run_tui()`` with the app's
    ``run`` method patched to abort immediately.

    Raises ImportError if Textual is unavailable.
    """
    cached = getattr(run_tui, "_app_class", None)
    if cached is not None:
        return cached

    import unittest.mock as _mock

    class _AbortRun(BaseException):
        pass

    with _mock.patch("textual.app.App.run", side_effect=_AbortRun):
        try:
            run_tui()
        except _AbortRun:
            pass

    result = getattr(run_tui, "_app_class", None)
    if result is None:
        raise ImportError("SwarmeeTUI class could not be constructed (is Textual installed?)")
    return result


def run_tui() -> int:
    """Run the full-screen TUI if Textual is installed."""
    try:
        textual_app = importlib.import_module("textual.app")
        textual_binding = importlib.import_module("textual.binding")
        textual_containers = importlib.import_module("textual.containers")
        textual_screen = importlib.import_module("textual.screen")
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
    ModalScreen = textual_screen.ModalScreen
    Binding = textual_binding.Binding
    Horizontal = textual_containers.Horizontal
    Vertical = textual_containers.Vertical
    VerticalScroll = textual_containers.VerticalScroll
    Button = textual_widgets.Button
    Header = textual_widgets.Header
    Footer = textual_widgets.Footer
    Input = textual_widgets.Input
    Static = textual_widgets.Static
    TextArea = textual_widgets.TextArea

    from swarmee_river.tui.views.agents import wire_agents_widgets
    from swarmee_river.tui.views.bundles import wire_bundles_widgets
    from swarmee_river.tui.views.engage import wire_engage_widgets
    from swarmee_river.tui.views.scaffold import wire_scaffold_widgets
    from swarmee_river.tui.views.settings import (
        env_spec_by_key,
        wire_settings_widgets,
    )
    from swarmee_river.tui.views.sidebar import compose_sidebar
    from swarmee_river.tui.widgets import (
        ActionSheet,
        CommandPalette,
        ConsentPrompt,
        ContextBudgetBar,
        ErrorActionPrompt,
        PlanStepRow,
        StatusBar,
        ThinkingBar,
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

        def _replace_text(self, text: str) -> None:
            self.clear()
            for method_name in ("insert", "insert_text_at_cursor"):
                method = getattr(self, method_name, None)
                if callable(method):
                    with contextlib.suppress(Exception):
                        method(text)
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
            if app is not None and hasattr(app, "_on_prompt_text_changed"):
                app._on_prompt_text_changed(self.text)

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

            # ── Ctrl+K / Ctrl+Space: action sheet ──
            if (
                key in {"ctrl+k", "ctrl+space", "ctrl+@"}
                and app is not None
                and hasattr(app, "action_open_action_sheet")
            ):
                event.stop()
                event.prevent_default()
                app.action_open_action_sheet()
                return

            # ── Ctrl+T: transcript mode toggle ──
            if key == "ctrl+t" and app is not None and hasattr(app, "action_toggle_transcript_mode"):
                event.stop()
                event.prevent_default()
                app.action_toggle_transcript_mode()
                return

            # ── Sidebar resize shortcuts should win over TextArea cursor-nav ──
            if app is not None:
                if is_widen_side_key(event) and hasattr(app, "action_widen_side"):
                    event.stop()
                    event.prevent_default()
                    app.action_widen_side()
                    return
                if is_widen_transcript_key(event) and hasattr(app, "action_widen_transcript"):
                    event.stop()
                    event.prevent_default()
                    app.action_widen_transcript()
                    return

            # ── Arrow keys: prompt history (when palette hidden) ──
            if key == "up" and app is not None and hasattr(app, "_prompt_history"):
                next_index, next_draft, entry = prompt_history_previous(
                    app._prompt_history,
                    current_index=app._history_index,
                    draft_text=getattr(app, "_history_draft_text", None),
                    current_text=self.text,
                )
                if entry is not None:
                    event.stop()
                    event.prevent_default()
                    app._history_index = next_index
                    app._history_draft_text = next_draft
                    self._replace_text(entry)
                    return
            if key == "down" and app is not None and hasattr(app, "_prompt_history"):
                next_index, next_draft, entry = prompt_history_next(
                    app._prompt_history,
                    current_index=app._history_index,
                    draft_text=getattr(app, "_history_draft_text", None),
                )
                if entry is not None:
                    event.stop()
                    event.prevent_default()
                    app._history_index = next_index
                    app._history_draft_text = next_draft
                    self._replace_text(entry)
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
                with contextlib.suppress(AttributeError, TypeError):
                    self.insert(" ")
                    return
                with contextlib.suppress(AttributeError, TypeError):
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

    class ConfirmPlanCancelScreen(ModalScreen[bool]):
        """Confirmation modal before clearing full plan state."""

        DEFAULT_CSS = """
        ConfirmPlanCancelScreen {
            align: center middle;
            background: rgba(0, 0, 0, 0.7);
        }
        #plan_cancel_confirm_dialog {
            width: 56;
            height: auto;
            border: round $warning;
            background: $panel;
            padding: 1 2;
            layout: vertical;
        }
        #plan_cancel_confirm_title {
            height: auto;
            color: $warning;
            text-style: bold;
            padding: 0 0 1 0;
        }
        #plan_cancel_confirm_body {
            height: auto;
            color: $text;
            padding: 0 0 1 0;
        }
        #plan_cancel_confirm_actions {
            height: auto;
            layout: horizontal;
        }
        #plan_cancel_confirm_actions Button {
            width: 1fr;
            margin: 0 1 0 0;
        }
        """

        def compose(self) -> Any:  # type: ignore[override]
            with Vertical(id="plan_cancel_confirm_dialog"):
                yield Static("Cancel plan?", id="plan_cancel_confirm_title")
                yield Static(
                    "This clears the current plan and refinement state.\nTranscript history is preserved.",
                    id="plan_cancel_confirm_body",
                )
                with Horizontal(id="plan_cancel_confirm_actions"):
                    yield Button("Back", id="plan_cancel_confirm_back", variant="default", compact=True)
                    yield Button("Confirm", id="plan_cancel_confirm_confirm", variant="error", compact=True)

        def on_button_pressed(self, event: Any) -> None:
            button_id = str(getattr(getattr(event, "button", None), "id", "")).strip().lower()
            if button_id == "plan_cancel_confirm_confirm":
                self.dismiss(True)
                return
            self.dismiss(False)

    class SwarmeeTUI(
        TranscriptMixin,
        ThinkingMixin,
        ToolsMixin,
        PlanMixin,
        AgentStudioMixin,
        ContextSourcesMixin,
        SessionMixin,
        ArtifactsMixin,
        SettingsMixin,
        DaemonMixin,
        PromptUIMixin,
        OutputMixin,
        AppBase,
    ):
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
            width: 1fr;
        }

        #transcript, #transcript_text {
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
        #transcript_text {
            display: none;
        }

        #side {
            width: 1fr;
            height: 1fr;
            layout: vertical;
        }

        #side_tabs {
            height: 1fr;
        }

        #side_tabs > Tabs {
            height: 3;
            margin: 0 0 1 0;
            padding: 0;
            background: #1e241b;
            border: round #3b4a35;
        }

        #side_tabs > Tabs Tab {
            text-style: bold;
            color: #b8b8b8;
            padding: 0 2;
        }

        #side_tabs > Tabs Tab.-active {
            color: #f3f3f3;
            background: #2f3d29;
        }

        /* ── Engage tab ── */
        #engage_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #engage_view_switch {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #engage_view_switch Button {
            width: 1fr;
            min-width: 12;
            margin: 0 1 0 0;
        }

        #engage_view_switch Button:last-child {
            margin-right: 0;
        }

        #engage_plan_view {
            height: 1fr;
            layout: vertical;
        }

        #engage_planning_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #engage_execution_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #engage_plan_items {
            display: none;
            height: 1fr;
            min-height: 8;
            border: round #3b3b3b;
            padding: 0 1;
            margin: 0 0 1 0;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        #engage_plan_summary_scroll {
            display: none;
            height: auto;
            max-height: 8;
            margin: 0 0 1 0;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        #engage_plan_summary {
            height: auto;
            color: $text;
            padding: 0 0 1 0;
        }

        #engage_plan_questions {
            display: none;
            height: auto;
            max-height: 14;
            border: round #3b3b3b;
            padding: 0 1;
            margin: 0 0 1 0;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        #engage_plan_actions_row {
            display: none;
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #engage_plan_actions_row Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #engage_plan_actions_row Button:last-child {
            margin-right: 0;
        }

        #engage_session_view {
            display: none;
            height: 1fr;
            layout: vertical;
        }

        #engage_orchestrator_status {
            height: auto;
            color: $accent;
            padding: 0 0 1 0;
        }

        /* ── Help text (muted hints below headers) ── */
        #agent_overview_help, #agent_overview_model_help, #agent_tools_help, #agent_team_help,
        #kbs_empty_state {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        /* ── Session panel scrollbars ── */
        #session_panel {
            scrollbar-background: #2f2f2f;
            scrollbar-color: #7f7f7f;
        }

        /* ── Tooling tab ── */
        #tooling_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #tooling_view_switch {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #tooling_view_switch Button {
            width: 1fr;
            min-width: 10;
            margin: 0 1 0 0;
        }

        #tooling_view_switch Button:last-child {
            margin-right: 0;
        }

        #tooling_tools_view {
            height: 1fr;
            layout: vertical;
        }

        #tooling_tools_table,
        #tooling_prompts_table,
        #tooling_sops_table,
        #tooling_kbs_table,
        #session_timeline_table,
        #session_artifacts_table,
        #agent_overview_table,
        #agent_builder_table,
        #bundles_table,
        #settings_models_table,
        #settings_env_table {
            height: 2fr;
            border: round #3b3b3b;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        #tooling_tools_table > .datatable--cursor,
        #tooling_prompts_table > .datatable--cursor,
        #tooling_sops_table > .datatable--cursor,
        #tooling_kbs_table > .datatable--cursor,
        #session_timeline_table > .datatable--cursor,
        #session_artifacts_table > .datatable--cursor,
        #agent_overview_table > .datatable--cursor,
        #agent_builder_table > .datatable--cursor,
        #bundles_table > .datatable--cursor,
        #settings_models_table > .datatable--cursor,
        #settings_env_table > .datatable--cursor {
            background: $accent 30%;
            color: $text;
        }

        #tooling_tools_table > .datatable--header,
        #tooling_prompts_table > .datatable--header,
        #tooling_sops_table > .datatable--header,
        #tooling_kbs_table > .datatable--header,
        #session_timeline_table > .datatable--header,
        #session_artifacts_table > .datatable--header,
        #agent_overview_table > .datatable--header,
        #agent_builder_table > .datatable--header,
        #bundles_table > .datatable--header,
        #settings_models_table > .datatable--header,
        #settings_env_table > .datatable--header {
            background: #2f2f2f;
            color: $text;
        }

        #tooling_tools_view SidebarDetail,
        #tooling_sops_view SidebarDetail,
        #tooling_kbs_view SidebarDetail {
            height: 1fr;
        }

        #tooling_prompts_view {
            display: none;
            height: 1fr;
            layout: vertical;
        }

        #tooling_prompts_table {
            height: 1fr;
            min-height: 8;
            margin: 0 0 1 0;
        }

        #tooling_prompt_content_input {
            height: 3fr;
            min-height: 16;
            margin: 0;
            border: round #3b3b3b;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        #tooling_prompt_actions {
            height: auto;
            layout: horizontal;
            margin: 1 0 0 0;
        }

        #tooling_prompt_actions Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #tooling_prompt_actions Button:last-child {
            margin-right: 0;
        }

        #tooling_sops_view {
            display: none;
            height: 1fr;
            layout: vertical;
        }

        #tooling_kbs_view {
            display: none;
            height: 1fr;
            layout: vertical;
        }

        /* ── Settings tab ── */
        #settings_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #settings_view_switch {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #settings_view_switch Button {
            width: 1fr;
            min-width: 12;
            margin: 0 1 0 0;
        }

        #settings_general_view {
            height: 1fr;
            scrollbar-background: #2f2f2f;
            scrollbar-color: #7f7f7f;
        }

        #settings_models_view {
            display: none;
            height: 1fr;
            layout: vertical;
        }

        #settings_advanced_view {
            display: none;
            height: 1fr;
            layout: vertical;
        }

        #settings_general_header, #settings_models_header, #settings_env_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #settings_general_summary, #settings_models_summary {
            height: auto;
            color: $accent;
            padding: 0 0 1 0;
        }

        #settings_models_defaults_row {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #settings_models_defaults_row Select {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #settings_auth_row, #settings_aws_profile_row {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #settings_auth_row Button {
            width: 1fr;
            min-width: 12;
            margin: 0 1 0 0;
        }

        #settings_aws_profile_row Input {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #settings_auth_status {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #settings_models_detail, #settings_env_detail {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #settings_models_actions {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #settings_models_actions Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #settings_models_actions Button:last-child {
            margin-right: 0;
        }

        .settings-section-label {
            height: auto;
            color: $text-muted;
            text-style: bold;
            padding: 0 0 0 0;
            margin: 0 0 0 0;
        }

        #settings_general_runtime_row,
        #settings_interrupt_control_row,
        #settings_interrupt_control_actions,
        #settings_general_context_row,
        #settings_general_context_budget_row,
        #settings_general_context_budget_actions,
        #settings_general_features_row,
        #settings_general_guardrails_row,
        #settings_scope_row {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }
        #settings_general_runtime_row Button,
        #settings_general_features_row Button,
        #settings_general_guardrails_row Button {
            width: 1fr;
            min-width: 10;
            margin: 0 1 0 0;
        }

        #settings_general_runtime_row Button:last-child,
        #settings_general_features_row Button:last-child,
        #settings_general_guardrails_row Button:last-child {
            margin-right: 0;
        }

        #settings_general_context_row Select {
            width: 1fr;
            margin: 0 1 0 0;
        }
        #settings_general_context_budget_row Select,
        #settings_general_context_budget_row Input {
            width: 1fr;
            margin: 0 1 0 0;
        }
        #settings_interrupt_control_row Input,
        #settings_interrupt_control_row Select {
            width: 1fr;
            margin: 0 1 0 0;
        }
        #settings_interrupt_control_actions Button {
            width: 1fr;
            min-width: 10;
            margin: 0 1 0 0;
        }
        #settings_general_context_budget_actions Button {
            width: 1fr;
            min-width: 10;
            margin: 0 1 0 0;
        }
        #settings_interrupt_control_actions Button:last-child {
            margin-right: 0;
        }
        #settings_general_context_budget_actions Button:last-child {
            margin-right: 0;
        }

        #settings_scope_current {
            height: auto;
            color: $accent;
            padding: 0 0 0 0;
        }

        #settings_scope_row Input {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #settings_env_edit_row, #settings_env_actions {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #settings_env_edit_row Select, #settings_env_edit_row Input {
            width: 1fr;
            margin: 0 0 0 0;
        }

        #settings_env_actions Button {
            width: 1fr;
            min-width: 8;
            margin: 0 1 0 0;
        }

        #settings_env_actions Button:last-child {
            margin-right: 0;
        }

        #settings_directory_tree {
            height: auto;
            max-height: 16;
            border: round #3b3b3b;
            margin: 0 0 1 0;
        }

        #plan, #agent_summary {
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

        #sops_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #artifacts_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #artifacts_detail {
            height: 1fr;
        }

        #session_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #session_view_switch {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #session_view_switch Button {
            width: 1fr;
            min-width: 12;
            margin: 0 1 0 0;
        }

        #session_view_switch Button:last-child {
            margin-right: 0;
        }

        #session_timeline_view {
            height: 1fr;
            layout: vertical;
        }

        #session_artifacts_view {
            display: none;
            height: 1fr;
            layout: vertical;
        }

        #session_timeline_table {
            margin: 0 0 1 0;
            min-height: 10;
        }

        #session_timeline_detail {
            height: 1fr;
        }

        #session_artifacts_table {
            margin: 0 0 1 0;
            min-height: 10;
        }

        #session_artifacts_detail {
            height: 1fr;
        }

        #agent_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #agent_view_switch {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_view_switch Button {
            width: 1fr;
            min-width: 12;
            margin: 0 1 0 0;
        }

        #agent_overview_view, #agent_builder_view, #bundles_panel {
            height: 1fr;
            layout: vertical;
        }

        #agent_builder_scroll {
            height: 1fr;
            scrollbar-background: #2f2f2f;
            scrollbar-color: #7f7f7f;
        }

        #agent_builder_help, #bundles_help {
            height: auto;
            color: $text-muted;
            margin: 0 0 1 0;
        }

        #agent_overview_model_row {
            height: auto;
            layout: horizontal;
            align: left middle;
            margin: 0 0 1 0;
        }

        #agent_overview_model_label {
            width: auto;
            min-width: 13;
            margin: 0 1 0 0;
            color: $text-muted;
        }

        #agent_overview_model_row Select {
            width: 30;
            min-width: 18;
            max-width: 1fr;
        }

        #agent_summary_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #agent_overview_table, #agent_builder_table, #bundles_table {
            width: 1fr;
            margin: 0 0 1 0;
            min-height: 8;
        }

        #agent_overview_detail, #agent_builder_agent_detail, #bundles_detail {
            height: 1fr;
        }

        #agent_builder_agent_meta_row, #agent_builder_model_row,
        #agent_builder_prompt_asset_meta_row, #agent_builder_prompt_asset_actions,
        #agent_builder_tools_row, #agent_builder_sops_row, #agent_builder_kb_row,
        #bundles_meta_row, #bundles_actions_primary, #bundles_actions_secondary {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_builder_agent_id, #agent_builder_agent_name, #bundle_id, #bundle_name {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_builder_agent_name, #bundle_name {
            margin: 0;
        }

        #agent_builder_agent_prompt {
            height: 8;
            margin: 0 0 1 0;
        }

        #agent_builder_agent_summary,
        #agent_builder_agent_prompt_refs,
        #agent_builder_prompt_asset_name,
        #agent_builder_prompt_asset_id,
        #agent_builder_prompt_asset_tags {
            margin: 0 0 1 0;
        }

        #agent_builder_tools_row Static,
        #agent_builder_sops_row Static,
        #agent_builder_kb_row Static {
            width: 1fr;
            margin: 0 1 0 0;
            color: $text-muted;
        }

        #agent_builder_tools_row Button,
        #agent_builder_sops_row Button,
        #agent_builder_kb_row Button {
            width: auto;
            min-width: 12;
        }

        #agent_builder_prompt_asset_meta_row Input, #agent_builder_prompt_asset_actions Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_builder_prompt_asset_meta_row Input:last-child, #agent_builder_prompt_asset_actions Button:last-child {
            margin: 0;
        }

        #agent_builder_model_row Select {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_builder_model_row Select:last-child {
            margin: 0;
        }

        #bundles_actions_primary Button,
        #bundles_actions_secondary Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #bundles_actions_primary Button:last-child,
        #bundles_actions_secondary Button:last-child {
            margin: 0;
        }

        #agent_profile_status, #agent_overview_status, #agent_builder_status, #bundles_status {
            height: auto;
            color: $text-muted;
        }

        #agent_builder_agent_activated {
            margin: 0 0 1 0;
        }

        #settings_safety_header {
            height: auto;
            color: $text-muted;
            padding: 1 0 0 0;
        }

        #settings_safety_tool_consent, #settings_safety_tool_allowlist, #settings_safety_tool_blocklist {
            margin: 0 0 1 0;
        }

        #settings_safety_actions {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #settings_safety_actions Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #settings_safety_actions Button:last-child {
            margin-right: 0;
        }

        /* ── Responsive core button groups (sidebar-width driven) ── */
        .layout-narrow #engage_view_switch,
        .layout-narrow #tooling_view_switch,
        .layout-narrow #tooling_prompt_actions,
        .layout-narrow #session_view_switch,
        .layout-narrow #agent_overview_model_row,
        .layout-narrow #bundles_actions_primary,
        .layout-narrow #bundles_actions_secondary,
        .layout-narrow #settings_general_runtime_row,
        .layout-narrow #settings_interrupt_control_actions,
        .layout-narrow #settings_general_features_row,
        .layout-narrow #settings_general_guardrails_row,
        .layout-narrow #settings_models_actions,
        .layout-narrow #settings_env_actions,
        .layout-narrow #settings_safety_actions {
            layout: vertical;
        }

        .layout-narrow #engage_view_switch Button,
        .layout-narrow #tooling_view_switch Button,
        .layout-narrow #session_view_switch Button,
        .layout-narrow #bundles_actions_primary Button,
        .layout-narrow #bundles_actions_secondary Button,
        .layout-narrow #settings_general_runtime_row Button,
        .layout-narrow #settings_interrupt_control_actions Button,
        .layout-narrow #settings_general_features_row Button,
        .layout-narrow #settings_general_guardrails_row Button,
        .layout-narrow #settings_models_actions Button,
        .layout-narrow #settings_env_actions Button,
        .layout-narrow #settings_safety_actions Button {
            width: 1fr;
            min-width: 0;
            margin: 0 0 1 0;
        }

        .layout-narrow #engage_view_switch Button:last-child,
        .layout-narrow #tooling_view_switch Button:last-child,
        .layout-narrow #session_view_switch Button:last-child,
        .layout-narrow #bundles_actions_primary Button:last-child,
        .layout-narrow #bundles_actions_secondary Button:last-child,
        .layout-narrow #settings_general_runtime_row Button:last-child,
        .layout-narrow #settings_interrupt_control_actions Button:last-child,
        .layout-narrow #settings_general_features_row Button:last-child,
        .layout-narrow #settings_general_guardrails_row Button:last-child,
        .layout-narrow #settings_models_actions Button:last-child,
        .layout-narrow #settings_env_actions Button:last-child,
        .layout-narrow #settings_safety_actions Button:last-child {
            margin-bottom: 0;
        }

        .layout-narrow #agent_overview_model_row Static,
        .layout-narrow #agent_overview_model_row Select {
            width: 1fr;
            min-width: 0;
            margin: 0 0 1 0;
        }

        .layout-narrow #agent_overview_model_row Select:last-child {
            margin-bottom: 0;
        }

        .layout-narrow #engage_plan_actions_row {
            layout: vertical;
        }

        .layout-narrow #engage_plan_actions_row Button {
            width: 1fr;
            min-width: 0;
            margin: 0 0 1 0;
        }

        .layout-narrow #engage_plan_actions_row Button:last-child {
            margin-bottom: 0;
        }

        #settings_safety_status {
            height: auto;
            color: $text-muted;
        }

        #sops_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #plan_actions {
            display: none;
            height: auto;
        }

        #consent_prompt {
            display: none;
            height: auto;
            min-height: 5;
            margin: 0 0 0 0;
        }

        #error_action_prompt {
            display: none;
            height: auto;
            min-height: 0;
            margin: 0 0 0 0;
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
            width: 1fr;
        }

        Select > SelectCurrent {
            border: tall #4a5568;
            background: #1f2733;
            color: $text;
        }

        Select:focus > SelectCurrent,
        Select.-expanded > SelectCurrent {
            border: tall $accent;
            background: #253244;
        }

        SelectCurrent .arrow {
            color: $accent;
        }

        Select > SelectOverlay {
            border: tall #4a5568;
            background: #141a23;
            color: $text;
            max-height: 14;
        }

        Select > SelectOverlay > .option-list--option {
            color: $text;
        }

        Select > SelectOverlay > .option-list--option-highlighted {
            background: #324760;
            color: $text;
            text-style: bold;
        }

        Select > SelectOverlay > .option-list--option-hover {
            background: #2a3b52;
            color: $text;
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
            Binding("f5", "submit_prompt", "Send prompt", show=False),
            ("escape", "interrupt_run", "Interrupt run"),
            ("ctrl+t", "toggle_transcript_mode", "Toggle transcript mode"),
            Binding("ctrl+k", "open_action_sheet", "Actions", priority=True, show=False),
            Binding("ctrl+p", "open_action_sheet", "Actions", priority=True),
            Binding("ctrl+space", "open_action_sheet", "Actions", priority=True, show=False),
            ("ctrl+c", "copy_selection", "Copy selection"),
            ("meta+c", "copy_selection", "Copy selection"),
            ("super+c", "copy_selection", "Copy selection"),
            ("ctrl+d", "quit", "Quit"),
            ("tab", "focus_prompt", "Focus prompt"),
            Binding("ctrl+left", "widen_side", "Widen side", priority=True),
            Binding("ctrl+right", "widen_transcript", "Widen transcript", priority=True),
            Binding("ctrl+shift+left", "widen_side", "Widen side", priority=True),
            Binding("ctrl+shift+right", "widen_transcript", "Widen transcript", priority=True),
            Binding("alt+left", "widen_side", "Widen side", priority=True, show=False),
            Binding("alt+right", "widen_transcript", "Widen transcript", priority=True, show=False),
            Binding("ctrl+h", "widen_side", "Widen side", priority=True, show=False),
            Binding("ctrl+l", "widen_transcript", "Widen transcript", priority=True, show=False),
            ("f6", "widen_side", "Widen side"),
            ("f7", "widen_transcript", "Widen transcript"),
            ("ctrl+f", "search_transcript", "Search"),
        ]

        state: AppState
        _last_prompt: str | None = None
        _last_run_auto_approve: bool = False
        _default_auto_approve: bool = False
        _consent_active: bool = False
        _consent_buffer: list[str] = []
        _consent_history_lines: list[str] = []
        _consent_tool_name: str = "tool"
        _consent_prompt_nonce: int = 0
        _consent_hide_timer: Any = None
        _context_sources: list[dict[str, str]] = []
        _context_sop_names: list[str] = []
        _context_add_mode: str | None = None
        _context_ready_for_sync: bool = False
        _sop_catalog: list[dict[str, str]] = []
        _active_sop_names: set[str] = set()
        _sops_ready_for_sync: bool = False
        # Conversation view state
        _current_assistant_chunks: list[str] = []
        _streaming_buffer: list[str] = []
        _streaming_flush_timer: Any = None
        _streaming_last_flush_mono: float = 0.0
        _tool_progress_pending_ids: set[str] = set()
        _tool_progress_flush_timer: Any = None
        _current_assistant_model: str | None = None
        _current_assistant_timestamp: str | None = None
        _assistant_completion_seen_turn: bool = False
        _assistant_placeholder_written: bool = False
        _stream_render_warning_emitted_turn: bool = False
        _structured_assistant_seen_turn: bool = False
        _raw_assistant_lines_suppressed_turn: int = 0
        _last_structured_assistant_text_turn: str = ""
        _callback_event_trace_turn: list[str] = []
        _active_assistant_message: Any = None  # AssistantMessage | None
        _active_reasoning_block: Any = None  # ReasoningBlock | None
        _current_thinking: bool = False
        _thinking_buffer: list[str] = []
        _thinking_char_count: int = 0
        _thinking_display_timer: Any = None
        _thinking_animation_timer: Any = None
        _thinking_min_visible_timer: Any = None
        _thinking_started_mono: float | None = None
        _thinking_frame_index: int = 0
        _last_thinking_text: str = ""
        _thinking_seen_turn: bool = False
        _thinking_unavailable_notice_emitted_turn: bool = False
        _active_thinking_indicator: Any = None  # ThinkingIndicator | None
        _tool_blocks: dict[str, dict[str, Any]] = {}
        _tool_pending_start: dict[str, float] = {}
        _tool_pending_start_timers: dict[str, Any] = {}
        _transcript_mode: str = "rich"
        _transcript_fallback_lines: list[str] = []
        _consent_prompt_widget: Any = None  # ConsentPrompt | None
        _error_action_prompt_widget: Any = None  # ErrorActionPrompt | None
        _pending_error_action: dict[str, Any] | None = None
        _context_sources_list: Any = None  # VerticalScroll | None
        _context_input: Any = None  # Input | None
        _context_sop_select: Any = None  # Select | None
        _session_header: Any = None  # SidebarHeader | None
        _session_view_timeline_button: Any = None  # Button | None
        _session_view_issues_button: Any = None  # Button | None (backward compat)
        _session_view_artifacts_button: Any = None  # Button | None
        _session_timeline_view: Any = None  # Vertical | None
        _session_issues_view: Any = None  # Vertical | None (backward compat)
        _session_artifacts_view: Any = None  # Vertical | None
        _session_timeline_header: Any = None  # SidebarHeader | None
        _session_timeline_table: Any = None  # DataTable | None
        _session_timeline_detail: Any = None  # SidebarDetail | None
        _session_artifacts_header: Any = None  # SidebarHeader | None
        _session_artifacts_table: Any = None  # DataTable | None
        _session_artifacts_detail: Any = None  # SidebarDetail | None
        _session_issue_list: Any = None  # SidebarList | None
        _session_issue_detail: Any = None  # SidebarDetail | None
        _artifacts_header: Any = None  # SidebarHeader | None
        _artifacts_table: Any = None  # DataTable | None
        _artifacts_detail: Any = None  # SidebarDetail | None
        _command_palette: Any = None  # CommandPalette | None
        _action_sheet: Any = None  # ActionSheet | None
        _action_sheet_mode: str = "root"
        _action_sheet_previous_focus: Any = None
        _status_bar: Any = None  # StatusBar | None
        _thinking_bar: Any = None  # ThinkingBar | None
        _prompt_metrics: Any = None  # ContextBudgetBar | None
        _prompt_input_tokens_est: int | None = None
        _prompt_estimate_timer: Any = None
        _pending_prompt_estimate_text: str = ""
        _last_assistant_text: str = ""
        _last_transcript_dedup_line: str = ""
        _last_transcript_dedup_count: int = 0
        _prompt_history: list[str] = []
        _history_index: int = -1
        _history_draft_text: str | None = None
        _MAX_PROMPT_HISTORY: int = 50
        _TRANSCRIPT_MAX_LINES: int = 5000
        _split_ratio: int = 2
        _search_active: bool = False
        _agent_view_profile_button: Any = None  # Button | None (legacy alias to overview)
        _agent_view_tools_button: Any = None  # Button | None
        _agent_view_team_button: Any = None  # Button | None
        _agent_view_overview_button: Any = None  # Button | None
        _agent_view_builder_button: Any = None  # Button | None
        _agent_overview_view: Any = None  # Vertical | None
        _agent_builder_view: Any = None  # Vertical | None
        _agent_builder_scroll: Any = None  # VerticalScroll | None
        _agent_overview_header: Any = None  # SidebarHeader | None
        _agent_overview_table: Any = None  # DataTable | None
        _agent_overview_detail: Any = None  # SidebarDetail | None
        _agent_overview_status: Any = None  # Static | None
        _agent_profile_select: Any = None  # Select | None (legacy)
        _agent_builder_auto_delegate_checkbox: Any = None  # Checkbox | None
        _agent_builder_table: Any = None  # DataTable | None
        _agent_builder_detail: Any = None  # SidebarDetail | None
        _agent_builder_agent_id_input: Any = None  # Input | None
        _agent_builder_agent_name_input: Any = None  # Input | None
        _agent_builder_agent_summary_input: Any = None  # Input | None
        _agent_builder_agent_prompt_input: Any = None  # TextArea | None
        _agent_builder_agent_prompt_refs_input: Any = None  # Input | None
        _agent_builder_prompt_asset_name_input: Any = None  # Input | None
        _agent_builder_prompt_asset_id_input: Any = None  # Input | None
        _agent_builder_prompt_asset_tags_input: Any = None  # Input | None
        _agent_builder_agent_provider_select: Any = None  # Select | None
        _agent_builder_agent_tier_select: Any = None  # Select | None
        _agent_builder_tools_summary: Any = None  # Static | None
        _agent_builder_sops_summary: Any = None  # Static | None
        _agent_builder_kb_summary: Any = None  # Static | None
        _agent_builder_tools_draft: list[str] = []
        _agent_builder_sops_draft: list[str] = []
        _agent_builder_kb_draft: str = ""
        _agent_builder_agent_activated_checkbox: Any = None  # Checkbox | None
        _agent_builder_status: Any = None  # Static | None
        _agent_profile_view: Any = None  # Vertical | None
        _agent_tools_view: Any = None  # Vertical | None
        _agent_team_view: Any = None  # Vertical | None
        _agent_summary: Any = None  # TextArea | None
        _agent_profile_list: Any = None  # SidebarList | None
        _agent_tools_header: Any = None  # SidebarHeader | None
        _agent_tools_list: Any = None  # SidebarList | None
        _agent_tools_detail: Any = None  # SidebarDetail | None
        _agent_tools_override_consent_input: Any = None  # Input | None
        _agent_tools_override_allowlist_input: Any = None  # Input | None
        _agent_tools_override_blocklist_input: Any = None  # Input | None
        _agent_tools_override_status: Any = None  # Static | None
        _agent_team_header: Any = None  # SidebarHeader | None
        _agent_team_list: Any = None  # SidebarList | None
        _agent_team_detail: Any = None  # SidebarDetail | None
        _agent_team_preset_id_input: Any = None  # Input | None
        _agent_team_preset_name_input: Any = None  # Input | None
        _agent_team_preset_description_input: Any = None  # Input | None
        _agent_team_preset_spec_input: Any = None  # TextArea | None
        _agent_team_status: Any = None  # Static | None
        _agent_profile_id_input: Any = None  # Input | None
        _agent_profile_name_input: Any = None  # Input | None
        _agent_profile_status: Any = None  # Static | None
        # Bundles tab
        _bundles_panel: Any = None  # Vertical | None
        _bundles_header: Any = None  # SidebarHeader | None
        _bundles_table: Any = None  # DataTable | None
        _bundles_detail: Any = None  # SidebarDetail | None
        _bundle_id_input: Any = None  # Input | None
        _bundle_name_input: Any = None  # Input | None
        _bundles_status: Any = None  # Static | None
        # Engage tab
        _engage_view_plan_button: Any = None
        _engage_view_session_button: Any = None
        _engage_plan_view: Any = None
        _engage_session_view: Any = None
        _engage_orchestrator_status: Any = None  # Static | None
        _engage_plan_summary_scroll: Any = None  # VerticalScroll | None
        _engage_plan_summary: Any = None  # Static | None
        _engage_plan_questions: Any = None  # VerticalScroll | None
        _engage_plan_items: Any = None  # VerticalScroll | None
        # Tooling tab
        _tooling_view_prompts_button: Any = None
        _tooling_view_tools_button: Any = None
        _tooling_view_sops_button: Any = None
        _tooling_view_kbs_button: Any = None
        _tooling_prompts_view: Any = None
        _tooling_tools_view: Any = None
        _tooling_sops_view: Any = None
        _tooling_kbs_view: Any = None
        _tooling_prompts_header: Any = None
        _tooling_prompts_table: Any = None
        _tooling_prompt_content_input: Any = None
        _tooling_tools_header: Any = None
        _tooling_tools_table: Any = None
        _tooling_tools_detail: Any = None
        _tooling_sops_header: Any = None
        _tooling_sops_table: Any = None
        _tooling_sops_detail: Any = None
        _tooling_kbs_header: Any = None
        _tooling_kbs_table: Any = None
        _kbs_detail: Any = None
        # Settings tab
        _settings_view_general_button: Any = None
        _settings_view_models_button: Any = None
        _settings_view_advanced_button: Any = None
        _settings_general_view: Any = None
        _settings_models_view: Any = None
        _settings_advanced_view: Any = None
        _settings_general_summary: Any = None  # Static | None
        _settings_models_summary: Any = None  # Static | None
        _settings_models_table: Any = None  # DataTable | None
        _settings_models_detail: Any = None  # Static | None
        _settings_auth_status: Any = None  # Static | None
        _settings_aws_profile_input: Any = None  # Input | None
        _settings_diag_level_select: Any = None  # Select | None
        _settings_diag_redact_toggle: Any = None  # Button | None
        _settings_diag_retention_input: Any = None  # Input | None
        _settings_diag_max_bytes_input: Any = None  # Input | None
        _settings_diag_status: Any = None  # Static | None
        _settings_env_category_select: Any = None  # Select | None
        _settings_env_detail: Any = None  # Static | None
        _settings_env_value_select: Any = None  # Select | None
        _settings_env_value_input: Any = None  # Input | None
        _settings_safety_tool_consent_input: Any = None  # Input | None
        _settings_safety_tool_allowlist_input: Any = None  # Input | None
        _settings_safety_tool_blocklist_input: Any = None  # Input | None
        _settings_safety_status: Any = None  # Static | None
        _settings_toggle_auto_approve_button: Any = None  # Button | None
        _settings_toggle_bypass_consent_button: Any = None  # Button | None
        _settings_toggle_esc_interrupt_button: Any = None  # Button | None
        _settings_interrupt_timeout_input: Any = None  # Input | None
        _settings_interrupt_force_restart_select: Any = None  # Select | None
        _settings_general_context_manager_select: Any = None  # Select | None
        _settings_general_preflight_select: Any = None  # Select | None
        _settings_general_preflight_level_select: Any = None  # Select | None
        _settings_general_context_budget_mode_select: Any = None  # Select | None
        _settings_general_context_budget_input: Any = None  # Input | None
        _settings_toggle_swarm_button: Any = None  # Button | None
        _settings_toggle_log_events_button: Any = None  # Button | None
        _settings_toggle_project_map_button: Any = None  # Button | None
        _settings_toggle_limit_tool_results_button: Any = None  # Button | None
        _settings_toggle_truncate_results_button: Any = None  # Button | None
        _settings_toggle_log_redact_button: Any = None  # Button | None
        _settings_toggle_freeze_tools_button: Any = None  # Button | None
        _settings_bedrock_read_timeout_input: Any = None  # Input | None
        _settings_bedrock_connect_timeout_input: Any = None  # Input | None
        _settings_bedrock_max_retries_input: Any = None  # Input | None
        _settings_env_table: Any = None  # DataTable | None
        _settings_scope_current: Any = None  # Static | None
        _settings_directory_tree: Any = None  # DirectoryTree | None
        _settings_env_selected_key: str | None = None
        _settings_models_selected_id: str | None = None
        _pre_settings_split_ratio: int | None = None
        _model_select_programmatic_value: str | None = None
        _model_select_target_value: str | None = None
        _model_select_target_until_mono: float | None = None
        _pending_connect_payload: dict[str, Any] | None = None
        _pending_connect_retry_payload: dict[str, Any] | None = None
        _runtime_proxy_recovery_attempted: set[str] = set()
        _auth_connect_screen: Any = None
        _auth_connect_provider: str | None = None
        _auth_connect_capture_warnings: bool = False
        _auth_connect_completion_announced: bool = False
        _thread_dispatch_backlog: Any = None  # deque[(callback, args, kwargs, attempts)] | None
        _thread_dispatch_dropped_total: int = 0
        _thread_dispatch_dropped_pending: int = 0
        _thread_dispatch_last_warning_mono: float = 0.0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.state = AppState()

        def compose(self) -> Any:
            yield Header()
            with Horizontal(id="panes"):
                yield VerticalScroll(id="transcript")
                yield TextArea(
                    text="",
                    read_only=True,
                    show_line_numbers=False,
                    id="transcript_text",
                    soft_wrap=True,
                )
                yield from compose_sidebar()
            yield CommandPalette(id="command_palette")
            yield ActionSheet(id="action_sheet")
            yield ThinkingBar(id="thinking_bar")
            yield StatusBar(id="status_bar")
            yield ErrorActionPrompt(id="error_action_prompt")
            yield ConsentPrompt(id="consent_prompt")
            with Vertical(id="prompt_box"):
                yield PromptTextArea(
                    text="",
                    placeholder="Type prompt. Enter submits, Shift+Enter inserts newline.",
                    id="prompt",
                    soft_wrap=True,
                )
                with Horizontal(id="prompt_bottom"):
                    yield ContextBudgetBar(id="prompt_metrics")
            yield Footer()

        def on_mount(self) -> None:
            self._bind_ui_widgets()
            self._apply_startup_env()
            self._reset_ui_panels()
            self._update_responsive_layout_classes()
            self._initialize_agent_studio()
            self._refresh_all_views()
            self._display_startup_banner()
            self._load_session()
            if self.state.daemon.session_id:
                self._schedule_session_timeline_refresh(delay=0.1)
            self._refresh_agent_summary()
            self._spawn_daemon()

        def _sidebar_width(self) -> int:
            side = None
            with contextlib.suppress(Exception):
                side = self.query_one("#side", Vertical)
            if side is not None:
                with contextlib.suppress(Exception):
                    width = int(getattr(getattr(side, "content_region", None), "width", 0) or 0)
                    if width > 0:
                        return width
                with contextlib.suppress(Exception):
                    width = int(getattr(getattr(side, "region", None), "width", 0) or 0)
                    if width > 0:
                        return width
                with contextlib.suppress(Exception):
                    width = int(getattr(getattr(side, "size", None), "width", 0) or 0)
                    if width > 0:
                        return width
            app_width = int(getattr(getattr(self, "size", None), "width", 0) or 0)
            if app_width <= 0:
                return 0
            return max(1, app_width // 3)

        def _update_responsive_layout_classes(self) -> None:
            sidebar_width = self._sidebar_width()
            is_narrow = sidebar_width < 36
            is_wide = sidebar_width >= 46
            is_medium = not is_narrow and not is_wide
            self.set_class(is_wide, "layout-wide")
            self.set_class(is_medium, "layout-medium")
            self.set_class(is_narrow, "layout-narrow")

        def on_resize(self, event: Any) -> None:
            _ = event
            self._update_responsive_layout_classes()
            self.refresh()

        def _bind_ui_widgets(self) -> None:
            self._command_palette = self.query_one("#command_palette", CommandPalette)
            self._action_sheet = self.query_one("#action_sheet", ActionSheet)
            self._thinking_bar = self.query_one("#thinking_bar", ThinkingBar)
            self._status_bar = self.query_one("#status_bar", StatusBar)
            self._consent_prompt_widget = self.query_one("#consent_prompt", ConsentPrompt)
            self._error_action_prompt_widget = self.query_one("#error_action_prompt", ErrorActionPrompt)
            self._prompt_metrics = self.query_one("#prompt_metrics", ContextBudgetBar)
            wire_engage_widgets(self)
            wire_agents_widgets(self)
            wire_bundles_widgets(self)
            wire_scaffold_widgets(self)
            wire_settings_widgets(self)

        def _apply_startup_env(self) -> None:
            from swarmee_river.settings import load_settings
            from swarmee_river.state_paths import set_state_dir_override
            from swarmee_river.utils.provider_utils import resolve_bedrock_runtime_profile

            self._apply_project_settings_env_overrides()
            settings = load_settings()
            self._default_auto_approve = bool(settings.runtime.auto_approve)
            set_state_dir_override(settings.runtime.state_dir, cwd=Path.cwd())
            # Internal bridging: allow Bedrock profile selection without treating AWS_PROFILE
            # as supported end-user configuration.
            bedrock = settings.models.providers.get("bedrock")
            extra = dict(getattr(bedrock, "extra", {}) or {})
            aws_profile = str(extra.get("aws_profile") or "").strip()
            resolved_profile, _aws_ok, _aws_source, _aws_warning = resolve_bedrock_runtime_profile(aws_profile)
            if resolved_profile:
                os.environ["AWS_PROFILE"] = resolved_profile

        def _reset_ui_panels(self) -> None:
            self._status_bar.set_model(self._current_model_summary())
            self.query_one("#prompt", PromptTextArea).focus()
            self._reset_plan_panel()
            self._reset_issues_panel()
            self._reset_session_timeline_panel()
            self._reset_artifacts_panel()
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            self._set_engage_view_mode("plan")
            self._set_tooling_view_mode("tools")
            self._set_settings_view_mode("general")
            self._set_session_view_mode("timeline")
            self._set_context_add_mode(None)

        def _initialize_agent_studio(self) -> None:
            self._bootstrap_legacy_profiles_cleanup()
            self._refresh_context_sop_options()
            self._render_context_sources_panel()
            self._refresh_sop_catalog()
            self._refresh_tooling_sops_table()
            self._reload_saved_bundles()
            self._refresh_agent_tool_catalog()
            self._set_agent_tools_override_form_values(self.state.agent_studio.session_safety_overrides)
            self._set_agent_studio_view_mode("overview")
            if self.state.agent_studio.saved_bundles:
                first_bundle = self.state.agent_studio.saved_bundles[0]
                self._load_bundle_into_draft(first_bundle)
                self.state.bundles.selected_bundle_id = str(first_bundle.get("id", "")).strip() or None
            else:
                self._new_agent_builder_draft_from_session(announce=False)
            self._render_agent_builder_panel()
            self._render_agent_overview_panel()
            self._render_bundles_panel()
            # Tooling tab initialization
            self._refresh_tooling_prompts_list()
            self._refresh_tooling_tools_list()
            self._refresh_tooling_kbs_table()

        def _refresh_all_views(self) -> None:
            self._refresh_agent_summary()
            self._refresh_model_select()
            self._refresh_orchestrator_status()
            self._refresh_plan_actions_visibility()
            self._refresh_settings_general()
            self._refresh_settings_models()
            self._render_bundles_panel()
            self.title = "Swarmee"
            self.sub_title = self._current_model_summary()
            self._update_prompt_placeholder()

        def _display_startup_banner(self) -> None:
            from swarmee_river.utils.welcome_utils import SWARMEE_BANNER

            for banner_line in SWARMEE_BANNER.strip().splitlines():
                self._mount_transcript_widget(banner_line, plain_text=banner_line)
            self._write_transcript("Starting Swarmee daemon...")
            self._write_transcript(self.sub_title)
            fallback_notice = resolve_model_fallback_notice()
            if fallback_notice:
                self._write_transcript(f"[model] {fallback_notice}")
            self._write_transcript(
                "Tips: use /commands in the prompt, Builder for roster edits, and Bundles for save/apply."
            )
            transcript = self.query_one("#transcript", VerticalScroll)
            with contextlib.suppress(Exception):
                transcript.scroll_end(animate=False)
            self._set_transcript_mode("rich", notify=False)

        # _set_agents_view_mode removed — Agents tab uses _set_agent_studio_view_mode

        # ── Tooling tab helpers ─────────────────────────────────────────

        def _build_action_sheet_actions(self) -> tuple[str, list[dict[str, str]]]:
            if self._action_sheet_mode == "tier_menu":
                options, _selected = self._model_select_options()
                tier_actions: list[dict[str, str]] = [
                    {"id": "tiers:back", "icon": "←", "label": "Back", "shortcut": "Esc"},
                ]
                for _label, value in options:
                    parsed = parse_model_select_value(value)
                    if parsed is None:
                        continue
                    provider_name, tier_name = parsed
                    tier_actions.append(
                        {
                            "id": f"tier:{value}",
                            "icon": "◌",
                            "label": f"{provider_name}/{tier_name}",
                            "shortcut": "Enter",
                        }
                    )
                return "Switch Model Tier", tier_actions

            if self._consent_active:
                return (
                    "Consent Pending",
                    [
                        {"id": "consent:y", "icon": "✓", "label": "Allow", "shortcut": "y"},
                        {"id": "consent:n", "icon": "✗", "label": "Deny", "shortcut": "n"},
                        {"id": "consent:a", "icon": "★", "label": "Always allow", "shortcut": "a"},
                        {"id": "consent:v", "icon": "🚫", "label": "Never allow", "shortcut": "v"},
                    ],
                )

            if self.state.daemon.query_active:
                return (
                    "Run Actions",
                    [
                        {"id": "run:stop", "icon": "■", "label": "Stop run", "shortcut": "Esc"},
                        {"id": "view:plan", "icon": "▶", "label": "View plan progress", "shortcut": "P"},
                        {"id": "view:artifacts", "icon": "⚠", "label": "View session artifacts", "shortcut": "A"},
                    ],
                )

            if self.state.plan.pending_record:
                return (
                    "Plan Review",
                    [
                        {"id": "plan:approve", "icon": "✓", "label": "Approve plan", "shortcut": "/approve"},
                        {"id": "plan:replan", "icon": "↻", "label": "Replan", "shortcut": "/replan"},
                        {"id": "plan:clear", "icon": "⌫", "label": "Clear plan", "shortcut": "/clearplan"},
                        {"id": "plan:edit", "icon": "✎", "label": "Edit plan", "shortcut": "Future"},
                    ],
                )

            actions: list[dict[str, str]] = [
                {"id": "idle:new_query", "icon": "✍", "label": "New query", "shortcut": "Tab"},
                {"id": "idle:plan_mode", "icon": "🧭", "label": "Plan mode", "shortcut": "/plan"},
                {"id": "idle:run_mode", "icon": "▶", "label": "Run mode", "shortcut": "/run"},
            ]
            if self.state.daemon.available_restore_session_id:
                actions.append({"id": "idle:restore", "icon": "↺", "label": "Restore session", "shortcut": "/restore"})
            actions.extend(
                [
                    {"id": "idle:compact", "icon": "⇢", "label": "Compact context", "shortcut": "/compact"},
                    {"id": "idle:tiers", "icon": "⚙", "label": "Switch model tier", "shortcut": "Enter"},
                ]
            )
            return "Actions", actions

        def _show_action_sheet(self) -> None:
            sheet = self._action_sheet
            if sheet is None:
                return
            title, actions = self._build_action_sheet_actions()
            sheet.set_actions(title=title, actions=actions)
            sheet.show_sheet(focus=True)

        def _dismiss_action_sheet(self, *, restore_focus: bool = True) -> None:
            sheet = self._action_sheet
            if sheet is not None:
                sheet.hide_sheet()
            previous = self._action_sheet_previous_focus
            self._action_sheet_previous_focus = None
            self._action_sheet_mode = "root"
            if restore_focus and previous is not None:
                with contextlib.suppress(Exception):
                    previous.focus()

        def action_open_action_sheet(self) -> None:
            sheet = self._action_sheet
            if sheet is None:
                return
            if sheet.is_visible:
                self._dismiss_action_sheet(restore_focus=True)
                return
            if self._command_palette is not None:
                self._command_palette.hide()
            self._action_sheet_previous_focus = getattr(self, "focused", None)
            self._action_sheet_mode = "root"
            self._show_action_sheet()

        def _execute_action_sheet_action(self, action_id: str) -> None:
            action = action_id.strip().lower()
            if not action:
                return
            if action == "tiers:back":
                self._action_sheet_mode = "root"
                self._show_action_sheet()
                return
            if action.startswith("tier:"):
                value = action_id.split(":", 1)[1].strip()
                self._set_model_tier_from_value(value)
                return

            if action == "idle:new_query":
                self.action_focus_prompt()
                return
            if action == "idle:plan_mode":
                self._seed_prompt_with_command("/plan")
                return
            if action == "idle:run_mode":
                self._seed_prompt_with_command("/run")
                return
            if action == "idle:restore":
                self._restore_available_session()
                return
            if action == "idle:compact":
                self._request_context_compact()
                return
            if action == "idle:tiers":
                self._action_sheet_mode = "tier_menu"
                self._show_action_sheet()
                return

            if action == "run:stop":
                self._stop_run()
                return
            if action == "view:plan":
                self._switch_side_tab("tab_engage")
                self._set_engage_view_mode("plan")
                return
            if action == "view:artifacts":
                self._switch_side_tab("tab_engage")
                self._set_engage_view_mode("session")
                self._set_session_view_mode("artifacts")
                return

            if action.startswith("consent:"):
                choice = action.split(":", 1)[1].strip().lower()
                self._submit_consent_choice(choice)
                return

            if action == "plan:approve":
                self._dispatch_plan_action("approve")
                return
            if action == "plan:replan":
                self._dispatch_plan_action("replan")
                return
            if action == "plan:clear":
                self._dispatch_plan_action("clearplan")
                return
            if action == "plan:edit":
                self._seed_prompt_with_command("/plan")
                self._write_transcript_line("[plan] inline editing is not available yet; adjust prompt and submit.")
                return

        def _notify(self, message: str, *, severity: str = "information", timeout: float | None = 2.5) -> None:
            with contextlib.suppress(Exception):
                self.notify(message, severity=severity, timeout=timeout)

        def action_quit(self) -> None:
            self.state.daemon.is_shutting_down = True
            timer = self._prompt_estimate_timer
            self._prompt_estimate_timer = None
            if timer is not None:
                with contextlib.suppress(RuntimeError):
                    timer.stop()
            timeline_timer = self.state.session.timeline_refresh_timer
            self.state.session.timeline_refresh_timer = None
            if timeline_timer is not None:
                with contextlib.suppress(RuntimeError):
                    timeline_timer.stop()
            status_timer = self.state.daemon.status_timer
            self.state.daemon.status_timer = None
            if status_timer is not None:
                with contextlib.suppress(RuntimeError):
                    status_timer.stop()
            self._cancel_streaming_flush_timer()
            self._cancel_tool_progress_flush_timer()
            self._clear_pending_tool_starts()
            self._reset_thinking_state()
            if self.state.daemon.proc is not None and self.state.daemon.proc.poll() is None:
                self._shutdown_transport(self.state.daemon.proc)
            if self.state.daemon.runner_thread is not None and self.state.daemon.runner_thread.is_alive():
                with contextlib.suppress(Exception):
                    self.state.daemon.runner_thread.join(timeout=1.0)
            with contextlib.suppress(Exception):
                self._save_session()
            self.exit(return_code=0)

        def on_key(self, event: Any) -> None:
            key = str(getattr(event, "key", "")).lower()

            if key in {"ctrl+k", "ctrl+space", "ctrl+@"}:
                event.stop()
                event.prevent_default()
                self.action_open_action_sheet()
                return

            if self._consent_active and key in _CONSENT_CHOICES:
                event.stop()
                event.prevent_default()
                self._submit_consent_choice(key)
                return

            if key == "escape" and self._error_action_prompt_widget is not None:
                try:
                    visible = str(self._error_action_prompt_widget.styles.display) != "none"
                except Exception:
                    visible = False
                if visible:
                    event.stop()
                    event.prevent_default()
                    self._reset_error_action_prompt()
                    self.action_focus_prompt()
                    return

            if key in {"ctrl+c", "meta+c", "super+c", "cmd+c", "command+c"}:
                event.stop()
                event.prevent_default()
                self.action_copy_selection()
                return

            if is_widen_side_key(event):
                event.stop()
                event.prevent_default()
                self.action_widen_side()
                return

            if is_widen_transcript_key(event):
                event.stop()
                event.prevent_default()
                self.action_widen_transcript()
                return

        def on_checkbox_changed(self, event: Any) -> None:
            checkbox = getattr(event, "checkbox", None)
            checkbox_id = str(getattr(checkbox, "id", "")).strip()
            if not checkbox_id:
                return
            # Plan step checkbox toggles
            if checkbox_id.startswith("plan_step_cb_"):
                try:
                    index = int(checkbox_id.split("_")[-1])
                except (ValueError, IndexError):
                    return
                with contextlib.suppress(Exception):
                    row = self.query_one(f"#plan_step_row_{index}", PlanStepRow)
                    row.toggle_comment_visibility()
                return
            if checkbox_id == "agent_builder_agent_activated":
                if self.state.agent_studio.builder_form_syncing:
                    return
                self._set_agent_builder_status("Agent draft changes pending.")
                self._set_agent_draft_dirty(True)
                return

        def on_select_changed(self, event: Any) -> None:
            select_widget = getattr(event, "select", None)
            select_id = str(getattr(select_widget, "id", "")).strip().lower()
            if select_id == "tooling_tools_source_filter":
                self._refresh_tooling_tools_list()
                return
            if select_id == "settings_env_category":
                self._refresh_settings_env_list()
                self._refresh_settings_env_detail(self._settings_env_selected_key)
                return
            if select_id == "settings_general_context_manager":
                value = str(getattr(event, "value", "")).strip()
                if value in {"summarize", "sliding", "none"}:
                    self._persist_project_setting_env_override("SWARMEE_CONTEXT_MANAGER", value)
                    self._write_transcript_line(f"[settings] context manager set to {value}.")
                return
            if select_id == "settings_general_preflight":
                value = str(getattr(event, "value", "")).strip()
                if value in {"enabled", "disabled"}:
                    self._persist_project_setting_env_override("SWARMEE_PREFLIGHT", value)
                    self._write_transcript_line(f"[settings] preflight {value}.")
                return
            if select_id == "settings_general_preflight_level":
                value = str(getattr(event, "value", "")).strip()
                if value in {"summary", "summary+tree", "summary+files"}:
                    self._persist_project_setting_env_override("SWARMEE_PREFLIGHT_LEVEL", value)
                    self._write_transcript_line(f"[settings] preflight level set to {value}.")
                return
            if select_id == "settings_general_context_budget_mode":
                mode = str(getattr(event, "value", "")).strip().lower()
                input_widget = self._settings_general_context_budget_input
                if input_widget is not None:
                    with contextlib.suppress(Exception):
                        input_widget.styles.display = "block" if mode == "custom" else "none"
                return
            if select_id in {"settings_models_provider_select", "settings_models_default_tier_select"}:
                if self.state.daemon.model_select_syncing:
                    return
                # Ignore programmatic/default updates during view refresh; only persist
                # when the user actively changes the selector.
                has_focus = bool(getattr(select_widget, "has_focus", False))
                if not has_focus:
                    return
                self._save_models_default_selection()
                self._refresh_model_select()
                self._refresh_settings_models()
                self._refresh_agent_summary()
                self._write_transcript_line("[settings] saved model defaults.")
                return
            if select_id in {"agent_builder_agent_provider", "agent_builder_agent_tier"}:
                if self.state.agent_studio.builder_form_syncing:
                    return
                self._set_agent_builder_status("Agent draft changes pending.")
                self._set_agent_draft_dirty(True)
                return
            if select_id not in {"model_select"}:
                return

            value = str(getattr(event, "value", "")).strip()
            has_focus = bool(getattr(select_widget, "has_focus", False))
            if not should_process_model_select_change(
                value=value,
                model_select_syncing=bool(self.state.daemon.model_select_syncing),
                has_focus=has_focus,
                programmatic_value=self._model_select_programmatic_value,
            ):
                if should_ignore_programmatic_model_select_change(
                    value=value,
                    programmatic_value=self._model_select_programmatic_value,
                ):
                    self._model_select_programmatic_value = None
                return
            if self._model_select_programmatic_value is not None:
                self._model_select_programmatic_value = None
            if value == _MODEL_AUTO_VALUE:
                self.state.daemon.pending_model_select_value = None
                self.state.daemon.model_provider_override = None
                self.state.daemon.model_tier_override = None
                with contextlib.suppress(Exception):
                    self._persist_quick_model_selection(provider=None, tier=None)
            elif value == _MODEL_LOADING_VALUE:
                return
            elif "|" in value:
                provider, tier = value.split("|", 1)
                requested_provider = provider.strip().lower()
                requested_tier = tier.strip().lower()
                if not requested_provider or not requested_tier:
                    return
                if should_ignore_model_select_reversion_during_target(
                    requested_value=f"{requested_provider}|{requested_tier}",
                    current_value=(
                        f"{str(self.state.daemon.provider or '').strip().lower()}|"
                        f"{str(self.state.daemon.tier or '').strip().lower()}"
                    ),
                    target_value=self._model_select_target_value,
                    target_until_mono=self._model_select_target_until_mono,
                    now_mono=time.monotonic(),
                ):
                    return
                with contextlib.suppress(Exception):
                    self._persist_quick_model_selection(provider=requested_provider, tier=requested_tier)
                self._pin_model_select_target(requested_provider, requested_tier)
                if (
                    self.state.daemon.ready
                    and self.state.daemon.proc is not None
                    and self.state.daemon.proc.poll() is None
                ):
                    current_tier = (self.state.daemon.tier or "").strip().lower()
                    current_provider = (self.state.daemon.provider or "").strip().lower()
                    if requested_provider == current_provider and requested_tier == current_tier:
                        self.state.daemon.pending_model_select_value = None
                        self.state.daemon.model_provider_override = requested_provider or None
                        self.state.daemon.model_tier_override = requested_tier or None
                        self._update_header_status()
                        self._update_prompt_placeholder()
                        if self._status_bar is not None:
                            self._status_bar.set_model(self._current_model_summary())
                        return
                    if self.state.daemon.query_active:
                        if should_skip_active_run_tier_warning(
                            requested_provider=requested_provider,
                            requested_tier=requested_tier,
                            pending_value=self.state.daemon.pending_model_select_value,
                        ):
                            self.state.daemon.model_provider_override = requested_provider or None
                            self.state.daemon.model_tier_override = requested_tier or None
                            self._update_header_status()
                            self._update_prompt_placeholder()
                            if self._status_bar is not None:
                                self._status_bar.set_model(self._current_model_summary())
                            return
                        self.state.daemon.pending_model_select_value = f"{requested_provider}|{requested_tier}"
                        self.state.daemon.model_provider_override = requested_provider or None
                        self.state.daemon.model_tier_override = requested_tier or None
                        self._update_header_status()
                        self._update_prompt_placeholder()
                        if self._status_bar is not None:
                            self._status_bar.set_model(self._current_model_summary())
                        return
                    else:
                        # Persist desired selection locally; the next query command carries `tier` and applies
                        # atomically in daemon before invocation.
                        self.state.daemon.pending_model_select_value = f"{requested_provider}|{requested_tier}"
                else:
                    self.state.daemon.pending_model_select_value = None
                self.state.daemon.model_provider_override = requested_provider or None
                self.state.daemon.model_tier_override = requested_tier or None
            self._update_header_status()
            self._update_prompt_placeholder()
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())
            self._refresh_agent_summary()
            # Keep selector changes quiet; status/header already reflect updates.

        def on_tabbed_content_tab_activated(self, event: Any) -> None:
            """Auto-expand sidebar when Settings tab is active, restore on leave."""
            tab = getattr(event, "tab", None)
            pane = getattr(event, "pane", None)
            pane_id = str(getattr(pane, "id", "") or getattr(tab, "id", "")).strip()
            self._sync_settings_sidebar_autosize(pane_id)

        def on_directory_tree_directory_selected(self, event: Any) -> None:
            """Populate scope path input when a directory is selected in the tree."""
            path = getattr(event, "path", None)
            if path is None:
                return
            with contextlib.suppress(Exception):
                scope_input = self.query_one("#settings_scope_path_input", Input)
                scope_input.value = str(path)

        def on_sidebar_list_selection_changed(self, event: Any) -> None:
            sidebar_list = getattr(event, "sidebar_list", None)
            if sidebar_list is None:
                return

            if sidebar_list is self._session_issue_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_issue = self._session_issue_by_id(selected_id)
                self._set_session_issue_selection(selected_issue)
                return

        def on_data_table_row_selected(self, event: Any) -> None:
            table = getattr(event, "data_table", None)
            if table is not None and table is self._tooling_tools_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._tooling_select_tool(selected_id)
                return
            if table is not None and table is self._tooling_sops_table:
                row_key = getattr(event, "row_key", None)
                if row_key is None:
                    return
                selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                self._tooling_select_sop(selected_id)
                if not selected_id:
                    return
                next_active = selected_id not in self._active_sop_names
                self._set_sop_active(selected_id, next_active, sync=True, announce=True)
                return
            if table is not None and table is self._tooling_prompts_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._tooling_select_prompt(selected_id)
                    cursor = getattr(table, "cursor_coordinate", None)
                    col_index = int(getattr(cursor, "column", -1) or -1)
                    column_key = ""
                    with contextlib.suppress(Exception):
                        if table.is_valid_column_index(col_index):
                            col = table.get_column_at(col_index)
                            raw_key = getattr(col, "key", "")
                            if hasattr(raw_key, "value"):
                                column_key = str(raw_key.value).strip().lower()
                            else:
                                column_key = str(raw_key).strip().lower()
                    if column_key in {"name", "id", "tags"} and selected_id:
                        self._tooling_prompt_open_table_cell_editor(selected_id, column_key)
                    elif column_key == "used_by" and selected_id:
                        self._tooling_prompt_open_used_by_editor(selected_id)
                return
            if table is not None and table is self._tooling_kbs_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._tooling_select_kb(selected_id)
                return
            if table is not None and table is self._session_timeline_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._set_session_timeline_selection(self._session_timeline_event_by_id(selected_id))
                return
            if table is not None and table is self._session_artifacts_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._set_artifact_selection(self._artifact_entry_by_item_id(selected_id))
                return
            if table is not None and table is self._agent_overview_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    selected_item = next(
                        (
                            item
                            for item in self.state.agent_studio.activated_items
                            if str(item.get("id", "")).strip() == selected_id
                        ),
                        None,
                    )
                    self._set_agent_overview_selection(selected_item)
                return
            if table is not None and table is self._agent_builder_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._set_agent_builder_selection(self._agent_builder_item_by_id(selected_id))
                return
            if table is not None and table is self._bundles_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._set_bundle_selection(self._bundle_by_id(selected_id))
                return
            if table is not None and table is self._settings_models_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._settings_models_selected_id = (
                        selected_id if selected_id and not selected_id.startswith("__") else None
                    )
                    self._refresh_settings_model_detail()
                return
            if table is not None and table is self._settings_env_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._settings_env_selected_key = (
                        selected_id if selected_id and not selected_id.startswith("__") else None
                    )
                    self._refresh_settings_env_detail(self._settings_env_selected_key)
                return

        def on_data_table_row_highlighted(self, event: Any) -> None:
            table = getattr(event, "data_table", None)
            if table is not None and table is self._tooling_tools_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._tooling_select_tool(selected_id)
                return
            if table is not None and table is self._tooling_prompts_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._tooling_select_prompt(selected_id)
                return
            if table is not None and table is self._tooling_sops_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._tooling_select_sop(selected_id)
                return
            if table is not None and table is self._tooling_kbs_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._tooling_select_kb(selected_id)
                return
            if table is not None and table is self._session_timeline_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._set_session_timeline_selection(self._session_timeline_event_by_id(selected_id))
                return
            if table is not None and table is self._session_artifacts_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._set_artifact_selection(self._artifact_entry_by_item_id(selected_id))
                return
            if table is not None and table is self._agent_overview_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    selected_item = next(
                        (
                            item
                            for item in self.state.agent_studio.activated_items
                            if str(item.get("id", "")).strip() == selected_id
                        ),
                        None,
                    )
                    self._set_agent_overview_selection(selected_item)
                return
            if table is not None and table is self._agent_builder_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._set_agent_builder_selection(self._agent_builder_item_by_id(selected_id))
                return
            if table is not None and table is self._bundles_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._set_bundle_selection(self._bundle_by_id(selected_id))
                return
            if table is not None and table is self._settings_models_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._settings_models_selected_id = (
                        selected_id if selected_id and not selected_id.startswith("__") else None
                    )
                    self._refresh_settings_model_detail()
                return
            if table is not None and table is self._settings_env_table:
                row_key = getattr(event, "row_key", None)
                if row_key is not None:
                    selected_id = str(row_key.value if hasattr(row_key, "value") else row_key).strip()
                    self._settings_env_selected_key = (
                        selected_id if selected_id and not selected_id.startswith("__") else None
                    )
                    self._refresh_settings_env_detail(self._settings_env_selected_key)
                return

        def on_sidebar_detail_action_selected(self, event: Any) -> None:
            detail = getattr(event, "detail", None)
            action_id = str(getattr(event, "action_id", "")).strip().lower()
            if detail is None or not action_id:
                return

            if detail is self._session_issue_detail:
                issue = self._session_issue_by_id(self.state.session.selected_issue_id)
                if issue is None:
                    self._notify("Select an issue first.", severity="warning")
                    return
                tool_use_id = str(issue.get("tool_use_id", "")).strip()
                if action_id == "session_issue_retry_tool":
                    self._session_retry_tool(tool_use_id)
                    return
                if action_id == "session_issue_skip_tool":
                    self._session_skip_tool(tool_use_id)
                    return
                if action_id == "session_issue_escalate_tier":
                    self._session_escalate_tier(str(issue.get("next_tier", "")).strip())
                    return
                if action_id == "session_issue_interrupt":
                    self._session_interrupt()
                    return

            if detail is self._session_timeline_detail:
                selected_event = self._session_timeline_event_by_id(self.state.session.timeline_selected_event_id)
                if selected_event is None:
                    self._notify("Select a timeline event first.", severity="warning")
                    return
                if action_id == "session_timeline_copy_json":
                    payload = _json.dumps(selected_event, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                    self._copy_text(payload, label="timeline event json")
                    return
                if action_id == "session_timeline_copy_summary":
                    summary = summarize_session_timeline_event(selected_event).rstrip() + "\n"
                    self._copy_text(summary, label="timeline event summary")
                    return

            if detail is self._artifacts_detail:
                selected = self._artifact_entry_by_item_id(self.state.artifacts.selected_item_id)
                if selected is None:
                    self._notify("Select an artifact first.", severity="warning")
                    return
                path = str(selected.get("path", "")).strip()
                if action_id == "artifact_action_open":
                    self._open_artifact_path(path)
                    return
                if action_id == "artifact_action_copy_path":
                    self._copy_selected_artifact_path()
                    return
                if action_id == "artifact_action_add_context":
                    self._add_selected_artifact_as_context()
                    return

        def on_input_changed(self, event: Any) -> None:
            input_widget = getattr(event, "input", None)
            input_id = str(getattr(input_widget, "id", "")).strip().lower()
            if input_id == "tooling_tools_search":
                self._refresh_tooling_tools_list()
                return
            # Plan step comments — no action needed on change
            if input_id.startswith("plan_step_comment_"):
                return
            if input_id in {
                "agent_tools_override_consent",
                "agent_tools_override_allowlist",
                "agent_tools_override_blocklist",
            }:
                if self.state.agent_studio.tools_form_syncing:
                    return
                self._set_agent_tools_status("Override draft changes pending.")
                return
            if input_id in {"agent_team_preset_id", "agent_team_preset_name", "agent_team_preset_description"}:
                if self.state.agent_studio.team_form_syncing:
                    return
                self._set_agent_team_status("Team preset draft changes pending.")
                self._set_agent_draft_dirty(True)
                return
            if input_id in {
                "agent_builder_agent_id",
                "agent_builder_agent_name",
                "agent_builder_agent_summary",
                "agent_builder_agent_prompt_refs",
                "agent_builder_prompt_asset_name",
                "agent_builder_prompt_asset_id",
                "agent_builder_prompt_asset_tags",
            }:
                if self.state.agent_studio.builder_form_syncing:
                    return
                self._set_agent_builder_status("Agent draft changes pending.")
                self._set_agent_draft_dirty(True)
                return
            if input_id in {"bundle_id", "bundle_name"}:
                if self.state.bundles.form_syncing:
                    return
                self._set_bundles_status("Bundle draft changes pending.")
                return
            if input_id in {
                "settings_safety_tool_consent",
                "settings_safety_tool_allowlist",
                "settings_safety_tool_blocklist",
            }:
                if self.state.agent_studio.tools_form_syncing:
                    return
                self._set_agent_tools_status("Override draft changes pending.")
                return

        def on_text_area_changed(self, event: Any) -> None:
            text_area = getattr(event, "text_area", None)
            text_area_id = str(getattr(text_area, "id", "")).strip().lower()
            if text_area_id == "agent_team_preset_spec":
                if self.state.agent_studio.team_form_syncing:
                    return
                self._set_agent_team_status("Team preset draft changes pending.")
                self._set_agent_draft_dirty(True)
                return
            if text_area_id == "agent_builder_agent_prompt":
                if self.state.agent_studio.builder_form_syncing:
                    return
                self._set_agent_builder_status("Agent draft changes pending.")
                self._set_agent_draft_dirty(True)

        def on_action_sheet_action_selected(self, event: Any) -> None:
            action_id = str(getattr(event, "action_id", "")).strip()
            event.stop()
            if not action_id:
                self._dismiss_action_sheet(restore_focus=True)
                return
            # Keep the sheet open when traversing sub-menus; otherwise close first.
            if action_id.startswith("tier:") or action_id == "tiers:back" or action_id == "idle:tiers":
                self._execute_action_sheet_action(action_id)
                return
            self._dismiss_action_sheet(restore_focus=False)
            self._execute_action_sheet_action(action_id)

        def on_action_sheet_dismissed(self, event: Any) -> None:
            event.stop()
            self._dismiss_action_sheet(restore_focus=True)

        def on_button_pressed(self, event: Any) -> None:
            button_id = str(getattr(getattr(event, "button", None), "id", "")).strip().lower()
            if button_id == "engage_view_plan":
                self._set_engage_view_mode("plan")
                return
            if button_id == "engage_view_session":
                self._set_engage_view_mode("session")
                return
            if button_id == "engage_start_plan":
                from textual.widgets import TextArea

                if self.state.plan.pre_planning_split_ratio is None:
                    self.state.plan.pre_planning_split_ratio = self._split_ratio
                if self._split_ratio > 1:
                    self.action_widen_side()
                self._set_engage_view_mode("plan")
                plan_prompt = str(self.query_one("#plan", TextArea).text or "").strip()
                if not plan_prompt:
                    self._notify("Enter a planning prompt in the Plan box first.", severity="warning", timeout=4.0)
                    return
                self._start_run(plan_prompt, auto_approve=False, mode="plan")
                return
            if button_id == "engage_continue_plan":
                self._handle_planning_continue()
                return
            if button_id == "engage_clear_plan":
                self._clear_planning_view()
                return
            if button_id == "engage_cancel_plan":
                self.push_screen(ConfirmPlanCancelScreen(), self._handle_plan_cancel_confirmation)
                return
            if button_id == "tooling_view_prompts":
                self._set_tooling_view_mode("prompts")
                return
            if button_id == "tooling_view_tools":
                self._set_tooling_view_mode("tools")
                return
            if button_id == "tooling_view_sops":
                self._set_tooling_view_mode("sops")
                return
            if button_id == "tooling_view_kbs":
                self._set_tooling_view_mode("kbs")
                return
            # Tooling tab action buttons
            if button_id == "tooling_prompt_new":
                self._tooling_prompt_new()
                return
            if button_id == "tooling_prompt_save":
                self._tooling_prompt_save()
                return
            if button_id == "tooling_prompt_delete":
                self._tooling_prompt_delete()
                return
            if button_id == "tooling_tools_s3_import":
                self._tooling_s3_import("tools")
                return
            if button_id == "tooling_tools_tag_manager":
                self._tooling_tools_open_tag_manager()
                return
            if button_id == "tooling_sops_s3_import":
                self._tooling_s3_import("sops")
                return
            if button_id == "tooling_kbs_s3_import":
                self._tooling_s3_import("kbs")
                return
            if button_id == "settings_view_general":
                self._set_settings_view_mode("general")
                return
            if button_id == "settings_view_models":
                self._set_settings_view_mode("models")
                return
            if button_id == "settings_view_advanced":
                self._set_settings_view_mode("advanced")
                return
            if button_id == "settings_toggle_auto_approve":
                self._default_auto_approve = not self._default_auto_approve
                self._persist_project_setting_env_override(
                    "SWARMEE_AUTO_APPROVE",
                    "true" if self._default_auto_approve else "false",
                )
                self._update_prompt_placeholder()
                self._refresh_settings_general()
                state_label = "enabled" if self._default_auto_approve else "disabled"
                self._write_transcript_line(f"[settings] auto-approve {state_label}.")
                return
            if button_id == "settings_toggle_bypass_consent":
                from swarmee_river.settings import load_settings

                settings = load_settings()
                tool_consent = str(getattr(settings.safety, "tool_consent", "ask") or "ask").strip().lower()
                bypass = tool_consent == "allow"
                new_val = "false" if bypass else "true"
                self._persist_project_setting_env_override("BYPASS_TOOL_CONSENT", new_val)
                self._refresh_settings_general()
                self._write_transcript_line(
                    f"[settings] tool consent {'allow' if new_val == 'true' else 'ask'}."
                )
                return
            if button_id == "settings_toggle_esc_interrupt":
                from swarmee_river.settings import load_settings

                settings = load_settings()
                new_val = "disabled" if bool(settings.runtime.esc_interrupt_enabled) else "enabled"
                self._persist_project_setting_env_override("SWARMEE_ESC_INTERRUPT", new_val)
                self._refresh_settings_general()
                self._write_transcript_line(f"[settings] ESC interrupt {new_val}.")
                return
            if button_id == "settings_interrupt_control_apply":
                self._apply_interrupt_control_settings()
                return
            if button_id == "settings_interrupt_control_reset":
                self._reset_interrupt_control_settings()
                return
            if button_id == "settings_general_context_budget_apply":
                self._apply_context_budget_setting(
                    str(getattr(self._settings_general_context_budget_input, "value", "")).strip()
                )
                return
            if button_id == "settings_general_context_budget_reset":
                self._persist_project_context_budget_tokens(None)
                self._write_transcript_line("[settings] context budget reset to auto.")
                return
            if button_id == "settings_toggle_swarm":
                from swarmee_river.settings import load_settings

                settings = load_settings()
                new_val = "false" if bool(settings.runtime.swarm_enabled) else "true"
                self._persist_project_setting_env_override("SWARMEE_SWARM_ENABLED", new_val)
                self._refresh_settings_general()
                self._write_transcript_line(f"[settings] swarm {'enabled' if new_val == 'true' else 'disabled'}.")
                return
            if button_id == "settings_toggle_log_events":
                from swarmee_river.settings import load_settings

                settings = load_settings()
                new_val = "false" if bool(settings.diagnostics.log_events) else "true"
                self._persist_project_setting_env_override("SWARMEE_LOG_EVENTS", new_val)
                self._refresh_settings_general()
                self._write_transcript_line(f"[settings] log events {new_val}.")
                return
            if button_id == "settings_toggle_project_map":
                from swarmee_river.settings import load_settings

                settings = load_settings()
                new_val = "disabled" if bool(settings.runtime.project_map_enabled) else "enabled"
                self._persist_project_setting_env_override("SWARMEE_PROJECT_MAP", new_val)
                self._refresh_settings_general()
                self._write_transcript_line(f"[settings] project map {new_val}.")
                return
            if button_id == "settings_toggle_limit_tool_results":
                from swarmee_river.settings import load_settings

                settings = load_settings()
                new_val = "false" if bool(settings.runtime.limit_tool_results) else "true"
                self._persist_project_setting_env_override("SWARMEE_LIMIT_TOOL_RESULTS", new_val)
                self._refresh_settings_general()
                self._write_transcript_line(f"[settings] limit tool results {new_val}.")
                return
            if button_id == "settings_toggle_truncate_results":
                from swarmee_river.settings import load_settings

                settings = load_settings()
                new_val = "false" if bool(settings.context.truncate_results) else "true"
                self._persist_project_setting_env_override("SWARMEE_TRUNCATE_RESULTS", new_val)
                self._refresh_settings_general()
                self._write_transcript_line(f"[settings] truncate results {new_val}.")
                return
            if button_id == "settings_toggle_log_redact":
                from swarmee_river.settings import load_settings

                settings = load_settings()
                on = bool(settings.diagnostics.redact) and bool(settings.diagnostics.log_redact)
                new_val = "false" if on else "true"
                self._persist_project_setting_env_override("SWARMEE_DIAG_REDACT", new_val)
                self._persist_project_setting_env_override("SWARMEE_LOG_REDACT", new_val)
                self._refresh_settings_general()
                self._write_transcript_line(f"[settings] log redact {new_val}.")
                return
            if button_id == "settings_diag_bundle":
                self._create_diagnostics_bundle()
                return
            if button_id == "settings_toggle_freeze_tools":
                from swarmee_river.settings import load_settings

                settings = load_settings()
                new_val = "false" if bool(settings.runtime.freeze_tools) else "true"
                self._persist_project_setting_env_override("SWARMEE_FREEZE_TOOLS", new_val)
                self._refresh_settings_general()
                self._write_transcript_line(f"[settings] freeze tools {new_val}.")
                return
            if button_id == "settings_aws_profile_apply":
                self._apply_settings_aws_profile(self._settings_aws_profile_value(), announce=True)
                return
            if button_id == "settings_auth_connect_copilot":
                self._request_provider_connect("github_copilot")
                return
            if button_id == "settings_auth_connect_aws":
                profile = self._settings_aws_profile_value()
                self._apply_settings_aws_profile(profile, announce=False)
                self._request_provider_connect("bedrock", profile=profile or None)
                return
            if button_id == "settings_auth_refresh":
                self._refresh_settings_models()
                self._write_transcript_line("[settings] refreshed model/auth status.")
                return
            if button_id == "settings_models_open_manager":
                self._open_settings_model_manager()
                return
            if button_id == "settings_env_apply":
                key = (self._settings_env_selected_key or "").strip()
                spec = env_spec_by_key(key)
                if spec is None:
                    self._write_transcript_line("[settings] select an environment variable first.")
                    return
                value_input = str(getattr(self._settings_env_value_input, "value", "")).strip()
                value_select = str(getattr(self._settings_env_value_select, "value", "")).strip()
                if spec.choices and value_select not in {"", "__none__"}:
                    value = value_select
                else:
                    value = value_input or (value_select if value_select not in {"", "__none__"} else "")
                if value:
                    self._persist_project_setting_env_override(spec.key, value)
                    self._write_transcript_line(f"[settings] set {spec.key}")
                else:
                    self._persist_project_setting_env_override(spec.key, None)
                    self._write_transcript_line(f"[settings] unset {spec.key}")
                self._refresh_settings_env_list()
                self._refresh_settings_env_detail(spec.key)
                self._refresh_settings_models()
                return
            if button_id == "settings_env_default":
                key = (self._settings_env_selected_key or "").strip()
                spec = env_spec_by_key(key)
                if spec is None:
                    self._write_transcript_line("[settings] select an environment variable first.")
                    return
                if spec.default and spec.default != "(unset)":
                    self._persist_project_setting_env_override(spec.key, spec.default)
                    self._write_transcript_line(f"[settings] applied default for {spec.key}")
                else:
                    self._persist_project_setting_env_override(spec.key, None)
                    self._write_transcript_line(f"[settings] no explicit default for {spec.key}; variable unset.")
                self._refresh_settings_env_list()
                self._refresh_settings_env_detail(spec.key)
                self._refresh_settings_models()
                return
            if button_id == "settings_env_unset":
                key = (self._settings_env_selected_key or "").strip()
                spec = env_spec_by_key(key)
                if spec is None:
                    self._write_transcript_line("[settings] select an environment variable first.")
                    return
                self._persist_project_setting_env_override(spec.key, None)
                self._write_transcript_line(f"[settings] unset {spec.key}")
                self._refresh_settings_env_list()
                self._refresh_settings_env_detail(spec.key)
                self._refresh_settings_models()
                return
            if button_id == "settings_safety_apply":
                self._apply_agent_tools_safety_overrides(reset=False)
                return
            if button_id == "settings_safety_reset":
                self._apply_agent_tools_safety_overrides(reset=True)
                return
            if button_id == "settings_set_scope":
                import contextlib as _ctx

                with _ctx.suppress(Exception):
                    path_input = self.query_one("#settings_scope_path_input", Input)
                    path_val = path_input.value.strip()
                    if path_val:
                        target = Path(path_val).expanduser().resolve()
                        swarmee_dir = target / ".swarmee"
                        swarmee_dir.mkdir(parents=True, exist_ok=True)
                        self._persist_project_setting_env_override("SWARMEE_STATE_DIR", str(swarmee_dir))
                        self._refresh_settings_scope_display()
                        self._refresh_settings_env_list()
                        self._refresh_settings_env_detail(self._settings_env_selected_key)
                        self._write_transcript_line(f"[settings] Scope set to {swarmee_dir}")
                return
            if button_id in {"agent_view_profile", "agent_view_overview"}:
                self._set_agent_studio_view_mode("overview")
                return
            if button_id == "agent_view_builder":
                self._set_agent_studio_view_mode("builder")
                return
            if button_id == "bundle_new":
                self._new_bundle_draft()
                return
            if button_id == "bundle_save":
                self._save_bundle_from_draft()
                return
            if button_id == "bundle_delete":
                self._delete_selected_bundle()
                return
            if button_id == "bundle_load_draft":
                self._load_selected_bundle_into_draft()
                return
            if button_id == "bundle_apply":
                self._apply_bundle_selection()
                return
            if button_id == "agent_builder_open_manager":
                self._open_agent_manager()
                return
            if button_id == "agent_builder_prompt_promote":
                self._promote_inline_agent_prompt_to_asset()
                return
            if button_id == "agent_builder_edit_tools":
                self._edit_agent_builder_capability("tools")
                return
            if button_id == "agent_builder_edit_sops":
                self._edit_agent_builder_capability("sops")
                return
            if button_id == "agent_builder_edit_kb":
                self._edit_agent_builder_capability("kb")
                return
            if button_id == "session_view_timeline":
                self._set_session_view_mode("timeline")
                return
            if button_id == "session_view_artifacts":
                self._set_session_view_mode("artifacts")
                return
            if button_id.startswith("context_remove_"):
                suffix = button_id.removeprefix("context_remove_")
                with contextlib.suppress(ValueError):
                    self._remove_context_source(int(suffix))
                return
            if button_id == "context_add_file":
                self._set_context_add_mode("file")
                return
            if button_id == "context_add_note":
                self._set_context_add_mode("note")
                return
            if button_id == "context_add_sop":
                self._set_context_add_mode("sop")
                return
            if button_id == "context_add_kb":
                self._set_context_add_mode("kb")
                return
            if button_id in {"context_add_commit", "context_sop_commit"}:
                self._commit_context_add_from_ui()
                return
            if button_id in {"context_add_cancel", "context_sop_cancel"}:
                self._set_context_add_mode(None)
                return
            if button_id == "consent_choice_y":
                self._submit_consent_choice("y")
                return
            if button_id == "consent_choice_n":
                self._submit_consent_choice("n")
                return
            if button_id == "consent_choice_a":
                self._submit_consent_choice("a")
                return
            if button_id == "consent_choice_v":
                self._submit_consent_choice("v")
                return
            if button_id == "error_action_retry_tool":
                self._retry_failed_tool()
                return
            if button_id == "error_action_skip_tool":
                self._skip_failed_tool()
                return
            if button_id == "error_action_escalate":
                self._resume_after_error(escalate=True)
                return
            if button_id == "error_action_continue":
                self._resume_after_error(escalate=False)
                return
            if button_id == "error_action_dismiss":
                self._reset_error_action_prompt()
                self.action_focus_prompt()
                return
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

            if self._handle_copy_command(normalized):
                return

            if self._handle_pre_run_command(text):
                return

            if self.state.daemon.query_active:
                self._write_transcript_line("[run] already running; use /stop.")
                return

            if self._handle_post_run_command(text):
                return

            if text.startswith("/") or text.startswith(":"):
                self._write_transcript_line(f"[run] unknown command: {text}")
                return

            self._start_run(text, auto_approve=self._default_auto_approve, mode="execute")

        def _handle_plan_cancel_confirmation(self, confirmed: bool | None) -> None:
            if not confirmed:
                return
            self._cancel_plan_and_reset()

    run_tui._app_class = SwarmeeTUI  # expose for testing

    try:
        SwarmeeTUI().run()
    except KeyboardInterrupt:
        return 130
    return 0
