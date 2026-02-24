"""Optional Textual app scaffold for `swarmee tui`."""

from __future__ import annotations

import asyncio
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
from datetime import datetime
from pathlib import Path
from typing import Any

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.error_classification import (
    ERROR_CATEGORY_AUTH_ERROR,
    ERROR_CATEGORY_ESCALATABLE,
    ERROR_CATEGORY_FATAL,
    ERROR_CATEGORY_TOOL_ERROR,
    ERROR_CATEGORY_TRANSIENT,
    classify_error_message,
    normalize_error_category,
)
from swarmee_river.profiles import AgentProfile, delete_profile, list_profiles, save_profile
from swarmee_river.runtime_service.client import ensure_runtime_broker
from swarmee_river.session.graph_index import (
    build_session_graph_index,
    load_session_graph_index,
    write_session_graph_index,
)
from swarmee_river.state_paths import logs_dir, sessions_dir
from swarmee_river.tui.agent_studio import (
    _normalized_tool_name_list,
    _policy_tier_profile,
    build_agent_policy_lens,
    build_agent_team_sidebar_items,
    build_agent_tools_safety_sidebar_items,
    build_team_preset_run_prompt,
    normalize_agent_studio_view_mode,
    normalize_session_safety_overrides,
    normalize_team_preset,
    normalize_team_presets,
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
from swarmee_river.tui.event_router import handle_daemon_event as _handle_daemon_event_router
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
from swarmee_river.tui.model_select import (
    _MODEL_AUTO_VALUE,
    _MODEL_LOADING_VALUE,
    choose_daemon_model_select_value,
    choose_model_summary_parts,
    daemon_model_select_options,
    model_select_options,
    parse_model_select_value,
    resolve_model_config_summary,
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
    _DaemonTransport,
    _SocketTransport,
    _SubprocessTransport,
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
_TOOL_FAST_COMPLETE_SUPPRESS_START_S = 0.5
_THINKING_DISPLAY_DEBOUNCE_S = 0.2
_THINKING_ANIMATION_INTERVAL_S = 0.5
_THINKING_EXPORT_MAX_CHARS = 5000
_TRANSIENT_TOAST_TIMEOUT_S = 5.0
_FATAL_TOAST_TIMEOUT_S = 3600.0

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
    session_issue_actions,
    session_timeline_actions,
    summarize_session_timeline_event,
)


def _sanitize_profile_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


def _default_profile_id(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"profile-{ts.strftime('%Y%m%d-%H%M%S')}"


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


def classify_tui_error_event(event: dict[str, Any]) -> dict[str, Any]:
    message = str(event.get("message", event.get("text", ""))).strip()
    category_hint = normalize_error_category(event.get("category"))
    tool_use_id = str(event.get("tool_use_id", "")).strip() or None
    classified = classify_error_message(message, category_hint=category_hint, tool_use_id=tool_use_id)
    retry_after_raw = event.get("retry_after_s")
    retry_after_s: int | None = None
    if isinstance(retry_after_raw, (int, float)):
        retry_after_s = int(retry_after_raw)
    elif isinstance(retry_after_raw, str) and retry_after_raw.strip().isdigit():
        retry_after_s = int(retry_after_raw.strip())
    next_tier = str(event.get("next_tier", "")).strip() or None
    return {
        "message": message,
        "category": str(classified.get("category", ERROR_CATEGORY_FATAL)),
        "retryable": bool(event.get("retryable", classified.get("retryable", False))),
        "tool_use_id": str(classified.get("tool_use_id", "")).strip() or None,
        "retry_after_s": retry_after_s if isinstance(retry_after_s, int) and retry_after_s > 0 else None,
        "next_tier": next_tier,
    }


def summarize_error_for_toast(error_info: dict[str, Any]) -> tuple[str, str, float | None]:
    category = str(error_info.get("category", ERROR_CATEGORY_FATAL))
    message = str(error_info.get("message", "")).strip()
    retry_after_s = error_info.get("retry_after_s")

    if category == ERROR_CATEGORY_TRANSIENT:
        delay = int(retry_after_s) if isinstance(retry_after_s, int) and retry_after_s > 0 else 1
        return f"Rate limited - retrying in {delay}s", "warning", _TRANSIENT_TOAST_TIMEOUT_S

    if category == ERROR_CATEGORY_TOOL_ERROR:
        tool_use_id = str(error_info.get("tool_use_id", "")).strip()
        if tool_use_id:
            return f"Tool failed ({tool_use_id})", "error", 6.0
        return "Tool execution failed", "error", 6.0

    if category == ERROR_CATEGORY_ESCALATABLE:
        return "Model/context limit hit - escalation available", "warning", 8.0

    if category == ERROR_CATEGORY_AUTH_ERROR:
        return "Auth/permissions error - check credentials", "error", 10.0

    if message:
        first = message.splitlines()[0].strip()
        if first:
            return first[:140], "error", _FATAL_TOAST_TIMEOUT_S
    return "Fatal error", "error", _FATAL_TOAST_TIMEOUT_S


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
    Button = textual_widgets.Button
    Checkbox = textual_widgets.Checkbox
    Header = textual_widgets.Header
    Footer = textual_widgets.Footer
    Input = textual_widgets.Input
    RichLog = textual_widgets.RichLog
    Select = textual_widgets.Select
    Static = textual_widgets.Static
    TabbedContent = textual_widgets.TabbedContent
    TextArea = textual_widgets.TextArea

    from swarmee_river.tui.views.agent_studio import wire_agent_studio_widgets
    from swarmee_river.tui.views.session import wire_session_widgets
    from swarmee_river.tui.views.sidebar import compose_sidebar
    from swarmee_river.tui.widgets import (
        ActionSheet,
        CommandPalette,
        ConsentPrompt,
        ContextBudgetBar,
        ErrorActionPrompt,
        SidebarDetail,
        SidebarHeader,
        SidebarList,
        StatusBar,
        ThinkingBar,
        extract_consent_tool_name,
        format_tool_input_oneliner,
        render_agent_profile_summary_text,
        render_assistant_message,
        render_plan_panel,
        render_system_message,
        render_tool_details_panel,
        render_tool_heartbeat_line,
        render_tool_progress_chunk,
        render_tool_result_line,
        render_tool_start_line_with_input,
        render_user_message,
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

        #context_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
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

        #artifacts_list {
            margin: 0 0 1 0;
            min-height: 10;
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

        #session_timeline_view, #session_issues_view {
            height: 1fr;
            layout: vertical;
        }

        #session_timeline_list, #session_issue_list {
            margin: 0 0 1 0;
            min-height: 10;
        }

        #session_timeline_detail, #session_issue_detail {
            height: 1fr;
        }

        #session_issue_list {
            margin: 0 0 1 0;
            min-height: 10;
        }

        #session_issue_detail {
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

        #agent_profile_view, #agent_tools_view, #agent_team_view {
            height: 1fr;
            layout: vertical;
        }

        #agent_summary_header, #agent_profiles_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #agent_profile_list {
            width: 1fr;
            margin: 0 0 1 0;
            min-height: 8;
        }

        #agent_profile_meta_row {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_profile_id, #agent_profile_name {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_profile_name {
            margin: 0;
        }

        #agent_profile_actions {
            height: auto;
            margin: 0 0 1 0;
        }

        #agent_profile_status {
            height: auto;
            color: $text-muted;
        }

        #agent_tools_list, #agent_team_list {
            width: 1fr;
            margin: 0 0 1 0;
            min-height: 8;
        }

        #agent_tools_detail, #agent_team_detail {
            height: 1fr;
        }

        #agent_tools_overrides_header {
            height: auto;
            color: $text-muted;
            padding: 1 0 0 0;
        }

        #agent_tools_override_consent, #agent_tools_override_allowlist, #agent_tools_override_blocklist {
            width: 1fr;
            margin: 0 0 1 0;
        }

        #agent_tools_override_actions {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_tools_override_actions Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_tools_override_status {
            height: auto;
            color: $text-muted;
        }

        #agent_team_editor_header {
            height: auto;
            color: $text-muted;
            padding: 1 0 0 0;
        }

        #agent_team_meta_row {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_team_preset_id, #agent_team_preset_name {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_team_preset_name {
            margin: 0;
        }

        #agent_team_preset_description {
            width: 1fr;
            margin: 0 0 1 0;
        }

        #agent_team_preset_spec {
            width: 1fr;
            height: 8;
            margin: 0 0 1 0;
        }

        #agent_team_actions {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_team_actions Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_team_status {
            height: auto;
            color: $text-muted;
        }

        #context_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #sops_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #context_sources_list {
            height: 1fr;
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

        #sop_list {
            height: 1fr;
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

        .context-source-row {
            layout: horizontal;
            height: auto;
            margin: 0;
            padding: 0;
        }

        .sop-row {
            height: auto;
            margin: 0 0 1 0;
            padding: 0;
            layout: vertical;
            border-bottom: solid #333333;
        }

        .sop-row-header {
            height: auto;
            layout: horizontal;
            margin: 0;
            padding: 0;
        }

        .sop-source-label {
            width: auto;
            min-width: 14;
            color: $text-muted;
            padding: 0 0 0 1;
        }

        .sop-preview {
            color: $text-muted;
            padding: 0 0 0 3;
        }

        .context-source-label {
            width: 1fr;
            color: $text;
            padding: 0 1 0 0;
        }

        .context-remove-btn {
            width: 3;
            min-width: 3;
            margin: 0;
            padding: 0;
        }

        #context_add_row {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #context_add_row Button {
            width: 1fr;
            min-width: 7;
            margin: 0 1 0 0;
        }

        #context_input_row, #context_sop_row {
            height: auto;
            layout: horizontal;
            margin: 0;
        }

        #context_input, #context_sop_select {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #plan_actions {
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
            ("ctrl+t", "toggle_transcript_mode", "Toggle transcript mode"),
            Binding("ctrl+k", "open_action_sheet", "Actions", priority=True),
            Binding("ctrl+space", "open_action_sheet", "Actions", priority=True),
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
        _sop_toggle_id_to_name: dict[str, str] = {}
        _sops_ready_for_sync: bool = False
        # Conversation view state
        _current_assistant_chunks: list[str] = []
        _streaming_buffer: list[str] = []
        _streaming_flush_timer: Any = None
        _tool_progress_pending_ids: set[str] = set()
        _tool_progress_flush_timer: Any = None
        _current_assistant_model: str | None = None
        _current_assistant_timestamp: str | None = None
        _assistant_placeholder_written: bool = False
        _current_thinking: bool = False
        _thinking_buffer: list[str] = []
        _thinking_char_count: int = 0
        _thinking_display_timer: Any = None
        _thinking_animation_timer: Any = None
        _thinking_started_mono: float | None = None
        _thinking_frame_index: int = 0
        _last_thinking_text: str = ""
        _tool_blocks: dict[str, dict[str, Any]] = {}
        _tool_pending_start: dict[str, float] = {}
        _tool_pending_start_timers: dict[str, Any] = {}
        _transcript_mode: str = "rich"
        _transcript_fallback_lines: list[str] = []
        _consent_prompt_widget: Any = None  # ConsentPrompt | None
        _error_action_prompt_widget: Any = None  # ErrorActionPrompt | None
        _pending_error_action: dict[str, Any] | None = None
        _context_sources_list: Any = None  # VerticalScroll | None
        _sop_list: Any = None  # VerticalScroll | None
        _context_input: Any = None  # Input | None
        _context_sop_select: Any = None  # Select | None
        _session_header: Any = None  # SidebarHeader | None
        _session_view_timeline_button: Any = None  # Button | None
        _session_view_issues_button: Any = None  # Button | None
        _session_timeline_view: Any = None  # Vertical | None
        _session_issues_view: Any = None  # Vertical | None
        _session_timeline_header: Any = None  # SidebarHeader | None
        _session_timeline_list: Any = None  # SidebarList | None
        _session_timeline_detail: Any = None  # SidebarDetail | None
        _session_issue_list: Any = None  # SidebarList | None
        _session_issue_detail: Any = None  # SidebarDetail | None
        _artifacts_header: Any = None  # SidebarHeader | None
        _artifacts_list: Any = None  # SidebarList | None
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
        _prompt_history: list[str] = []
        _history_index: int = -1
        _MAX_PROMPT_HISTORY: int = 50
        _TRANSCRIPT_MAX_LINES: int = 5000
        _split_ratio: int = 2
        _search_active: bool = False
        _agent_view_profile_button: Any = None  # Button | None
        _agent_view_tools_button: Any = None  # Button | None
        _agent_view_team_button: Any = None  # Button | None
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

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.state = AppState()

        def compose(self) -> Any:
            yield Header()
            with Horizontal(id="panes"):
                yield RichLog(id="transcript")
                yield TextArea(
                    text="",
                    read_only=True,
                    show_line_numbers=False,
                    id="transcript_text",
                    soft_wrap=True,
                )
                yield from compose_sidebar(context_select_placeholder=_CONTEXT_SELECT_PLACEHOLDER)
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
                    yield Select(
                        options=[("Loading model info...", _MODEL_LOADING_VALUE)],
                        allow_blank=False,
                        id="model_select",
                        compact=True,
                    )
            yield Footer()

        def on_mount(self) -> None:
            self._command_palette = self.query_one("#command_palette", CommandPalette)
            self._action_sheet = self.query_one("#action_sheet", ActionSheet)
            self._thinking_bar = self.query_one("#thinking_bar", ThinkingBar)
            self._status_bar = self.query_one("#status_bar", StatusBar)
            self._consent_prompt_widget = self.query_one("#consent_prompt", ConsentPrompt)
            self._error_action_prompt_widget = self.query_one("#error_action_prompt", ErrorActionPrompt)
            self._context_sources_list = self.query_one("#context_sources_list", VerticalScroll)
            self._sop_list = self.query_one("#sop_list", VerticalScroll)
            self._context_input = self.query_one("#context_input", Input)
            self._context_sop_select = self.query_one("#context_sop_select", Select)
            wire_session_widgets(self)
            self._artifacts_header = self.query_one("#artifacts_header", SidebarHeader)
            self._artifacts_list = self.query_one("#artifacts_list", SidebarList)
            self._artifacts_detail = self.query_one("#artifacts_detail", SidebarDetail)
            wire_agent_studio_widgets(self)
            self._prompt_metrics = self.query_one("#prompt_metrics", ContextBudgetBar)
            self._status_bar.set_model(self._current_model_summary())
            self.query_one("#prompt", PromptTextArea).focus()
            self._reset_plan_panel()
            self._reset_issues_panel()
            self._reset_session_timeline_panel()
            self._reset_artifacts_panel()
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            self._set_session_view_mode("timeline")
            self._set_context_add_mode(None)
            self._refresh_context_sop_options()
            self._render_context_sources_panel()
            self._refresh_sop_catalog()
            self._render_sop_panel()
            self._reload_saved_profiles()
            self._render_agent_tools_panel()
            self._render_agent_team_panel()
            self._set_agent_studio_view_mode("profile")
            if self.state.agent_studio.saved_profiles:
                self._load_profile_into_draft(self.state.agent_studio.saved_profiles[0])
            else:
                self._new_agent_profile_draft(announce=False)
            self._refresh_agent_summary()
            self._refresh_model_select()
            self.title = "Swarmee"
            self.sub_title = self._current_model_summary()
            self._update_prompt_placeholder()
            # Show ASCII art banner at the top of the transcript.
            # Write plain lines so selection/export keeps exact banner text.
            from swarmee_river.utils.welcome_utils import SWARMEE_BANNER

            for banner_line in SWARMEE_BANNER.strip().splitlines():
                self._mount_transcript_widget(banner_line, plain_text=banner_line)
            self._write_transcript("Starting Swarmee daemon...")
            self._write_transcript(self.sub_title)
            self._write_transcript("Tips: use /commands in the prompt and the Agent tab for profile actions.")
            transcript = self.query_one("#transcript", RichLog)
            with contextlib.suppress(Exception):
                transcript.auto_scroll = True
            with contextlib.suppress(Exception):
                transcript.max_lines = self._TRANSCRIPT_MAX_LINES
            self._set_transcript_mode("rich", notify=False)
            self._load_session()
            if self.state.daemon.session_id:
                self._schedule_session_timeline_refresh(delay=0.1)
            self._refresh_agent_summary()
            self._spawn_daemon()

        def _record_transcript_fallback(self, text: str) -> None:
            clean = sanitize_output_text(text).rstrip("\n")
            if not clean:
                return
            self._transcript_fallback_lines.extend(clean.splitlines())
            if len(self._transcript_fallback_lines) > self._TRANSCRIPT_MAX_LINES:
                self._transcript_fallback_lines = self._transcript_fallback_lines[-self._TRANSCRIPT_MAX_LINES :]

        def _sync_transcript_text_widget(self, *, scroll_to_end: bool = True) -> None:
            text_widget = self.query_one("#transcript_text", TextArea)
            text = "\n".join(self._transcript_fallback_lines).rstrip()
            if text:
                text += "\n"
            text_widget.load_text(text)
            if scroll_to_end:
                self._scroll_transcript_text_to_end()

        def _scroll_transcript_text_to_end(self) -> None:
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
            normalized = mode.strip().lower()
            if normalized not in {"rich", "text"}:
                return
            rich_widget = self.query_one("#transcript", RichLog)
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
            """Write a renderable into the transcript RichLog."""
            transcript = self.query_one("#transcript", RichLog)
            transcript.write(renderable)
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
            self._mount_transcript_widget(render_system_message(line), plain_text=line)

        def _cancel_thinking_display_timer(self) -> None:
            timer = self._thinking_display_timer
            self._thinking_display_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _cancel_thinking_animation_timer(self) -> None:
            timer = self._thinking_animation_timer
            self._thinking_animation_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _thinking_elapsed_s(self) -> float:
            started = self._thinking_started_mono
            if not isinstance(started, float):
                return 0.0
            return max(0.0, time.monotonic() - started)

        def _thinking_preview(self) -> str:
            for chunk in reversed(self._thinking_buffer):
                text = sanitize_output_text(str(chunk or "")).strip()
                if text:
                    return text
            return ""

        def _render_thinking_bar(self) -> None:
            bar = self._thinking_bar
            if bar is None:
                return
            bar.show_thinking(
                char_count=max(0, int(self._thinking_char_count)),
                elapsed_s=self._thinking_elapsed_s(),
                preview=self._thinking_preview(),
                frame_index=self._thinking_frame_index,
            )

        def _on_thinking_display_timer(self) -> None:
            self._thinking_display_timer = None
            if self._current_thinking:
                self._render_thinking_bar()

        def _schedule_thinking_display_update(self) -> None:
            self._cancel_thinking_display_timer()
            self._thinking_display_timer = self.set_timer(_THINKING_DISPLAY_DEBOUNCE_S, self._on_thinking_display_timer)

        def _on_thinking_animation_tick(self) -> None:
            if not self._current_thinking:
                return
            self._thinking_frame_index = (self._thinking_frame_index + 1) % 3
            self._render_thinking_bar()

        def _ensure_thinking_animation_timer(self) -> None:
            if self._thinking_animation_timer is not None:
                return
            self._thinking_animation_timer = self.set_interval(
                _THINKING_ANIMATION_INTERVAL_S,
                self._on_thinking_animation_tick,
            )

        def _reset_thinking_state(self) -> None:
            self._cancel_thinking_display_timer()
            self._cancel_thinking_animation_timer()
            self._current_thinking = False
            self._thinking_buffer = []
            self._thinking_char_count = 0
            self._thinking_started_mono = None
            self._thinking_frame_index = 0
            bar = self._thinking_bar
            if bar is not None:
                with contextlib.suppress(Exception):
                    bar.hide_thinking()

        def _record_thinking_event(self, thinking_text: str) -> None:
            chunk = sanitize_output_text(str(thinking_text or ""))
            first_event = not self._current_thinking and not self._thinking_buffer and self._thinking_char_count == 0
            if first_event:
                self._current_thinking = True
                self._thinking_started_mono = time.monotonic()
                self._thinking_frame_index = 0
                self._ensure_thinking_animation_timer()
            else:
                self._current_thinking = True
            if chunk:
                self._thinking_buffer.append(chunk)
                self._thinking_char_count += len(chunk)
            self._render_thinking_bar()
            self._schedule_thinking_display_update()

        def _dismiss_thinking(self, *, emit_summary: bool = False) -> None:
            """Hide thinking indicator and optionally persist a transcript summary."""
            had_thinking = bool(
                self._current_thinking
                or self._thinking_started_mono is not None
                or self._thinking_char_count > 0
                or self._thinking_buffer
            )
            if not had_thinking:
                bar = self._thinking_bar
                if bar is not None:
                    with contextlib.suppress(Exception):
                        bar.hide_thinking()
                return

            elapsed_s = self._thinking_elapsed_s()
            full_thinking = "".join(self._thinking_buffer)
            self._last_thinking_text = full_thinking
            char_count = self._thinking_char_count

            if emit_summary:
                elapsed_label = max(0, int(round(elapsed_s)))
                summary_line = f"💭 Thought for {elapsed_label}s"
                if char_count > 0:
                    summary_line += f" ({char_count:,} chars)"
                self._mount_transcript_widget(render_system_message(summary_line), plain_text=summary_line)

            self._reset_thinking_state()

        def _cancel_streaming_flush_timer(self) -> None:
            timer = self._streaming_flush_timer
            self._streaming_flush_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
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
            if not self._streaming_buffer:
                return
            text = "".join(self._streaming_buffer)
            self._streaming_buffer = []
            if not text:
                return
            self._current_assistant_chunks.append(text)
            # Render streamed assistant text incrementally; avoid waiting for text_complete.
            self._mount_transcript_widget(text, plain_text=text)
            self._assistant_placeholder_written = True

        def _cancel_tool_progress_flush_timer(self) -> None:
            timer = self._tool_progress_flush_timer
            self._tool_progress_flush_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
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

        def _write_transcript_line(self, line: str) -> None:
            """Write a plain text line to the transcript (used for TUI-internal messages)."""
            if self.state.daemon.query_active:
                self.state.daemon.turn_output_chunks.append(sanitize_output_text(f"[tui] {line}\n"))
            self._write_transcript(line)

        def _tool_input_summary(self, tool_name: str, tool_input: Any) -> str:
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
                with contextlib.suppress(Exception):
                    timer.stop()

        def _clear_pending_tool_starts(self) -> None:
            for tool_use_id in list(self._tool_pending_start_timers.keys()):
                self._cancel_tool_start_timer(tool_use_id)
            self._tool_pending_start_timers = {}
            self._tool_pending_start = {}

        def _emit_tool_start_line(self, tool_use_id: str) -> bool:
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
            self._mount_transcript_widget(
                render_tool_start_line_with_input(tool_name, tool_input=tool_input, tool_use_id=tool_use_id),
                plain_text=self._tool_start_plain_text(tool_name, tool_input),
            )
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
            record = self._tool_blocks.get(tool_use_id)
            if record is None:
                return False
            now = time.monotonic()
            last = float(record.get("last_progress_render_mono", 0.0))
            pending = str(record.get("pending_output", ""))
            if pending:
                if force or (now - last) >= _TOOL_PROGRESS_RENDER_INTERVAL_S:
                    stream = str(record.get("pending_stream", "stdout") or "stdout")
                    self._mount_transcript_widget(
                        render_tool_progress_chunk(pending, stream=stream),
                        plain_text=pending,
                    )
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
            self._mount_transcript_widget(
                render_tool_heartbeat_line(tool_name, elapsed_s=elapsed_s, tool_use_id=tool_use_id),
                plain_text=f"⚙ {tool_name} running... ({elapsed_s:.1f}s)",
            )
            record["last_progress_render_mono"] = now
            record["last_heartbeat_rendered_s"] = elapsed_s
            return True

        def _call_from_thread_safe(self, callback: Any, *args: Any, **kwargs: Any) -> None:
            if self.state.daemon.is_shutting_down:
                return
            with contextlib.suppress(Exception):
                self.call_from_thread(callback, *args, **kwargs)

        def _warn_run_active_tier_change_once(self) -> None:
            if self.state.daemon.run_active_tier_warning_emitted:
                return
            self.state.daemon.run_active_tier_warning_emitted = True
            self._write_transcript_line(_RUN_ACTIVE_TIER_WARNING)

        def _write_user_input(self, text: str) -> None:
            timestamp = self._turn_timestamp()
            plain = f"YOU> {text}\n{timestamp}"
            self._mount_transcript_widget(render_user_message(text, timestamp=timestamp), plain_text=plain)

        def _write_user_message(self, text: str, *, timestamp: str | None = None) -> None:
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
            return datetime.now().strftime("%I:%M %p").lstrip("0")

        def _append_plain_text(self, text: str) -> None:
            """Write a plain text line for non-event fallback output."""
            if not text.strip():
                return
            self._mount_transcript_widget(text, plain_text=text)

        def _set_plan_panel(self, content: str) -> None:
            self.state.plan.text = content
            plan_panel = self.query_one("#plan", TextArea)
            text = content if content.strip() else "(no plan)"
            plan_panel.load_text(text)
            plan_panel.scroll_end(animate=False)

        def _extract_plan_step_descriptions(self, plan_json: dict[str, Any]) -> list[str]:
            steps_raw = plan_json.get("steps", [])
            if not isinstance(steps_raw, list):
                return []
            steps: list[str] = []
            for step in steps_raw:
                if isinstance(step, str):
                    desc = step.strip()
                elif isinstance(step, dict):
                    desc = str(step.get("description", step.get("title", step))).strip()
                else:
                    desc = str(step).strip()
                if desc:
                    steps.append(desc)
            return steps

        def _refresh_plan_status_bar(self) -> None:
            if self._status_bar is None:
                return
            if not self.state.daemon.query_active:
                self._status_bar.set_plan_step(current=None, total=None)
                return
            total = self.state.plan.current_steps_total
            if total <= 0:
                self._status_bar.set_plan_step(current=None, total=None)
                return
            current: int | None = None
            if isinstance(self.state.plan.current_active_step, int) and self.state.plan.current_active_step >= 0:
                current = self.state.plan.current_active_step + 1
            else:
                completed = sum(1 for item in self.state.plan.current_step_statuses if item == "completed")
                if completed >= total:
                    current = total
                elif completed > 0:
                    current = completed
            self._status_bar.set_plan_step(current=current, total=total)

        def _render_plan_panel_from_status(self) -> None:
            if self.state.plan.current_steps_total <= 0 or not self.state.plan.current_steps:
                return
            text_lines: list[str] = []
            if self.state.plan.current_summary:
                text_lines.append(self.state.plan.current_summary)
                text_lines.append("")
            for index, desc in enumerate(self.state.plan.current_steps, start=1):
                status = (
                    self.state.plan.current_step_statuses[index - 1]
                    if index - 1 < len(self.state.plan.current_step_statuses)
                    else "pending"
                )
                marker = "☐"
                if status == "in_progress":
                    marker = "▶"
                elif status == "completed":
                    marker = "☑"
                text_lines.append(f"{marker} {index}. {desc}")
            text_lines.append("")
            text_lines.append("/approve  /replan  /clearplan")
            self._set_plan_panel("\n".join(text_lines))
            self._refresh_plan_status_bar()

        def _selected_agent_profile_id(self) -> str | None:
            sidebar_list = self._agent_profile_list
            if sidebar_list is None:
                return None
            getter = getattr(sidebar_list, "selected_id", None)
            if not callable(getter):
                return None
            value = str(getter() or "").strip()
            if not value or value == _AGENT_PROFILE_SELECT_NONE:
                return None
            return value

        def _lookup_saved_profile(self, profile_id: str | None) -> AgentProfile | None:
            target = str(profile_id or "").strip()
            if not target:
                return None
            for profile in self.state.agent_studio.saved_profiles:
                if profile.id == target:
                    return profile
            return None

        def _set_agent_form_values(self, *, profile_id: str, profile_name: str) -> None:
            self.state.agent_studio.form_syncing = True
            try:
                if self._agent_profile_id_input is not None:
                    self._agent_profile_id_input.value = profile_id
                if self._agent_profile_name_input is not None:
                    self._agent_profile_name_input.value = profile_name
            finally:
                self.state.agent_studio.form_syncing = False

        def _agent_tools_item_by_id(self, item_id: str | None) -> dict[str, Any] | None:
            target = str(item_id or "").strip()
            if not target:
                return None
            for item in self.state.agent_studio.tools_items:
                if str(item.get("id", "")).strip() == target:
                    return item
            return None

        def _set_agent_tools_status(self, message: str) -> None:
            widget = self._agent_tools_override_status
            if widget is None:
                return
            text = message.strip() if isinstance(message, str) else ""
            widget.update(text)

        def _set_agent_tools_override_form_values(self, overrides: dict[str, Any] | None) -> None:
            normalized = normalize_session_safety_overrides(overrides)
            consent = str(normalized.get("tool_consent", "")).strip().lower()
            allow = _normalized_tool_name_list(normalized.get("tool_allowlist"))
            block = _normalized_tool_name_list(normalized.get("tool_blocklist"))
            self.state.agent_studio.tools_form_syncing = True
            try:
                if self._agent_tools_override_consent_input is not None:
                    self._agent_tools_override_consent_input.value = consent
                if self._agent_tools_override_allowlist_input is not None:
                    self._agent_tools_override_allowlist_input.value = ", ".join(allow)
                if self._agent_tools_override_blocklist_input is not None:
                    self._agent_tools_override_blocklist_input.value = ", ".join(block)
            finally:
                self.state.agent_studio.tools_form_syncing = False

        def _parse_agent_tools_csv_list(self, value: str) -> list[str]:
            if not value.strip():
                return []
            tokens = [token.strip() for token in value.split(",")]
            return _normalized_tool_name_list(tokens)

        def _agent_tools_form_update_payload(self) -> dict[str, Any] | None:
            raw_consent = str(getattr(self._agent_tools_override_consent_input, "value", "")).strip().lower()
            if raw_consent and raw_consent not in _AGENT_TOOL_CONSENT_VALUES:
                self._notify("tool_consent must be ask|allow|deny.", severity="warning")
                return None
            allow_csv = str(getattr(self._agent_tools_override_allowlist_input, "value", ""))
            block_csv = str(getattr(self._agent_tools_override_blocklist_input, "value", ""))
            allow_list = self._parse_agent_tools_csv_list(allow_csv)
            block_list = self._parse_agent_tools_csv_list(block_csv)
            return {
                "tool_consent": raw_consent or None,
                "tool_allowlist": allow_list or None,
                "tool_blocklist": block_list or None,
            }

        def _current_agent_policy_tier_name(self) -> str | None:
            return (
                str(self.state.daemon.tier or "").strip().lower()
                or str(self.state.daemon.model_tier_override or "").strip().lower()
                or None
            )

        def _refresh_agent_tools_policy_lens(self) -> None:
            self.state.agent_studio.tools_policy_lens = build_agent_policy_lens(
                tier_name=self._current_agent_policy_tier_name(),
                overrides=self.state.agent_studio.session_safety_overrides,
            )

        def _apply_agent_tools_safety_overrides(self, *, reset: bool = False) -> None:
            proc = self.state.daemon.proc
            if self.state.daemon.query_active:
                self._set_agent_tools_status("Cannot update overrides while a run is active.")
                self._notify("Cannot update overrides while a run is active.", severity="warning")
                return
            if not self.state.daemon.ready or proc is None or proc.poll() is not None:
                self._set_agent_tools_status("Daemon is not ready.")
                self._notify("Daemon is not ready.", severity="warning")
                return
            payload = (
                {"tool_consent": None, "tool_allowlist": None, "tool_blocklist": None}
                if reset
                else self._agent_tools_form_update_payload()
            )
            if payload is None:
                return
            command: dict[str, Any] = {"cmd": "set_safety_overrides"}
            command.update(payload)
            if not send_daemon_command(proc, command):
                self._set_agent_tools_status("Failed to send override update.")
                self._notify("Failed to send safety override update.", severity="warning")
                return
            if reset:
                self._set_agent_tools_status("Resetting overrides...")
            else:
                self._set_agent_tools_status("Applying overrides...")

        def _set_agent_tools_selection(self, item: dict[str, Any] | None) -> None:
            detail = self._agent_tools_detail
            if detail is None:
                return
            if item is None:
                self.state.agent_studio.tools_selected_item_id = None
                detail.set_preview("(no tools/safety items)")
                detail.set_actions([])
                return
            self.state.agent_studio.tools_selected_item_id = str(item.get("id", "")).strip() or None
            detail.set_preview(render_agent_tools_safety_detail_text(item, self.state.agent_studio.tools_policy_lens))
            detail.set_actions([])

        def _agent_team_item_by_id(self, item_id: str | None) -> dict[str, Any] | None:
            target = str(item_id or "").strip()
            if not target:
                return None
            for item in self.state.agent_studio.team_items:
                if str(item.get("id", "")).strip() == target:
                    return item
            return None

        def _set_agent_team_status(self, message: str) -> None:
            widget = self._agent_team_status
            if widget is None:
                return
            text = message.strip() if isinstance(message, str) else ""
            widget.update(text)

        def _set_agent_team_form_values(self, preset: dict[str, Any] | None) -> None:
            normalized = normalize_team_preset(preset) if isinstance(preset, dict) else None
            preset_id = str(normalized.get("id", "")).strip() if normalized else ""
            name = str(normalized.get("name", "")).strip() if normalized else ""
            description = str(normalized.get("description", "")).strip() if normalized else ""
            spec_text = (
                _json.dumps(normalized.get("spec", {}), ensure_ascii=False, indent=2, sort_keys=True)
                if normalized
                else "{}"
            )
            self.state.agent_studio.team_form_syncing = True
            try:
                if self._agent_team_preset_id_input is not None:
                    self._agent_team_preset_id_input.value = preset_id
                if self._agent_team_preset_name_input is not None:
                    self._agent_team_preset_name_input.value = name
                if self._agent_team_preset_description_input is not None:
                    self._agent_team_preset_description_input.value = description
                if self._agent_team_preset_spec_input is not None:
                    self._agent_team_preset_spec_input.load_text(spec_text)
            finally:
                self.state.agent_studio.team_form_syncing = False

        def _agent_team_form_payload(self) -> dict[str, Any] | None:
            raw_name = str(getattr(self._agent_team_preset_name_input, "value", "")).strip()
            if not raw_name:
                self._notify("Preset name is required.", severity="warning")
                return None
            raw_id = str(getattr(self._agent_team_preset_id_input, "value", "")).strip()
            preset_id = _sanitize_profile_token(raw_id or raw_name)
            description = str(getattr(self._agent_team_preset_description_input, "value", "")).strip()
            spec_text = str(getattr(self._agent_team_preset_spec_input, "text", "")).strip() or "{}"
            try:
                spec = _json.loads(spec_text)
            except Exception:
                self._notify("Preset spec must be valid JSON.", severity="warning")
                return None
            if not isinstance(spec, dict):
                self._notify("Preset spec must be a JSON object.", severity="warning")
                return None
            normalized = normalize_team_preset(
                {"id": preset_id, "name": raw_name, "description": description, "spec": spec}
            )
            if normalized is None:
                self._notify("Preset is invalid.", severity="warning")
                return None
            return normalized

        def _selected_team_preset(self) -> dict[str, Any] | None:
            selected_item = self._agent_team_item_by_id(self.state.agent_studio.team_selected_item_id)
            if selected_item is None:
                return None
            return normalize_team_preset(selected_item.get("preset"))

        def _set_prompt_editor_text(self, text: str) -> bool:
            content = str(text or "")
            if not content.strip():
                return False
            with contextlib.suppress(Exception):
                prompt_widget = self.query_one("#prompt", PromptTextArea)
                prompt_widget.clear()
                loader = getattr(prompt_widget, "load_text", None)
                if callable(loader):
                    loader(content)
                else:
                    prompt_widget.insert(content)
                prompt_widget.focus()
                return True
            return False

        def _new_agent_team_preset_draft(self) -> None:
            seed = normalize_team_preset(
                {
                    "id": _default_team_preset_id(),
                    "name": _default_team_preset_name(),
                    "description": "",
                    "spec": {},
                }
            )
            if seed is None:
                return
            self.state.agent_studio.team_selected_item_id = None
            self._set_agent_team_form_values(seed)
            self._set_agent_team_status("New team preset draft.")
            self._set_agent_draft_dirty(True, note="Team preset draft updated.")

        def _save_agent_team_preset_draft(self) -> None:
            payload = self._agent_team_form_payload()
            if payload is None:
                return
            selected_id = str(self.state.agent_studio.team_selected_item_id or "").strip()
            saved_id = str(payload.get("id", "")).strip()
            next_presets = [dict(item) for item in normalize_team_presets(self.state.agent_studio.team_presets)]
            if selected_id and selected_id != saved_id:
                next_presets = [item for item in next_presets if str(item.get("id", "")).strip() != selected_id]
            replaced = False
            for idx, existing in enumerate(next_presets):
                if str(existing.get("id", "")).strip() == saved_id:
                    next_presets[idx] = payload
                    replaced = True
                    break
            if not replaced:
                next_presets.append(payload)
            self.state.agent_studio.team_presets = normalize_team_presets(next_presets)
            self.state.agent_studio.team_selected_item_id = saved_id
            self._render_agent_team_panel()
            self._set_agent_team_status(f"Saved preset '{payload['name']}' in draft.")
            self._set_agent_draft_dirty(True, note=f"Team preset '{payload['name']}' saved in draft.")

        def _delete_selected_agent_team_preset(self) -> None:
            selected = self._selected_team_preset()
            if selected is None:
                self._notify("Select a team preset first.", severity="warning")
                return
            selected_id = str(selected.get("id", "")).strip()
            next_presets = [
                item for item in self.state.agent_studio.team_presets if str(item.get("id", "")).strip() != selected_id
            ]
            self.state.agent_studio.team_presets = normalize_team_presets(next_presets)
            self.state.agent_studio.team_selected_item_id = None
            self._render_agent_team_panel()
            self._set_agent_team_status(f"Deleted preset '{selected.get('name', selected_id)}' from draft.")
            self._set_agent_draft_dirty(True, note=f"Team preset '{selected_id}' removed from draft.")

        def _insert_agent_team_preset_run_prompt(self, *, run_now: bool = False) -> None:
            preset = self._selected_team_preset()
            if preset is None:
                preset = self._agent_team_form_payload()
            if preset is None:
                return
            prompt = build_team_preset_run_prompt(preset)
            if not prompt:
                self._notify("Failed to build team preset prompt.", severity="warning")
                return
            if not self._set_prompt_editor_text(prompt):
                self._notify("Prompt editor is unavailable.", severity="warning")
                return
            self._set_agent_team_status(f"Inserted run prompt for '{preset['name']}'.")
            if run_now:
                self.action_submit_prompt()

        def _set_agent_team_selection(self, item: dict[str, Any] | None) -> None:
            detail = self._agent_team_detail
            if detail is None:
                return
            if item is None:
                self.state.agent_studio.team_selected_item_id = None
                detail.set_preview("(no team items)")
                detail.set_actions([])
                self._set_agent_team_form_values(None)
                return
            self.state.agent_studio.team_selected_item_id = str(item.get("id", "")).strip() or None
            detail.set_preview(render_agent_team_detail_text(item))
            detail.set_actions([])
            preset = normalize_team_preset(item.get("preset"))
            self._set_agent_team_form_values(preset)

        def _refresh_agent_tools_header(self) -> None:
            header = self._agent_tools_header
            if header is None:
                return
            effective = (
                self.state.agent_studio.tools_policy_lens.get("effective", {})
                if isinstance(self.state.agent_studio.tools_policy_lens.get("effective"), dict)
                else {}
            )
            overrides = (
                self.state.agent_studio.tools_policy_lens.get("session_overrides", {})
                if isinstance(self.state.agent_studio.tools_policy_lens.get("session_overrides"), dict)
                else {}
            )
            consent = str(effective.get("tool_consent", "ask")).strip().lower() or "ask"
            header.set_badges([f"consent {consent}", f"overrides {len(overrides)}"])

        def _refresh_agent_team_header(self) -> None:
            header = self._agent_team_header
            if header is None:
                return
            header.set_badges([f"presets {len(self.state.agent_studio.team_presets)}"])

        def _render_agent_tools_panel(self) -> None:
            self._refresh_agent_tools_policy_lens()
            self.state.agent_studio.tools_items = [
                dict(item) for item in build_agent_tools_safety_sidebar_items(self.state.agent_studio.tools_policy_lens)
            ]
            list_widget = self._agent_tools_list
            if list_widget is not None:
                selected_id = self.state.agent_studio.tools_selected_item_id
                if not selected_id and self.state.agent_studio.tools_items:
                    selected_id = str(self.state.agent_studio.tools_items[0].get("id", "")).strip()
                list_widget.set_items(self.state.agent_studio.tools_items, selected_id=selected_id, emit=False)
                selected_id = list_widget.selected_id()
                selected_item = self._agent_tools_item_by_id(selected_id)
                if selected_item is None and self.state.agent_studio.tools_items:
                    selected_item = self.state.agent_studio.tools_items[0]
                    with contextlib.suppress(Exception):
                        list_widget.select_by_id(str(selected_item.get("id", "")), emit=False)
                self._set_agent_tools_selection(selected_item)
            else:
                self._set_agent_tools_selection(
                    self.state.agent_studio.tools_items[0] if self.state.agent_studio.tools_items else None
                )
            self._refresh_agent_tools_header()
            self._set_agent_tools_override_form_values(self.state.agent_studio.session_safety_overrides)

        def _render_agent_team_panel(self) -> None:
            self.state.agent_studio.team_presets = normalize_team_presets(self.state.agent_studio.team_presets)
            self.state.agent_studio.team_items = [
                dict(item) for item in build_agent_team_sidebar_items(self.state.agent_studio.team_presets)
            ]
            list_widget = self._agent_team_list
            if list_widget is not None:
                selected_id = self.state.agent_studio.team_selected_item_id
                if not selected_id and self.state.agent_studio.team_items:
                    selected_id = str(self.state.agent_studio.team_items[0].get("id", "")).strip()
                list_widget.set_items(self.state.agent_studio.team_items, selected_id=selected_id, emit=False)
                selected_id = list_widget.selected_id()
                selected_item = self._agent_team_item_by_id(selected_id)
                if selected_item is None and self.state.agent_studio.team_items:
                    selected_item = self.state.agent_studio.team_items[0]
                    with contextlib.suppress(Exception):
                        list_widget.select_by_id(str(selected_item.get("id", "")), emit=False)
                self._set_agent_team_selection(selected_item)
            else:
                self._set_agent_team_selection(
                    self.state.agent_studio.team_items[0] if self.state.agent_studio.team_items else None
                )
            self._refresh_agent_team_header()

        def _set_agent_studio_view_mode(self, mode: str) -> None:
            normalized = normalize_agent_studio_view_mode(mode)
            self.state.agent_studio.view_mode = normalized

            profile_view = self._agent_profile_view
            tools_view = self._agent_tools_view
            team_view = self._agent_team_view
            if profile_view is not None:
                profile_view.styles.display = "block" if normalized == "profile" else "none"
            if tools_view is not None:
                tools_view.styles.display = "block" if normalized == "tools" else "none"
            if team_view is not None:
                team_view.styles.display = "block" if normalized == "team" else "none"

            profile_button = self._agent_view_profile_button
            tools_button = self._agent_view_tools_button
            team_button = self._agent_view_team_button
            if profile_button is not None:
                profile_button.variant = "primary" if normalized == "profile" else "default"
            if tools_button is not None:
                tools_button.variant = "primary" if normalized == "tools" else "default"
            if team_button is not None:
                team_button.variant = "primary" if normalized == "team" else "default"

        def _kb_id_from_context_sources(self) -> str | None:
            for source in self._context_sources:
                source_type = str(source.get("type", "")).strip().lower()
                if source_type != "kb":
                    continue
                kb_id = str(source.get("id", "")).strip()
                if kb_id:
                    return kb_id
            return None

        def _session_effective_profile(self) -> AgentProfile:
            provider_name, tier_name, _model_id = choose_model_summary_parts(
                daemon_provider=self.state.daemon.provider,
                daemon_tier=self.state.daemon.tier,
                daemon_model_id=self.state.daemon.model_id,
                daemon_tiers=self.state.daemon.tiers,
                pending_value=self.state.daemon.pending_model_select_value,
                override_provider=self.state.daemon.model_provider_override,
                override_tier=self.state.daemon.model_tier_override,
            )
            current = self.state.agent_studio.effective_profile
            return AgentProfile(
                id=(current.id if current is not None else "session-effective"),
                name=(current.name if current is not None else "Session Effective"),
                provider=provider_name,
                tier=tier_name,
                system_prompt_snippets=(list(current.system_prompt_snippets) if current is not None else []),
                context_sources=self._context_sources_payload(),
                active_sops=sorted(self._active_sop_names),
                knowledge_base_id=(
                    self._kb_id_from_context_sources() or (current.knowledge_base_id if current else None)
                ),
                team_presets=(normalize_team_presets(current.team_presets) if current is not None else []),
            )

        def _set_agent_status(self, message: str) -> None:
            widget = self._agent_profile_status
            if widget is None:
                return
            text = message.strip() if isinstance(message, str) else ""
            widget.update(text)

        def _set_agent_draft_dirty(self, dirty: bool, *, note: str | None = None) -> None:
            self.state.agent_studio.draft_dirty = bool(dirty)
            if self.state.agent_studio.draft_dirty:
                base = "Draft changes pending."
            else:
                base = "Draft synced."
            if isinstance(note, str) and note.strip():
                self._set_agent_status(f"{base} {note.strip()}")
            else:
                self._set_agent_status(base)
            if self._agent_profile_list is not None:
                self._reload_saved_profiles()

        def _reload_saved_profiles(self, *, selected_id: str | None = None) -> None:
            self.state.agent_studio.saved_profiles = sorted(
                list_profiles(),
                key=lambda item: (item.name.lower(), item.id.lower()),
            )
            sidebar_list = self._agent_profile_list
            if sidebar_list is None:
                return

            items: list[dict[str, str]] = [
                {
                    "id": _AGENT_PROFILE_SELECT_NONE,
                    "title": "Draft / Session",
                    "subtitle": "Unsaved local draft",
                    "state": "syncing" if self.state.agent_studio.draft_dirty else "default",
                }
            ]
            for profile in self.state.agent_studio.saved_profiles:
                profile_subtitle_parts = [profile.id]
                model_summary = "/".join(
                    part for part in [str(profile.provider or "").strip(), str(profile.tier or "").strip()] if part
                )
                if model_summary:
                    profile_subtitle_parts.append(model_summary)
                items.append(
                    {
                        "id": profile.id,
                        "title": profile.name,
                        "subtitle": " | ".join(profile_subtitle_parts),
                        "state": (
                            "active"
                            if self.state.agent_studio.effective_profile
                            and self.state.agent_studio.effective_profile.id == profile.id
                            else "default"
                        ),
                    }
                )

            getter = getattr(sidebar_list, "selected_id", None)
            current_value = str(getter() or "").strip() if callable(getter) else ""
            candidate = selected_id if selected_id else current_value
            saved_ids = {profile.id for profile in self.state.agent_studio.saved_profiles}
            if candidate == _AGENT_PROFILE_SELECT_NONE:
                pass
            elif candidate not in saved_ids:
                candidate = (
                    self.state.agent_studio.saved_profiles[0].id
                    if self.state.agent_studio.saved_profiles
                    else _AGENT_PROFILE_SELECT_NONE
                )

            self.state.agent_studio.form_syncing = True
            try:
                setter = getattr(sidebar_list, "set_items", None)
                if callable(setter):
                    setter(items, selected_id=candidate, emit=False)
            finally:
                self.state.agent_studio.form_syncing = False

        def _refresh_agent_summary(self) -> None:
            summary = self._agent_summary
            if summary is None:
                return
            effective = self._session_effective_profile()
            self.state.agent_studio.effective_profile = effective
            summary.load_text(render_agent_profile_summary_text(effective.to_dict()))
            summary.scroll_home(animate=False)

        def _new_agent_profile_draft(self, *, announce: bool = True) -> None:
            snapshot = self._session_effective_profile()
            draft = AgentProfile(
                id=_default_profile_id(),
                name=_default_profile_name(),
                provider=snapshot.provider,
                tier=snapshot.tier,
                system_prompt_snippets=list(snapshot.system_prompt_snippets),
                context_sources=[dict(source) for source in snapshot.context_sources],
                active_sops=list(snapshot.active_sops),
                knowledge_base_id=snapshot.knowledge_base_id,
                team_presets=normalize_team_presets(snapshot.team_presets),
            )
            sidebar_list = self._agent_profile_list
            if sidebar_list is not None:
                self.state.agent_studio.form_syncing = True
                try:
                    select_by_id = getattr(sidebar_list, "select_by_id", None)
                    if callable(select_by_id):
                        select_by_id(_AGENT_PROFILE_SELECT_NONE, emit=False)
                finally:
                    self.state.agent_studio.form_syncing = False
            self.state.agent_studio.team_presets = normalize_team_presets(draft.team_presets)
            self.state.agent_studio.team_selected_item_id = None
            self._render_agent_team_panel()
            self._set_agent_form_values(profile_id=draft.id, profile_name=draft.name)
            self._set_agent_draft_dirty(True, note=("New profile draft." if announce else None))

        def _load_profile_into_draft(self, profile: AgentProfile) -> None:
            sidebar_list = self._agent_profile_list
            if sidebar_list is not None:
                self.state.agent_studio.form_syncing = True
                try:
                    select_by_id = getattr(sidebar_list, "select_by_id", None)
                    if callable(select_by_id):
                        select_by_id(profile.id, emit=False)
                finally:
                    self.state.agent_studio.form_syncing = False
            self.state.agent_studio.team_presets = normalize_team_presets(profile.team_presets)
            self.state.agent_studio.team_selected_item_id = None
            self._render_agent_team_panel()
            self._set_agent_form_values(profile_id=profile.id, profile_name=profile.name)
            self._set_agent_draft_dirty(False, note=f"Loaded profile '{profile.name}'.")

        def _profile_from_draft(self) -> AgentProfile:
            selected_profile = self._lookup_saved_profile(self._selected_agent_profile_id())
            seed = selected_profile if selected_profile is not None else self._session_effective_profile()

            raw_id = str(getattr(self._agent_profile_id_input, "value", "")).strip()
            raw_name = str(getattr(self._agent_profile_name_input, "value", "")).strip()
            profile_id = _sanitize_profile_token(raw_id or seed.id or _default_profile_id())
            profile_name = raw_name or seed.name or _default_profile_name()

            return AgentProfile.from_dict(
                {
                    "id": profile_id,
                    "name": profile_name,
                    "provider": seed.provider,
                    "tier": seed.tier,
                    "system_prompt_snippets": list(seed.system_prompt_snippets),
                    "context_sources": [dict(source) for source in seed.context_sources],
                    "active_sops": list(seed.active_sops),
                    "knowledge_base_id": seed.knowledge_base_id,
                    "team_presets": normalize_team_presets(self.state.agent_studio.team_presets),
                }
            )

        def _save_agent_profile_draft(self) -> None:
            profile = self._profile_from_draft()
            saved = save_profile(profile)
            self._reload_saved_profiles(selected_id=saved.id)
            self._set_agent_form_values(profile_id=saved.id, profile_name=saved.name)
            self._set_agent_draft_dirty(False, note=f"Saved profile '{saved.name}'.")

        def _delete_selected_agent_profile(self) -> None:
            selected_id = self._selected_agent_profile_id()
            if not selected_id:
                self._notify("Select a saved profile first.", severity="warning")
                return
            removed = delete_profile(selected_id)
            if not removed:
                self._notify("Profile not found.", severity="warning")
                return
            self._reload_saved_profiles()
            if self.state.agent_studio.saved_profiles:
                self._load_profile_into_draft(self.state.agent_studio.saved_profiles[0])
                self._set_agent_draft_dirty(False, note=f"Deleted profile '{selected_id}'.")
            else:
                self._new_agent_profile_draft(announce=False)
                self._set_agent_draft_dirty(True, note=f"Deleted profile '{selected_id}'.")

        def _apply_agent_profile_draft(self) -> None:
            if self.state.daemon.query_active:
                self._write_transcript_line("[agent] cannot apply profile while a run is active.")
                return
            if not self.state.daemon.ready:
                self._write_transcript_line("[agent] daemon is not ready.")
                return
            proc = self.state.daemon.proc
            if proc is None or proc.poll() is not None:
                self._write_transcript_line("[agent] daemon is not running.")
                self.state.daemon.ready = False
                return
            profile = self._profile_from_draft()
            payload = {"cmd": "set_profile", "profile": profile.to_dict()}
            if not send_daemon_command(proc, payload):
                self._write_transcript_line("[agent] failed to send set_profile.")
                return
            self._set_agent_status(f"Applying profile '{profile.name}'...")

        def _reset_plan_panel(self) -> None:
            self._set_plan_panel("(no plan)")
            self.state.plan.current_steps_total = 0
            self.state.plan.current_summary = ""
            self.state.plan.current_steps = []
            self.state.plan.current_step_statuses = []
            self.state.plan.current_active_step = None
            self.state.plan.updates_seen = False
            self.state.plan.step_counter = 0
            self.state.plan.completion_announced = False
            self._refresh_plan_status_bar()

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
            self.state.session.view_mode = normalized

            timeline_view = self._session_timeline_view
            issues_view = self._session_issues_view
            if timeline_view is not None:
                timeline_view.styles.display = "block" if normalized == "timeline" else "none"
            if issues_view is not None:
                issues_view.styles.display = "block" if normalized == "issues" else "none"

            timeline_button = self._session_view_timeline_button
            issues_button = self._session_view_issues_button
            if timeline_button is not None:
                timeline_button.variant = "primary" if normalized == "timeline" else "default"
            if issues_button is not None:
                issues_button.variant = "primary" if normalized == "issues" else "default"

        def _schedule_session_timeline_refresh(self, *, delay: float = 0.35) -> None:
            timer = self.state.session.timeline_refresh_timer
            self.state.session.timeline_refresh_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
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
                except Exception:
                    existing_index = None
                try:
                    built_index = await asyncio.to_thread(build_session_graph_index, session_id, cwd=Path.cwd())
                    await asyncio.to_thread(write_session_graph_index, session_id, built_index)
                except Exception:
                    built_index = None
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
                    events_raw = index.get("events")
                    normalized_events: list[dict[str, Any]] = []
                    if isinstance(events_raw, list):
                        for offset, raw in enumerate(events_raw, start=1):
                            if not isinstance(raw, dict):
                                continue
                            event = dict(raw)
                            event.setdefault("id", f"timeline-{offset}")
                            normalized_events.append(event)
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

        def _update_header_status(self) -> None:
            counts = []
            if self.state.session.warning_count:
                counts.append(f"warn={self.state.session.warning_count}")
            if self.state.session.error_count:
                counts.append(f"err={self.state.session.error_count}")
            suffix = (" | " + " ".join(counts)) if counts else ""
            self.sub_title = f"{self._current_model_summary()}{suffix}"
            if self._status_bar is not None:
                self._status_bar.set_counts(
                    warnings=self.state.session.warning_count, errors=self.state.session.error_count
                )
            self._refresh_session_header()

        def _current_model_summary(self) -> str:
            provider_name, tier_name, model_id = choose_model_summary_parts(
                daemon_provider=self.state.daemon.provider,
                daemon_tier=self.state.daemon.tier,
                daemon_model_id=self.state.daemon.model_id,
                daemon_tiers=self.state.daemon.tiers,
                pending_value=self.state.daemon.pending_model_select_value,
                override_provider=self.state.daemon.model_provider_override,
                override_tier=self.state.daemon.model_tier_override,
            )
            if provider_name and tier_name:
                suffix = f" ({model_id})" if model_id else ""
                return f"Model: {provider_name}/{tier_name}{suffix}"
            return resolve_model_config_summary(
                provider_override=self.state.daemon.model_provider_override,
                tier_override=self.state.daemon.model_tier_override,
            )

        def _model_env_overrides(self) -> dict[str, str]:
            overrides: dict[str, str] = {}
            if self.state.daemon.model_provider_override:
                overrides["SWARMEE_MODEL_PROVIDER"] = self.state.daemon.model_provider_override
            if self.state.daemon.model_tier_override:
                overrides["SWARMEE_MODEL_TIER"] = self.state.daemon.model_tier_override
            return overrides

        def _refresh_model_select(self) -> None:
            if self.state.daemon.provider and self.state.daemon.tier and self.state.daemon.tiers:
                self._refresh_model_select_from_daemon(
                    provider=self.state.daemon.provider,
                    tier=self.state.daemon.tier,
                    tiers=self.state.daemon.tiers,
                )
                return

            options, selected_value = self._model_select_options()
            self._apply_model_select_options(options, selected_value)

        def _apply_model_select_options(self, options: list[tuple[str, str]], selected_value: str) -> None:
            selector = self.query_one("#model_select", Select)
            self.state.daemon.model_select_syncing = True
            try:
                selector.set_options(options)
                with contextlib.suppress(Exception):
                    selector.value = selected_value
            finally:
                self.state.daemon.model_select_syncing = False

        def _refresh_model_select_from_daemon(
            self,
            *,
            provider: str,
            tier: str,
            tiers: list[dict[str, Any]],
        ) -> None:
            options, selected_value = daemon_model_select_options(
                provider=provider,
                tier=tier,
                tiers=tiers,
                pending_value=self.state.daemon.pending_model_select_value,
                override_provider=self.state.daemon.model_provider_override,
                override_tier=self.state.daemon.model_tier_override,
            )
            self._apply_model_select_options(options, selected_value)

        def _model_select_options(self) -> tuple[list[tuple[str, str]], str]:
            if self.state.daemon.tiers and self.state.daemon.provider:
                return daemon_model_select_options(
                    provider=self.state.daemon.provider,
                    tier=(self.state.daemon.tier or ""),
                    tiers=self.state.daemon.tiers,
                    pending_value=self.state.daemon.pending_model_select_value,
                    override_provider=self.state.daemon.model_provider_override,
                    override_tier=self.state.daemon.model_tier_override,
                )
            return model_select_options(
                provider_override=self.state.daemon.model_provider_override,
                tier_override=self.state.daemon.model_tier_override,
            )

        def _handle_model_info(self, event: dict[str, Any]) -> None:
            provider = str(event.get("provider", "")).strip().lower()
            tier = str(event.get("tier", "")).strip().lower()
            model_id = event.get("model_id")
            tiers = event.get("tiers")

            self.state.daemon.provider = provider or None
            self.state.daemon.tier = tier or None
            self.state.daemon.model_id = (
                str(model_id).strip() if model_id is not None and str(model_id).strip() else None
            )
            self.state.daemon.tiers = tiers if isinstance(tiers, list) else []
            self.state.daemon.current_model = self.state.daemon.model_id or (
                f"{self.state.daemon.provider}/{self.state.daemon.tier}"
                if self.state.daemon.provider and self.state.daemon.tier
                else None
            )
            pending_value = (self.state.daemon.pending_model_select_value or "").strip().lower()
            if pending_value and "|" in pending_value:
                pending_provider, pending_tier = pending_value.split("|", 1)
                if pending_provider == provider and pending_tier == tier:
                    self.state.daemon.pending_model_select_value = None
                    pending_value = ""

            if not pending_value and self.state.daemon.provider and self.state.daemon.tier:
                self.state.daemon.model_provider_override = self.state.daemon.provider
                self.state.daemon.model_tier_override = self.state.daemon.tier

            if self.state.daemon.provider and self.state.daemon.tier:
                self._refresh_model_select_from_daemon(
                    provider=self.state.daemon.provider,
                    tier=self.state.daemon.tier,
                    tiers=self.state.daemon.tiers,
                )
            else:
                self._refresh_model_select()

            self._update_header_status()
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())
            self._refresh_agent_summary()
            self._render_agent_tools_panel()

        def _update_prompt_placeholder(self) -> None:
            input_widget = self.query_one("#prompt", PromptTextArea)
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
            with contextlib.suppress(Exception):
                tabs = self.query_one("#side_tabs", TabbedContent)
                tabs.active = tab_id

        def _seed_prompt_with_command(self, command: str) -> None:
            prompt_widget = self.query_one("#prompt", PromptTextArea)
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

        def _set_model_tier_from_value(self, value: str) -> None:
            parsed = parse_model_select_value(value)
            if parsed is None:
                return
            requested_provider, requested_tier = parsed
            self.state.daemon.pending_model_select_value = None
            self.state.daemon.model_provider_override = requested_provider or None
            self.state.daemon.model_tier_override = requested_tier or None
            self._refresh_model_select()
            self._update_header_status()
            self._update_prompt_placeholder()
            if (
                self.state.daemon.ready
                and self.state.daemon.proc is not None
                and self.state.daemon.proc.poll() is None
                and not self.state.daemon.query_active
            ):
                if not send_daemon_command(self.state.daemon.proc, {"cmd": "set_tier", "tier": requested_tier}):
                    self._write_transcript_line("[model] failed to send tier change to daemon.")
                else:
                    self.state.daemon.pending_model_select_value = f"{requested_provider}|{requested_tier}"
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())

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
                        {"id": "view:issues", "icon": "⚠", "label": "View session issues", "shortcut": "I"},
                    ],
                )

            if self.state.plan.pending_prompt:
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
                self._switch_side_tab("tab_plan")
                return
            if action == "view:issues":
                self._switch_side_tab("tab_session")
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

        def _estimate_prompt_tokens(self, text: str) -> int:
            # Lightweight heuristic for pre-send token estimate.
            return max(0, (len(text or "") + 3) // 4)

        def _apply_prompt_estimate(self) -> None:
            text = self._pending_prompt_estimate_text
            self._prompt_input_tokens_est = self._estimate_prompt_tokens(text)
            self._refresh_prompt_metrics()

        def _schedule_prompt_estimate_update(self, text: str) -> None:
            self._pending_prompt_estimate_text = text or ""
            timer = self._prompt_estimate_timer
            self._prompt_estimate_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()
            self._prompt_estimate_timer = self.set_timer(0.2, self._apply_prompt_estimate)

        def _on_prompt_text_changed(self, text: str) -> None:
            self._schedule_prompt_estimate_update(text)

        def _notify(self, message: str, *, severity: str = "information", timeout: float | None = 2.5) -> None:
            with contextlib.suppress(Exception):
                self.notify(message, severity=severity, timeout=timeout)

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
                self._write_transcript_line(
                    f"[copy] {label}: clipboard unavailable. Saved to artifact: {artifact_path}"
                )
            else:
                self._write_transcript_line(f"[copy] {label}: clipboard unavailable.")

        def _get_transcript_text(self) -> str:
            if self._transcript_fallback_lines:
                return "\n".join(self._transcript_fallback_lines).rstrip() + "\n"
            return ""

        def _get_richlog_selection_text(self, transcript: Any) -> str:
            if isinstance(transcript, TextArea):
                selected = transcript.selected_text or ""
                return selected if isinstance(selected, str) else ""
            return ""

        def _get_all_text(self) -> str:
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

        def _load_indexed_artifact_entries(self, *, limit: int = 200) -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            seen_paths: set[str] = set()
            try:
                store = ArtifactStore()
                indexed = store.list(limit=limit)
            except Exception:
                indexed = []

            for raw in indexed:
                normalized = normalize_artifact_index_entry(raw if isinstance(raw, dict) else {})
                if normalized is None:
                    continue
                path = str(normalized.get("path", "")).strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                entries.append(normalized)

            # Keep compatibility with legacy in-memory artifact paths that may not
            # have an index record (e.g., session logs).
            for raw_path in self.state.artifacts.recent_paths:
                path = str(raw_path or "").strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                entries.append(
                    {
                        "item_id": path,
                        "id": Path(path).name,
                        "name": Path(path).name,
                        "kind": "path",
                        "created_at": "",
                        "path": path,
                        "bytes": None,
                        "chars": None,
                        "meta": {},
                    }
                )
            return entries

        def _artifact_entry_by_item_id(self, item_id: str | None) -> dict[str, Any] | None:
            target = str(item_id or "").strip()
            if not target:
                return None
            for entry in self.state.artifacts.entries:
                if str(entry.get("item_id", "")).strip() == target:
                    return entry
            return None

        def _artifact_looks_textual(self, entry: dict[str, Any]) -> bool:
            path = str(entry.get("path", "")).strip()
            kind = str(entry.get("kind", "")).strip().lower()
            if kind in {"tui_transcript", "tool_result", "diagnostic", "project_map"}:
                return True
            suffix = Path(path).suffix.lower()
            if suffix in {".txt", ".md", ".json", ".jsonl", ".log", ".yaml", ".yml", ".csv", ".patch", ".diff"}:
                return True
            if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".tar", ".bz2"}:
                return False
            return True

        def _artifact_metadata_preview(self, entry: dict[str, Any]) -> str:
            lines = [
                f"Kind: {entry.get('kind', 'unknown')}",
                f"ID: {entry.get('id', '(none)')}",
                f"Name: {entry.get('name', '(none)')}",
                f"Created: {entry.get('created_at', '(unknown)')}",
                f"Path: {entry.get('path', '(none)')}",
            ]
            bytes_value = entry.get("bytes")
            chars_value = entry.get("chars")
            if isinstance(bytes_value, int):
                lines.append(f"Bytes: {bytes_value}")
            if isinstance(chars_value, int):
                lines.append(f"Chars: {chars_value}")
            meta = entry.get("meta")
            if isinstance(meta, dict) and meta:
                try:
                    lines.append("")
                    lines.append("Meta:")
                    lines.append(_json.dumps(meta, indent=2, ensure_ascii=False))
                except Exception:
                    pass
            return "\n".join(lines)

        def _artifact_preview_text(self, entry: dict[str, Any]) -> str:
            path = str(entry.get("path", "")).strip()
            if not path:
                return self._artifact_metadata_preview(entry)
            artifact_path = Path(path).expanduser()
            if not artifact_path.exists() or not artifact_path.is_file():
                return self._artifact_metadata_preview(entry) + "\n\nFile not found."
            if not self._artifact_looks_textual(entry):
                return self._artifact_metadata_preview(entry)
            try:
                store = ArtifactStore()
                body = store.read_text(artifact_path, max_chars=5000)
            except Exception as exc:
                return self._artifact_metadata_preview(entry) + f"\n\nFailed to read artifact: {exc}"
            header = self._artifact_metadata_preview(entry)
            return f"{header}\n\nPreview:\n{body}"

        def _set_artifact_selection(self, entry: dict[str, Any] | None) -> None:
            detail = self._artifacts_detail
            if detail is None:
                return
            if entry is None:
                self.state.artifacts.selected_item_id = None
                detail.set_preview("(no artifacts yet)")
                detail.set_actions([])
                return
            self.state.artifacts.selected_item_id = str(entry.get("item_id", "")).strip() or None
            detail.set_preview(self._artifact_preview_text(entry))
            detail.set_actions(
                [
                    {"id": "artifact_action_open", "label": "Open", "variant": "default"},
                    {"id": "artifact_action_copy_path", "label": "Copy path", "variant": "default"},
                    {"id": "artifact_action_add_context", "label": "Add context", "variant": "default"},
                ]
            )

        def _render_artifacts_panel(self) -> None:
            self.state.artifacts.entries = self._load_indexed_artifact_entries(limit=200)
            if self._artifacts_header is not None:
                badge_count = len(self.state.artifacts.entries)
                self._artifacts_header.set_badges([f"{badge_count} item{'s' if badge_count != 1 else ''}"])
            list_widget = self._artifacts_list
            if list_widget is None:
                return
            items = build_artifact_sidebar_items(self.state.artifacts.entries)
            selected_id = self.state.artifacts.selected_item_id
            if not selected_id and self.state.artifacts.entries:
                selected_id = str(self.state.artifacts.entries[0].get("item_id", "")).strip()
            list_widget.set_items(items, selected_id=selected_id, emit=False)
            selected_item_id = list_widget.selected_id()
            selected_entry = self._artifact_entry_by_item_id(selected_item_id)
            if selected_entry is None and self.state.artifacts.entries:
                selected_entry = self.state.artifacts.entries[0]
                list_widget.select_by_id(str(selected_entry.get("item_id", "")), emit=False)
            self._set_artifact_selection(selected_entry)

        def _get_artifacts_text(self) -> str:
            entries = self.state.artifacts.entries or self._load_indexed_artifact_entries(limit=200)
            if not entries:
                return "(no artifacts)\n"
            lines: list[str] = []
            for index, entry in enumerate(entries, start=1):
                kind = str(entry.get("kind", "unknown")).strip() or "unknown"
                artifact_id = str(entry.get("id", "")).strip() or "(no-id)"
                name = str(entry.get("name", "")).strip() or artifact_id
                created_at = str(entry.get("created_at", "")).strip() or "unknown time"
                path = str(entry.get("path", "")).strip() or "(no-path)"
                lines.append(f"{index}. {kind} | {name} | {created_at} | {path}")
            return "\n".join(lines).rstrip() + "\n"

        def _reset_artifacts_panel(self) -> None:
            self.state.artifacts.recent_paths = []
            self.state.artifacts.entries = []
            self.state.artifacts.selected_item_id = None
            self._render_artifacts_panel()

        def _add_artifact_paths(self, paths: list[str]) -> None:
            updated = add_recent_artifacts(self.state.artifacts.recent_paths, paths, max_items=20)
            if updated != self.state.artifacts.recent_paths:
                self.state.artifacts.recent_paths = updated
            self._render_artifacts_panel()

        def _context_source_label(self, source: dict[str, str]) -> str:
            source_type = source.get("type", "")
            if source_type == "file":
                return source.get("path", "")
            if source_type == "note":
                return source.get("text", "")
            if source_type == "sop":
                return source.get("name", "")
            if source_type == "kb":
                return source.get("id", "")
            if source_type == "url":
                return source.get("url", source.get("path", ""))
            return str(source)

        def _truncate_context_label(self, value: str, *, max_chars: int = _CONTEXT_SOURCE_MAX_LABEL) -> str:
            text = value.strip().replace("\n", " ")
            if len(text) <= max_chars:
                return text
            if max_chars <= 1:
                return text[:max_chars]
            return text[: max_chars - 1].rstrip() + "…"

        def _render_context_sources_panel(self) -> None:
            container = self._context_sources_list
            if container is None:
                with contextlib.suppress(Exception):
                    container = self.query_one("#context_sources_list", VerticalScroll)
                    self._context_sources_list = container
            if container is None:
                return

            for child in list(container.children):
                with contextlib.suppress(Exception):
                    child.remove()

            if not self._context_sources:
                container.mount(Static("[dim](no context sources attached)[/dim]"))
                return

            for index, source in enumerate(self._context_sources):
                source_type = source.get("type", "").strip().lower()
                icon = _CONTEXT_SOURCE_ICONS.get(source_type, "•")
                label = self._truncate_context_label(self._context_source_label(source))
                row = Horizontal(classes="context-source-row")
                container.mount(row)
                row.mount(Static(f"{icon} {label}", classes="context-source-label"))
                row.mount(
                    Button(
                        "✕",
                        id=f"context_remove_{index}",
                        classes="context-remove-btn",
                        compact=True,
                        variant="error",
                    )
                )

        def _refresh_context_sop_options(self) -> None:
            self._context_sop_names = discover_available_sop_names()
            selector = self._context_sop_select
            if selector is None:
                with contextlib.suppress(Exception):
                    selector = self.query_one("#context_sop_select", Select)
                    self._context_sop_select = selector
            if selector is None:
                return
            options: list[tuple[str, str]] = [("Select SOP...", _CONTEXT_SELECT_PLACEHOLDER)]
            options.extend((name, name) for name in self._context_sop_names)
            with contextlib.suppress(Exception):
                selector.set_options(options)
                selector.value = _CONTEXT_SELECT_PLACEHOLDER

        def _sop_checkbox_id(self, name: str) -> str:
            token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip().lower())
            token = token.strip("_") or "sop"
            return f"sop_toggle_{token}"

        def _refresh_sop_catalog(self) -> None:
            self._sop_catalog = discover_available_sops()
            available_names = {item.get("name", "").strip() for item in self._sop_catalog}
            self._active_sop_names = {name for name in self._active_sop_names if name in available_names}
            self._sops_ready_for_sync = bool(self._active_sop_names)

        def _lookup_sop(self, name: str) -> dict[str, str] | None:
            target = name.strip().lower()
            for item in self._sop_catalog:
                sop_name = str(item.get("name", "")).strip()
                if sop_name.lower() == target:
                    return item
            return None

        def _render_sop_panel(self) -> None:
            container = self._sop_list
            if container is None:
                with contextlib.suppress(Exception):
                    container = self.query_one("#sop_list", VerticalScroll)
                    self._sop_list = container
            if container is None:
                return

            self._sop_toggle_id_to_name = {}
            for child in list(container.children):
                with contextlib.suppress(Exception):
                    child.remove()

            if not self._sop_catalog:
                container.mount(Static("[dim](no SOPs found)[/dim]"))
                return

            for sop in self._sop_catalog:
                name = str(sop.get("name", "")).strip()
                if not name:
                    continue
                source = str(sop.get("source", "")).strip() or "unknown"
                preview = str(sop.get("first_paragraph_preview", "")).strip() or "(no preview available)"
                checkbox_id = self._sop_checkbox_id(name)
                self._sop_toggle_id_to_name[checkbox_id] = name
                row = Vertical(classes="sop-row")
                container.mount(row)
                header = Horizontal(classes="sop-row-header")
                row.mount(header)
                header.mount(Checkbox(name, id=checkbox_id, value=(name in self._active_sop_names)))
                header.mount(Static(f"[dim]{source}[/dim]", classes="sop-source-label"))
                row.mount(Static(preview, classes="sop-preview"))

        def _sync_active_sops_with_daemon(self, *, notify_on_failure: bool = False) -> bool:
            proc = self.state.daemon.proc
            if not self.state.daemon.ready or proc is None or proc.poll() is not None:
                self._sops_ready_for_sync = bool(self._active_sop_names)
                return False
            for name in sorted(self._active_sop_names):
                record = self._lookup_sop(name)
                if record is None:
                    continue
                content = str(record.get("content", "")).strip()
                if not content:
                    continue
                payload = {"cmd": "set_sop", "name": name, "content": content}
                if not send_daemon_command(proc, payload):
                    self._sops_ready_for_sync = True
                    if notify_on_failure:
                        self._write_transcript_line("[sop] failed to sync active SOPs with daemon.")
                    return False
            self._sops_ready_for_sync = False
            return True

        def _set_sop_active(self, name: str, active: bool, *, sync: bool = True, announce: bool = True) -> bool:
            record = self._lookup_sop(name)
            if record is None:
                self._write_transcript_line(f"[sop] unknown SOP: {name}")
                return False
            sop_name = str(record.get("name", "")).strip()
            if not sop_name:
                return False

            changed = False
            if active and sop_name not in self._active_sop_names:
                self._active_sop_names.add(sop_name)
                changed = True
            elif (not active) and sop_name in self._active_sop_names:
                self._active_sop_names.remove(sop_name)
                changed = True

            if not changed:
                return True

            self._render_sop_panel()
            self._save_session()
            self._refresh_agent_summary()

            if sync:
                proc = self.state.daemon.proc
                if not self.state.daemon.ready or proc is None or proc.poll() is not None:
                    self._sops_ready_for_sync = bool(self._active_sop_names)
                else:
                    payload = {
                        "cmd": "set_sop",
                        "name": sop_name,
                        "content": (str(record.get("content", "")).strip() if active else None),
                    }
                    if not send_daemon_command(proc, payload):
                        self._sops_ready_for_sync = True
                        self._write_transcript_line("[sop] failed to sync with daemon.")
                    else:
                        self._sops_ready_for_sync = False

            if announce:
                status = "activated" if active else "deactivated"
                self._write_transcript_line(f"[sop] {status}: {sop_name}")
            return True

        def _sop_list_lines(self) -> list[str]:
            if not self._sop_catalog:
                return ["[sop] no SOPs found."]
            lines: list[str] = ["[sop] available:"]
            for index, sop in enumerate(self._sop_catalog, start=1):
                name = str(sop.get("name", "")).strip()
                source = str(sop.get("source", "")).strip() or "unknown"
                marker = "✓" if name in self._active_sop_names else " "
                lines.append(f"{index}. [{marker}] {name} ({source})")
            return lines

        def _handle_sop_command(self, argument: str) -> bool:
            raw = (argument or "").strip()
            if not raw:
                self._write_transcript_line(_SOP_USAGE_TEXT)
                return True
            lowered = raw.lower()

            if lowered == "list":
                for line in self._sop_list_lines():
                    self._write_transcript_line(line)
                return True

            if lowered.startswith("activate "):
                target = raw.split(maxsplit=1)[1].strip()
                if not target:
                    self._write_transcript_line(_SOP_USAGE_TEXT)
                    return True
                self._set_sop_active(target, True, sync=True, announce=True)
                return True

            if lowered.startswith("deactivate "):
                target = raw.split(maxsplit=1)[1].strip()
                if not target:
                    self._write_transcript_line(_SOP_USAGE_TEXT)
                    return True
                self._set_sop_active(target, False, sync=True, announce=True)
                return True

            if lowered.startswith("preview "):
                target = raw.split(maxsplit=1)[1].strip()
                if not target:
                    self._write_transcript_line(_SOP_USAGE_TEXT)
                    return True
                record = self._lookup_sop(target)
                if record is None:
                    self._write_transcript_line(f"[sop] unknown SOP: {target}")
                    return True
                name = str(record.get("name", target)).strip()
                source = str(record.get("source", "")).strip() or "unknown"
                content = str(record.get("content", "")).strip()
                if not content:
                    self._write_transcript_line(f"[sop] no content available for {name}")
                    return True
                markdown = f"# SOP: {name}\n\n[dim]Source: {source}[/dim]\n\n{content}"
                self._mount_transcript_widget(
                    render_assistant_message(markdown),
                    plain_text=f"SOP: {name}\n\n{content}",
                )
                return True

            self._write_transcript_line(_SOP_USAGE_TEXT)
            return True

        def _set_context_add_mode(self, mode: str | None) -> None:
            normalized = (mode or "").strip().lower() or None
            self._context_add_mode = normalized
            input_row = self.query_one("#context_input_row", Horizontal)
            sop_row = self.query_one("#context_sop_row", Horizontal)
            input_widget = self._context_input
            if input_widget is None:
                with contextlib.suppress(Exception):
                    input_widget = self.query_one("#context_input", Input)
                    self._context_input = input_widget

            if normalized in _CONTEXT_INPUT_SOURCE_TYPES:
                input_row.styles.display = "block"
                sop_row.styles.display = "none"
                if input_widget is not None:
                    if normalized == "file":
                        input_widget.placeholder = "Enter file path"
                    elif normalized == "kb":
                        input_widget.placeholder = "Enter knowledge base ID"
                    else:
                        input_widget.placeholder = "Enter context note"
                    with contextlib.suppress(Exception):
                        input_widget.value = ""
                        input_widget.focus()
                return

            if normalized == _CONTEXT_SOP_SOURCE_TYPE:
                input_row.styles.display = "none"
                sop_row.styles.display = "block"
                self._refresh_context_sop_options()
                if self._context_sop_select is not None:
                    with contextlib.suppress(Exception):
                        self._context_sop_select.focus()
                return

            input_row.styles.display = "none"
            sop_row.styles.display = "none"

        def _context_sources_payload(self) -> list[dict[str, str]]:
            payload: list[dict[str, str]] = []
            for source in self._context_sources:
                source_type = source.get("type", "")
                if source_type == "file":
                    path = source.get("path", "").strip()
                    if path:
                        payload.append({"type": "file", "path": path, "id": source.get("id", "")})
                elif source_type == "note":
                    text = source.get("text", "").strip()
                    if text:
                        payload.append({"type": "note", "text": text, "id": source.get("id", "")})
                elif source_type == "sop":
                    name = source.get("name", "").strip()
                    if name:
                        payload.append({"type": "sop", "name": name, "id": source.get("id", "")})
                elif source_type == "kb":
                    kb_id = source.get("id", "").strip()
                    if kb_id:
                        payload.append({"type": "kb", "id": kb_id})
                elif source_type == "url":
                    url = source.get("url", "").strip()
                    if url:
                        payload.append({"type": "url", "url": url, "id": source.get("id", "")})
            return payload

        def _sync_context_sources_with_daemon(self, *, notify_on_failure: bool = False) -> bool:
            proc = self.state.daemon.proc
            if not self.state.daemon.ready or proc is None or proc.poll() is not None:
                self._context_ready_for_sync = True
                return False
            payload = {"cmd": "set_context_sources", "sources": self._context_sources_payload()}
            if send_daemon_command(proc, payload):
                self._context_ready_for_sync = False
                return True
            self._context_ready_for_sync = True
            if notify_on_failure:
                self._write_transcript_line("[context] failed to sync sources with daemon.")
            return False

        def _set_context_sources(self, sources: list[dict[str, str]], *, sync: bool = True) -> None:
            self._context_sources = _normalize_context_sources(sources)
            self._render_context_sources_panel()
            self._save_session()
            self._refresh_agent_summary()
            if sync:
                self._sync_context_sources_with_daemon(notify_on_failure=True)

        def _add_context_source(self, source: dict[str, str]) -> None:
            normalized = _normalize_context_source(source)
            if normalized is None:
                return
            updated = [*self._context_sources, normalized]
            self._set_context_sources(updated, sync=True)

        def _remove_context_source(self, index: int) -> None:
            if index < 0 or index >= len(self._context_sources):
                self._write_transcript_line(f"[context] invalid index: {index + 1}.")
                return
            updated = [item for i, item in enumerate(self._context_sources) if i != index]
            self._set_context_sources(updated, sync=True)

        def _clear_context_sources(self) -> None:
            self._set_context_sources([], sync=True)

        def _context_list_lines(self) -> list[str]:
            if not self._context_sources:
                return ["[context] no sources attached."]
            lines: list[str] = ["[context] active sources:"]
            for index, source in enumerate(self._context_sources, start=1):
                source_type = source.get("type", "unknown")
                label = self._truncate_context_label(self._context_source_label(source), max_chars=96)
                lines.append(f"{index}. {source_type}: {label}")
            return lines

        def _commit_context_add_from_ui(self) -> None:
            mode = (self._context_add_mode or "").strip().lower()
            if mode in _CONTEXT_INPUT_SOURCE_TYPES:
                if self._context_input is None:
                    return
                raw_value = str(getattr(self._context_input, "value", "")).strip()
                if not raw_value:
                    self._notify("Context value is required.", severity="warning")
                    return
                source: dict[str, str]
                if mode == "file":
                    source = {"type": "file", "path": raw_value, "id": _sanitize_context_source_id(uuid.uuid4().hex)}
                elif mode == "kb":
                    source = {"type": "kb", "id": raw_value}
                else:
                    source = {"type": "note", "text": raw_value, "id": _sanitize_context_source_id(uuid.uuid4().hex)}
                self._add_context_source(source)
                self._set_context_add_mode(None)
                self._write_transcript_line(f"[context] added {mode} source.")
                return

            if mode == _CONTEXT_SOP_SOURCE_TYPE:
                selector = self._context_sop_select
                selected = str(getattr(selector, "value", "")).strip() if selector is not None else ""
                if not selected or selected == _CONTEXT_SELECT_PLACEHOLDER:
                    self._notify("Select an SOP first.", severity="warning")
                    return
                self._add_context_source(
                    {
                        "type": "sop",
                        "name": selected,
                        "id": _sanitize_context_source_id(f"sop-{selected}"),
                    }
                )
                self._set_context_add_mode(None)
                self._write_transcript_line(f"[context] added sop source: {selected}")

        def _handle_context_command(self, argument: str) -> bool:
            raw = (argument or "").strip()
            if not raw:
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True

            lowered = raw.lower()
            if lowered == "list":
                for line in self._context_list_lines():
                    self._write_transcript_line(line)
                return True

            if lowered == "clear":
                self._clear_context_sources()
                self._write_transcript_line("[context] cleared all sources.")
                return True

            if lowered.startswith("remove "):
                token = raw.split(maxsplit=1)[1].strip()
                try:
                    index = int(token) - 1
                except ValueError:
                    self._write_transcript_line("Usage: /context remove <index>")
                    return True
                self._remove_context_source(index)
                return True

            add_match = re.match(r"^add\s+(file|note|sop|kb)\s+(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
            if add_match is None:
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True
            source_type = add_match.group(1).strip().lower()
            value = add_match.group(2).strip()
            if not value:
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True

            if source_type == "file":
                self._add_context_source(
                    {
                        "type": "file",
                        "path": value,
                        "id": _sanitize_context_source_id(uuid.uuid4().hex),
                    }
                )
            elif source_type == "note":
                self._add_context_source(
                    {
                        "type": "note",
                        "text": value,
                        "id": _sanitize_context_source_id(uuid.uuid4().hex),
                    }
                )
            elif source_type == "sop":
                self._add_context_source(
                    {"type": "sop", "name": value, "id": _sanitize_context_source_id(f"sop-{value}")}
                )
            elif source_type == "kb":
                self._add_context_source({"type": "kb", "id": value})
            else:
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True

            self._write_transcript_line(f"[context] added {source_type} source.")
            return True

        def _record_consent_history(self, line: str) -> None:
            entry = line.strip()
            if not entry:
                return
            self._consent_history_lines.append(entry)
            if len(self._consent_history_lines) > 200:
                self._consent_history_lines = self._consent_history_lines[-200:]

        def _show_thinking_text(self) -> None:
            current_text = "".join(self._thinking_buffer).strip()
            text = current_text or (self._last_thinking_text or "").strip()
            if not text:
                self._write_transcript_line("No thinking content from this turn.")
                return

            total_chars = len(text)
            if total_chars > _THINKING_EXPORT_MAX_CHARS:
                shown = text[-_THINKING_EXPORT_MAX_CHARS:]
                self._write_transcript_line(
                    f"[thinking] showing last {_THINKING_EXPORT_MAX_CHARS:,} of {total_chars:,} chars."
                )
                text = shown
            else:
                self._write_transcript_line(f"[thinking] showing {total_chars:,} chars.")

            self._mount_transcript_widget(render_system_message(text), plain_text=text)

        def _cancel_consent_hide_timer(self) -> None:
            timer = self._consent_hide_timer
            self._consent_hide_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _complete_consent_prompt_hide(self, expected_nonce: int) -> None:
            self._consent_hide_timer = None
            if expected_nonce != self._consent_prompt_nonce:
                return
            widget = self._consent_prompt_widget
            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.hide_prompt()
            with contextlib.suppress(Exception):
                self.query_one("#prompt", TextArea).focus()

        def _schedule_consent_prompt_hide(self, *, delay: float = 1.0) -> None:
            self._cancel_consent_hide_timer()
            nonce = self._consent_prompt_nonce
            self._consent_hide_timer = self.set_timer(
                delay,
                lambda: self._complete_consent_prompt_hide(nonce),
            )

        def _show_consent_prompt(self, *, context: str, options: list[str] | None = None, alert: bool = True) -> None:
            widget = self._consent_prompt_widget
            if widget is None:
                with contextlib.suppress(Exception):
                    widget = self.query_one("#consent_prompt", ConsentPrompt)
                    self._consent_prompt_widget = widget
            if widget is None:
                return
            self._cancel_consent_hide_timer()
            self._consent_prompt_nonce += 1
            self._consent_active = True
            self._consent_tool_name = extract_consent_tool_name(context)
            normalized_options = options or ["y", "n", "a", "v"]
            widget.set_prompt(context=context, options=normalized_options, alert=alert)

        def _reset_consent_panel(self) -> None:
            self._cancel_consent_hide_timer()
            self._consent_prompt_nonce += 1
            self._consent_active = False
            self._consent_buffer = []
            self._consent_tool_name = "tool"
            widget = self._consent_prompt_widget
            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.hide_prompt()

        def _reset_error_action_prompt(self) -> None:
            self._pending_error_action = None
            widget = self._error_action_prompt_widget
            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.hide_prompt()

        def _next_available_tier_name(self) -> str | None:
            current_tier = (self.state.daemon.tier or "").strip().lower()
            available = [
                str(item.get("name", "")).strip().lower()
                for item in self.state.daemon.tiers
                if isinstance(item, dict) and bool(item.get("available"))
            ]
            if not available:
                return None
            if current_tier in available:
                idx = available.index(current_tier)
                for candidate in available[idx + 1 :]:
                    if candidate:
                        return candidate
            for candidate in available:
                if candidate and candidate != current_tier:
                    return candidate
            return None

        def _show_tool_error_actions(self, *, tool_use_id: str, tool_name: str) -> None:
            widget = self._error_action_prompt_widget
            if widget is None:
                return
            self._pending_error_action = {
                "kind": "tool",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
            }
            with contextlib.suppress(Exception):
                widget.show_tool_error(tool_name=tool_name, tool_use_id=tool_use_id)

        def _show_escalation_actions(self, *, next_tier: str | None = None) -> None:
            widget = self._error_action_prompt_widget
            if widget is None:
                return
            resolved_next = (next_tier or "").strip().lower() or self._next_available_tier_name()
            self._pending_error_action = {
                "kind": "escalation",
                "next_tier": resolved_next or None,
            }
            with contextlib.suppress(Exception):
                widget.show_escalation(next_tier=resolved_next or None)

        def _resume_after_error(self, *, escalate: bool) -> None:
            if self.state.daemon.query_active:
                self._write_transcript_line("[run] already running; use /stop.")
                return
            prompt = (self._last_prompt or "").strip()
            if not prompt:
                self._write_transcript_line("[run] no previous prompt to continue.")
                self._reset_error_action_prompt()
                return
            proc = self.state.daemon.proc
            if proc is None or proc.poll() is not None or not self.state.daemon.ready:
                self._write_transcript_line("[run] daemon is not ready. Use /daemon restart.")
                self._reset_error_action_prompt()
                return
            pending = self._pending_error_action or {}
            if escalate:
                next_tier = str(pending.get("next_tier", "")).strip().lower()
                if next_tier:
                    if not send_daemon_command(proc, {"cmd": "set_tier", "tier": next_tier}):
                        self._write_transcript_line("[model] failed to request tier escalation.")
                        return
                    self._write_transcript_line(f"[model] escalated to {next_tier}.")
                else:
                    self._write_transcript_line("[model] no higher tier available; continuing on current tier.")
            self._reset_error_action_prompt()
            self._start_run(prompt, auto_approve=self._last_run_auto_approve, mode="execute")

        def _retry_failed_tool(self) -> None:
            action = self._pending_error_action or {}
            tool_use_id = str(action.get("tool_use_id", "")).strip()
            if not tool_use_id:
                self._write_transcript_line("[recovery] no failed tool selected.")
                self._reset_error_action_prompt()
                return
            proc = self.state.daemon.proc
            if proc is None or proc.poll() is not None or not self.state.daemon.ready:
                self._write_transcript_line("[recovery] daemon is not ready.")
                self._reset_error_action_prompt()
                return
            if send_daemon_command(proc, {"cmd": "retry_tool", "tool_use_id": tool_use_id}):
                self._write_transcript_line(f"[recovery] retry requested for tool {tool_use_id}.")
                self._reset_error_action_prompt()
            else:
                self._write_transcript_line("[recovery] failed to send retry request.")

        def _skip_failed_tool(self) -> None:
            action = self._pending_error_action or {}
            tool_use_id = str(action.get("tool_use_id", "")).strip()
            if not tool_use_id:
                self._write_transcript_line("[recovery] no failed tool selected.")
                self._reset_error_action_prompt()
                return
            proc = self.state.daemon.proc
            if proc is None or proc.poll() is not None or not self.state.daemon.ready:
                self._write_transcript_line("[recovery] daemon is not ready.")
                self._reset_error_action_prompt()
                return
            if send_daemon_command(proc, {"cmd": "skip_tool", "tool_use_id": tool_use_id}):
                self._write_transcript_line(f"[recovery] skip requested for tool {tool_use_id}.")
                self._reset_error_action_prompt()
            else:
                self._write_transcript_line("[recovery] failed to send skip request.")

        def _apply_consent_capture(self, line: str) -> None:
            previously_active = self._consent_active
            next_active, next_buffer = update_consent_capture(
                self._consent_active,
                self._consent_buffer,
                line,
                max_lines=20,
            )
            if next_active != self._consent_active or next_buffer != self._consent_buffer:
                self._consent_active = next_active
                self._consent_buffer = next_buffer
                if self._consent_active:
                    context = "\n".join(self._consent_buffer[-4:])
                    self._show_consent_prompt(
                        context=context,
                        options=["y", "n", "a", "v"],
                        alert=not previously_active and next_active,
                    )

        def _consent_decision_line(self, choice: str) -> str:
            tool_name = self._consent_tool_name or "tool"
            if choice == "y":
                return f"✓ {tool_name} allowed"
            if choice == "n":
                return f"✗ {tool_name} denied"
            if choice == "a":
                return f"✓ {tool_name} always allowed (session)"
            if choice == "v":
                return f"✗ {tool_name} never allowed (session)"
            return f"[consent] response: {choice}"

        def _submit_consent_choice(self, choice: str) -> None:
            normalized_choice = choice.strip().lower()
            if normalized_choice not in _CONSENT_CHOICES:
                self._write_transcript_line("Usage: /consent <y|n|a|v>")
                return
            if not self._consent_active:
                self._write_transcript_line("[consent] no active prompt.")
                return
            if self.state.daemon.proc is None or self.state.daemon.proc.poll() is not None:
                self._write_transcript_line("[consent] daemon is not running.")
                self._reset_consent_panel()
                return
            decision_line = self._consent_decision_line(normalized_choice)
            self._write_transcript(decision_line)
            self._record_consent_history(decision_line)
            self._consent_active = False
            self._consent_buffer = []
            approved = normalized_choice in {"y", "a"}
            widget = self._consent_prompt_widget
            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.show_confirmation(decision_line, approved=approved)
            self._schedule_consent_prompt_hide(delay=1.0)
            if not send_daemon_command(
                self.state.daemon.proc, {"cmd": "consent_response", "choice": normalized_choice}
            ):
                self._write_transcript_line("[consent] failed to send response (stdin unavailable).")
                return

        def _finalize_assistant_message(self) -> None:
            self._cancel_streaming_flush_timer()
            self._flush_streaming_buffer()
            if not self._current_assistant_chunks:
                self._current_assistant_model = None
                self._current_assistant_timestamp = None
                self._assistant_placeholder_written = False
                return

            full_text = "".join(self._current_assistant_chunks)
            self._last_assistant_text = full_text
            model = self._current_assistant_model
            timestamp = self._current_assistant_timestamp or self._turn_timestamp()
            plain_lines = [full_text]
            meta_parts = [part for part in [model, timestamp] if isinstance(part, str) and part.strip()]
            if meta_parts:
                plain_lines.append(" · ".join(meta_parts))
            if not self._assistant_placeholder_written:
                self._mount_transcript_widget(
                    render_assistant_message(full_text, model=model, timestamp=timestamp),
                    plain_text="\n".join(plain_lines),
                )
            elif meta_parts:
                meta_line = " · ".join(meta_parts)
                self._mount_transcript_widget(render_system_message(meta_line), plain_text=meta_line)

            self._current_assistant_chunks = []
            self._streaming_buffer = []
            self._current_assistant_model = None
            self._current_assistant_timestamp = None
            self._assistant_placeholder_written = False

        def _handle_output_line(self, line: str, raw_line: str | None = None) -> None:
            if self.state.daemon.query_active:
                chunk = raw_line if raw_line is not None else (line + "\n")
                self.state.daemon.turn_output_chunks.append(sanitize_output_text(chunk))
            sanitized = sanitize_output_text(line)
            # Try structured JSONL first (emitted by TuiCallbackHandler).
            tui_event = parse_tui_event(sanitized)
            if tui_event is not None:
                _handle_daemon_event_router(self, tui_event)
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
                self.state.session.error_count += 1
                self._write_issue(text)
                self._update_header_status()
            elif event.kind == "warning":
                text = event.text
                if not text.startswith("WARN:"):
                    text = f"WARN: {text}"
                self.state.session.warning_count += 1
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
            _handle_daemon_event_router(self, event)

        def _classify_tui_error_event(self, event: dict[str, Any]) -> dict[str, Any]:
            return classify_tui_error_event(event)

        def _summarize_error_for_toast(self, error_info: dict[str, Any]) -> tuple[str, str, float | None]:
            return summarize_error_for_toast(error_info)

        def render_plan_panel(self, plan_json: dict[str, Any]) -> Any:
            return render_plan_panel(plan_json)

        def render_system_message(self, text: str) -> Any:
            return render_system_message(text)

        def render_tool_result_line(
            self,
            tool_name: str,
            *,
            status: str,
            duration_s: float,
            tool_input: dict | None = None,
            tool_use_id: str | None = None,
        ) -> Any:
            return render_tool_result_line(
                tool_name,
                status=status,
                duration_s=duration_s,
                tool_input=tool_input,
                tool_use_id=tool_use_id,
            )

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
            self.state.daemon.run_active_tier_warning_emitted = False
            if self.state.daemon.status_timer is not None:
                self.state.daemon.status_timer.stop()
                self.state.daemon.status_timer = None
            elapsed = (
                time.time() - self.state.daemon.run_start_time if self.state.daemon.run_start_time is not None else 0.0
            )
            if self._status_bar is not None:
                self._status_bar.set_state("idle")
                self._status_bar.set_elapsed(elapsed)
                self._status_bar.set_plan_step(current=None, total=None)
            self.state.daemon.run_start_time = None
            self.state.daemon.query_active = False
            self._clear_pending_tool_starts()
            self._cancel_tool_progress_flush_timer()
            self._tool_progress_pending_ids = set()
            for tool_use_id in list(self._tool_blocks.keys()):
                self._flush_tool_progress_render(tool_use_id, force=True)

            run_tool_count = self.state.daemon.run_tool_count
            completion_line = (
                f"[run] completed in {elapsed:.1f}s "
                f"({run_tool_count} tool calls, status={exit_status})"
            )
            self._write_transcript(completion_line)

            self._finalize_assistant_message()
            self._dismiss_thinking(emit_summary=True)

            output_text = "".join(self.state.daemon.turn_output_chunks)
            self.state.daemon.turn_output_chunks = []
            transcript_path = self._persist_run_transcript(
                pid=(self.state.daemon.proc.pid if self.state.daemon.proc is not None else None),
                session_id=self.state.daemon.session_id,
                prompt=self._last_prompt or "",
                auto_approve=self._last_run_auto_approve,
                exit_code=0 if exit_status == "ok" else 1,
                output_text=output_text,
            )
            if transcript_path:
                self._add_artifact_paths([transcript_path])

            log_path = self._discover_session_log_path(self.state.daemon.session_id)
            if log_path:
                self._add_artifact_paths([log_path])

            if not self.state.plan.received_structured_plan:
                extracted_plan = extract_plan_section_from_output(sanitize_output_text(output_text))
                if extracted_plan:
                    self.state.plan.pending_prompt = self._last_prompt
                    self._set_plan_panel(extracted_plan)
                    self._write_transcript_line(render_tui_hint_after_plan())

            self._reset_consent_panel()
            self.state.plan.received_structured_plan = False
            self._save_session()

        def _handle_daemon_exit(self, proc: _DaemonTransport, *, return_code: int) -> None:
            if self.state.daemon.proc is not proc:
                return
            was_query_active = self.state.daemon.query_active
            self.state.daemon.ready = False
            self.state.daemon.pending_model_select_value = None
            self.state.daemon.query_active = False
            self._context_ready_for_sync = bool(self._context_sources)
            self._sops_ready_for_sync = bool(self._active_sop_names)
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            self._clear_pending_tool_starts()
            self.state.daemon.proc = None
            self.state.daemon.runner_thread = None

            if self.state.daemon.status_timer is not None:
                self.state.daemon.status_timer.stop()
                self.state.daemon.status_timer = None
            if self._status_bar is not None:
                self._status_bar.set_state("idle")

            if was_query_active:
                self._finalize_turn(exit_status="error")
            else:
                self._reset_thinking_state()
            if self.state.daemon.is_shutting_down:
                return
            self._write_transcript_line(f"[daemon] exited unexpectedly (code {return_code}).")
            self._write_transcript_line("[daemon] run /daemon restart to restart the background agent.")

        def _stream_daemon_output(self, proc: _DaemonTransport) -> None:
            try:
                while True:
                    raw_line = proc.read_line()
                    if raw_line == "":
                        break
                    self._call_from_thread_safe(self._handle_output_line, raw_line.rstrip("\n"), raw_line)
            except Exception as exc:
                self._call_from_thread_safe(self._write_transcript_line, f"[daemon] output stream error: {exc}")
            finally:
                return_code = 0
                with contextlib.suppress(Exception):
                    return_code = proc.wait()
                self._call_from_thread_safe(self._handle_daemon_exit, proc, return_code=return_code)

        def _shutdown_transport(self, proc: _DaemonTransport) -> None:
            send_daemon_command(proc, {"cmd": "shutdown"})
            with contextlib.suppress(Exception):
                proc.wait(timeout=3.0)
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.close()

        def _request_daemon_shutdown(self) -> None:
            proc = self.state.daemon.proc
            if proc is None or proc.poll() is not None:
                self.state.daemon.ready = False
                self._write_transcript_line("[daemon] already stopped.")
                return
            self.state.daemon.ready = False
            self.state.daemon.is_shutting_down = True
            payload = {"cmd": "shutdown_service"} if isinstance(proc, _SocketTransport) else {"cmd": "shutdown"}
            if send_daemon_command(proc, payload):
                self._write_transcript_line("[daemon] shutdown requested.")
                return
            self.state.daemon.is_shutting_down = False
            self._write_transcript_line("[daemon] failed to send shutdown command.")

        def _spawn_daemon(self, *, restart: bool = False) -> None:
            self.state.daemon.is_shutting_down = False
            proc = self.state.daemon.proc
            if proc is not None and proc.poll() is None:
                if restart:
                    self.state.daemon.pending_model_select_value = None
                    self._shutdown_transport(proc)
                    self.state.daemon.proc = None
                else:
                    return

            requested_session_id = (self.state.daemon.session_id or "").strip() or uuid.uuid4().hex
            self.state.daemon.session_id = requested_session_id
            daemon: _DaemonTransport | None = None
            broker_error: Exception | None = None

            try:
                ensure_runtime_broker(cwd=Path.cwd())
            except Exception as exc:
                broker_error = exc

            try:
                daemon = _SocketTransport.connect(
                    session_id=requested_session_id,
                    cwd=Path.cwd(),
                    client_name="swarmee-tui",
                    surface="tui",
                )
            except Exception as exc:
                if broker_error is None:
                    broker_error = exc

            if daemon is None:
                try:
                    daemon_proc = spawn_swarmee_daemon(
                        session_id=requested_session_id,
                        env_overrides=self._model_env_overrides(),
                    )
                    daemon = _SubprocessTransport(daemon_proc)
                except Exception as exc:
                    self.state.daemon.ready = False
                    self._write_transcript_line(f"[daemon] failed to start: {exc}")
                    return

            self.state.daemon.proc = daemon
            self.state.daemon.ready = False
            self._context_ready_for_sync = bool(self._context_sources)
            self._sops_ready_for_sync = bool(self._active_sop_names)
            self.state.daemon.runner_thread = threading.Thread(
                target=self._stream_daemon_output,
                args=(daemon,),
                daemon=True,
                name="swarmee-tui-daemon-stream",
            )
            self.state.daemon.runner_thread.start()
            if isinstance(daemon, _SocketTransport):
                self._write_transcript_line("[daemon] connected to runtime broker, waiting for ready event.")
            else:
                if broker_error is not None and not isinstance(broker_error, FileNotFoundError):
                    self._write_transcript_line(
                        f"[daemon] runtime broker unavailable ({broker_error}); using local daemon."
                    )
                self._write_transcript_line("[daemon] started, waiting for ready event.")
            self._save_session()

        def _tick_status(self) -> None:
            if self.state.daemon.run_start_time is not None and self._status_bar is not None:
                self._status_bar.set_elapsed(time.time() - self.state.daemon.run_start_time)

        def _start_run(self, prompt: str, *, auto_approve: bool, mode: str | None = None) -> None:
            if not self.state.daemon.ready:
                self._write_transcript_line("[run] daemon is not ready. Use /daemon restart.")
                return
            proc = self.state.daemon.proc
            if proc is None or proc.poll() is not None:
                self._write_transcript_line("[run] daemon is not running. Use /daemon restart.")
                self.state.daemon.ready = False
                return
            if self.state.daemon.query_active:
                self._write_transcript_line("[run] already running; use /stop.")
                return
            self._dismiss_action_sheet(restore_focus=False)
            self._sync_selected_model_before_run()

            self.state.plan.pending_prompt = None
            self._reset_artifacts_panel()
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            self._reset_issues_panel()
            self._current_assistant_chunks = []
            self._streaming_buffer = []
            self._cancel_streaming_flush_timer()
            self._current_assistant_model = None
            self._current_assistant_timestamp = None
            self._assistant_placeholder_written = False
            self._reset_thinking_state()
            self._last_thinking_text = ""
            self._tool_blocks = {}
            self._clear_pending_tool_starts()
            self._tool_progress_pending_ids = set()
            self._cancel_tool_progress_flush_timer()
            self.state.daemon.run_tool_count = 0
            self.state.daemon.run_start_time = time.time()
            self.state.daemon.run_active_tier_warning_emitted = False
            self.state.plan.step_counter = 0
            self.state.plan.completion_announced = False
            mode_normalized = (mode or "").strip().lower()
            if mode_normalized == "execute" and self.state.plan.current_steps_total > 0:
                self.state.plan.current_step_statuses = ["pending"] * self.state.plan.current_steps_total
                self.state.plan.current_active_step = None
                self.state.plan.updates_seen = False
                self._render_plan_panel_from_status()
            else:
                self.state.plan.current_steps_total = 0
                self.state.plan.current_summary = ""
                self.state.plan.current_steps = []
                self.state.plan.current_step_statuses = []
                self.state.plan.current_active_step = None
                self.state.plan.updates_seen = False
            self.state.plan.received_structured_plan = False
            self.state.daemon.turn_output_chunks = []
            self.state.daemon.last_usage = None
            self.state.daemon.last_cost_usd = None
            if self._status_bar is not None:
                self._status_bar.set_state("running")
                self._status_bar.set_tool_count(0)
                self._status_bar.set_elapsed(0.0)
                self._status_bar.set_model(self._current_model_summary())
                self._status_bar.set_usage(None, cost_usd=None)
                self._status_bar.set_context(
                    prompt_tokens_est=self.state.daemon.last_prompt_tokens_est,
                    budget_tokens=self.state.daemon.last_budget_tokens,
                )
                if mode_normalized == "execute" and self.state.plan.current_steps_total > 0:
                    self._refresh_plan_status_bar()
                else:
                    self._status_bar.set_plan_step(current=None, total=None)
            self._refresh_prompt_metrics()
            if self.state.daemon.status_timer is not None:
                self.state.daemon.status_timer.stop()
            self.state.daemon.status_timer = self.set_interval(1.0, self._tick_status)
            self._last_prompt = prompt
            self._last_run_auto_approve = auto_approve
            self.state.daemon.query_active = True
            desired_tier = ""
            pending_value = (self.state.daemon.pending_model_select_value or "").strip().lower()
            if "|" in pending_value:
                _pending_provider, pending_tier = pending_value.split("|", 1)
                desired_tier = pending_tier.strip().lower()
            if not desired_tier:
                desired_tier = (self.state.daemon.model_tier_override or "").strip().lower()
            if not desired_tier:
                with contextlib.suppress(Exception):
                    selector = self.query_one("#model_select", Select)
                    selected_value = str(getattr(selector, "value", "")).strip()
                    parsed = parse_model_select_value(selected_value)
                    if parsed is not None:
                        _provider_name, parsed_tier = parsed
                        desired_tier = parsed_tier.strip().lower()
            command: dict[str, Any] = {
                "cmd": "query",
                "text": prompt,
                "auto_approve": bool(auto_approve),
            }
            if desired_tier:
                command["tier"] = desired_tier
            if mode:
                command["mode"] = mode
            if not send_daemon_command(proc, command):
                self.state.daemon.query_active = False
                if self.state.daemon.status_timer is not None:
                    self.state.daemon.status_timer.stop()
                    self.state.daemon.status_timer = None
                if self._status_bar is not None:
                    self._status_bar.set_state("idle")
                self._write_transcript_line("[run] failed to send query to daemon.")

        def _stop_run(self) -> None:
            proc = self.state.daemon.proc
            if proc is None or proc.poll() is not None:
                self._write_transcript_line("[run] no active run.")
                self.state.daemon.ready = False
                self._reset_consent_panel()
                self._reset_error_action_prompt()
                return
            if not self.state.daemon.query_active:
                self._write_transcript_line("[run] no active run.")
                self._reset_consent_panel()
                self._reset_error_action_prompt()
                return
            self._flush_all_streaming_buffers()
            self._clear_pending_tool_starts()
            self._dismiss_thinking(emit_summary=True)
            if send_daemon_command(proc, {"cmd": "interrupt"}):
                self._write_transcript_line("[run] interrupt requested.")
            else:
                self._write_transcript_line("[run] failed to send interrupt.")
            self._reset_consent_panel()
            self._reset_error_action_prompt()

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

        def action_quit(self) -> None:
            self.state.daemon.is_shutting_down = True
            timer = self._prompt_estimate_timer
            self._prompt_estimate_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()
            timeline_timer = self.state.session.timeline_refresh_timer
            self.state.session.timeline_refresh_timer = None
            if timeline_timer is not None:
                with contextlib.suppress(Exception):
                    timeline_timer.stop()
            self._cancel_streaming_flush_timer()
            self._cancel_tool_progress_flush_timer()
            self._clear_pending_tool_starts()
            self._reset_thinking_state()
            if self.state.daemon.proc is not None and self.state.daemon.proc.poll() is None:
                self._shutdown_transport(self.state.daemon.proc)
            if self.state.daemon.runner_thread is not None and self.state.daemon.runner_thread.is_alive():
                with contextlib.suppress(Exception):
                    self.state.daemon.runner_thread.join(timeout=1.0)
            self._save_session()
            self.exit(return_code=0)

        def action_copy_transcript(self) -> None:
            self._copy_text(self._get_transcript_text(), label="transcript")

        def action_copy_plan(self) -> None:
            self._copy_text((self.state.plan.text or "").rstrip() + "\n", label="plan")

        def action_copy_issues(self) -> None:
            self._flush_issue_repeats()
            payload = (
                ("\n".join(self.state.session.issue_lines).rstrip() + "\n") if self.state.session.issue_lines else ""
            )
            self._copy_text(payload, label="issues")

        def action_copy_artifacts(self) -> None:
            self._copy_text(self._get_artifacts_text(), label="artifacts")

        def action_focus_prompt(self) -> None:
            self.query_one("#prompt", PromptTextArea).focus()

        def action_submit_prompt(self) -> None:
            prompt_widget = self.query_one("#prompt", PromptTextArea)
            text = (prompt_widget.text or "").strip()
            prompt_widget.clear()
            if text:
                self._prompt_history.append(text)
                if len(self._prompt_history) > self._MAX_PROMPT_HISTORY:
                    self._prompt_history = self._prompt_history[-self._MAX_PROMPT_HISTORY :]
                self._history_index = -1
                self._handle_user_input(text)

        def action_interrupt_run(self) -> None:
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
                transcript_widget = self.query_one("#transcript", RichLog)

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

        def action_widen_side(self) -> None:
            if self._split_ratio > 1:
                self._split_ratio -= 1
                self._apply_split_ratio()

        def action_widen_transcript(self) -> None:
            if self._split_ratio < 4:
                self._split_ratio += 1
                self._apply_split_ratio()

        def _apply_split_ratio(self) -> None:
            transcript = self.query_one("#transcript", RichLog)
            transcript_text = self.query_one("#transcript_text", TextArea)
            side = self.query_one("#side", Vertical)
            transcript.styles.width = f"{self._split_ratio}fr"
            transcript_text.styles.width = f"{self._split_ratio}fr"
            side.styles.width = "1fr"
            self.refresh(layout=True)

        def action_toggle_transcript_mode(self) -> None:
            self._toggle_transcript_mode()

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
            term_lower = term.lower()
            transcript_text = self._get_transcript_text()
            if term_lower in transcript_text.lower():
                with contextlib.suppress(Exception):
                    self.query_one("#transcript", RichLog).scroll_end(animate=True)
                self._write_transcript_line("[search] found match in transcript.")
                return
            self._write_transcript_line(f"[search] no match for '{term}'.")

        def _request_context_compact(self) -> None:
            if self.state.daemon.query_active:
                self._write_transcript_line("[compact] unavailable while a run is active.")
                return
            if not self.state.daemon.ready:
                self._write_transcript_line("[compact] daemon is not ready. Use /daemon restart.")
                return
            proc = self.state.daemon.proc
            if proc is None or proc.poll() is not None:
                self.state.daemon.ready = False
                self._write_transcript_line("[compact] daemon is not running. Use /daemon restart.")
                return
            self._notify("Compacting context...", severity="information", timeout=5.0)
            if send_daemon_command(proc, {"cmd": "compact"}):
                self._write_transcript_line("[compact] requested context compaction.")
            else:
                self._write_transcript_line("[compact] failed to send compact command.")

        def _expand_tool_call(self, tool_use_id: str) -> None:
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

        def _open_artifact_path(self, path: str) -> None:
            resolved = str(path or "").strip()
            if not resolved:
                self._write_transcript_line("[open] invalid artifact path.")
                return
            editor = os.environ.get("EDITOR", "")
            try:
                if editor:
                    subprocess.Popen([editor, resolved])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", resolved])
                elif shutil.which("xdg-open"):
                    subprocess.Popen(["xdg-open", resolved])
                else:
                    self._write_transcript_line(f"[open] set $EDITOR. Path: {resolved}")
                    return
                self._write_transcript_line(f"[open] opened: {resolved}")
            except Exception as exc:
                self._write_transcript_line(f"[open] failed: {exc}")

        def _open_artifact(self, index_str: str) -> None:
            try:
                index = int(index_str.strip()) - 1
            except ValueError:
                self._write_transcript_line("Usage: /open <number>")
                return
            self._render_artifacts_panel()
            entries = self.state.artifacts.entries
            if index < 0 or index >= len(entries):
                self._write_transcript_line(f"[open] invalid index. {len(entries)} artifacts available.")
                return
            path = str(entries[index].get("path", "")).strip()
            self._open_artifact_path(path)

        def _copy_selected_artifact_path(self) -> None:
            selected = self._artifact_entry_by_item_id(self.state.artifacts.selected_item_id)
            if selected is None:
                self._notify("Select an artifact first.", severity="warning")
                return
            path = str(selected.get("path", "")).strip()
            if not path:
                self._notify("Selected artifact has no path.", severity="warning")
                return
            self._copy_text(path, label="artifact path")

        def _add_selected_artifact_as_context(self) -> None:
            selected = self._artifact_entry_by_item_id(self.state.artifacts.selected_item_id)
            if selected is None:
                self._notify("Select an artifact first.", severity="warning")
                return
            path = str(selected.get("path", "")).strip()
            if not path:
                self._notify("Selected artifact has no path.", severity="warning")
                return
            payload = artifact_context_source_payload(path)
            self._add_context_source(payload)
            self._write_transcript_line(f"[context] added file source from artifact: {path}")

        def _session_retry_tool(self, tool_use_id: str) -> None:
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
            proc = self.state.daemon.proc
            if not self.state.daemon.ready or proc is None or proc.poll() is not None:
                self._write_transcript_line("[session] daemon is not ready.")
                return
            if send_daemon_command(proc, {"cmd": "interrupt"}):
                self._write_transcript_line("[session] interrupt requested.")
            else:
                self._write_transcript_line("[session] failed to send interrupt.")

        def _save_session(self) -> None:
            try:
                session_path = sessions_dir() / "tui_session.json"
                session_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "prompt_history": self._prompt_history[-self._MAX_PROMPT_HISTORY :],
                    "last_prompt": self._last_prompt,
                    "plan_text": self.state.plan.text,
                    "artifacts": self.state.artifacts.recent_paths,
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
                artifacts = data.get("artifacts", [])
                if artifacts:
                    self.state.artifacts.recent_paths = artifacts
                    self._render_artifacts_panel()
                self._context_sources = _normalize_context_sources(data.get("context_sources", []))
                self._render_context_sources_panel()
                self._context_ready_for_sync = bool(self._context_sources)
                loaded_active_sops = data.get("active_sop_names", [])
                if isinstance(loaded_active_sops, list):
                    self._active_sop_names = {str(item).strip() for item in loaded_active_sops if str(item).strip()}
                self._refresh_sop_catalog()
                self._render_sop_panel()
                self._sops_ready_for_sync = bool(self._active_sop_names)
                self.state.daemon.session_id = str(data.get("daemon_session_id", "")).strip() or None
                self.state.daemon.available_restore_session_id = (
                    str(data.get("available_restore_session_id", "")).strip() or None
                )
                restore_turn_count_raw = data.get("available_restore_turn_count", 0)
                try:
                    self.state.daemon.available_restore_turn_count = max(0, int(restore_turn_count_raw or 0))
                except (TypeError, ValueError):
                    self.state.daemon.available_restore_turn_count = 0
                # Do not restore model overrides from prior sessions.
                # The daemon-reported model_info is the source of truth for startup model state.
                self.state.daemon.model_provider_override = None
                self.state.daemon.model_tier_override = None
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

            if key in {"ctrl+c", "meta+c", "super+c", "cmd+c", "command+c"}:
                event.stop()
                event.prevent_default()
                self.action_copy_selection()
                return

        def on_checkbox_changed(self, event: Any) -> None:
            checkbox = getattr(event, "checkbox", None)
            checkbox_id = str(getattr(checkbox, "id", "")).strip()
            if not checkbox_id:
                return
            sop_name = self._sop_toggle_id_to_name.get(checkbox_id)
            if not sop_name:
                return
            value = bool(getattr(event, "value", False))
            self._set_sop_active(sop_name, value, sync=True, announce=True)

        def on_select_changed(self, event: Any) -> None:
            select_widget = getattr(event, "select", None)
            select_id = str(getattr(select_widget, "id", "")).strip().lower()
            if select_id != "model_select":
                return
            if self.state.daemon.model_select_syncing:
                return

            value = str(getattr(event, "value", "")).strip()
            if not value:
                return
            if value == _MODEL_AUTO_VALUE:
                self.state.daemon.pending_model_select_value = None
                self.state.daemon.model_provider_override = None
                self.state.daemon.model_tier_override = None
            elif value == _MODEL_LOADING_VALUE:
                return
            elif "|" in value:
                provider, tier = value.split("|", 1)
                requested_provider = provider.strip().lower()
                requested_tier = tier.strip().lower()
                if not requested_provider or not requested_tier:
                    return
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
            # The model selector is always visible; avoid transient notifications.

        def on_sidebar_list_selection_changed(self, event: Any) -> None:
            sidebar_list = getattr(event, "sidebar_list", None)
            if sidebar_list is None:
                return

            if sidebar_list is self._session_issue_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_issue = self._session_issue_by_id(selected_id)
                self._set_session_issue_selection(selected_issue)
                return

            if sidebar_list is self._session_timeline_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_event = self._session_timeline_event_by_id(selected_id)
                self._set_session_timeline_selection(selected_event)
                return

            if sidebar_list is self._artifacts_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_entry = self._artifact_entry_by_item_id(selected_id)
                self._set_artifact_selection(selected_entry)
                return

            if sidebar_list is self._agent_profile_list:
                if self.state.agent_studio.form_syncing:
                    return
                selected_id = str(getattr(event, "item_id", "")).strip()
                if not selected_id or selected_id == _AGENT_PROFILE_SELECT_NONE:
                    return
                profile = self._lookup_saved_profile(selected_id)
                if profile is None:
                    return
                self._load_profile_into_draft(profile)
                return

            if sidebar_list is self._agent_tools_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_item = self._agent_tools_item_by_id(selected_id)
                self._set_agent_tools_selection(selected_item)
                return

            if sidebar_list is self._agent_team_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_item = self._agent_team_item_by_id(selected_id)
                self._set_agent_team_selection(selected_item)
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
            if input_id in {"agent_profile_id", "agent_profile_name"}:
                if self.state.agent_studio.form_syncing:
                    return
                self._set_agent_draft_dirty(True)
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

        def on_text_area_changed(self, event: Any) -> None:
            text_area = getattr(event, "text_area", None)
            text_area_id = str(getattr(text_area, "id", "")).strip().lower()
            if text_area_id != "agent_team_preset_spec":
                return
            if self.state.agent_studio.team_form_syncing:
                return
            self._set_agent_team_status("Team preset draft changes pending.")
            self._set_agent_draft_dirty(True)

        def _sync_selected_model_before_run(self) -> None:
            selected_value: str | None = None
            with contextlib.suppress(Exception):
                selector = self.query_one("#model_select", Select)
                selected_value = str(getattr(selector, "value", "")).strip()
            parsed = parse_model_select_value(selected_value)
            if parsed is None:
                return

            requested_provider, requested_tier = parsed
            self.state.daemon.model_provider_override = requested_provider or None
            self.state.daemon.model_tier_override = requested_tier or None

            current_provider = (self.state.daemon.provider or "").strip().lower()
            current_tier = (self.state.daemon.tier or "").strip().lower()
            if (
                current_provider
                and current_tier
                and requested_provider == current_provider
                and requested_tier == current_tier
            ):
                self.state.daemon.pending_model_select_value = None
                return
            self.state.daemon.pending_model_select_value = f"{requested_provider}|{requested_tier}"

        def _dispatch_plan_action(self, action: str) -> None:
            normalized = action.strip().lower()
            if normalized == "approve":
                if not self.state.plan.pending_prompt:
                    self._write_transcript_line("[run] no pending plan.")
                    return
                self._start_run(self.state.plan.pending_prompt, auto_approve=True, mode="execute")
                return
            if normalized == "replan":
                if not self._last_prompt:
                    self._write_transcript_line("[run] no previous prompt to replan.")
                    return
                self._start_run(self._last_prompt, auto_approve=False, mode="plan")
                return
            if normalized == "clearplan":
                self.state.plan.pending_prompt = None
                self._reset_plan_panel()
                self._write_transcript_line("[run] plan cleared.")
                return

        def _restore_available_session(self) -> None:
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
            self._write_transcript_line("[session] starting fresh.")
            self._save_session()

        def _handle_copy_command(self, normalized: str) -> bool:
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

        def _handle_model_command(self, normalized: str) -> bool:
            command = classify_model_command(normalized)
            if command is None:
                return False
            action, argument = command

            if action == "help":
                self._write_transcript_line(self._current_model_summary())
                self._write_transcript_line(_MODEL_USAGE_TEXT)
                return True

            if action == "show":
                self._write_transcript_line(self._current_model_summary())
                return True

            if action == "list":
                options, _ = self._model_select_options()
                for label, _value in options:
                    self._write_transcript_line(f"- {label}")
                return True

            if action == "reset":
                self.state.daemon.pending_model_select_value = None
                self.state.daemon.model_provider_override = None
                self.state.daemon.model_tier_override = None
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] reset. {self._current_model_summary()}")
                return True

            if action == "provider":
                provider = (argument or "").strip()
                if not provider:
                    self._write_transcript_line("Usage: /model provider <name>")
                    return True
                self.state.daemon.pending_model_select_value = None
                self.state.daemon.model_provider_override = provider
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] provider set to {provider}.")
                if self.state.daemon.ready:
                    self._write_transcript_line("[model] restart daemon to apply provider changes.")
                self._write_transcript_line(self._current_model_summary())
                return True

            if action == "tier":
                tier = (argument or "").strip()
                if not tier:
                    self._write_transcript_line("Usage: /model tier <name>")
                    return True
                if (
                    self.state.daemon.ready
                    and self.state.daemon.proc is not None
                    and self.state.daemon.proc.poll() is None
                    and self.state.daemon.query_active
                ):
                    self._warn_run_active_tier_change_once()
                    return True
                self.state.daemon.pending_model_select_value = None
                self.state.daemon.model_tier_override = tier
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] tier set to {tier}.")
                if (
                    self.state.daemon.ready
                    and self.state.daemon.proc is not None
                    and self.state.daemon.proc.poll() is None
                    and not self.state.daemon.query_active
                ):
                    requested_provider = (
                        (self.state.daemon.model_provider_override or self.state.daemon.provider or "").strip().lower()
                    )
                    requested_tier = tier.strip().lower()
                    if not send_daemon_command(self.state.daemon.proc, {"cmd": "set_tier", "tier": tier}):
                        self.state.daemon.pending_model_select_value = None
                        self._write_transcript_line("[model] failed to send tier change to daemon.")
                    elif requested_provider and requested_tier:
                        self.state.daemon.pending_model_select_value = f"{requested_provider}|{requested_tier}"
                self._write_transcript_line(self._current_model_summary())
                return True

            return False

        def _handle_pre_run_command(self, text: str) -> bool:
            classified = classify_pre_run_command(text)
            if classified is None:
                return False

            action, argument = classified
            if action == "open":
                self._open_artifact(argument or "")
                return True
            if action == "open_usage":
                self._write_transcript_line(_OPEN_USAGE_TEXT)
                return True
            if action == "restore":
                self._restore_available_session()
                return True
            if action == "new":
                self._start_fresh_session()
                return True
            if action == "context":
                return self._handle_context_command(argument or "")
            if action == "context_usage":
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True
            if action == "sop":
                return self._handle_sop_command(argument or "")
            if action == "sop_usage":
                self._write_transcript_line(_SOP_USAGE_TEXT)
                return True
            if action == "expand":
                self._expand_tool_call(argument or "")
                return True
            if action == "expand_usage":
                self._write_transcript_line(_EXPAND_USAGE_TEXT)
                return True
            if action == "search":
                self._search_transcript(argument or "")
                return True
            if action == "search_usage":
                self._write_transcript_line(_SEARCH_USAGE_TEXT)
                return True
            if action == "text":
                self._toggle_transcript_mode()
                return True
            if action == "text_usage":
                self._write_transcript_line(_TEXT_USAGE_TEXT)
                return True
            if action == "thinking":
                self._show_thinking_text()
                return True
            if action == "thinking_usage":
                self._write_transcript_line(_THINKING_USAGE_TEXT)
                return True
            if action == "compact":
                self._request_context_compact()
                return True
            if action == "compact_usage":
                self._write_transcript_line(_COMPACT_USAGE_TEXT)
                return True
            if action == "stop":
                self._stop_run()
                return True
            if action == "exit":
                self.action_quit()
                return True
            if action == "daemon_restart":
                self._spawn_daemon(restart=True)
                return True
            if action == "daemon_stop":
                self._request_daemon_shutdown()
                return True
            if action == "consent_usage":
                self._write_transcript_line(_CONSENT_USAGE_TEXT)
                return True
            if action == "consent":
                self._submit_consent_choice((argument or "").strip())
                return True
            if action == "connect":
                provider = (argument or "").strip() or "github_copilot"
                proc = self.state.daemon.proc
                if not self.state.daemon.ready or proc is None or proc.poll() is not None:
                    self._write_transcript_line("[connect] daemon is not ready.")
                    return True
                if self.state.daemon.query_active:
                    self._write_transcript_line("[connect] cannot connect while a run is active.")
                    return True
                self._write_transcript_line(f"[connect] starting provider auth for {provider}...")
                if not send_daemon_command(
                    proc,
                    {"cmd": "connect", "provider": provider, "method": "device", "open_browser": True},
                ):
                    self._write_transcript_line("[connect] failed to send command.")
                return True
            if action == "auth_usage":
                self._write_transcript_line(_AUTH_USAGE_TEXT)
                return True
            if action == "auth":
                raw = (argument or "").strip()
                normalized = raw.lower()
                proc = self.state.daemon.proc
                if not self.state.daemon.ready or proc is None or proc.poll() is not None:
                    self._write_transcript_line("[auth] daemon is not ready.")
                    return True
                if self.state.daemon.query_active:
                    self._write_transcript_line("[auth] cannot run while a run is active.")
                    return True
                if not raw:
                    self._write_transcript_line(_AUTH_USAGE_TEXT)
                    return True
                if normalized in {"list", "ls"}:
                    send_daemon_command(proc, {"cmd": "auth", "action": "list"})
                    return True
                if normalized.startswith("logout"):
                    parts = raw.split()
                    provider = parts[1].strip() if len(parts) >= 2 else "github_copilot"
                    send_daemon_command(proc, {"cmd": "auth", "action": "logout", "provider": provider})
                    return True
                self._write_transcript_line(_AUTH_USAGE_TEXT)
                return True
            if action.startswith("model:"):
                normalized = text.lower()
                return self._handle_model_command(normalized)
            return False

        def _handle_post_run_command(self, text: str) -> bool:
            classified = classify_post_run_command(text)
            if classified is None:
                return False

            action, argument = classified

            if action == "approve":
                self._dispatch_plan_action("approve")
                return True

            if action == "replan":
                self._dispatch_plan_action("replan")
                return True

            if action == "clearplan":
                self._dispatch_plan_action("clearplan")
                return True

            if action == "plan_mode":
                self._default_auto_approve = False
                self._update_prompt_placeholder()
                self._write_transcript_line("[mode] auto-approve disabled for default prompts.")
                return True

            if action == "plan_prompt":
                prompt = (argument or "").strip()
                if not prompt:
                    self._write_transcript_line("Usage: /plan <prompt>")
                    return True
                self._start_run(prompt, auto_approve=False, mode="plan")
                return True

            if action == "run_mode":
                self._default_auto_approve = True
                self._update_prompt_placeholder()
                self._write_transcript_line("[mode] auto-approve enabled for default prompts.")
                return True

            if action == "run_prompt":
                prompt = (argument or "").strip()
                if not prompt:
                    self._write_transcript_line("Usage: /run <prompt>")
                    return True
                self._start_run(prompt, auto_approve=True, mode="execute")
                return True

            return False

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
            if button_id == "agent_view_profile":
                self._set_agent_studio_view_mode("profile")
                return
            if button_id == "agent_view_tools":
                self._set_agent_studio_view_mode("tools")
                return
            if button_id == "agent_view_team":
                self._set_agent_studio_view_mode("team")
                return
            if button_id == "agent_team_new":
                self._new_agent_team_preset_draft()
                return
            if button_id == "agent_team_save":
                self._save_agent_team_preset_draft()
                return
            if button_id == "agent_team_delete":
                self._delete_selected_agent_team_preset()
                return
            if button_id == "agent_team_insert_prompt":
                self._insert_agent_team_preset_run_prompt(run_now=False)
                return
            if button_id == "agent_team_run_now":
                self._insert_agent_team_preset_run_prompt(run_now=True)
                return
            if button_id == "agent_tools_overrides_apply":
                self._apply_agent_tools_safety_overrides(reset=False)
                return
            if button_id == "agent_tools_overrides_reset":
                self._apply_agent_tools_safety_overrides(reset=True)
                return
            if button_id == "session_view_timeline":
                self._set_session_view_mode("timeline")
                return
            if button_id == "session_view_issues":
                self._set_session_view_mode("issues")
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
            if button_id == "plan_action_approve":
                self._dispatch_plan_action("approve")
                return
            if button_id == "plan_action_replan":
                self._dispatch_plan_action("replan")
                return
            if button_id == "plan_action_clear":
                self._dispatch_plan_action("clearplan")
                return
            if button_id == "agent_profile_new":
                self._new_agent_profile_draft()
                return
            if button_id == "agent_profile_save":
                self._save_agent_profile_draft()
                return
            if button_id == "agent_profile_delete":
                self._delete_selected_agent_profile()
                return
            if button_id == "agent_profile_apply":
                self._apply_agent_profile_draft()
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

            self._start_run(text, auto_approve=self._default_auto_approve)

    try:
        SwarmeeTUI().run()
    except KeyboardInterrupt:
        return 130
    return 0
