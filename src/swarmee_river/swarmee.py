#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""
Swarmee - A minimal CLI interface for Swarmee River (built on Strands)
"""

import argparse
import asyncio
import csv
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ROOT_HELP_EPILOG = (
    "Runtime commands:\n"
    "  swarmee daemon status            Show runtime broker status\n"
    "  swarmee daemon start             Start runtime broker\n"
    "  swarmee daemon stop              Stop runtime broker\n"
    "  swarmee daemon stop all          Stop all runtime brokers across scopes\n"
    "  swarmee broker stop              Alias for 'swarmee daemon stop'\n"
    "  swarmee serve                    Run runtime broker in foreground\n"
    "  swarmee attach [--tail]          Attach to runtime broker session\n"
    "  swarmee diagnostics tail         Tail persisted diagnostics\n"
    "  swarmee diagnostics bundle       Create a redacted support bundle\n"
    "  swarmee diagnostics doctor       Print runtime/auth diagnostics report\n"
    "\n"
    "TUI command: swarmee tui"
)

warnings.filterwarnings(
    "ignore",
    message='Field name "json" in "Http_requestTool" shadows an attribute in parent "BaseModel"',
    category=UserWarning,
    module=r"pydantic\.main",
)

# Strands
from strands import Agent
from strands.types.exceptions import MaxTokensReachedException
from swarmee_river.utils.user_input import get_user_input

from swarmee_river.handlers.callback_handler import (
    callback_handler,
    configure_callback_handler_mode,
    set_interrupt_event,
)

try:
    from prompt_toolkit import HTML as _PromptHTML, PromptSession as _PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout as _patch_stdout
except Exception:
    _PromptHTML = None  # type: ignore[misc,assignment]
    _PromptSession = None  # type: ignore[misc,assignment]
    _patch_stdout = None  # type: ignore[assignment]
HTML: Any = _PromptHTML
PromptSession: Any = _PromptSession
patch_stdout: Any = _patch_stdout

try:
    from rich.console import Console as _RichConsole
    from rich.panel import Panel as _RichPanel
    from rich.text import Text as _RichText
except Exception:
    _RichConsole = None  # type: ignore[misc,assignment]
    _RichPanel = None  # type: ignore[misc,assignment]
    _RichText = None  # type: ignore[misc,assignment]
Console: Any = _RichConsole
Panel: Any = _RichPanel
Text: Any = _RichText

try:
    from swarmee_river.hooks.file_diff_review import FileDiffReviewHooks as _FileDiffReviewHooks
    from swarmee_river.hooks.jsonl_logger import JSONLLoggerHooks as _JSONLLoggerHooks
    from swarmee_river.hooks.tui_metrics import TuiMetricsHooks as _TuiMetricsHooks
    from swarmee_river.hooks.tool_consent import ToolConsentHooks as _ToolConsentHooks
    from swarmee_river.hooks.tool_policy import ToolPolicyHooks as _ToolPolicyHooks
    from swarmee_river.hooks.tool_result_limiter import ToolResultLimiterHooks as _ToolResultLimiterHooks

    _HAS_STRANDS_HOOKS = True
except Exception:
    _FileDiffReviewHooks = None  # type: ignore[misc,assignment]
    _JSONLLoggerHooks = None  # type: ignore[misc,assignment]
    _TuiMetricsHooks = None  # type: ignore[misc,assignment]
    _ToolConsentHooks = None  # type: ignore[misc,assignment]
    _ToolResultLimiterHooks = None  # type: ignore[misc,assignment]
    _ToolPolicyHooks = None  # type: ignore[misc,assignment]
    _HAS_STRANDS_HOOKS = False
FileDiffReviewHooks: Any = _FileDiffReviewHooks
JSONLLoggerHooks: Any = _JSONLLoggerHooks
TuiMetricsHooks: Any = _TuiMetricsHooks
ToolConsentHooks: Any = _ToolConsentHooks
ToolResultLimiterHooks: Any = _ToolResultLimiterHooks
ToolPolicyHooks: Any = _ToolPolicyHooks

try:
    from swarmee_river.hooks.tool_message_repair import ToolMessageRepairHooks as _ToolMessageRepairHooks
except Exception:
    _ToolMessageRepairHooks = None  # type: ignore[misc,assignment]
ToolMessageRepairHooks: Any = _ToolMessageRepairHooks

try:
    from swarmee_river.hooks.session_s3 import SessionS3Hooks as _SessionS3Hooks
except Exception:
    _SessionS3Hooks = None  # type: ignore[misc,assignment]
SessionS3Hooks: Any = _SessionS3Hooks
from swarmee_river.agent_runner import invoke_agent
from swarmee_river.auth.github_copilot import login_device_flow, save_api_key
from swarmee_river.auth.store import delete_provider_record, list_auth_records, normalize_provider_name
from swarmee_river.artifacts import ArtifactStore, tools_expected_from_plan
from swarmee_river.cli.builtin_commands import register_builtin_commands
from swarmee_river.cli.commands import CLIContext, CommandRegistry, handle_pack_command
from swarmee_river.cli.diagnostics import (
    render_config_command_for_surface,
    render_diagnostic_command_for_surface,
    render_diagnostics_command_for_surface,
)
from swarmee_river.cli.repl import run_repl
from swarmee_river.context.prompt_cache import PromptCacheState
from swarmee_river.diagnostics import append_session_event, append_session_issue
from swarmee_river.error_classification import (
    ERROR_CATEGORY_AUTH_ERROR,
    ERROR_CATEGORY_FATAL,
    ERROR_CATEGORY_ESCALATABLE,
    ERROR_CATEGORY_TRANSIENT,
    ERROR_CATEGORY_TOOL_ERROR,
    classify_error_message,
)
from swarmee_river.interrupts import (
    AgentInterruptedError,
    interrupt_watcher_from_env,
    pause_active_interrupt_watcher_for_input,
)
from swarmee_river.packs import (
    find_agent_bundle,
    enabled_sop_paths,
    enabled_system_prompts,
    list_agent_bundles,
    load_enabled_pack_tools,
    with_deleted_agent_bundle,
    with_upserted_agent_bundle,
)
from swarmee_river.planning import (
    PendingWorkPlan,
    WorkPlan,
    classify_intent,
    new_pending_work_plan,
    pending_work_plan_from_payload,
    structured_plan_prompt,
)
from swarmee_river.prompt_assets import (
    PromptAsset,
    ensure_prompt_assets_bootstrapped,
    load_prompt_assets,
    resolve_agent_prompt_text,
    resolve_orchestrator_prompt_from_agent,
    save_prompt_assets,
)
from swarmee_river.profiles.models import ORCHESTRATOR_AGENT_ID, AgentProfile, normalize_agent_definitions
from swarmee_river.project_map import build_context_snapshot, build_project_map
from swarmee_river.runtime_service.client import (
    RuntimeServiceClient,
    default_session_id_for_cwd,
    ensure_runtime_broker,
    runtime_hello_supports_capability,
    runtime_discovery_path,
    shutdown_runtime_broker,
)
from swarmee_river.runtime_service.server import RuntimeServiceServer
from swarmee_river.runtime_env import detect_runtime_environment, render_runtime_environment_section
from swarmee_river.session.models import SessionModelManager
from swarmee_river.session.graph_index import (
    build_session_graph_index,
    write_session_graph_index,
)
from swarmee_river.session.store import SessionStore
from swarmee_river.settings import SwarmeeSettings, apply_project_env_overrides, load_settings, save_settings
from swarmee_river.tools import _HIDDEN_RUNTIME_TOOL_NAMES, get_tools
from swarmee_river.utils.agent_runtime_utils import (
    build_base_system_prompt,
    plan_json_for_execution,
    render_plan_text,
    resolve_effective_sop_paths,
)
from swarmee_river.utils.fork_utils import create_shared_prefix_child_agent
from swarmee_river.utils import model_utils
from swarmee_river.utils.env_utils import load_env_file, truthy
from swarmee_river.utils.kb_utils import load_system_prompt, store_conversation_in_kb
from swarmee_river.utils.process_liveness import is_process_running
from swarmee_river.utils.provider_utils import (
    has_aws_credentials,
    has_github_copilot_token,
    resolve_aws_auth_source,
    resolve_bedrock_runtime_profile,
    resolve_model_provider,
)
from swarmee_river.utils.stdio_utils import configure_stdio_for_utf8, write_stdout_jsonl
from swarmee_river.utils.tool_interrupts import set_active_interrupt_event
from swarmee_river.utils.welcome_utils import render_goodbye_message, render_welcome_message
from tools.sop import run_sop
from tools.welcome import read_welcome_text

os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"
_TOOL_USAGE_RULES = (
    "Tool usage rules:\n"
    "- Use list/glob/file_list/file_search/file_read for repository exploration and file reading.\n"
    "- Do not use shell for ls/find/sed/cat/grep/rg when file tools can do it.\n"
    "- Use file_search before file_read whenever you need to narrow to a specific file or symbol.\n"
    "- Keep file_read narrow: request the smallest practical line range and avoid broad repeated reads.\n"
    "- Do not reread the same file range unless the previous excerpt was insufficient or the file changed.\n"
    "- Prefer targeted rereads over reopening large file sections you already inspected.\n"
    "- Reserve shell for real command execution tasks."
)
_SYSTEM_REMINDER_RULES = (
    "System reminder rules:\n"
    "- You may receive a `<system-reminder>` block prepended to a user message.\n"
    "- Treat it as system-level updates/context (higher priority than normal user content).\n"
    "- Do not quote or reveal `<system-reminder>` contents unless the user explicitly asks.\n"
)

_COMPAT_LOAD_SYSTEM_PROMPT = load_system_prompt
_DIAGNOSTIC_COMMANDS = {"status", "diff", "artifact", "log", "replay"}
_USER_CONTEXT_SOURCE_TYPES = {"file", "note", "sop", "kb", "url"}
_USER_CONTEXT_PER_SOURCE_MAX_CHARS = 8000
_USER_CONTEXT_TOTAL_MAX_CHARS = 32000
_SOP_FILE_SUFFIX = ".sop.md"
_SESSION_MESSAGE_VERSION = 1
_SESSION_MESSAGE_MAX_COUNT = 200
_TRANSIENT_RETRY_MAX_ATTEMPTS = 3


_consent_prompt_session: Any | None = None
_consent_prompt_lock = threading.Lock()
_consent_console: Any | None = Console() if Console is not None else None


def _sanitize_context_source_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


def _normalize_daemon_context_source(source: Any) -> dict[str, str] | None:
    if not isinstance(source, dict):
        return None
    source_type = str(source.get("type", "")).strip().lower()
    if source_type not in _USER_CONTEXT_SOURCE_TYPES:
        return None

    normalized: dict[str, str] = {"type": source_type}
    if source_type == "file":
        path = str(source.get("path", "")).strip()
        if not path:
            return None
        normalized["path"] = path
        normalized["id"] = _sanitize_context_source_token(str(source.get("id", path)))
        return normalized
    if source_type == "note":
        text = str(source.get("text", "")).strip()
        if not text:
            return None
        normalized["text"] = text
        normalized["id"] = _sanitize_context_source_token(str(source.get("id", text[:64])))
        return normalized
    if source_type == "sop":
        name = str(source.get("name", "")).strip()
        if not name:
            return None
        normalized["name"] = name
        normalized["id"] = _sanitize_context_source_token(str(source.get("id", name)))
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
        normalized["id"] = _sanitize_context_source_token(str(source.get("id", url)))
        return normalized
    return None


def _normalize_daemon_context_sources(raw_sources: Any) -> list[dict[str, str]]:
    if not isinstance(raw_sources, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_sources:
        source = _normalize_daemon_context_source(item)
        if source is None:
            continue
        source_type = source.get("type", "")
        value_key = (
            source.get("path")
            or source.get("text")
            or source.get("name")
            or source.get("url")
            or source.get("id")
            or ""
        ).strip()
        dedupe_key = (source_type, value_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(source)
    return normalized


_SAFETY_OVERRIDE_CONSENT_VALUES = {"ask", "allow", "deny"}


def _normalize_safety_override_tool_list(raw_value: Any) -> list[str] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, list):
        raise ValueError("must be a list of strings or null")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_value:
        token = str(item).strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(token)
    return normalized


def _normalize_session_safety_overrides_update(raw_update: Any) -> dict[str, Any]:
    if not isinstance(raw_update, dict):
        raise ValueError("set_safety_overrides payload must be an object")

    normalized: dict[str, Any] = {}
    if "tool_consent" in raw_update:
        raw_consent = raw_update.get("tool_consent")
        if raw_consent is None:
            normalized["tool_consent"] = None
        else:
            consent = str(raw_consent).strip().lower()
            if consent not in _SAFETY_OVERRIDE_CONSENT_VALUES:
                raise ValueError("set_safety_overrides.tool_consent must be ask|allow|deny")
            normalized["tool_consent"] = consent

    if "tool_allowlist" in raw_update:
        try:
            normalized["tool_allowlist"] = _normalize_safety_override_tool_list(raw_update.get("tool_allowlist"))
        except ValueError as exc:
            raise ValueError(f"set_safety_overrides.tool_allowlist {exc}") from exc

    if "tool_blocklist" in raw_update:
        try:
            normalized["tool_blocklist"] = _normalize_safety_override_tool_list(raw_update.get("tool_blocklist"))
        except ValueError as exc:
            raise ValueError(f"set_safety_overrides.tool_blocklist {exc}") from exc

    return normalized


def _normalized_session_safety_overrides_payload(raw_state: Any) -> dict[str, Any]:
    if not isinstance(raw_state, dict):
        return {}
    payload: dict[str, Any] = {}
    consent = str(raw_state.get("tool_consent", "")).strip().lower()
    if consent in _SAFETY_OVERRIDE_CONSENT_VALUES:
        payload["tool_consent"] = consent
    for key in ("tool_allowlist", "tool_blocklist"):
        raw_list = raw_state.get(key)
        if not isinstance(raw_list, list):
            continue
        normalized = [str(item).strip() for item in raw_list if str(item).strip()]
        if normalized:
            payload[key] = normalized
    return payload


def _write_stdout_jsonl(event: dict[str, Any]) -> None:
    write_stdout_jsonl(event)


configure_stdio_for_utf8()


def _tui_events_enabled() -> bool:
    """True when running as a subprocess spawned by the TUI."""
    return truthy(os.getenv("SWARMEE_TUI_EVENTS"))


def _is_context_window_overflow_error(exc: BaseException) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    markers = (
        "context window overflow",
        "context length",
        "maximum context",
        "too many tokens",
        "token limit exceeded",
    )
    return any(marker in text for marker in markers)


def _is_escalatable_retry_exception(exc: BaseException) -> bool:
    if isinstance(exc, MaxTokensReachedException):
        return True
    category = str(classify_error_message(str(exc or "")).get("category", "")).strip().lower()
    return category == ERROR_CATEGORY_ESCALATABLE


def _remove_hidden_runtime_tools(agent: Any) -> None:
    hidden = set(_HIDDEN_RUNTIME_TOOL_NAMES)
    if not hidden:
        return
    registry = getattr(getattr(agent, "tool_registry", None), "registry", None)
    if isinstance(registry, dict):
        for name in hidden:
            registry.pop(name, None)
    dynamic_tools = getattr(getattr(agent, "tool_registry", None), "dynamic_tools", None)
    if isinstance(dynamic_tools, dict):
        for name in hidden:
            dynamic_tools.pop(name, None)


def _emit_tui_event(event: dict[str, Any]) -> None:
    """Emit a structured JSONL event to stdout for TUI consumption."""
    if _tui_events_enabled():
        _write_stdout_jsonl(event)


def _build_tui_error_event(
    text: str,
    *,
    category_hint: str | None = None,
    tool_use_id: str | None = None,
    retry_after_s: int | None = None,
    next_tier: str | None = None,
) -> dict[str, Any]:
    classified = classify_error_message(text, category_hint=category_hint, tool_use_id=tool_use_id)
    payload: dict[str, Any] = {
        "event": "error",
        # Keep both fields for compatibility with older TUI parsers.
        "text": text,
        "message": text,
        "category": classified["category"],
        "retryable": bool(classified.get("retryable", False)),
    }
    resolved_tool = classified.get("tool_use_id")
    if isinstance(resolved_tool, str) and resolved_tool.strip():
        payload["tool_use_id"] = resolved_tool.strip()
    if isinstance(retry_after_s, int) and retry_after_s > 0:
        payload["retry_after_s"] = retry_after_s
    if isinstance(next_tier, str) and next_tier.strip():
        payload["next_tier"] = next_tier.strip().lower()
    return payload


def _emit_classified_tui_error(
    text: str,
    *,
    category_hint: str | None = None,
    tool_use_id: str | None = None,
    retry_after_s: int | None = None,
    next_tier: str | None = None,
) -> None:
    _emit_tui_event(
        _build_tui_error_event(
            text,
            category_hint=category_hint,
            tool_use_id=tool_use_id,
            retry_after_s=retry_after_s,
            next_tier=next_tier,
        )
    )


def _extract_replay_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _extract_replay_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    if isinstance(content, dict):
        if content.get("toolUse") or content.get("toolResult"):
            return ""
        for key in ("text", "data", "delta", "output_text", "outputText", "textDelta"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
        nested = content.get("content")
        if nested is not None:
            nested_text = _extract_replay_text(nested)
            if nested_text:
                return nested_text
    return ""


def _extract_text_from_message_for_replay(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    extracted = _extract_replay_text(content)
    if extracted:
        return extracted
    for key in ("text", "data", "output_text"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _turn_count_from_messages(messages: list[Any]) -> int:
    return sum(1 for item in messages if isinstance(item, dict) and str(item.get("role", "")).strip().lower() == "user")


def _event_loop_running() -> bool:
    with contextlib.suppress(RuntimeError):
        return asyncio.get_running_loop().is_running()
    return False


def _prompt_input_with_prompt_toolkit(
    prompt: str,
    *,
    default: str = "",
    keyboard_interrupt_return_default: bool = True,
) -> str:
    if PromptSession is None or patch_stdout is None or HTML is None:
        try:
            response = input(f"{prompt} ")
        except (KeyboardInterrupt, EOFError):
            if keyboard_interrupt_return_default:
                return str(default)
            raise
        return str(response or default)

    global _consent_prompt_session
    with _consent_prompt_lock:
        if _consent_prompt_session is None:
            _consent_prompt_session = PromptSession()
        session = _consent_prompt_session

    try:
        with patch_stdout(raw=True):
            response = session.prompt(HTML(f"{prompt} "), in_thread=True)
    except (KeyboardInterrupt, EOFError):
        if keyboard_interrupt_return_default:
            return str(default)
        raise

    return str(response or default)


def _prompt_input_with_stdin(
    prompt: str,
    *,
    default: str = "",
    keyboard_interrupt_return_default: bool = True,
) -> str:
    try:
        response = input(f"{prompt} ")
    except (KeyboardInterrupt, EOFError):
        if keyboard_interrupt_return_default:
            return str(default)
        raise
    return str(response or default)


def _get_user_input_compat(
    prompt: str,
    *,
    default: str = "",
    keyboard_interrupt_return_default: bool = True,
    prefer_prompt_toolkit_in_async: bool = False,
) -> str:
    if _event_loop_running() and prefer_prompt_toolkit_in_async:
        return _prompt_input_with_prompt_toolkit(
            prompt,
            default=default,
            keyboard_interrupt_return_default=keyboard_interrupt_return_default,
        )
    if _event_loop_running():
        return _prompt_input_with_stdin(
            prompt,
            default=default,
            keyboard_interrupt_return_default=keyboard_interrupt_return_default,
        )
    return str(
        get_user_input(
            prompt,
            default=default,
            keyboard_interrupt_return_default=keyboard_interrupt_return_default,
        )
    )


def _consent_text_with_hotkey_emphasis(message: str) -> Any:
    if Text is None:
        return message
    text = Text(message, style="gold1")
    for matched in re.finditer(r"\[[ynavYNAV]\]", message):
        text.stylize("bold underline", matched.start(), matched.end())
    return text


def _render_tool_consent_message(message: str) -> None:
    text = (message or "").strip()
    if not text:
        return

    if _consent_console is not None and Panel is not None and Text is not None:
        consent_text = _consent_text_with_hotkey_emphasis(text)
        _consent_console.print()
        _consent_console.print(
            Panel(
                consent_text,
                title=Text("Tool Consent", style="bold blue"),
                border_style="blue",
                expand=False,
                padding=(0, 1),
            )
        )
        return

    print(f"\n[tool consent] {text}")


def _build_resolved_invocation_state(
    *,
    invocation_state: dict[str, Any] | None,
    runtime_environment: dict[str, Any],
    model_manager: SessionModelManager,
    selected_provider: str,
    settings: SwarmeeSettings,
    structured_output_model: type[Any] | None,
    session_safety_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_state: dict[str, Any] = dict(invocation_state) if isinstance(invocation_state, dict) else {}
    sw_state = resolved_state.setdefault("swarmee", {})
    if isinstance(sw_state, dict):
        current_provider = normalize_provider_name(getattr(model_manager, "current_provider", "") or selected_provider)
        sw_state.setdefault("runtime_environment", dict(runtime_environment))
        sw_state["tier"] = model_manager.current_tier
        sw_state["provider"] = current_provider
        if current_provider == "bedrock":
            sw_state.setdefault("invoke_mode", "isolated")
        profile = settings.harness.tier_profiles.get(model_manager.current_tier)
        if profile is not None:
            sw_state["tool_profile"] = profile.to_dict()
        if sw_state.get("mode") == "plan" and structured_output_model is not None:
            model_tool_name = getattr(structured_output_model, "__name__", "").strip()
            if model_tool_name:
                existing = sw_state.get("plan_allowed_tools")
                merged: set[str] = {model_tool_name}
                if isinstance(existing, (list, tuple, set)):
                    merged.update(str(item).strip() for item in existing if str(item).strip())
                sw_state["plan_allowed_tools"] = sorted(merged)
        tiers = model_manager.list_tiers()
        current = next(
            (
                item
                for item in tiers
                if item.name == model_manager.current_tier
                and normalize_provider_name(getattr(item, "provider", current_provider)) == current_provider
            ),
            None,
        )
        if current is not None:
            sw_state["model_id"] = getattr(current, "model_id", None)
            sw_state["transport"] = getattr(current, "transport", None)
            sw_state["reasoning_effort"] = getattr(current, "reasoning_effort", None)
            sw_state["reasoning_mode"] = getattr(current, "reasoning_mode", None)
            sw_state["tooling_mode"] = getattr(current, "tooling_mode", None)
            sw_state["tooling_discovery"] = getattr(current, "tooling_discovery", None)
            sw_state["context_strategy"] = getattr(current, "context_strategy", None)
            sw_state["context_compaction"] = getattr(current, "context_compaction", None)
            resolve_budget = getattr(model_manager, "resolve_effective_context_budget", None)
            sw_state["context_budget_tokens"] = resolve_budget() if callable(resolve_budget) else None
            sw_state["supports_guardrails"] = getattr(current, "supports_guardrails", None)
            sw_state["supports_cache_tools"] = getattr(current, "supports_cache_tools", None)
            sw_state["supports_forced_tool_with_reasoning"] = getattr(
                current, "supports_forced_tool_with_reasoning", None
            )
            sw_state["bedrock_message_cache_enabled"] = _bedrock_message_cache_enabled(
                model_manager=model_manager,
                selected_provider=selected_provider,
            )
        sw_state["session_safety_overrides"] = _normalized_session_safety_overrides_payload(session_safety_overrides)
    return resolved_state


def _resolved_context_budget_tokens(
    *,
    settings: SwarmeeSettings,
    model_manager: SessionModelManager,
) -> int:
    return model_manager.resolve_effective_context_budget()


def _estimate_agent_prompt_tokens(agent: Any, *, chars_per_token: int = 4) -> tuple[int, int]:
    from swarmee_river.context.budgeted_summarizing_conversation_manager import (
        estimate_tokens_for_agent,
        estimate_tool_schema_chars,
    )

    tool_schema_chars = estimate_tool_schema_chars(getattr(agent, "tools", None))
    return estimate_tokens_for_agent(agent, chars_per_token=chars_per_token), tool_schema_chars


def _bedrock_message_cache_enabled(*, model_manager: SessionModelManager, selected_provider: str) -> bool:
    tiers = model_manager.list_tiers()
    current_provider = normalize_provider_name(getattr(model_manager, "current_provider", "") or selected_provider)
    current = next(
        (
            item
            for item in tiers
            if item.name == model_manager.current_tier
            and normalize_provider_name(getattr(item, "provider", current_provider)) == current_provider
        ),
        None,
    )
    if current is None:
        return False
    return current_provider == "bedrock" and str(getattr(current, "context_strategy", "")).strip().lower() in {
        "cache_safe",
        "long_running",
    }


def _emit_tui_context_event_if_enabled(
    agent: Any,
    *,
    settings: SwarmeeSettings,
    model_manager: SessionModelManager,
) -> None:
    if not _tui_events_enabled():
        return
    try:
        chars_per_token = 4
        prompt_tokens_est, tool_schema_chars = _estimate_agent_prompt_tokens(agent, chars_per_token=chars_per_token)
        budget = _resolved_context_budget_tokens(settings=settings, model_manager=model_manager)
        diagnostics: dict[str, Any] = {}
        manager = getattr(agent, "conversation_manager", None)
        cache_diag_fn = getattr(manager, "cache_diagnostics_for_agent", None)
        if callable(cache_diag_fn):
            raw_diag = cache_diag_fn(agent)
            if isinstance(raw_diag, dict):
                diagnostics.update(raw_diag)
        _emit_tui_event(
            {
                "event": "context",
                "prompt_tokens_est": prompt_tokens_est,
                "budget_tokens": budget,
                "chars_per_token": chars_per_token,
                "messages": len(getattr(agent, "messages", []) or []),
                "tool_schema_chars": tool_schema_chars,
                "bedrock_message_cache_enabled": _bedrock_message_cache_enabled(
                    model_manager=model_manager,
                    selected_provider=getattr(model_manager, "current_provider", "") or "bedrock",
                ),
                **diagnostics,
            }
        )
    except Exception:
        pass


def _build_session_meta_payload(
    *,
    settings: SwarmeeSettings,
    selected_provider: str,
    current_tier: str,
    active_sop_names: list[str] | None,
) -> dict[str, Any]:
    normalized_sops = [name.strip() for name in (active_sop_names or []) if str(name).strip()]
    enabled_packs = [
        {"name": p.name, "path": p.path, "enabled": p.enabled}
        for p in settings.packs.installed
        if getattr(p, "enabled", True)
    ]
    try:
        pm = build_project_map()
        git_root = pm.get("git_root")
    except Exception:
        git_root = None
    return {
        "cwd": str(Path.cwd()),
        "git_root": git_root,
        "provider": selected_provider,
        "tier": current_tier,
        "packs": enabled_packs,
        # Keep both fields for backwards compatibility with existing session metadata readers.
        "active_sop": normalized_sops[0] if normalized_sops else None,
        "active_sops": normalized_sops,
    }


def _build_model_info_event_payload(
    *,
    model_manager: SessionModelManager,
    selected_provider: str,
    tool_names: list[str] | None = None,
) -> dict[str, Any]:
    tiers = model_manager.list_tiers()
    current_provider = normalize_provider_name(getattr(model_manager, "current_provider", "") or selected_provider)
    current = next(
        (
            item
            for item in tiers
            if item.name == model_manager.current_tier
            and normalize_provider_name(getattr(item, "provider", current_provider)) == current_provider
        ),
        None,
    )
    normalized_tool_names = sorted({str(name).strip() for name in (tool_names or []) if str(name).strip()})
    message_cache_enabled = _bedrock_message_cache_enabled(
        model_manager=model_manager,
        selected_provider=selected_provider,
    )
    return {
        "event": "model_info",
        "provider": current.provider if current is not None else current_provider,
        "tier": model_manager.current_tier,
        "model_id": current.model_id if current is not None else None,
        "context_budget_tokens": model_manager.resolve_effective_context_budget(),
        "bedrock_message_cache_enabled": message_cache_enabled,
        "tool_names": normalized_tool_names,
        "tiers": [
            {
                "name": item.name,
                "provider": item.provider,
                "model_id": item.model_id,
                "display_name": item.display_name,
                "description": item.description,
                "transport": item.transport,
                "reasoning_effort": item.reasoning_effort,
                "reasoning_mode": item.reasoning_mode,
                "tooling_mode": item.tooling_mode,
                "tooling_discovery": item.tooling_discovery,
                "context_strategy": item.context_strategy,
                "context_compaction": item.context_compaction,
                "context_budget_tokens": item.context_max_prompt_tokens,
                "supports_guardrails": item.supports_guardrails,
                "supports_cache_tools": item.supports_cache_tools,
                "supports_forced_tool_with_reasoning": item.supports_forced_tool_with_reasoning,
                "available": item.available,
                "reason": item.reason,
            }
            for item in tiers
        ],
    }


def _render_auth_records_text() -> str:
    records = list_auth_records(include_opencode=True)
    if not records:
        return "No provider credentials found."
    lines = ["# Auth records", ""]
    for item in records:
        provider = str(item.get("provider", ""))
        source = str(item.get("source", ""))
        auth_type = str(item.get("type", "unknown"))
        details: list[str] = [auth_type, source]
        if item.get("has_key"):
            details.append("key")
        if item.get("has_refresh"):
            details.append("refresh")
        if item.get("has_access"):
            details.append("access")
        lines.append(f"- {provider}: {', '.join(details)}")
    return "\n".join(lines)


def _run_aws_identity_check(*, profile: str | None) -> tuple[bool, str]:
    command = ["aws", "sts", "get-caller-identity", "--output", "json"]
    resolved_profile = (profile or "").strip()
    if resolved_profile:
        command.extend(["--profile", resolved_profile])
    env = dict(os.environ)
    if resolved_profile:
        env["AWS_PROFILE"] = resolved_profile
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    except Exception as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    return False, stderr or stdout or f"aws sts failed (exit {result.returncode})"


def _connect_aws_credentials(
    *,
    profile: str | None,
    emit: Callable[[str], None],
) -> str:
    aws_path = shutil.which("aws")
    if not aws_path:
        raise RuntimeError("AWS CLI was not found on PATH. Install AWS CLI v2 and configure a profile.")
    explicit_profile = str(profile or "").strip()
    if explicit_profile:
        previous_profile = os.environ.get("AWS_PROFILE")
        os.environ["AWS_PROFILE"] = explicit_profile
        emit(f"Checking AWS credentials for profile '{explicit_profile}'...")
        ok, check_message = _run_aws_identity_check(profile=explicit_profile)
        if ok:
            has_creds, source = resolve_aws_auth_source()
            resolved_source = source if has_creds else "profile"
            emit(f"AWS credentials already valid for profile '{explicit_profile}' (source={resolved_source}).")
            return resolved_source
        _ambient_profile, ambient_has_creds, ambient_source, _ambient_warning = resolve_bedrock_runtime_profile(None)
        if ambient_has_creds:
            if previous_profile is None:
                os.environ.pop("AWS_PROFILE", None)
            else:
                os.environ["AWS_PROFILE"] = previous_profile
            emit(
                f"Profile '{explicit_profile}' is unavailable; using the default credential chain instead "
                f"(source={ambient_source})."
            )
            return ambient_source
        emit(f"AWS credentials unavailable for profile '{explicit_profile}'. Starting aws sso login...")
        login_cmd = [aws_path, "sso", "login", "--profile", explicit_profile]
        login_env = dict(os.environ)
        login_env["AWS_PROFILE"] = explicit_profile
        try:
            process = subprocess.Popen(
                login_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=login_env,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to start AWS SSO login: {exc}") from exc

        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = str(raw_line or "").strip()
                    if line:
                        emit(f"[aws] {line}")
        finally:
            return_code = process.wait()

        if return_code != 0:
            suffix = f" Last check: {check_message}" if check_message else ""
            raise RuntimeError(f"AWS SSO login failed for profile '{explicit_profile}'.{suffix}")

        ok_after, verify_message = _run_aws_identity_check(profile=explicit_profile)
        if not ok_after:
            raise RuntimeError(
                f"AWS login completed but credentials are still unavailable for profile '{explicit_profile}': "
                f"{verify_message}"
            )
        has_creds, source = resolve_aws_auth_source()
        resolved_source = source if has_creds else "profile"
        emit(f"AWS credentials refreshed for profile '{explicit_profile}' (source={resolved_source}).")
        return resolved_source

    emit("Checking AWS credentials via default credential chain...")
    ok, check_message = _run_aws_identity_check(profile=None)
    if ok:
        has_creds, source = resolve_aws_auth_source()
        resolved_source = source if has_creds else "unknown"
        emit(f"AWS credentials available via default chain (source={resolved_source}).")
        return resolved_source

    has_creds, source = resolve_aws_auth_source()
    if has_creds:
        emit(f"AWS credentials resolved from botocore source '{source}', but sts get-caller-identity failed.")
        return source
    raise RuntimeError(
        "AWS credentials are unavailable in the default credential chain. "
        "Use `swarmee connect aws <profile>` for SSO locally, or rely on role-based credentials (for example in "
        "SageMaker) without configuring a named profile."
        + (f" Last check: {check_message}" if check_message else "")
    )


def _handle_auth_cli_command(command: str, args: list[str]) -> tuple[bool, str]:
    cmd = (command or "").strip().lower()
    if cmd == "connect":
        provider = normalize_provider_name(args[0] if args else "github_copilot")
        if provider == "bedrock":
            profile = args[1] if len(args) >= 2 else None
            source = _connect_aws_credentials(profile=profile, emit=lambda line: print(line))
            resolved_profile = str(profile or "").strip()
            if resolved_profile:
                return True, f"AWS credentials connected for profile '{resolved_profile}' (source={source})."
            return True, f"AWS credentials connected via default chain (source={source})."
        if provider != "github_copilot":
            return True, "Usage: swarmee connect [github_copilot] | swarmee connect aws [profile]"
        result = login_device_flow(status=lambda line: print(line), open_browser=True)
        return True, f"GitHub Copilot connected.\nSaved credentials to: {result.get('path')}"

    if cmd != "auth":
        return False, ""

    subcmd = args[0].strip().lower() if args else "list"
    if subcmd in {"list", "ls"}:
        return True, _render_auth_records_text()
    if subcmd == "logout":
        provider = normalize_provider_name(args[1] if len(args) >= 2 else "github_copilot")
        if not provider:
            return True, "Usage: swarmee auth logout <provider>"
        deleted = delete_provider_record(provider)
        if deleted:
            return True, f"Removed saved credentials for provider: {provider}"
        return True, f"No saved credentials found for provider: {provider}"
    if subcmd == "login":
        provider_raw = "github_copilot"
        if len(args) >= 2 and not str(args[1]).strip().startswith("-"):
            provider_raw = args[1]
        provider = normalize_provider_name(provider_raw)
        if provider != "github_copilot":
            return True, "Only github_copilot login is currently supported."
        if "--api-key" in args:
            key = (os.getenv("SWARMEE_GITHUB_COPILOT_API_KEY") or "").strip()
            if not key:
                return True, (
                    "Set SWARMEE_GITHUB_COPILOT_API_KEY and re-run `swarmee auth login --api-key` to save it locally."
                )
            path = save_api_key(key)
            return True, f"Saved GitHub Copilot API key to: {path}"
        result = login_device_flow(status=lambda line: print(line), open_browser=True)
        return True, f"GitHub Copilot connected.\nSaved credentials to: {result.get('path')}"
    return (
        True,
        "Usage: swarmee auth list | swarmee auth login [github_copilot] [--api-key] | swarmee auth logout [provider]",
    )


def _build_conversation_manager(
    *,
    settings: SwarmeeSettings,
    window_size: int | None,
    per_turn: int | None,
    max_prompt_tokens: int | None = None,
    summarization_system_prompt: str | None = None,
    strategy: str | None = None,
    compaction_mode: str | None = None,
    stable_tool_names: list[str] | None = None,
) -> Any:
    manager_name = str(settings.context.manager or "summarize").strip().lower()
    resolved_strategy = str(strategy or settings.context.strategy or "balanced").strip().lower() or "balanced"
    resolved_compaction = str(compaction_mode or settings.context.compaction or "auto").strip().lower() or "auto"

    if manager_name in {"none", "null", "off", "disabled"}:
        try:
            from strands.agent.conversation_manager import NullConversationManager
        except Exception:
            return None
        return NullConversationManager()

    if manager_name in {"summarize", "summary", "summarizing"}:
        try:
            from swarmee_river.context.budgeted_summarizing_conversation_manager import (
                BudgetedSummarizingConversationManager,
                SWARMEE_SUMMARIZATION_SYSTEM_PROMPT,
            )
        except Exception:
            return None

        return BudgetedSummarizingConversationManager(
            max_prompt_tokens=max_prompt_tokens,
            summary_ratio=float(settings.context.summary_ratio),
            preserve_recent_messages=int(settings.context.preserve_recent_messages),
            summarization_system_prompt=summarization_system_prompt or SWARMEE_SUMMARIZATION_SYSTEM_PROMPT,
            strategy=resolved_strategy,
            compaction_mode=resolved_compaction,
            stable_tool_names=stable_tool_names,
        )

    # Default: sliding window
    try:
        from strands.agent.conversation_manager import SlidingWindowConversationManager
    except Exception:
        return None

    resolved_window_size = window_size if window_size is not None else int(settings.context.window_size)
    resolved_per_turn = per_turn if per_turn is not None else int(settings.context.per_turn)
    should_truncate_results = bool(settings.context.truncate_results)

    return SlidingWindowConversationManager(
        window_size=resolved_window_size,
        should_truncate_results=should_truncate_results,
        per_turn=resolved_per_turn,
    )


def _build_runtime_hooks(
    *,
    args: argparse.Namespace,
    settings: SwarmeeSettings,
    safety_settings: Any,
    auto_approve: bool,
    consent_prompt_fn: Callable[[str], str] | None,
) -> list[Any]:
    if not _HAS_STRANDS_HOOKS:
        return []

    def _consent_prompt(text: str, payload: dict[str, Any] | None = None) -> str:
        if callable(consent_prompt_fn):
            try:
                return str(consent_prompt_fn(text, payload) or "")
            except TypeError:
                return str(consent_prompt_fn(text) or "")
        if _tui_events_enabled():
            event_payload = {
                "event": "consent_prompt",
                "context": text,
                "options": ["y", "n", "a", "v"],
            }
            if isinstance(payload, dict):
                event_payload.update(payload)
            _emit_tui_event(event_payload)
            try:
                return input().strip()
            except (KeyboardInterrupt, EOFError):
                return ""
        callback_handler(force_stop=True)
        _render_tool_consent_message(text)
        with pause_active_interrupt_watcher_for_input():
            return _get_user_input_compat(
                "\n~ consent> ",
                default="",
                keyboard_interrupt_return_default=True,
                prefer_prompt_toolkit_in_async=False,
            )

    hooks = [
        JSONLLoggerHooks(),
        TuiMetricsHooks(pricing=settings.pricing),
        ToolPolicyHooks(safety_settings, runtime=settings.runtime),
        ToolConsentHooks(
            safety_settings,
            interactive=not bool(args.query),
            auto_approve=auto_approve,
            prompt=_consent_prompt,
        ),
        FileDiffReviewHooks(),
        ToolResultLimiterHooks(enabled=settings.runtime.limit_tool_results),
    ]
    if ToolMessageRepairHooks is not None:
        hooks.insert(2, ToolMessageRepairHooks())
    if settings.runtime.session_s3_bucket and SessionS3Hooks is not None:
        hooks.append(SessionS3Hooks(settings=settings))
    return hooks


def _build_runtime_tools(settings: SwarmeeSettings) -> tuple[dict[str, Any], list[Any]]:
    tools_dict = get_tools(settings)
    for name, tool_obj in load_enabled_pack_tools(settings).items():
        tools_dict.setdefault(name, tool_obj)
    return tools_dict, [tools_dict[name] for name in sorted(tools_dict)]


def _resolve_provider_and_model_manager(
    *,
    args: argparse.Namespace,
    settings: SwarmeeSettings,
) -> tuple[str, str | None, SessionModelManager]:
    selected_provider, provider_notice = resolve_model_provider(
        cli_provider=args.model_provider.stem if args.model_provider is not None else None,
        env_provider=None,
        settings_provider=settings.models.provider,
    )
    provider_explicit = bool(args.model_provider is not None) or bool(str(settings.models.provider or "").strip())
    if selected_provider == "openai" and not provider_explicit:
        compatibility = model_utils.probe_openai_responses_transport()
        if not compatibility.available:
            fallback_provider = "github_copilot" if has_github_copilot_token() else "bedrock"
            selected_provider = fallback_provider
            notice = (
                "Skipping implicit OpenAI provider selection because Swarmee's OpenAI Responses runtime "
                f"is unavailable. {compatibility.reason} Falling back to {fallback_provider}."
            )
            provider_notice = f"{provider_notice} {notice}".strip() if provider_notice else notice
    model_manager = SessionModelManager(settings, fallback_provider=selected_provider)
    model_manager.set_fallback_config(args.model_config)
    return selected_provider, provider_notice, model_manager


def _build_agent_runtime(
    args: argparse.Namespace,
    settings: SwarmeeSettings,
    auto_approve: bool,
    consent_prompt_fn: Callable[[str], str] | None,
    *,
    knowledge_base_id: str | None = None,
    settings_path_for_project: Path | None = None,
) -> dict[str, Any]:
    selected_provider, provider_notice, model_manager = _resolve_provider_and_model_manager(
        args=args,
        settings=settings,
    )
    model = model_manager.build_model()
    runtime_environment = detect_runtime_environment(cwd=Path.cwd())
    runtime_environment_prompt_section = render_runtime_environment_section(runtime_environment)

    tools_dict, tools = _build_runtime_tools(settings)

    pack_sop_paths = enabled_sop_paths(settings)
    pack_prompt_sections = enabled_system_prompts(settings)
    ensure_prompt_assets_bootstrapped()
    raw_system_prompt = resolve_orchestrator_prompt_from_agent(None)
    compat_system_prompt = str(load_system_prompt() or "").strip()
    if compat_system_prompt and compat_system_prompt not in raw_system_prompt:
        raw_system_prompt = (
            f"{compat_system_prompt}\n\n{raw_system_prompt}" if raw_system_prompt else compat_system_prompt
        )
    base_system_prompt = build_base_system_prompt(
        raw_system_prompt=raw_system_prompt,
        runtime_environment_prompt_section=runtime_environment_prompt_section,
        pack_prompt_sections=pack_prompt_sections,
        tool_usage_rules=_TOOL_USAGE_RULES,
        system_reminder_rules=_SYSTEM_REMINDER_RULES,
    )
    effective_sop_paths = resolve_effective_sop_paths(cli_sop_paths=args.sop_paths, pack_sop_paths=pack_sop_paths)

    initial_context_behavior = model_manager.current_context_behavior()
    initial_context_strategy = str(initial_context_behavior.strategy or settings.context.strategy or "balanced")
    initial_compaction_mode = str(initial_context_behavior.compaction or settings.context.compaction or "auto")
    summarization_system_prompt = None
    conversation_manager = _build_conversation_manager(
        settings=settings,
        window_size=args.window_size,
        per_turn=args.context_per_turn,
        max_prompt_tokens=model_manager.resolve_effective_context_budget(),
        summarization_system_prompt=summarization_system_prompt,
        strategy=initial_context_strategy,
        compaction_mode=initial_compaction_mode,
        stable_tool_names=sorted(tools_dict),
    )
    hooks = _build_runtime_hooks(
        args=args,
        settings=settings,
        safety_settings=settings.safety,
        auto_approve=auto_approve,
        consent_prompt_fn=consent_prompt_fn,
    )

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "system_prompt": base_system_prompt,
        "callback_handler": callback_handler,
        "load_tools_from_directory": not bool(settings.runtime.freeze_tools),
    }
    if conversation_manager is not None:
        agent_kwargs["conversation_manager"] = conversation_manager
    if hooks:
        agent_kwargs["hooks"] = hooks

    prompt_cache: PromptCacheState | None = None

    def create_agent(*, messages: Any | None = None, state: Any | None = None) -> Agent:
        kwargs = dict(agent_kwargs)
        if messages is not None:
            kwargs["messages"] = messages
        if state is not None:
            kwargs["state"] = state
        try:
            agent_instance = Agent(**kwargs)
        except TypeError:
            kwargs.pop("conversation_manager", None)
            kwargs.pop("hooks", None)
            kwargs.pop("messages", None)
            kwargs.pop("state", None)
            agent_instance = Agent(**kwargs)
        _remove_hidden_runtime_tools(agent_instance)
        agent_instance._swarmee_prompt_cache = prompt_cache
        agent_instance._swarmee_current_invocation_state = None
        return agent_instance

    agent = create_agent()

    def _refresh_conversation_manager_for_current_tier() -> None:
        nonlocal conversation_manager, summarization_system_prompt
        behavior = model_manager.current_context_behavior()
        strategy = str(behavior.strategy or settings.context.strategy or "balanced").strip().lower() or "balanced"
        compaction_mode = str(behavior.compaction or settings.context.compaction or "auto").strip().lower() or "auto"
        summarization_system_prompt = None
        conversation_manager = _build_conversation_manager(
            settings=settings,
            window_size=args.window_size,
            per_turn=args.context_per_turn,
            max_prompt_tokens=model_manager.resolve_effective_context_budget(),
            summarization_system_prompt=summarization_system_prompt,
            strategy=strategy,
            compaction_mode=compaction_mode,
            stable_tool_names=sorted(tools_dict),
        )
        if conversation_manager is not None:
            agent_kwargs["conversation_manager"] = conversation_manager
        else:
            agent_kwargs.pop("conversation_manager", None)
        with contextlib.suppress(Exception):
            agent.conversation_manager = conversation_manager

    interrupt_event = threading.Event()

    preflight_prompt_section: str | None = None
    project_map_prompt_section: str | None = None
    active_plan_prompt_section: str | None = None
    artifact_store = ArtifactStore()
    prompt_cache = PromptCacheState()
    agent._swarmee_prompt_cache = prompt_cache
    active_knowledge_base_id: str | None = knowledge_base_id
    user_context_sources: list[dict[str, str]] = []
    active_profile_system_prompt_snippets: list[str] = []
    active_profile_agents: list[dict[str, Any]] = []
    active_orchestrator_agent: dict[str, Any] | None = None
    auto_delegate_assistive = True
    user_context_active_reminder_keys: set[str] = set()
    user_context_file_cache: dict[str, dict[str, Any]] = {}
    user_context_sop_cache: dict[str, dict[str, Any]] = {}
    user_context_last_warning: str | None = None
    user_context_lock = threading.Lock()
    active_sop_overrides: dict[str, str] = {}
    active_sop_lock = threading.Lock()
    daemon_managed_sops = False
    session_safety_overrides: dict[str, Any] = {}

    def _prompt_assets_by_id() -> dict[str, PromptAsset]:
        ensure_prompt_assets_bootstrapped()
        return {asset.id: asset for asset in load_prompt_assets()}

    def _normalize_prompt_asset_payload(raw: Any) -> PromptAsset:
        if not isinstance(raw, dict):
            raise ValueError("prompt asset payload must be an object")
        return PromptAsset.from_dict(raw)

    def _orchestrator_agent_from_agents(agents: list[dict[str, Any]] | None) -> dict[str, Any] | None:
        for agent in normalize_agent_definitions(agents or []):
            if str(agent.get("id", "")).strip().lower() == ORCHESTRATOR_AGENT_ID:
                return dict(agent)
        return None

    def _orchestrator_prompt_refs(orchestrator_agent: dict[str, Any] | None) -> set[str]:
        refs = {
            str(item).strip().lower()
            for item in ((orchestrator_agent or {}).get("prompt_refs") or [])
            if str(item).strip()
        }
        return refs or {"orchestrator_base"}

    def _refresh_orchestrator_system_prompt() -> None:
        nonlocal base_system_prompt, summarization_system_prompt
        raw_system_prompt = resolve_orchestrator_prompt_from_agent(active_orchestrator_agent, _prompt_assets_by_id())
        compat_system_prompt = str(load_system_prompt() or "").strip()
        if compat_system_prompt and compat_system_prompt not in raw_system_prompt:
            raw_system_prompt = (
                f"{compat_system_prompt}\n\n{raw_system_prompt}" if raw_system_prompt else compat_system_prompt
            )
        base_system_prompt = build_base_system_prompt(
            raw_system_prompt=raw_system_prompt,
            runtime_environment_prompt_section=runtime_environment_prompt_section,
            pack_prompt_sections=pack_prompt_sections,
            tool_usage_rules=_TOOL_USAGE_RULES,
            system_reminder_rules=_SYSTEM_REMINDER_RULES,
        )
        agent_kwargs["system_prompt"] = base_system_prompt
        with contextlib.suppress(Exception):
            agent.system_prompt = base_system_prompt
        if settings.context.cache_safe_summary:
            summarization_system_prompt = base_system_prompt

    def get_prompt_assets_payload() -> dict[str, Any]:
        assets = load_prompt_assets()
        orchestrator_prompt_id = next(iter(_orchestrator_prompt_refs(active_orchestrator_agent)), "orchestrator_base")
        return {
            "event": "prompt_assets",
            "orchestrator_prompt_id": orchestrator_prompt_id,
            "assets": [asset.to_dict() for asset in assets],
        }

    def set_prompt_asset(raw_asset: Any) -> dict[str, Any]:
        asset = _normalize_prompt_asset_payload(raw_asset)
        assets_by_id = _prompt_assets_by_id()
        assets_by_id[asset.id] = asset
        save_prompt_assets(list(assets_by_id.values()))
        if asset.id in _orchestrator_prompt_refs(active_orchestrator_agent):
            _refresh_orchestrator_system_prompt()
            _refresh_query_context(interactive=True)
        return get_prompt_assets_payload()

    def delete_prompt_asset(prompt_id: str) -> dict[str, Any]:
        token = str(prompt_id or "").strip().lower()
        if not token:
            raise ValueError("delete_prompt_asset.id is required")
        if token in _orchestrator_prompt_refs(active_orchestrator_agent):
            raise ValueError("Cannot delete orchestrator-bound prompt asset")
        assets_by_id = _prompt_assets_by_id()
        if token not in assets_by_id:
            raise ValueError(f"Prompt asset not found: {token}")
        assets_by_id.pop(token, None)
        save_prompt_assets(list(assets_by_id.values()))
        return get_prompt_assets_payload()

    def _context_source_key(source: dict[str, str]) -> str:
        source_type = source.get("type", "").strip().lower()
        raw_value = (
            source.get("id")
            or source.get("path")
            or source.get("name")
            or source.get("text", "")[:64]
            or source.get("url")
            or uuid.uuid4().hex
        )
        token = _sanitize_context_source_token(raw_value)
        return f"user_context_{source_type}_{token}"

    def _context_per_source_limit() -> int:
        # Fixed guardrail (not end-user configurable via env).
        return _USER_CONTEXT_PER_SOURCE_MAX_CHARS

    def _context_total_limit() -> int:
        # Fixed guardrail (not end-user configurable via env).
        return _USER_CONTEXT_TOTAL_MAX_CHARS

    def _resolve_context_file_path(path_text: str) -> Path:
        candidate = Path(path_text).expanduser()
        return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()

    def _load_cached_text_for_path(
        *,
        path: Path,
        cache: dict[str, dict[str, Any]],
        max_chars: int,
    ) -> str | None:
        try:
            stat = path.stat()
        except Exception:
            return None
        cache_key = str(path)
        entry = cache.get(cache_key)
        if entry and int(entry.get("mtime_ns", -1)) == int(stat.st_mtime_ns):
            return str(entry.get("text", ""))
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        text = raw[:max_chars].strip()
        cache[cache_key] = {"mtime_ns": int(stat.st_mtime_ns), "text": text}
        return text

    def _iter_effective_sop_dirs() -> list[Path]:
        dirs: list[Path] = []
        cwd_sops = Path.cwd() / "sops"
        if cwd_sops.exists() and cwd_sops.is_dir():
            dirs.append(cwd_sops)
        if effective_sop_paths:
            for token in str(effective_sop_paths).split(os.pathsep):
                candidate = Path(token).expanduser()
                if candidate.exists() and candidate.is_dir():
                    dirs.append(candidate)
        deduped: list[Path] = []
        seen: set[str] = set()
        for directory in dirs:
            key = str(directory.resolve())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(directory)
        return deduped

    def _resolve_sop_file_path(name: str) -> Path | None:
        sop_name = name.strip()
        if not sop_name:
            return None
        filename = sop_name if sop_name.endswith(_SOP_FILE_SUFFIX) else f"{sop_name}{_SOP_FILE_SUFFIX}"
        for directory in _iter_effective_sop_dirs():
            candidate = directory / filename
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        return None

    def _set_active_sop_override(name: str, content: str | None) -> None:
        sop_name = name.strip()
        if not sop_name:
            return
        with active_sop_lock:
            if content is None:
                active_sop_overrides.pop(sop_name, None)
                return
            normalized = str(content).strip()
            if not normalized:
                active_sop_overrides.pop(sop_name, None)
                return
            active_sop_overrides[sop_name] = normalized

    def _snapshot_active_sop_overrides() -> dict[str, str]:
        with active_sop_lock:
            return dict(active_sop_overrides)

    def _list_active_sop_names() -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        if not daemon_managed_sops and ctx.active_sop_name:
            base = ctx.active_sop_name.strip()
            if base:
                lowered = base.lower()
                seen.add(lowered)
                names.append(base)
        for name in _snapshot_active_sop_overrides().keys():
            normalized = name.strip()
            if normalized:
                lowered = normalized.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                names.append(normalized)
        return names

    def _provider_for_tier_name(tier_name: str) -> str | None:
        requested_tier = (tier_name or "").strip().lower()
        if not requested_tier:
            return None
        for provider_name, provider_cfg in settings.models.providers.items():
            tiers = getattr(provider_cfg, "tiers", {}) or {}
            if requested_tier in tiers:
                provider = normalize_provider_name(provider_name)
                return provider or None
        global_tier = settings.models.tiers.get(requested_tier)
        if global_tier is not None:
            provider = normalize_provider_name(getattr(global_tier, "provider", ""))
            return provider or None
        return None

    def _current_provider_name() -> str | None:
        provider = normalize_provider_name(getattr(model_manager, "current_provider", ""))
        return provider or _provider_for_tier_name(model_manager.current_tier)

    def _resolve_sop_content(name: str) -> str | None:
        sop_name = name.strip()
        if not sop_name:
            return None
        try:
            sop_result = run_sop(action="get", name=sop_name, sop_paths=effective_sop_paths)
        except Exception:
            return None
        if not isinstance(sop_result, dict):
            return None
        if str(sop_result.get("status", "")).strip().lower() != "success":
            return None
        content_entries = sop_result.get("content")
        if not isinstance(content_entries, list) or not content_entries:
            return None
        first_entry = content_entries[0]
        if not isinstance(first_entry, dict):
            return None
        text = str(first_entry.get("text", "")).strip()
        return text or None

    def _set_daemon_sop_override(name: str, content: str | None) -> None:
        nonlocal daemon_managed_sops
        daemon_managed_sops = True
        _set_active_sop_override(name, content)

    def _replace_daemon_sop_overrides(sop_stack: list[tuple[str, str]]) -> list[str]:
        nonlocal daemon_managed_sops
        daemon_managed_sops = True
        with active_sop_lock:
            active_sop_overrides.clear()
            for raw_name, raw_content in sop_stack:
                sop_name = raw_name.strip()
                sop_content = raw_content.strip()
                if not sop_name or not sop_content:
                    continue
                active_sop_overrides[sop_name] = sop_content
            return list(active_sop_overrides.keys())

    def _queue_user_context_reminders() -> str | None:
        nonlocal user_context_active_reminder_keys, user_context_last_warning
        total_limit = _context_total_limit()
        per_source_limit = _context_per_source_limit()
        used_chars = 0
        seen_keys: set[str] = set()
        warning_messages: list[str] = []
        with user_context_lock:
            sources_snapshot = [dict(item) for item in user_context_sources]
            active_keys_snapshot = set(user_context_active_reminder_keys)

        for source in sources_snapshot:
            source_type = source.get("type", "").strip().lower()
            if source_type == "kb":
                continue

            key = _context_source_key(source)
            seen_keys.add(key)
            section = ""
            if source_type == "file":
                resolved_path = _resolve_context_file_path(source.get("path", ""))
                text = _load_cached_text_for_path(
                    path=resolved_path,
                    cache=user_context_file_cache,
                    max_chars=per_source_limit,
                )
                if text:
                    section = f"User Context File ({resolved_path}):\n{text}"
                else:
                    warning_messages.append(f"[context] failed to read file source: {resolved_path}")
            elif source_type == "note":
                text = source.get("text", "").strip()
                section = f"User Context Note:\n{text[:per_source_limit]}".strip() if text else ""
            elif source_type == "sop":
                sop_name = source.get("name", "").strip()
                sop_path = _resolve_sop_file_path(sop_name)
                sop_text = ""
                if sop_path is not None:
                    loaded = _load_cached_text_for_path(
                        path=sop_path,
                        cache=user_context_sop_cache,
                        max_chars=per_source_limit,
                    )
                    sop_text = loaded or ""
                else:
                    try:
                        sop_result = run_sop(action="get", name=sop_name, sop_paths=effective_sop_paths)
                        if sop_result.get("status") == "success":
                            sop_text = str(sop_result.get("content", [{}])[0].get("text", "")).strip()[
                                :per_source_limit
                            ]
                    except Exception:
                        sop_text = ""
                if sop_text:
                    section = f"User SOP Context ({sop_name}):\n{sop_text}"
                else:
                    warning_messages.append(f"[context] failed to resolve sop source: {sop_name}")
            elif source_type == "url":
                url_text = source.get("url", "").strip()
                section = f"User Context URL reference:\n{url_text[:per_source_limit]}".strip() if url_text else ""

            section = section.strip()
            if not section:
                prompt_cache.queue_if_changed(key, "")
                continue

            if used_chars >= total_limit:
                prompt_cache.queue_if_changed(key, "")
                warning_messages.append(
                    f"[context] user context exceeded {total_limit} chars; truncated before source {key}."
                )
                continue

            remaining = total_limit - used_chars
            if len(section) > remaining:
                section = section[:remaining].rstrip()
                warning_messages.append(
                    f"[context] user context exceeded {total_limit} chars; additional content was truncated."
                )

            used_chars += len(section)
            prompt_cache.queue_if_changed(key, section)

        for stale_key in active_keys_snapshot - seen_keys:
            prompt_cache.queue_if_changed(stale_key, "")
        with user_context_lock:
            user_context_active_reminder_keys = seen_keys

        warning = warning_messages[0] if warning_messages else None
        if warning != user_context_last_warning:
            user_context_last_warning = warning
            return warning
        return None

    def set_user_context_sources(raw_sources: Any) -> list[dict[str, str]]:
        nonlocal user_context_sources, user_context_active_reminder_keys, active_knowledge_base_id
        normalized = _normalize_daemon_context_sources(raw_sources)
        with user_context_lock:
            old_keys = {_context_source_key(item) for item in user_context_sources if item.get("type") != "kb"}
            new_keys = {_context_source_key(item) for item in normalized if item.get("type") != "kb"}
            user_context_active_reminder_keys = new_keys
            user_context_sources = normalized
        for stale_key in old_keys - new_keys:
            prompt_cache.queue_if_changed(stale_key, "")
        kb_override = next((item.get("id", "").strip() for item in normalized if item.get("type") == "kb"), "")
        active_knowledge_base_id = kb_override or knowledge_base_id
        return [dict(item) for item in normalized]

    def current_knowledge_base_id() -> str | None:
        return active_knowledge_base_id

    def apply_session_safety_overrides(raw_update: Any) -> dict[str, Any]:
        nonlocal session_safety_overrides
        normalized_update = _normalize_session_safety_overrides_update(raw_update)
        next_overrides = dict(_normalized_session_safety_overrides_payload(session_safety_overrides))
        for key in ("tool_consent", "tool_allowlist", "tool_blocklist"):
            if key not in normalized_update:
                continue
            value = normalized_update.get(key)
            if value is None:
                next_overrides.pop(key, None)
                continue
            if key in {"tool_allowlist", "tool_blocklist"} and isinstance(value, list) and not value:
                next_overrides.pop(key, None)
                continue
            next_overrides[key] = value
        session_safety_overrides = _normalized_session_safety_overrides_payload(next_overrides)
        return dict(session_safety_overrides)

    def current_session_safety_overrides() -> dict[str, Any]:
        return dict(_normalized_session_safety_overrides_payload(session_safety_overrides))

    def compact_context() -> dict[str, Any]:
        manager = conversation_manager
        before_tokens: int | None = None
        after_tokens: int | None = None
        compacted = False
        compacted_read_results = 0
        summary_passes = 0
        trimmed_messages = 0
        warning: str | None = None
        effective_budget = model_manager.resolve_effective_context_budget()
        compact_result: dict[str, Any] | None = None

        with contextlib.suppress(Exception):
            before_tokens, _tool_schema_chars = _estimate_agent_prompt_tokens(agent, chars_per_token=4)

        if manager is None:
            warning = "Context compaction is unavailable for this session."
        else:
            try:
                compact_fn = getattr(manager, "compact_to_budget", None)
                reduce_fn = getattr(manager, "reduce_context", None)
                apply_fn = getattr(manager, "apply_management", None)
                if callable(compact_fn):
                    compact_result = compact_fn(agent)
                    compacted_read_results = int(compact_result.get("compacted_read_results", 0) or 0)
                    summary_passes = int(compact_result.get("summary_passes", 0) or 0)
                    trimmed_messages = int(compact_result.get("trimmed_messages", 0) or 0)
                    compacted = bool(
                        summary_passes > 0
                        or trimmed_messages > 0
                        or compacted_read_results > 0
                    )
                elif callable(reduce_fn):
                    reduce_fn(agent, e=None)
                    summary_passes = 1
                    compacted = True
                elif callable(apply_fn):
                    apply_fn(agent)
                    summary_passes = 1
                    compacted = True
                else:
                    warning = "Current conversation manager does not support compaction."
            except Exception as exc:
                warning = f"Context compaction failed: {exc}"

        with contextlib.suppress(Exception):
            after_tokens, _tool_schema_chars = _estimate_agent_prompt_tokens(agent, chars_per_token=4)
        if (
            isinstance(before_tokens, int)
            and isinstance(after_tokens, int)
            and after_tokens >= before_tokens
            and compacted
            and warning is None
        ):
            compacted = False
        return {
            "compacted": compacted,
            "before_tokens_est": before_tokens,
            "after_tokens_est": after_tokens,
            "summary_passes": summary_passes,
            "trimmed_messages": trimmed_messages,
            "within_budget": bool(isinstance(after_tokens, int) and after_tokens <= effective_budget),
            "budget_tokens": effective_budget,
            "compacted_read_results": compacted_read_results,
            "compaction_headroom_tokens": compact_result.get("compaction_headroom_tokens")
            if isinstance(compact_result, dict)
            else None,
            "warning": warning,
            "fork_kind": compact_result.get("fork_kind") if isinstance(compact_result, dict) else None,
            "fork_parent_message_count": compact_result.get("fork_parent_message_count")
            if isinstance(compact_result, dict)
            else None,
            "fork_prefix_hash": compact_result.get("fork_prefix_hash") if isinstance(compact_result, dict) else None,
            "fork_extra_prompt_chars": compact_result.get("fork_extra_prompt_chars")
            if isinstance(compact_result, dict)
            else None,
            "fork_used_pending_reminder": compact_result.get("fork_used_pending_reminder")
            if isinstance(compact_result, dict)
            else None,
        }

    def refresh_system_prompt(welcome_text_local: str) -> None:
        nonlocal auto_delegate_assistive
        prompt_cache.queue_if_changed("project_map", project_map_prompt_section)
        prompt_cache.queue_if_changed("preflight", preflight_prompt_section)
        prompt_cache.queue_if_changed("active_plan", active_plan_prompt_section)
        snippet_sections: list[str] = []
        for idx, snippet in enumerate(active_profile_system_prompt_snippets, start=1):
            text = snippet.strip()
            if not text:
                continue
            snippet_sections.append(f"Profile System Snippet {idx}:\n{text}")
        prompt_cache.queue_if_changed("profile_system_prompt_snippets", "\n\n".join(snippet_sections))
        if args.include_welcome_in_prompt and welcome_text_local:
            prompt_cache.queue_if_changed("welcome", f"Welcome Text Reference:\n{welcome_text_local}")

        active_sop_sections: list[str] = []
        active_sop_names_seen: set[str] = set()

        def _append_active_sop_section(name: str, content: str) -> None:
            sop_name = name.strip()
            sop_body = content.strip()
            if not sop_name or not sop_body:
                return
            token = sop_name.lower()
            if token in active_sop_names_seen:
                return
            active_sop_names_seen.add(token)
            active_sop_sections.append(f"Active SOP ({sop_name}):\n{sop_body}")

        if not daemon_managed_sops and ctx.active_sop_name:
            sop_text = ""
            try:
                sop_result = run_sop(
                    action="get",
                    name=ctx.active_sop_name,
                    sop_paths=effective_sop_paths,
                )
                if sop_result.get("status") == "success":
                    sop_text = sop_result.get("content", [{}])[0].get("text", "")
            except Exception:
                sop_text = ""
            _append_active_sop_section(ctx.active_sop_name, sop_text)

        override_snapshot = _snapshot_active_sop_overrides()
        for sop_name, sop_content in override_snapshot.items():
            _append_active_sop_section(sop_name, sop_content)

        if active_sop_sections:
            prompt_cache.queue_if_changed("active_sop", "\n\n".join(active_sop_sections))
        else:
            prompt_cache.queue_if_changed("active_sop", "")

        if auto_delegate_assistive:
            activated_agents = [
                item
                for item in active_profile_agents
                if isinstance(item, dict)
                and str(item.get("id", "")).strip().lower() != ORCHESTRATOR_AGENT_ID
                and bool(item.get("activated"))
            ]
            assets_by_id = _prompt_assets_by_id()
            guidance_lines: list[str] = []
            for idx, agent_def in enumerate(activated_agents, start=1):
                name = str(agent_def.get("name", "")).strip() or f"agent_{idx}"
                summary = str(agent_def.get("summary", "")).strip()
                prompt = resolve_agent_prompt_text(agent_def, assets_by_id)
                tools = [str(tool).strip() for tool in agent_def.get("tool_names", []) if str(tool).strip()]
                line = f"{idx}. {name}"
                if summary:
                    line += f" - {summary}"
                if tools:
                    line += f" [tools: {', '.join(tools)}]"
                if prompt:
                    line += f"\n   Instructions: {prompt[:220]}"
                guidance_lines.append(line)
            if guidance_lines:
                prompt_cache.queue_if_changed(
                    "assistive_delegation",
                    (
                        "Assistive Delegation Roster:\n"
                        "When a task matches one of these roles, prefer delegating via a single `swarm` call.\n"
                        "If delegation does not fit, continue normally.\n\n" + "\n".join(guidance_lines)
                    ),
                )
            else:
                prompt_cache.queue_if_changed("assistive_delegation", "")
        else:
            prompt_cache.queue_if_changed("assistive_delegation", "")

        warning_text = _queue_user_context_reminders()
        if warning_text:
            _emit_tui_event({"event": "warning", "text": warning_text})

    def run_agent(
        query: str,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[Any] | None = None,
        structured_output_prompt: str | None = None,
    ) -> Any:
        nonlocal agent
        _emit_tui_context_event_if_enabled(agent, settings=settings, model_manager=model_manager)
        invocation_state = _build_resolved_invocation_state(
            invocation_state=invocation_state,
            runtime_environment=runtime_environment,
            model_manager=model_manager,
            selected_provider=selected_provider,
            settings=settings,
            structured_output_model=structured_output_model,
            session_safety_overrides=session_safety_overrides,
        )

        # Inject agent-level context metadata for LLM call inspector hooks.
        sw = invocation_state.get("swarmee")
        if isinstance(sw, dict):
            sys_prompt = getattr(agent, "system_prompt", None) or ""
            sw["system_prompt_chars"] = len(sys_prompt)
            agent_tools = getattr(agent, "tools", None) or []
            sw["tool_count"] = len(agent_tools)
            try:
                from swarmee_river.context.budgeted_summarizing_conversation_manager import estimate_tool_schema_chars

                sw["tool_schema_chars"] = estimate_tool_schema_chars(agent_tools)
            except Exception:
                sw["tool_schema_chars"] = 0

        interrupt_event.clear()
        set_interrupt_event(interrupt_event)
        set_active_interrupt_event(interrupt_event)
        try:
            with interrupt_watcher_from_env(interrupt_event):
                try:
                    agent._swarmee_current_invocation_state = invocation_state
                    budget_manager = conversation_manager
                    effective_budget = model_manager.resolve_effective_context_budget()
                    compact_result: dict[str, Any] | None = None
                    compact_fn = getattr(budget_manager, "compact_to_budget", None)
                    if callable(compact_fn):
                        compact_result = compact_fn(agent)
                    if isinstance(compact_result, dict):
                        before_tokens = compact_result.get("before_tokens_est")
                        after_tokens = compact_result.get("after_tokens_est")
                        summary_passes = int(compact_result.get("summary_passes", 0) or 0)
                        trimmed_messages = int(compact_result.get("trimmed_messages", 0) or 0)
                        compacted_read_results = int(compact_result.get("compacted_read_results", 0) or 0)
                        within_budget = bool(compact_result.get("within_budget", False))
                        compacted = bool(summary_passes > 0 or trimmed_messages > 0 or compacted_read_results > 0)
                        _emit_tui_context_event_if_enabled(agent, settings=settings, model_manager=model_manager)
                        if compacted or not within_budget:
                            _emit_tui_event(
                                {
                                    "event": "compact_complete",
                                    "automatic": True,
                                    "compacted": compacted,
                                    "before_tokens_est": before_tokens,
                                    "after_tokens_est": after_tokens,
                                    "budget_tokens": effective_budget,
                                    "summary_passes": summary_passes,
                                    "trimmed_messages": trimmed_messages,
                                    "compacted_read_results": compacted_read_results,
                                    "compaction_headroom_tokens": compact_result.get("compaction_headroom_tokens"),
                                    "warning": compact_result.get("warning"),
                                    "fork_kind": compact_result.get("fork_kind"),
                                    "fork_parent_message_count": compact_result.get("fork_parent_message_count"),
                                    "fork_prefix_hash": compact_result.get("fork_prefix_hash"),
                                    "fork_extra_prompt_chars": compact_result.get("fork_extra_prompt_chars"),
                                    "fork_used_pending_reminder": compact_result.get("fork_used_pending_reminder"),
                                }
                            )
                        if not within_budget:
                            after_label = f"{int(after_tokens):,}" if isinstance(after_tokens, int) else "unknown"
                            error_msg = (
                                "Prompt still exceeds the configured context budget after compaction.\n"
                                f"- Current budget: {effective_budget:,} tokens\n"
                                f"- Current estimate after compaction: {after_label} tokens\n"
                                "- Fix: open Settings > General and increase Context Budget, "
                                "or shorten the conversation."
                            )
                            _emit_classified_tui_error(error_msg, category_hint=ERROR_CATEGORY_ESCALATABLE)
                            if not _tui_events_enabled():
                                print(f"\n{error_msg}")
                            raise RuntimeError(error_msg)
                    system_reminder = prompt_cache.pop_reminder()
                    agent._swarmee_current_invocation_state = invocation_state
                    return invoke_agent(
                        agent,
                        query,
                        callback_handler=callback_handler,
                        interrupt_event=interrupt_event,
                        invocation_state=invocation_state,
                        system_reminder=system_reminder or None,
                        structured_output_model=structured_output_model,
                        structured_output_prompt=structured_output_prompt,
                    )
                except MaxTokensReachedException:
                    callback_handler(force_stop=True)
                    configured = (
                        args.max_output_tokens
                        if isinstance(getattr(args, "max_output_tokens", None), int)
                        else settings.models.max_output_tokens
                    )
                    configured_label = (
                        str(configured)
                        if isinstance(configured, int) and configured > 0
                        else "(default)"
                    )
                    error_msg = (
                        "Error: Response hit the max output token limit.\n"
                        f"- Current max: {configured_label}\n"
                        "- Fix: increase `models.max_output_tokens` in `.swarmee/settings.json` "
                        "(or pass --max-output-tokens), or ask for a shorter response.\n"
                        "- Resetting agent loop so you can continue."
                    )
                    _emit_classified_tui_error(error_msg, category_hint=ERROR_CATEGORY_ESCALATABLE)
                    if not _tui_events_enabled():
                        print(f"\n{error_msg}")
                    agent = create_agent()
                    raise
                finally:
                    agent._swarmee_current_invocation_state = None
        finally:
            set_active_interrupt_event(None)
            set_interrupt_event(None)

    def _escalation_attempts() -> int:
        return max(1, model_manager.max_escalations_per_task + 1)

    def _maybe_escalate_tier(*, attempted: set[str]) -> bool:
        prev_tier = model_manager.current_tier
        if not model_manager.maybe_escalate(agent, attempted=attempted):
            return False
        if model_manager.current_tier != prev_tier and not _tui_events_enabled():
            print(f"\n[auto-escalation] tier: {prev_tier} -> {model_manager.current_tier}")
        agent_kwargs["model"] = agent.model
        return True

    def _retry_with_escalation(
        *,
        run_once: Callable[[], Any],
        retryable_exceptions: tuple[type[Exception], ...],
        non_retryable_exceptions: tuple[type[BaseException], ...] = (),
        retryable_filter: Callable[[Exception], bool] | None = None,
    ) -> Any:
        attempted: set[str] = set()
        max_attempts = _escalation_attempts()
        last_error: Exception | None = None

        for _attempt in range(max_attempts):
            try:
                return run_once()
            except non_retryable_exceptions:
                raise
            except retryable_exceptions as exc:
                if retryable_filter is not None and not retryable_filter(exc):
                    raise
                last_error = exc

            if not _maybe_escalate_tier(attempted=attempted):
                break

        if last_error is not None:
            raise last_error
        raise RuntimeError("Execution failed")

    def _build_plan_invocation_state(
        *,
        plan_run_id: str,
        revision_count: int,
    ) -> dict[str, Any]:
        return {
            "swarmee": {
                "mode": "plan",
                "plan_run_id": plan_run_id,
                "plan_revision_count": max(0, int(revision_count or 0)),
                "suppress_reasoning_ui": True,
                "suppress_user_stall_warnings": True,
            }
        }

    def _invoke_planning_branch(
        user_request: str,
        *,
        plan_run_id: str,
        revision_count: int = 0,
    ) -> PendingWorkPlan:
        child_agent, snapshot = create_shared_prefix_child_agent(
            parent_agent=agent,
            kind="plan_revision" if revision_count > 0 else "plan",
            seed_instruction=None,
            callback_handler=callback_handler,
        )
        invocation_state = _build_resolved_invocation_state(
            invocation_state=_build_plan_invocation_state(
                plan_run_id=plan_run_id,
                revision_count=revision_count,
            ),
            runtime_environment=runtime_environment,
            model_manager=model_manager,
            selected_provider=selected_provider,
            settings=settings,
            structured_output_model=WorkPlan,
            session_safety_overrides=session_safety_overrides,
        )
        sw = invocation_state.get("swarmee")
        if isinstance(sw, dict):
            sys_prompt = getattr(child_agent, "system_prompt", None) or ""
            sw["system_prompt_chars"] = len(sys_prompt)
            agent_tools = getattr(child_agent, "tools", None) or []
            sw["tool_count"] = len(agent_tools)
            try:
                from swarmee_river.context.budgeted_summarizing_conversation_manager import estimate_tool_schema_chars

                sw["tool_schema_chars"] = estimate_tool_schema_chars(agent_tools)
            except Exception:
                sw["tool_schema_chars"] = 0

        interrupt_event.clear()
        set_interrupt_event(interrupt_event)
        set_active_interrupt_event(interrupt_event)
        try:
            with interrupt_watcher_from_env(interrupt_event):
                child_agent._swarmee_current_invocation_state = invocation_state
                try:
                    result = invoke_agent(
                        child_agent,
                        user_request,
                        callback_handler=callback_handler,
                        interrupt_event=interrupt_event,
                        invocation_state=invocation_state,
                        system_reminder=snapshot.pending_reminder or None,
                        structured_output_model=WorkPlan,
                        structured_output_prompt=structured_plan_prompt(),
                    )
                finally:
                    child_agent._swarmee_current_invocation_state = None
        finally:
            set_active_interrupt_event(None)
            set_interrupt_event(None)

        plan = getattr(result, "structured_output", None)
        if not isinstance(plan, WorkPlan):
            raise RuntimeError(
                "Plan generation failed to produce a valid WorkPlan. "
                "Try :replan, shorten the request, or change tier."
            )

        artifact_store.write_text(
            kind="plan",
            text=plan.model_dump_json(indent=2),
            suffix="json",
            metadata={"request": user_request, "plan_run_id": plan_run_id},
        )
        return new_pending_work_plan(
            original_request=user_request,
            plan=plan,
            revision_count=revision_count,
            plan_run_id=plan_run_id,
        )

    def _generate_plan(user_request: str, *, plan_context: dict[str, Any] | None = None) -> PendingWorkPlan:
        context = plan_context if isinstance(plan_context, dict) else {}
        original_request = str(context.get("original_request") or user_request).strip() or user_request
        revision_count_raw = context.get("revision_count", 0)
        try:
            revision_count = max(0, int(revision_count_raw or 0))
        except (TypeError, ValueError):
            revision_count = 0
        feedback_history_raw = context.get("feedback_history")
        feedback_history = (
            [dict(item) for item in feedback_history_raw if isinstance(item, dict)]
            if isinstance(feedback_history_raw, list)
            else []
        )
        try:
            pending = _invoke_planning_branch(
                user_request,
                plan_run_id=uuid.uuid4().hex,
                revision_count=revision_count,
            )
        except MaxTokensReachedException as exc:
            raise RuntimeError(
                "Plan generation hit the max output token limit. "
                "Increase models.max_output_tokens, ask for a shorter plan, or change tier."
            ) from exc
        if original_request != pending.original_request or feedback_history:
            pending = new_pending_work_plan(
                original_request=original_request,
                plan=pending.current_plan,
                revision_count=revision_count,
                feedback_history=feedback_history,
                plan_run_id=pending.plan_run_id,
            )
        return pending

    def _swap_agent(messages: Any | None, state: Any | None) -> None:
        nonlocal agent
        agent = create_agent(messages=messages, state=state)
        ctx.agent = agent

    def _build_session_meta() -> dict[str, Any]:
        return _build_session_meta_payload(
            settings=settings,
            selected_provider=selected_provider,
            current_tier=model_manager.current_tier,
            active_sop_names=_list_active_sop_names(),
        )

    def _execute_with_plan(user_request: str, pending_plan: PendingWorkPlan, *, welcome_text_local: str) -> Any:
        nonlocal active_plan_prompt_section
        plan = pending_plan.current_plan
        allowed = {tool_name for tool_name in tools_expected_from_plan(plan) if tool_name != "WorkPlan"}
        allowed.add("plan_progress")
        allowed_tools = sorted(allowed)
        steps = list(plan.steps or [])
        step_descriptions = [str(step.description).strip() for step in steps]
        invocation_state = {
            "swarmee": {
                "mode": "execute",
                "enforce_plan": True,
                "allowed_tools": allowed_tools,
                "plan_step_count": len(step_descriptions),
                "plan_step_descriptions": step_descriptions,
                "plan_run_id": pending_plan.plan_run_id,
            }
        }
        plan_json_payload = plan_json_for_execution(plan)

        lines: list[str] = [
            "Active execution plan:",
            f"Summary: {plan.summary}",
        ]
        if step_descriptions:
            lines.append("Steps:")
            for idx, desc in enumerate(step_descriptions, start=1):
                lines.append(f"{idx}. {desc}")
        else:
            lines.append("Steps: (none)")
        lines.extend(
            [
                "",
                (
                    "You are executing the following plan. Before starting each step, emit a brief status "
                    "message indicating which step you are beginning."
                ),
                "Format: 'Starting step N: <description>'.",
                "After completing a step, emit: 'Completed step N.'",
                (
                    "Use the `plan_progress` tool before each step (status=in_progress) and after each step "
                    "(status=completed)."
                ),
                "The `plan_progress` tool accepts step (1-based) or step_index (0-based), status, and optional note.",
            ]
        )
        active_plan_prompt_section = "\n".join(lines).strip()

        approved_plan_section = (
            "Approved Plan (execute ONLY this plan; if you need changes, ask to :replan):\n" + plan_json_payload
        )
        prompt_cache.queue_one_off(approved_plan_section)
        refresh_system_prompt(welcome_text_local)
        try:

            def _run_execute_once() -> Any:
                result = run_agent(user_request, invocation_state=invocation_state)
                if knowledge_base_id:
                    with contextlib.suppress(Exception):
                        agent.tool.store_in_kb(
                            content=(f"Approved plan for request:\n{user_request}\n\n{plan_json_payload}\n"),
                            title=f"Plan: {user_request[:50]}{'...' if len(user_request) > 50 else ''}",
                            knowledge_base_id=knowledge_base_id,
                            record_direct_tool_call=False,
                        )
                return result

            return _retry_with_escalation(
                run_once=_run_execute_once,
                retryable_exceptions=(Exception,),
                non_retryable_exceptions=(AgentInterruptedError,),
                retryable_filter=lambda exc: _is_escalatable_retry_exception(exc)
                and not bool(
                    isinstance(invocation_state.get("swarmee"), dict)
                    and isinstance(invocation_state["swarmee"].get("invoke_diag"), dict)
                    and invocation_state["swarmee"]["invoke_diag"].get("saw_tool_activity")
                ),
            )
        finally:
            active_plan_prompt_section = None
            refresh_system_prompt(welcome_text_local)

    registry = CommandRegistry()
    register_builtin_commands(registry)

    settings_path = settings_path_for_project or (Path.cwd() / ".swarmee" / "settings.json")

    ctx = CLIContext(
        agent=agent,
        agent_kwargs=agent_kwargs,
        tools_dict=tools_dict,
        model_manager=model_manager,
        knowledge_base_id=knowledge_base_id,
        effective_sop_paths=effective_sop_paths,
        welcome_text="",
        auto_approve=auto_approve,
        selected_provider=selected_provider,
        settings=settings,
        settings_path=settings_path,
        refresh_system_prompt=lambda: refresh_system_prompt(ctx.welcome_text),
        generate_plan=_generate_plan,
        execute_with_plan=lambda req, plan: _execute_with_plan(req, plan, welcome_text_local=ctx.welcome_text),
        render_plan=render_plan_text,
        run_agent=run_agent,
        store_conversation=lambda user_input, response: store_conversation_in_kb(
            agent, user_input, response, knowledge_base_id
        ),
        output=print,
        stop_spinners=lambda: callback_handler(force_stop=True),
        build_session_meta=_build_session_meta,
        swap_agent=_swap_agent,
        refresh_conversation_manager=_refresh_conversation_manager_for_current_tier,
    )
    ctx.active_sop_name = args.sop
    ctx.session_store = SessionStore()

    def _refresh_query_context(*, interactive: bool) -> None:
        nonlocal preflight_prompt_section, project_map_prompt_section
        profile = settings.harness.tier_profiles.get(model_manager.current_tier)
        snapshot = build_context_snapshot(
            artifact_store=artifact_store,
            interactive=interactive,
            runtime=settings.runtime,
            default_preflight_level=profile.preflight_level if profile else None,
        )
        preflight_prompt_section = snapshot.preflight_prompt_section
        project_map_prompt_section = snapshot.project_map_prompt_section
        ctx.refresh_system_prompt()

    ctx.refresh_query_context = lambda *, interactive=True: _refresh_query_context(interactive=interactive)

    def _current_model_info_event() -> dict[str, Any]:
        return _build_model_info_event_payload(
            model_manager=model_manager,
            selected_provider=selected_provider,
            tool_names=sorted(tools_dict.keys()),
        )

    def apply_profile(raw_profile: Any) -> dict[str, Any]:
        nonlocal active_profile_system_prompt_snippets, active_profile_agents
        nonlocal active_orchestrator_agent, auto_delegate_assistive

        normalized = AgentProfile.from_dict(raw_profile)
        requested_tier = (normalized.tier or "").strip().lower()
        requested_provider = normalize_provider_name(normalized.provider)

        if requested_tier:
            tier_provider = _provider_for_tier_name(requested_tier)
            if tier_provider is None:
                raise ValueError(f"Unknown tier: {requested_tier}")
            if requested_provider and requested_provider != tier_provider:
                raise ValueError("set_profile.profile.provider does not match the requested tier's provider")
        elif requested_provider:
            current_provider = _current_provider_name()
            if current_provider and requested_provider != current_provider:
                raise ValueError("set_profile.profile.provider requires a matching profile.tier for runtime apply")

        resolved_sops: list[tuple[str, str]] = []
        for sop_name in normalized.active_sops:
            sop_content = _resolve_sop_content(sop_name)
            if sop_content is None:
                raise ValueError(f"set_profile.active_sops contains unknown SOP: {sop_name}")
            resolved_sops.append((sop_name, sop_content))

        next_context_sources = [dict(item) for item in normalized.context_sources]
        if normalized.knowledge_base_id:
            next_context_sources = [
                item for item in next_context_sources if str(item.get("type", "")).strip().lower() != "kb"
            ]
            next_context_sources.append({"type": "kb", "id": normalized.knowledge_base_id})
        normalized_sources = _normalize_daemon_context_sources(next_context_sources)

        if requested_tier:
            model_manager.set_selection(
                ctx.agent,
                provider_name=requested_provider or _current_provider_name(),
                tier_name=requested_tier,
            )
            agent_kwargs["model"] = ctx.agent.model
            _refresh_conversation_manager_for_current_tier()
            _refresh_query_context(interactive=True)

        active_profile_system_prompt_snippets = list(normalized.system_prompt_snippets)
        active_profile_agents = [dict(item) for item in normalized.agents]
        active_orchestrator_agent = _orchestrator_agent_from_agents(active_profile_agents)
        _refresh_orchestrator_system_prompt()
        auto_delegate_assistive = bool(normalized.auto_delegate_assistive)
        applied_sources = set_user_context_sources(normalized_sources)
        _replace_daemon_sop_overrides(resolved_sops)
        ctx.refresh_system_prompt()

        current_provider = _current_provider_name()
        current_kb = current_knowledge_base_id()
        current_kb = current_kb.strip() if isinstance(current_kb, str) else None
        return {
            "id": normalized.id,
            "name": normalized.name,
            "provider": current_provider,
            "tier": model_manager.current_tier,
            "system_prompt_snippets": list(active_profile_system_prompt_snippets),
            "context_sources": [dict(item) for item in applied_sources],
            "active_sops": _list_active_sop_names(),
            "knowledge_base_id": current_kb or None,
            "agents": [dict(item) for item in active_profile_agents],
            "auto_delegate_assistive": auto_delegate_assistive,
            "team_presets": [dict(item) for item in normalized.team_presets],
        }

    def get_bundles_payload() -> dict[str, Any]:
        nonlocal settings
        settings = load_settings(settings_path)
        return {
            "event": "bundles_catalog",
            "bundles": list_agent_bundles(settings),
        }

    def set_bundle(raw_bundle: Any) -> dict[str, Any]:
        nonlocal settings
        if not isinstance(raw_bundle, dict):
            raise ValueError("set_bundle.bundle must be an object")
        bundle_id = str(raw_bundle.get("id", "")).strip()
        if not bundle_id:
            raise ValueError("set_bundle.bundle.id is required")
        bundle_name = str(raw_bundle.get("name", "")).strip() or bundle_id
        normalized = AgentProfile.from_dict(
            {
                "id": bundle_id,
                "name": bundle_name,
                "provider": raw_bundle.get("provider"),
                "tier": raw_bundle.get("tier"),
                "system_prompt_snippets": raw_bundle.get("system_prompt_snippets") or [],
                "context_sources": raw_bundle.get("context_sources") or [],
                "active_sops": raw_bundle.get("active_sops") or [],
                "knowledge_base_id": raw_bundle.get("knowledge_base_id"),
                "agents": raw_bundle.get("agents") or [],
                "auto_delegate_assistive": raw_bundle.get("auto_delegate_assistive", True),
                "team_presets": raw_bundle.get("team_presets") or [],
            }
        )
        settings = with_upserted_agent_bundle(settings, normalized.to_dict())
        save_settings(settings, settings_path)
        return get_bundles_payload()

    def delete_bundle(bundle_id: str) -> dict[str, Any]:
        nonlocal settings
        token = str(bundle_id or "").strip()
        if not token:
            raise ValueError("delete_bundle.id is required")
        if find_agent_bundle(settings, token) is None:
            raise ValueError(f"Bundle not found: {token}")
        settings = with_deleted_agent_bundle(settings, token)
        save_settings(settings, settings_path)
        return get_bundles_payload()

    def apply_bundle(bundle_id: str) -> dict[str, Any]:
        nonlocal settings
        token = str(bundle_id or "").strip()
        if not token:
            raise ValueError("apply_bundle.id is required")
        settings = load_settings(settings_path)
        bundle = find_agent_bundle(settings, token)
        if bundle is None:
            raise ValueError(f"Bundle not found: {token}")
        return apply_profile(bundle)

    return {
        "selected_provider": selected_provider,
        "provider_notice": provider_notice,
        "model_manager": model_manager,
        "agent_kwargs": agent_kwargs,
        "interrupt_event": interrupt_event,
        "create_agent": create_agent,
        "run_agent": run_agent,
        "render_plan": render_plan_text,
        "generate_plan": _generate_plan,
        "execute_with_plan": _execute_with_plan,
        "ctx": ctx,
        "registry": registry,
        "refresh_conversation_manager": _refresh_conversation_manager_for_current_tier,
        "refresh_query_context": _refresh_query_context,
        "current_model_info_event": _current_model_info_event,
        "set_user_context_sources": set_user_context_sources,
        "set_daemon_sop_override": _set_daemon_sop_override,
        "current_knowledge_base_id": current_knowledge_base_id,
        "compact_context": compact_context,
        "apply_profile": apply_profile,
        "apply_session_safety_overrides": apply_session_safety_overrides,
        "current_session_safety_overrides": current_session_safety_overrides,
        "get_prompt_assets_payload": get_prompt_assets_payload,
        "set_prompt_asset": set_prompt_asset,
        "delete_prompt_asset": delete_prompt_asset,
        "get_bundles_payload": get_bundles_payload,
        "set_bundle": set_bundle,
        "delete_bundle": delete_bundle,
        "apply_bundle": apply_bundle,
    }


def _handle_agent_interrupt(exc: AgentInterruptedError) -> None:
    callback_handler(force_stop=True)
    if not _tui_events_enabled():
        print(f"\n{str(exc)}")


def _run_query_with_optional_plan(
    *,
    query_text: str,
    forced_mode: str | None,
    auto_approve: bool,
    welcome_text: str,
    generate_plan: Callable[[str, dict[str, Any] | None], PendingWorkPlan],
    execute_with_plan: Callable[[str, PendingWorkPlan, str], Any],
    run_agent: Callable[[str], Any],
    classify_intent_fn: Callable[[str], str],
    on_plan: Callable[[PendingWorkPlan], None],
    approved_plan: PendingWorkPlan | None = None,
    plan_context: dict[str, Any] | None = None,
) -> tuple[PendingWorkPlan | None, Any | None, bool]:
    mode = (forced_mode or "").strip().lower()
    if mode == "execute":
        if approved_plan is not None:
            return approved_plan, execute_with_plan(query_text, approved_plan, welcome_text), True
        return None, run_agent(query_text), True

    should_plan = mode == "plan" or classify_intent_fn(query_text) == "work"
    if not should_plan:
        return None, run_agent(query_text), True

    plan = generate_plan(query_text, plan_context=plan_context)
    on_plan(plan)
    if not auto_approve:
        return plan, None, False

    return plan, execute_with_plan(query_text, plan, welcome_text), True


def _build_serve_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarmee serve", description="Run shared runtime broker.")
    parser.add_argument("--port", type=int, default=0, help="Port for runtime broker (0 chooses an open port).")
    parser.add_argument(
        "--state-dir",
        type=str,
        default=None,
        help="Override runtime state directory (defaults to <cwd>/.swarmee).",
    )
    parser.add_argument(
        "--broker-log-path",
        type=str,
        default=None,
        help="Broker log path recorded in discovery payload (defaults to <state_dir>/diagnostics/broker.log).",
    )
    parser.add_argument(
        "--diag-session-events-path",
        type=str,
        default=None,
        help=(
            "Session events JSONL path template (defaults to "
            "<state_dir>/diagnostics/sessions/{session_id}.jsonl)."
        ),
    )
    return parser


def _build_attach_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarmee attach", description="Attach to shared runtime broker.")
    parser.add_argument("--session", type=str, default=None, help="Session ID to attach to.")
    parser.add_argument("--cwd", type=str, default=None, help="Session cwd for attach command.")
    parser.add_argument("--tail", action="store_true", help="Tail events only (no interactive prompt loop).")
    parser.add_argument(
        "--state-dir",
        type=str,
        default=None,
        help="Override runtime state directory (defaults to <cwd>/.swarmee).",
    )
    return parser


def _build_daemon_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarmee daemon", description="Manage shared runtime daemon.")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["start", "stop", "status"],
        default="status",
        help="Daemon action (default: status).",
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=["all"],
        default=None,
        help="Optional stop target (only 'all' is supported).",
    )
    parser.add_argument("--cwd", type=str, default=None, help="Scope cwd for daemon state resolution.")
    parser.add_argument(
        "--state-dir",
        type=str,
        default=None,
        help="Override runtime state directory.",
    )
    return parser


def _runtime_broker_pids() -> list[int]:
    def _collect_pids_from_entries(entries: list[tuple[str, str]]) -> list[int]:
        pids: list[int] = []
        self_pid = os.getpid()
        for pid_text, command in entries:
            if "swarmee_river.swarmee" not in command:
                continue
            if re.search(r"(?:^|\s)serve(?:\s|$)", command) is None:
                continue
            try:
                pid = int(str(pid_text).strip())
            except ValueError:
                continue
            if pid <= 0 or pid == self_pid:
                continue
            pids.append(pid)
        return sorted(set(pids))

    if os.name == "nt":
        windows_commands: list[list[str]] = [
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Select-Object ProcessId,CommandLine | "
                "ConvertTo-Csv -NoTypeInformation",
            ],
            ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
        ]
        for command in windows_commands:
            try:
                output = subprocess.check_output(
                    command,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception:
                continue

            entries: list[tuple[str, str]] = []
            with contextlib.suppress(Exception):
                reader = csv.DictReader(output.splitlines())
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    pid_text = str(
                        row.get("ProcessId")
                        or row.get("PID")
                        or row.get("pid")
                        or row.get("processid")
                        or ""
                    ).strip()
                    command_text = str(
                        row.get("CommandLine")
                        or row.get("commandline")
                        or row.get("Command")
                        or row.get("command")
                        or ""
                    ).strip()
                    if not pid_text or not command_text:
                        continue
                    entries.append((pid_text, command_text))

            pids = _collect_pids_from_entries(entries)
            if pids:
                return pids
        return []

    if os.name != "posix":
        return []
    try:
        output = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,command="],
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []

    entries: list[tuple[str, str]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        entries.append((pid_text, command))
    return _collect_pids_from_entries(entries)


def _stop_all_runtime_brokers(*, timeout_s: float = 6.0) -> tuple[int, int]:
    pids = _runtime_broker_pids()
    if not pids:
        return 0, 0

    import signal as _signal

    term_signal = getattr(_signal, "SIGTERM", None)
    force_signal = getattr(_signal, "SIGKILL", term_signal)

    if term_signal is None:
        return 0, len(pids)

    for pid in pids:
        with contextlib.suppress(Exception):
            os.kill(pid, term_signal)

    deadline = time.monotonic() + max(0.5, float(timeout_s))
    alive: set[int] = set(pids)
    while alive and time.monotonic() < deadline:
        for pid in list(alive):
            if not _is_process_running(pid):
                alive.discard(pid)
        if alive:
            time.sleep(0.1)

    for pid in list(alive):
        with contextlib.suppress(Exception):
            if force_signal is not None:
                os.kill(pid, force_signal)

    force_deadline = time.monotonic() + 1.0
    while alive and time.monotonic() < force_deadline:
        for pid in list(alive):
            if not _is_process_running(pid):
                alive.discard(pid)
        if alive:
            time.sleep(0.05)

    stopped = len(pids) - len(alive)
    failed = len(alive)
    return stopped, failed


def _is_process_running(pid: int) -> bool:
    return is_process_running(pid)


def _print_attach_event(event: dict[str, Any], *, streaming_state: dict[str, bool]) -> None:
    etype = str(event.get("event", "")).strip().lower()
    text = str(event.get("text", event.get("message", "")))

    if etype in {"text_delta", "message_delta", "output_text_delta", "delta"}:
        chunk = text
        if chunk:
            print(chunk, end="", flush=True)
            streaming_state["open"] = True
        return

    if etype in {"text_complete", "message_complete", "output_text_complete", "complete"}:
        if streaming_state["open"]:
            print()
            streaming_state["open"] = False
        return

    if etype == "turn_complete":
        if streaming_state["open"]:
            print()
            streaming_state["open"] = False
        status = str(event.get("exit_status", "ok")).strip()
        print(f"[turn] complete ({status})")
        return

    if etype == "consent_prompt":
        if streaming_state["open"]:
            print()
            streaming_state["open"] = False
        context = str(event.get("context", "")).strip()
        print("[consent] prompt received.")
        if context:
            print(context)
        print("Use /consent <y|n|a|v> to respond.")
        return

    if etype in {"warning", "error"}:
        if streaming_state["open"]:
            print()
            streaming_state["open"] = False
        prefix = "error" if etype == "error" else "warn"
        line = text.strip() or json.dumps(event, ensure_ascii=False)
        print(f"[{prefix}] {line}")
        return

    if etype in {"ready", "model_info", "attached", "session_available", "session_restored", "replay_complete"}:
        if streaming_state["open"]:
            print()
            streaming_state["open"] = False
        print(f"[{etype}] {json.dumps(event, ensure_ascii=False)}")
        return

    if text.strip():
        if streaming_state["open"]:
            print()
            streaming_state["open"] = False
        print(f"[{etype or 'event'}] {text.strip()}")
        return

    if streaming_state["open"]:
        print()
        streaming_state["open"] = False
    print(json.dumps(event, ensure_ascii=False))


def _run_serve_command(raw_args: list[str]) -> int:
    from swarmee_river.state_paths import set_state_dir_override
    from swarmee_river.settings import load_settings

    parser = _build_serve_command_parser()
    args = parser.parse_args(raw_args)
    serve_cwd = Path.cwd().resolve()
    settings = load_settings(serve_cwd / ".swarmee" / "settings.json")
    override = (
        args.state_dir.strip()
        if isinstance(args.state_dir, str) and args.state_dir.strip()
        else settings.runtime.state_dir
    )
    set_state_dir_override(override, cwd=serve_cwd)

    async def _serve() -> None:
        server = RuntimeServiceServer(
            port=int(args.port),
            broker_log_path=args.broker_log_path,
            session_events_path_template=args.diag_session_events_path,
        )
        await server.start()
        print(f"[runtime] listening on {server.host}:{server.port}")
        print(f"[runtime] discovery file: {server.runtime_file}")
        print("[runtime] press Ctrl+C to stop.")

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _request_stop() -> None:
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, _request_stop)

        try:
            done, pending = await asyncio.wait(
                [asyncio.ensure_future(stop_event.wait()), asyncio.ensure_future(server._stopped.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                with contextlib.suppress(Exception):
                    task.result()
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            await server.stop()

    try:
        asyncio.run(_serve())
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: failed to start runtime broker: {exc}")
        return 1


def _run_attach_command(raw_args: list[str]) -> int:
    from swarmee_river.state_paths import set_state_dir_override
    from swarmee_river.settings import load_settings

    parser = _build_attach_command_parser()
    args = parser.parse_args(raw_args)

    attach_cwd = Path(args.cwd).expanduser() if isinstance(args.cwd, str) and args.cwd.strip() else Path.cwd()
    attach_cwd = attach_cwd.resolve()
    if not attach_cwd.exists() or not attach_cwd.is_dir():
        print(f"Error: attach cwd does not exist or is not a directory: {attach_cwd}")
        return 1

    settings = load_settings(attach_cwd / ".swarmee" / "settings.json")
    override = (
        args.state_dir.strip()
        if isinstance(args.state_dir, str) and args.state_dir.strip()
        else settings.runtime.state_dir
    )
    set_state_dir_override(override, cwd=attach_cwd)

    session_id = (
        (str(args.session).strip() if isinstance(args.session, str) else "")
        or default_session_id_for_cwd(attach_cwd)
    )

    try:
        discovery = ensure_runtime_broker(cwd=attach_cwd)
    except Exception as exc:
        print(f"Error: failed to start runtime broker: {exc}")
        return 1

    try:
        client = RuntimeServiceClient.from_discovery_file(discovery)
        client.connect()
    except Exception as exc:
        print(f"Error: failed to connect to runtime broker: {exc}")
        return 1

    try:
        hello = client.hello(client_name="swarmee-attach", surface="cli")
        if hello is None:
            print("Error: runtime broker closed connection during hello.")
            return 1
        if str(hello.get("event", "")).strip().lower() == "error":
            print(f"Error: {hello.get('message', hello)}")
            return 1

        attach = client.attach(session_id=session_id, cwd=str(attach_cwd))
        if attach is None:
            print("Error: runtime broker closed connection during attach.")
            return 1
        if str(attach.get("event", "")).strip().lower() == "error":
            print(f"Error: {attach.get('message', attach)}")
            return 1

        print(f"[runtime] attached to session {session_id} (cwd={attach_cwd})")
        streaming_state: dict[str, bool] = {"open": False}

        stop_reader = threading.Event()

        def _reader_loop() -> None:
            while not stop_reader.is_set():
                try:
                    event = client.read_event()
                except Exception as exc:
                    if not stop_reader.is_set():
                        if streaming_state["open"]:
                            print()
                            streaming_state["open"] = False
                        print(f"[runtime] read error: {exc}")
                    break
                if event is None:
                    if not stop_reader.is_set():
                        if streaming_state["open"]:
                            print()
                            streaming_state["open"] = False
                        print("[runtime] connection closed.")
                    break
                _print_attach_event(event, streaming_state=streaming_state)

        reader_thread = threading.Thread(target=_reader_loop, daemon=True, name="swarmee-runtime-attach-reader")
        reader_thread.start()

        if args.tail:
            print("[runtime] tail mode active. Ctrl+C to stop.")
            try:
                while reader_thread.is_alive():
                    reader_thread.join(timeout=0.2)
            except KeyboardInterrupt:
                pass
            return 0

        print("Enter prompts to send `query`. Commands: /consent <y|n|a|v>, /stop, /shutdown, /exit")
        while True:
            try:
                line = input("~ ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            lowered = line.lower()
            if lowered in {"exit", "quit", "/exit", ":exit"}:
                break
            if lowered in {"/stop", ":stop"}:
                client.send_command({"cmd": "interrupt"})
                continue
            if lowered in {"/shutdown", ":shutdown", "/daemon stop"}:
                client.send_command({"cmd": "shutdown_service"})
                break
            if lowered.startswith("/consent "):
                choice = lowered.split(maxsplit=1)[1].strip()
                if choice not in {"y", "n", "a", "v"}:
                    print("Usage: /consent <y|n|a|v>")
                    continue
                client.send_command({"cmd": "consent_response", "choice": choice})
                continue
            client.send_command({"cmd": "query", "text": line})
        return 0
    except Exception as exc:
        print(f"Error: attach failed: {exc}")
        return 1
    finally:
        stop_reader = locals().get("stop_reader")
        reader_thread = locals().get("reader_thread")
        if isinstance(stop_reader, threading.Event):
            stop_reader.set()
        client.close()
        if isinstance(reader_thread, threading.Thread) and reader_thread.is_alive():
            reader_thread.join(timeout=1.0)


def _run_daemon_command(raw_args: list[str]) -> int:
    from swarmee_river.state_paths import set_state_dir_override
    from swarmee_river.settings import load_settings

    parser = _build_daemon_command_parser()
    args = parser.parse_args(raw_args)

    daemon_cwd = Path(args.cwd).expanduser() if isinstance(args.cwd, str) and args.cwd.strip() else Path.cwd()
    daemon_cwd = daemon_cwd.resolve()
    if not daemon_cwd.exists() or not daemon_cwd.is_dir():
        print(f"Error: cwd does not exist or is not a directory: {daemon_cwd}")
        return 1

    settings = load_settings(daemon_cwd / ".swarmee" / "settings.json")
    override = (
        args.state_dir.strip()
        if isinstance(args.state_dir, str) and args.state_dir.strip()
        else settings.runtime.state_dir
    )
    set_state_dir_override(override, cwd=daemon_cwd)

    action = str(args.action or "status").strip().lower()
    target = str(args.target or "").strip().lower()
    if action == "start":
        if target:
            print("Error: 'all' target is only valid with 'stop'.")
            return 1
        try:
            discovery = ensure_runtime_broker(cwd=daemon_cwd)
            print(f"[daemon] running (discovery: {discovery})")
            return 0
        except Exception as exc:
            print(f"Error: failed to start runtime broker: {exc}")
            return 1

    if action == "stop":
        if target == "all":
            stopped_count, failed_count = _stop_all_runtime_brokers()
            if stopped_count == 0 and failed_count == 0:
                print("[daemon] no runtime brokers found.")
                return 0
            if failed_count:
                print(f"[daemon] stopped {stopped_count} broker(s), failed to stop {failed_count}.")
                return 1
            print(f"[daemon] stopped {stopped_count} broker(s).")
            return 0
        stopped = shutdown_runtime_broker(cwd=daemon_cwd)
        if stopped:
            print("[daemon] stopped.")
            return 0
        print("[daemon] not running.")
        return 0
    if target:
        print("Error: 'all' target is only valid with 'stop'.")
        return 1

    discovery = runtime_discovery_path(cwd=daemon_cwd)
    if not discovery.exists():
        print("[daemon] not running.")
        return 0
    client: RuntimeServiceClient | None = None
    try:
        client = RuntimeServiceClient.from_discovery_file(discovery, timeout_s=1.0)
        client.connect()
        hello = client.hello(client_name="swarmee-daemon-status", surface="cli")
        running = isinstance(hello, dict) and str(hello.get("event", "")).strip().lower() == "hello_ack"
    except Exception:
        running = False
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()
    if running:
        print(f"[daemon] running (discovery: {discovery})")
    else:
        print("[daemon] not running.")
    return 0


def _run_settings_command(raw_args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="swarmee settings", description="Project settings maintenance commands.")
    sub = parser.add_subparsers(dest="settings_cmd")

    migrate_parser = sub.add_parser(
        "migrate",
        help="Migrate legacy `.swarmee/settings.json` `env.*` overrides into structured settings fields.",
    )
    migrate_parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Settings path (default: ./.swarmee/settings.json)",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print migration output without rewriting the file.",
    )

    args = parser.parse_args(raw_args or ["migrate"])
    cmd = str(getattr(args, "settings_cmd", "") or "migrate").strip().lower()
    if cmd != "migrate":
        parser.print_help()
        return 1

    from swarmee_river.settings import migrate_legacy_env_overrides, normalize_project_env_overrides

    raw_path = str(getattr(args, "path", "") or "").strip()
    path = Path(raw_path).expanduser().resolve() if raw_path else (Path.cwd() / ".swarmee" / "settings.json")
    if not path.exists() or not path.is_file():
        print(f"Error: settings file not found: {path}")
        return 1

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error: failed to read settings JSON: {exc}")
        return 1
    if not isinstance(payload, dict):
        print("Error: settings JSON must be an object.")
        return 1

    raw_env = payload.get("env")
    if not isinstance(raw_env, dict) or not raw_env:
        print("No legacy `env` section found; nothing to migrate.")
        return 0

    internal_env = normalize_project_env_overrides(raw_env)
    migrated_payload, migrated_map, dropped = migrate_legacy_env_overrides(payload)
    if internal_env:
        migrated_payload["env"] = internal_env
    else:
        migrated_payload.pop("env", None)

    migrated_keys = sorted(migrated_map.keys())
    dropped_keys = sorted({str(k).strip() for k in dropped if str(k).strip()})

    print("# Settings Migration")
    print(f"- file: {path}")
    print(f"- migrated: {len(migrated_keys)} key(s)")
    print(f"- dropped: {len(dropped_keys)} key(s)")
    if migrated_keys:
        print("\n## Migrated")
        for k in migrated_keys:
            print(f"- {k} -> {migrated_map.get(k)}")
    if dropped_keys:
        print("\n## Dropped")
        for k in dropped_keys:
            if k in {"OPENAI_API_KEY", "SWARMEE_GITHUB_COPILOT_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"}:
                print(f"- {k} (secret; set via OS environment or provider auth store, not settings.json)")
            else:
                print(f"- {k} (no longer supported)")

    if bool(getattr(args, "dry_run", False)):
        print("\n(dry-run; file not rewritten)")
        return 0

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(migrated_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"Error: failed to write migrated settings: {exc}")
        return 1

    print("\nWrote migrated settings.")
    return 0


def _build_session_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarmee session", description="Session graph and branching commands.")
    sub = parser.add_subparsers(dest="session_cmd")

    list_parser = sub.add_parser("list", help="List recent sessions.")
    list_parser.add_argument("--limit", type=int, default=20, help="Max sessions to print.")

    index_parser = sub.add_parser("index", help="Build and persist session graph index.")
    index_parser.add_argument("--session", type=str, default=None, help="Session ID.")

    export_parser = sub.add_parser("export", help="Export session graph data.")
    export_parser.add_argument("--session", type=str, default=None, help="Session ID.")
    export_parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Export format.",
    )
    export_parser.add_argument("--out", type=str, default=None, help="Output path. Defaults to stdout.")

    branch_parser = sub.add_parser("branch", help="Create a branched session from a turn.")
    branch_parser.add_argument("--from-session", required=True, help="Source session ID.")
    branch_parser.add_argument("--turn", required=True, type=int, help="1-based user turn index.")
    branch_parser.add_argument("--new-session", default=None, help="Optional target session ID.")
    return parser


def _resolve_session_id(raw_session_id: str | None) -> str | None:
    session_id = str(raw_session_id or "").strip()
    return session_id or None


def _render_session_list(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "No sessions found."
    lines = ["session_id\tupdated_at\tturn_count\tcwd"]
    for entry in entries:
        session_id = str(entry.get("id", "")).strip()
        updated_at = str(entry.get("updated_at") or entry.get("created_at") or "").strip()
        turn_count = entry.get("turn_count")
        turn_text = str(turn_count) if isinstance(turn_count, int) else ""
        cwd = str(entry.get("cwd", "")).strip()
        lines.append(f"{session_id}\t{updated_at}\t{turn_text}\t{cwd}")
    return "\n".join(lines)


def _summarize_plan(value: Any) -> str | None:
    if isinstance(value, dict):
        summary = str(value.get("summary", "")).strip()
        if summary:
            return summary
        steps = value.get("steps")
        if isinstance(steps, list) and steps:
            first = steps[0]
            if isinstance(first, dict):
                desc = str(first.get("description", "")).strip()
                if desc:
                    return desc
        return None
    if isinstance(value, str):
        summary = value.strip()
        return summary or None
    return None


def _render_session_export_markdown(
    *,
    session_id: str,
    index: dict[str, Any],
    last_plan_summary: str | None,
) -> str:
    stats = index.get("stats") if isinstance(index.get("stats"), dict) else {}
    tools = index.get("tools") if isinstance(index.get("tools"), dict) else {}
    tool_counts = tools.get("counts") if isinstance(tools.get("counts"), dict) else {}
    turns = index.get("turns") if isinstance(index.get("turns"), list) else []
    events = index.get("events") if isinstance(index.get("events"), list) else []

    lines: list[str] = [
        "# Session Export",
        "",
        f"- Session ID: `{session_id}`",
        f"- Generated: {str(index.get('generated_at', '')).strip() or '(unknown)'}",
        f"- Turns: {int(stats.get('turns', 0)) if isinstance(stats.get('turns'), int) else 0}",
        f"- Tool calls: {int(stats.get('tools', 0)) if isinstance(stats.get('tools'), int) else 0}",
        f"- Errors: {int(stats.get('errors', 0)) if isinstance(stats.get('errors'), int) else 0}",
        "",
        "## Tool Summary",
    ]

    if isinstance(tool_counts, dict) and tool_counts:
        for tool_name in sorted(tool_counts):
            count = tool_counts.get(tool_name)
            lines.append(f"- `{tool_name}`: {count}")
    else:
        lines.append("(no tool calls)")

    lines.append("")
    lines.append("## Errors")
    error_events: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        error = event.get("error")
        if isinstance(error, str) and error.strip():
            tool_name = str(event.get("tool", "")).strip()
            ts = str(event.get("ts", "")).strip()
            prefix = f"{ts} " if ts else ""
            if tool_name:
                error_events.append(f"- {prefix}`{tool_name}`: {error.strip()}")
            else:
                error_events.append(f"- {prefix}{error.strip()}")
        elif event.get("event") == "after_tool_call" and event.get("success") is False:
            tool_name = str(event.get("tool", "")).strip() or "unknown_tool"
            ts = str(event.get("ts", "")).strip()
            prefix = f"{ts} " if ts else ""
            error_events.append(f"- {prefix}`{tool_name}`: tool call failed")
    if error_events:
        lines.extend(error_events)
    else:
        lines.append("(no errors)")

    lines.append("")
    lines.append("## Last Plan")
    lines.append(last_plan_summary if isinstance(last_plan_summary, str) and last_plan_summary.strip() else "(none)")
    lines.append("")
    lines.append("## Turns")

    turn_rows = [item for item in turns if isinstance(item, dict)]
    if not turn_rows:
        lines.append("(no turns)")
    else:
        for turn in turn_rows:
            turn_id = turn.get("turn_id")
            user_text = str(turn.get("user_text", "")).strip()
            assistant_text = str(turn.get("assistant_text", "")).strip()
            lines.extend(
                [
                    f"### Turn {turn_id}",
                    "",
                    "User:",
                    user_text or "(empty)",
                    "",
                    "Assistant:",
                    assistant_text or "(empty)",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def _is_counted_user_turn_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    role = str(message.get("role", "")).strip().lower()
    if role != "user":
        return False
    return bool(_extract_text_from_message_for_replay(message).strip())


def _slice_messages_through_turn(messages: list[Any], turn_number: int) -> tuple[list[Any], int]:
    if turn_number <= 0:
        raise ValueError("--turn must be >= 1")
    total_turns = 0
    next_turn_start: int | None = None
    for idx, message in enumerate(messages):
        if not _is_counted_user_turn_message(message):
            continue
        total_turns += 1
        if total_turns == turn_number + 1:
            next_turn_start = idx
            break
    if total_turns < turn_number:
        raise ValueError(f"Requested turn {turn_number} but session has only {total_turns} turns")
    end_idx = next_turn_start if next_turn_start is not None else len(messages)
    return list(messages[:end_idx]), total_turns


def _run_session_command(raw_args: list[str]) -> int:
    parser = _build_session_command_parser()
    args = parser.parse_args(raw_args or ["list"])
    command = str(args.session_cmd or "list").strip().lower()

    store = SessionStore()
    try:
        if command == "list":
            limit = max(1, int(args.limit))
            print(_render_session_list(store.list(limit=limit)))
            return 0

        if command == "index":
            session_id = _resolve_session_id(args.session)
            if not session_id:
                print("Error: --session is required")
                return 1
            index = build_session_graph_index(session_id, cwd=Path.cwd())
            path = write_session_graph_index(session_id, index)
            stats = index.get("stats") if isinstance(index.get("stats"), dict) else {}
            turns = int(stats.get("turns", 0)) if isinstance(stats.get("turns"), int) else 0
            tools = int(stats.get("tools", 0)) if isinstance(stats.get("tools"), int) else 0
            errors = int(stats.get("errors", 0)) if isinstance(stats.get("errors"), int) else 0
            print(f"Indexed session: {session_id}")
            print(f"Path: {path}")
            print(f"Turns: {turns}")
            print(f"Tool calls: {tools}")
            print(f"Errors: {errors}")
            return 0

        if command == "export":
            session_id = _resolve_session_id(args.session)
            if not session_id:
                print("Error: --session is required")
                return 1
            index = build_session_graph_index(session_id, cwd=Path.cwd())
            _, _, _, last_plan = store.load(session_id)
            last_plan_summary = _summarize_plan(last_plan)

            output_format = str(args.format).strip().lower()
            if output_format == "json":
                text = json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            else:
                text = _render_session_export_markdown(
                    session_id=session_id,
                    index=index,
                    last_plan_summary=last_plan_summary,
                )

            if isinstance(args.out, str) and args.out.strip():
                out_path = Path(args.out).expanduser()
                if not out_path.is_absolute():
                    out_path = (Path.cwd() / out_path).resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(text, encoding="utf-8")
                print(f"Wrote export: {out_path}")
                return 0

            print(text, end="")
            return 0

        if command == "branch":
            from_session_id = str(args.from_session).strip()
            turn_number = int(args.turn)
            requested_new_session = str(args.new_session).strip() if isinstance(args.new_session, str) else ""

            source_meta = store.read_meta(from_session_id)
            source_messages = store.load_messages(from_session_id, max_messages=1000000)
            truncated_messages, total_turns = _slice_messages_through_turn(source_messages, turn_number)

            copied_meta = dict(source_meta)
            copied_meta.pop("id", None)
            now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            copied_meta["created_at"] = now
            copied_meta["updated_at"] = now
            copied_meta["branched_from_session_id"] = from_session_id
            copied_meta["branched_from_turn"] = turn_number
            copied_meta["turn_count"] = turn_number
            copied_meta["message_count"] = len(truncated_messages)

            new_session_id = store.create(
                meta=copied_meta,
                session_id=requested_new_session if requested_new_session else None,
            )
            store.save_messages(
                new_session_id,
                truncated_messages,
                max_messages=max(1, len(truncated_messages)),
            )
            print(f"Branched session: {new_session_id}")
            print(f"From: {from_session_id}")
            print(f"Turn: {turn_number}/{total_turns}")
            print(f"Messages: {len(truncated_messages)}")
            return 0

        print("Error: unknown session command")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def main() -> None:
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Swarmee - An enterprise analytics + coding assistant",
        epilog=_ROOT_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="*", help="Query to process")
    parser.add_argument(
        "--kb",
        "--knowledge-base",
        dest="knowledge_base_id",
        help="Knowledge base ID to use for retrievals",
    )
    parser.add_argument(
        "--model-provider",
        type=model_utils.load_path,
        default=None,
        help="Model provider to use for inference",
    )
    parser.add_argument(
        "--model-config",
        type=model_utils.load_config,
        default="{}",
        help="Model config as JSON string or path",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Max output tokens for the model (overrides `models.max_output_tokens`).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Conversation history window size (overrides `context.window_size`).",
    )
    parser.add_argument(
        "--context-per-turn",
        type=int,
        default=None,
        help="Max history items to drop per truncation pass (overrides `context.per_turn`).",
    )
    parser.add_argument(
        "--context-manager",
        type=str,
        default=None,
        choices=["summarize", "sliding", "none"],
        help="Context management strategy (overrides `context.manager`).",
    )
    parser.add_argument(
        "--context-budget-tokens",
        type=int,
        default=None,
        help="Approximate max prompt tokens before summarization (overrides `context.max_prompt_tokens`).",
    )
    parser.add_argument(
        "--include-welcome-in-prompt",
        action="store_true",
        help="Append the welcome text into the system prompt (not recommended for large welcomes)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-approve plans and tool consent prompts (useful for automation).",
    )
    parser.add_argument(
        "--tui-daemon",
        action="store_true",
        help="Run as long-lived JSONL daemon for the TUI.",
    )
    parser.add_argument(
        "--sop",
        type=str,
        default=None,
        help="Activate an SOP by name (requires `strands-agents-sops` or local `./sops/*.sop.md`)",
    )
    parser.add_argument(
        "--sop-paths",
        type=str,
        default=None,
        help="OS-separated directories containing `*.sop.md` files.",
    )
    args, extra_args = parser.parse_known_args()
    configure_callback_handler_mode(tui_events=True if args.tui_daemon else None)

    # Load .env early for local dev credentials/config.
    load_env_file()

    command = args.query[0].strip().lower() if args.query else ""
    sub = args.query[1:] if len(args.query) > 1 else []

    if command == "serve":
        raise SystemExit(_run_serve_command([*sub, *extra_args]))
    if command == "attach":
        raise SystemExit(_run_attach_command([*sub, *extra_args]))
    if command == "daemon":
        raise SystemExit(_run_daemon_command([*sub, *extra_args]))
    if command == "broker":
        raise SystemExit(_run_daemon_command([*sub, *extra_args]))
    if command == "session":
        raise SystemExit(_run_session_command([*sub, *extra_args]))
    if command == "settings":
        raise SystemExit(_run_settings_command([*sub, *extra_args]))
    if command == "diagnostics":
        if extra_args:
            sub = [*sub, *extra_args]
        print(render_diagnostics_command_for_surface(args=sub, cwd=Path.cwd(), surface="cli"))
        return
    if extra_args:
        parser.error(f"unrecognized arguments: {' '.join(extra_args)}")

    daemon_session_id: str | None = None
    if args.tui_daemon:
        # Internal runtime broker uses SWARMEE_SESSION_ID to bind session state.
        daemon_session_id = (os.getenv("SWARMEE_SESSION_ID") or "").strip() or uuid.uuid4().hex
        os.environ["SWARMEE_SESSION_ID"] = daemon_session_id

    settings_path_for_project = Path.cwd() / ".swarmee" / "settings.json"
    if args.tui_daemon:
        apply_project_env_overrides(settings_path_for_project, overwrite=True)
    settings = load_settings(settings_path_for_project)
    # Apply state-dir override once at startup.
    from swarmee_river.state_paths import set_state_dir_override

    set_state_dir_override(settings.runtime.state_dir, cwd=Path.cwd())

    # Internal bridging: Bedrock profile selection lives in structured settings, but
    # external AWS SDKs honor AWS_PROFILE.
    bedrock = settings.models.providers.get("bedrock")
    bedrock_extra = dict(getattr(bedrock, "extra", {}) or {})
    aws_profile = str(bedrock_extra.get("aws_profile") or "").strip()
    resolved_aws_profile, _has_aws_creds, _aws_source, _aws_warning = resolve_bedrock_runtime_profile(aws_profile)
    if resolved_aws_profile:
        os.environ["AWS_PROFILE"] = resolved_aws_profile

    # Apply CLI overrides to the in-memory settings object (no env mutation).
    from dataclasses import replace

    if args.max_output_tokens is not None:
        settings = replace(settings, models=replace(settings.models, max_output_tokens=int(args.max_output_tokens)))
    if args.context_manager:
        settings = replace(
            settings,
            context=replace(settings.context, manager=str(args.context_manager).strip().lower()),
        )
    if args.context_budget_tokens:
        settings = replace(
            settings,
            context=replace(settings.context, max_prompt_tokens=max(1, int(args.context_budget_tokens))),
        )

    # Resolve KB from CLI first, then project settings.
    knowledge_base_id = args.knowledge_base_id or settings.runtime.knowledge_base_id
    auto_approve = bool(args.yes) or bool(settings.runtime.auto_approve)
    # Explicit opt-in to launch the full-screen Textual UI.
    if command == "tui":
        from swarmee_river.tui.app import run_tui

        raise SystemExit(run_tui())

    # Pack management is CLI-first: `swarmee pack ...`.
    if command == "pack":
        output_text = handle_pack_command(
            args=sub,
            settings=settings,
            settings_path=Path.cwd() / ".swarmee" / "settings.json",
        )
        print(output_text)
        return

    handled_auth, auth_text = _handle_auth_cli_command(command, sub)
    if handled_auth:
        if auth_text:
            print(auth_text)
        return

    # Lightweight diagnostics commands that do not invoke the model.
    if command in ({"config"} | _DIAGNOSTIC_COMMANDS):
        cmd = command

        if cmd == "config":
            selected_provider, provider_notice, model_manager = _resolve_provider_and_model_manager(
                args=args,
                settings=settings,
            )
            if provider_notice:
                print(f"[provider] {provider_notice}")

            print(
                render_config_command_for_surface(
                    args=sub,
                    cwd=Path.cwd(),
                    settings_path=settings_path_for_project,
                    settings=settings,
                    selected_provider=selected_provider,
                    model_manager=model_manager,
                    knowledge_base_id=knowledge_base_id,
                    effective_sop_paths=args.sop_paths,
                    auto_approve=auto_approve,
                    surface="cli",
                )
            )
            return

        if cmd in _DIAGNOSTIC_COMMANDS:
            print(
                render_diagnostic_command_for_surface(
                    cmd=cmd,
                    args=sub,
                    cwd=Path.cwd(),
                    surface="cli",
                    current_session_id=(os.getenv("SWARMEE_SESSION_ID") or "").strip() or None,
                )
            )
            return

    daemon_consent_event = threading.Event()
    daemon_consent_lock = threading.Lock()
    daemon_consent_state: dict[str, str] = {"response": ""}

    def _set_daemon_consent_response(choice: str) -> None:
        with daemon_consent_lock:
            daemon_consent_state["response"] = choice

    def _consume_daemon_consent_response() -> str:
        with daemon_consent_lock:
            response = daemon_consent_state.get("response", "")
            daemon_consent_state["response"] = ""
            return response

    def _daemon_consent_prompt(text: str, payload: dict[str, Any] | None = None) -> str:
        event_payload = {
            "event": "consent_prompt",
            "context": text,
            "options": ["y", "n", "a", "v"],
        }
        if isinstance(payload, dict):
            event_payload.update(payload)
        _write_stdout_jsonl(event_payload)
        _set_daemon_consent_response("")
        daemon_consent_event.clear()
        daemon_consent_event.wait()
        return _consume_daemon_consent_response().strip()

    try:
        runtime = _build_agent_runtime(
            args,
            settings,
            auto_approve,
            _daemon_consent_prompt if args.tui_daemon else None,
            knowledge_base_id=knowledge_base_id,
            settings_path_for_project=settings_path_for_project,
        )
    except Exception as exc:
        if not args.tui_daemon:
            raise
        startup_text = str(exc).strip() or "Daemon startup failed"
        daemon_session_id = (os.getenv("SWARMEE_SESSION_ID") or "").strip() or None
        startup_event = _build_tui_error_event(startup_text, category_hint=ERROR_CATEGORY_FATAL)
        startup_event["phase"] = "startup"
        _write_stdout_jsonl(startup_event)
        _write_stdout_jsonl({"event": "warning", "text": f"Daemon startup failed: {startup_text}"})
        with contextlib.suppress(Exception):
            append_session_event(session_id=daemon_session_id, event=startup_event, cwd=Path.cwd())
        with contextlib.suppress(Exception):
            append_session_issue(
                session_id=daemon_session_id,
                line=f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())} startup failed: {startup_text}",
                cwd=Path.cwd(),
            )
        raise SystemExit(1) from exc

    selected_provider = runtime["selected_provider"]
    selection_is_auto = not bool(args.model_provider is not None or str(settings.models.provider or "").strip())
    provider_notice = runtime["provider_notice"]
    if provider_notice:
        if args.tui_daemon:
            _write_stdout_jsonl({"event": "warning", "text": f"[provider] {provider_notice}"})
        else:
            print(f"[provider] {provider_notice}")

    model_manager = runtime["model_manager"]
    agent_kwargs = runtime["agent_kwargs"]
    interrupt_event = runtime["interrupt_event"]
    run_agent = runtime["run_agent"]
    _render_plan = runtime["render_plan"]
    _generate_plan = runtime["generate_plan"]
    _execute_with_plan = runtime["execute_with_plan"]
    ctx = runtime["ctx"]
    registry = runtime["registry"]
    _refresh_conversation_manager = runtime["refresh_conversation_manager"]
    _refresh_query_context = runtime["refresh_query_context"]
    _current_model_info_event = runtime["current_model_info_event"]
    _set_user_context_sources = runtime["set_user_context_sources"]
    _set_daemon_sop_override = runtime["set_daemon_sop_override"]
    _current_knowledge_base_id = runtime["current_knowledge_base_id"]
    _compact_context = runtime["compact_context"]
    _apply_profile = runtime["apply_profile"]
    _apply_session_safety_overrides = runtime["apply_session_safety_overrides"]
    _current_session_safety_overrides = runtime["current_session_safety_overrides"]
    _get_prompt_assets_payload = runtime["get_prompt_assets_payload"]
    _set_prompt_asset = runtime["set_prompt_asset"]
    _delete_prompt_asset = runtime["delete_prompt_asset"]
    _get_bundles_payload = runtime["get_bundles_payload"]
    _set_bundle = runtime["set_bundle"]
    _delete_bundle = runtime["delete_bundle"]
    _apply_bundle = runtime["apply_bundle"]

    def _active_runtime_provider_name() -> str:
        provider = normalize_provider_name(getattr(model_manager, "current_provider", "") or selected_provider)
        normalized = normalize_provider_name(provider)
        if normalized:
            return normalized
        return normalize_provider_name(selected_provider)

    def _best_effort_retrieve(query_text: str, *, warn_on_error: bool) -> None:
        kb_for_turn = _current_knowledge_base_id()
        if not kb_for_turn:
            return
        try:
            ctx.agent.tool.retrieve(text=query_text, knowledgeBaseId=kb_for_turn)
        except Exception as e:
            if warn_on_error and not _tui_events_enabled():
                print(f"[warn] retrieve failed: {e}")

    def _emit_plan_event(plan: PendingWorkPlan, *, write_jsonl: bool, echo_console: bool) -> str:
        rendered = _render_plan(plan)
        event = {
            "event": "plan",
            "plan_json": plan.current_plan.model_dump(),
            "pending_plan": plan.model_dump(),
            "plan_run_id": plan.plan_run_id,
            "original_request": plan.original_request,
            "revision_count": plan.revision_count,
            "rendered": rendered,
        }
        if write_jsonl:
            _write_stdout_jsonl(event)
        else:
            _emit_tui_event(event)
        if echo_console and not _tui_events_enabled():
            print(rendered)
        return rendered

    def _run_tui_daemon() -> None:
        resolved_session_id = daemon_session_id or (os.getenv("SWARMEE_SESSION_ID") or "").strip() or uuid.uuid4().hex
        active_session_id = resolved_session_id
        persist_lock = threading.Lock()

        session_store = ctx.session_store if isinstance(ctx.session_store, SessionStore) else SessionStore()
        refresh_lock = threading.Lock()
        refresh_state = {"running": False, "generation": 0, "refreshed_generation": -1}

        # Hard-coded: non-secret env tuning knobs are no longer supported.
        refresh_timeout_s = 6.0

        def _mark_query_context_stale(*, reason: str) -> None:
            del reason
            with refresh_lock:
                refresh_state["generation"] += 1

        def _refresh_query_context_guarded(
            *,
            interactive: bool,
            phase: str,
            timeout_s: float | None = None,
        ) -> bool:
            timeout = refresh_timeout_s if timeout_s is None else max(0.1, float(timeout_s))
            with refresh_lock:
                if refresh_state["refreshed_generation"] == refresh_state["generation"]:
                    return True
                if refresh_state["running"]:
                    _write_stdout_jsonl(
                        {
                            "event": "warning",
                            "text": (
                                "Context refresh already in progress during "
                                f"{phase}; continuing without waiting."
                            ),
                        }
                    )
                    return False
                refresh_state["running"] = True

            done = threading.Event()
            error_holder: dict[str, Exception] = {}

            def _worker() -> None:
                try:
                    _refresh_query_context(interactive=interactive)
                except Exception as exc:
                    error_holder["error"] = exc
                finally:
                    with refresh_lock:
                        refresh_state["running"] = False
                    done.set()

            threading.Thread(
                target=_worker,
                daemon=True,
                name="swarmee-context-refresh",
            ).start()

            if not done.wait(timeout):
                _write_stdout_jsonl(
                    {
                        "event": "warning",
                        "text": (
                            f"Context refresh timed out after {timeout:.1f}s during {phase}; "
                            "continuing without blocking."
                        ),
                    }
                )
                return False

            error = error_holder.get("error")
            if error is not None:
                _write_stdout_jsonl(
                    {
                        "event": "warning",
                        "text": f"Context refresh failed during {phase}: {error}",
                    }
                )
                return False
            with refresh_lock:
                refresh_state["refreshed_generation"] = refresh_state["generation"]
            return True

        def _switch_runtime_provider(
            provider: str,
            *,
            preferred_tier: str | None = None,
            warning_text: str | None = None,
            allow_unavailable: bool = False,
        ) -> bool:
            nonlocal selected_provider, selection_is_auto
            normalized = normalize_provider_name(provider)
            if not normalized:
                return False
            previous_provider = normalize_provider_name(
                getattr(model_manager, "current_provider", "") or selected_provider
            )
            previous_default = normalize_provider_name(
                getattr(model_manager, "_default_provider", "") or previous_provider
            )
            previous_fallback = normalize_provider_name(
                getattr(model_manager, "_fallback_provider", "") or previous_provider
            )
            model_manager.current_provider = normalized
            model_manager._default_provider = normalized
            model_manager._fallback_provider = normalized
            model_manager.set_fallback_config(args.model_config)
            target_tier = str(
                preferred_tier or model_manager.current_tier or settings.models.default_tier or "balanced"
            )
            target_tier = target_tier.strip().lower() or "balanced"
            known = {
                item.name
                for item in model_manager.list_tiers()
                if normalize_provider_name(item.provider) == normalized
            }
            available = {
                item.name
                for item in model_manager.list_tiers()
                if item.available and normalize_provider_name(item.provider) == normalized
            }
            tier_choices = known if allow_unavailable else available
            if target_tier not in tier_choices:
                if "balanced" in tier_choices:
                    target_tier = "balanced"
                elif tier_choices:
                    target_tier = sorted(tier_choices)[0]
                else:
                    model_manager.current_provider = previous_provider
                    model_manager._default_provider = previous_default
                    model_manager._fallback_provider = previous_fallback
                    return False
            try:
                model_manager.set_selection(ctx.agent, provider_name=normalized, tier_name=target_tier)
            except Exception:
                model_manager.current_provider = previous_provider
                model_manager._default_provider = previous_default
                model_manager._fallback_provider = previous_fallback
                raise
            selected_provider = normalized
            selection_is_auto = False
            agent_kwargs["model"] = ctx.agent.model
            _refresh_conversation_manager()
            _mark_query_context_stale(reason=f"provider switch to {normalized}")
            if warning_text:
                _write_stdout_jsonl({"event": "warning", "text": warning_text})
            _write_stdout_jsonl(_current_model_info_event())
            return True

        def _apply_runtime_model_selection(
            *,
            provider: str | None,
            tier: str,
            allow_unavailable: bool = False,
        ) -> None:
            nonlocal selected_provider, selection_is_auto
            requested_provider = normalize_provider_name(
                provider or _active_runtime_provider_name() or selected_provider
            )
            requested_tier = str(tier or "").strip().lower()
            if not requested_provider or not requested_tier:
                raise ValueError("Both provider and tier are required")
            if requested_provider != _active_runtime_provider_name():
                if not _switch_runtime_provider(
                    requested_provider,
                    preferred_tier=requested_tier,
                    allow_unavailable=allow_unavailable,
                ):
                    raise ValueError(f"Unknown provider/tier selection: {requested_provider}/{requested_tier}")
                return
            model_manager.set_selection(ctx.agent, provider_name=requested_provider, tier_name=requested_tier)
            selected_provider = requested_provider
            selection_is_auto = False
            agent_kwargs["model"] = ctx.agent.model
            _refresh_conversation_manager()
            _mark_query_context_stale(reason="set_model")
            _refresh_query_context_guarded(interactive=True, phase="set_model")
            _write_stdout_jsonl(_current_model_info_event())

        def _persist_messages_async(*, session_id: str, messages_snapshot: list[Any]) -> None:
            def _worker() -> None:
                with persist_lock:
                    try:
                        session_store.save_messages(
                            session_id,
                            messages_snapshot,
                            max_messages=_SESSION_MESSAGE_MAX_COUNT,
                            version=_SESSION_MESSAGE_VERSION,
                        )
                    except Exception as e:
                        _write_stdout_jsonl({"event": "warning", "text": f"Failed to persist session messages: {e}"})

            threading.Thread(target=_worker, daemon=True, name="swarmee-session-save").start()

        def _same_cwd(meta_cwd: str) -> bool:
            if not meta_cwd:
                return False
            try:
                return Path(meta_cwd).expanduser().resolve() == Path.cwd().resolve()
            except Exception:
                return str(meta_cwd).strip() == str(Path.cwd())

        def _find_available_session_for_cwd() -> tuple[str, int, int] | None:
            with contextlib.suppress(Exception):
                entries = session_store.list(limit=200)
                for entry in entries:
                    sid = str(entry.get("id", "")).strip()
                    if not sid:
                        continue
                    if not _same_cwd(str(entry.get("cwd", "")).strip()):
                        continue
                    messages = session_store.load_messages(sid, max_messages=_SESSION_MESSAGE_MAX_COUNT)
                    if not messages:
                        continue
                    turn_count = _turn_count_from_messages(messages)
                    return sid, turn_count, len(messages)
            return None

        def _restore_session(session_id: str) -> tuple[int, int]:
            nonlocal active_session_id
            sid = session_id.strip()
            if not sid:
                raise ValueError("restore_session.session_id is required")

            messages = session_store.load_messages(sid, max_messages=_SESSION_MESSAGE_MAX_COUNT)
            if not isinstance(messages, list):
                messages = []
            with contextlib.suppress(Exception):
                session_store.save_messages(
                    sid,
                    messages,
                    max_messages=_SESSION_MESSAGE_MAX_COUNT,
                )

            try:
                ctx.agent.messages = messages
            except Exception:
                state = getattr(ctx.agent, "state", None)
                with contextlib.suppress(Exception):
                    ctx.swap_agent(messages, state)
                    _mark_query_context_stale(reason="session restore")
                    _refresh_query_context_guarded(
                        interactive=True,
                        phase="session restore",
                    )
            else:
                _mark_query_context_stale(reason="session restore")

            active_session_id = sid
            os.environ["SWARMEE_SESSION_ID"] = sid
            with contextlib.suppress(Exception):
                session_store.save(sid, meta=ctx.build_session_meta())

            turn_count = _turn_count_from_messages(messages)
            _write_stdout_jsonl(
                {
                    "event": "session_restored",
                    "session_id": sid,
                    "turn_count": turn_count,
                    "message_count": len(messages),
                }
            )

            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "")).strip().lower()
                if role not in {"user", "assistant"}:
                    continue
                text = _extract_text_from_message_for_replay(message).strip()
                if not text:
                    continue
                event_payload: dict[str, Any] = {
                    "event": "replay_turn",
                    "role": role,
                    "text": text,
                    "timestamp": str(message.get("timestamp") or message.get("ts") or message.get("created_at") or ""),
                }
                if role == "assistant":
                    model_value = message.get("model") or message.get("model_id") or message.get("name")
                    if isinstance(model_value, str) and model_value.strip():
                        event_payload["model"] = model_value.strip()
                _write_stdout_jsonl(event_payload)

            _write_stdout_jsonl({"event": "replay_complete", "turn_count": turn_count})
            return turn_count, len(messages)

        with contextlib.suppress(Exception):
            session_store.save(active_session_id, meta=ctx.build_session_meta())
            os.environ["SWARMEE_SESSION_ID"] = active_session_id

        _write_stdout_jsonl({"event": "ready", "session_id": resolved_session_id})
        _write_stdout_jsonl(_current_model_info_event())
        _write_stdout_jsonl({"event": "safety_overrides", "overrides": _current_session_safety_overrides()})
        _write_stdout_jsonl(_get_prompt_assets_payload())

        available = _find_available_session_for_cwd()
        if available is not None:
            available_sid, available_turn_count, _available_message_count = available
            _write_stdout_jsonl(
                {
                    "event": "session_available",
                    "session_id": available_sid,
                    "turn_count": available_turn_count,
                }
            )
        _refresh_query_context_guarded(interactive=True, phase="daemon startup handshake")

        worker_thread: threading.Thread | None = None
        worker_lock = threading.Lock()
        last_query_text: str = ""
        last_query_auto_approve: bool = auto_approve

        def _worker_is_running() -> bool:
            with worker_lock:
                return worker_thread is not None and worker_thread.is_alive()

        def _next_available_tier() -> str | None:
            current = str(model_manager.current_tier or "").strip().lower()
            current_provider = _active_runtime_provider_name()
            with contextlib.suppress(Exception):
                tiers = [
                    tier.name.strip().lower()
                    for tier in model_manager.list_tiers()
                    if tier.available and normalize_provider_name(tier.provider) == current_provider
                ]
                if current in tiers:
                    idx = tiers.index(current)
                    for candidate in tiers[idx + 1 :]:
                        if candidate:
                            return candidate
                for candidate in tiers:
                    if candidate and candidate != current:
                        return candidate
            return None

        def _emit_turn_error(text: str, *, category_hint: str | None = None) -> None:
            _write_stdout_jsonl(_build_tui_error_event(text, category_hint=category_hint))
            _write_stdout_jsonl({"event": "turn_complete", "exit_status": "error"})

        def _set_daemon_consent_waiting(waiting: bool) -> None:
            _set_daemon_consent_response("")
            if waiting:
                daemon_consent_event.clear()
            else:
                daemon_consent_event.set()

        def _start_query_worker_thread(
            *,
            query_text: str,
            turn_auto_approve: bool,
            mode: str | None = None,
            approved_plan: PendingWorkPlan | None = None,
            plan_context: dict[str, Any] | None = None,
        ) -> None:
            nonlocal worker_thread
            interrupt_event.clear()
            _set_daemon_consent_waiting(True)
            with worker_lock:
                worker_thread = threading.Thread(
                    target=_run_query_worker,
                    kwargs={
                        "query_text": query_text,
                        "turn_auto_approve": turn_auto_approve,
                        "mode": mode,
                        "approved_plan": approved_plan,
                        "plan_context": dict(plan_context) if isinstance(plan_context, dict) else None,
                    },
                    daemon=True,
                )
                worker_thread.start()

        def _run_query_worker(
            query_text: str,
            *,
            turn_auto_approve: bool,
            mode: str | None = None,
            approved_plan: PendingWorkPlan | None = None,
            plan_context: dict[str, Any] | None = None,
        ) -> None:
            exit_status = "ok"
            response: Any | None = None
            executed = False
            forced_mode = (mode or "").strip().lower()
            if forced_mode not in {"", "plan", "execute"}:
                forced_mode = ""
            transient_attempts = 0

            try:
                while True:
                    try:
                        _refresh_query_context_guarded(
                            interactive=False,
                            phase="query preflight",
                        )
                        _best_effort_retrieve(query_text, warn_on_error=False)
                        if _active_runtime_provider_name() == "bedrock" and not has_aws_credentials():
                            if selection_is_auto and has_github_copilot_token() and _switch_runtime_provider(
                                "github_copilot",
                                preferred_tier=model_manager.current_tier,
                                warning_text=(
                                    "[provider] AWS credentials are unavailable for Bedrock; "
                                    "falling back to GitHub Copilot for this session."
                                ),
                            ):
                                continue
                            _write_stdout_jsonl(
                                _build_tui_error_event(
                                    "AWS credentials are unavailable for Bedrock. "
                                    "Run connect aws [profile] or configure another provider, then retry.",
                                    category_hint=ERROR_CATEGORY_AUTH_ERROR,
                                )
                            )
                            exit_status = "error"
                            break

                        _, response, executed = _run_query_with_optional_plan(
                            query_text=query_text,
                            forced_mode=forced_mode,
                            auto_approve=turn_auto_approve,
                            welcome_text=ctx.welcome_text,
                            generate_plan=_generate_plan,
                            execute_with_plan=lambda req, plan, welcome: _execute_with_plan(
                                req, plan, welcome_text_local=welcome
                            ),
                            run_agent=lambda req: run_agent(req, invocation_state={"swarmee": {"mode": "execute"}}),
                            classify_intent_fn=classify_intent,
                            on_plan=lambda plan: (
                                setattr(ctx, "last_plan", plan.current_plan),
                                _emit_plan_event(plan, write_jsonl=True, echo_console=False),
                            ),
                            approved_plan=approved_plan,
                            plan_context=plan_context,
                        )

                        kb_for_turn = _current_knowledge_base_id()
                        if executed and kb_for_turn:
                            store_conversation_in_kb(ctx.agent, query_text, response, kb_for_turn)
                        break
                    except AgentInterruptedError:
                        callback_handler(force_stop=True)
                        exit_status = "interrupted"
                        break
                    except MaxTokensReachedException as exc:
                        callback_handler(force_stop=True)
                        next_tier = _next_available_tier()
                        _write_stdout_jsonl(
                            _build_tui_error_event(
                                str(exc),
                                category_hint=ERROR_CATEGORY_ESCALATABLE,
                                next_tier=next_tier,
                            )
                        )
                        exit_status = "error"
                        break
                    except Exception as e:
                        callback_handler(force_stop=True)
                        text = str(e)
                        classified = classify_error_message(text)
                        category = str(classified.get("category", "")).strip().lower()
                        if category == ERROR_CATEGORY_TRANSIENT and transient_attempts < _TRANSIENT_RETRY_MAX_ATTEMPTS:
                            retry_after_s = min(30, 2**transient_attempts)
                            transient_attempts += 1
                            _write_stdout_jsonl(
                                _build_tui_error_event(
                                    text,
                                    category_hint=ERROR_CATEGORY_TRANSIENT,
                                    retry_after_s=retry_after_s,
                                )
                            )
                            time.sleep(retry_after_s)
                            continue

                        if category == ERROR_CATEGORY_TRANSIENT and transient_attempts >= _TRANSIENT_RETRY_MAX_ATTEMPTS:
                            next_tier = _next_available_tier()
                            _write_stdout_jsonl(
                                _build_tui_error_event(
                                    text,
                                    category_hint=ERROR_CATEGORY_ESCALATABLE,
                                    next_tier=next_tier,
                                )
                            )
                        else:
                            next_tier = _next_available_tier() if category == ERROR_CATEGORY_ESCALATABLE else None
                            _write_stdout_jsonl(
                                _build_tui_error_event(
                                    text,
                                    category_hint=category,
                                    next_tier=next_tier,
                                )
                            )
                        exit_status = "error"
                        break
            finally:
                if exit_status == "ok":
                    snapshot_raw = getattr(ctx.agent, "messages", [])
                    if isinstance(snapshot_raw, list):
                        messages_snapshot = list(snapshot_raw)
                    elif isinstance(snapshot_raw, tuple):
                        messages_snapshot = list(snapshot_raw)
                    else:
                        messages_snapshot = []
                    _persist_messages_async(session_id=active_session_id, messages_snapshot=messages_snapshot)
                _write_stdout_jsonl(_current_model_info_event())
                _write_stdout_jsonl({"event": "turn_complete", "exit_status": exit_status})

        while True:
            raw_line = sys.stdin.readline()
            if raw_line == "":
                break
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except Exception:
                _write_stdout_jsonl({"event": "warning", "text": "Invalid daemon command JSON"})
                continue

            if not isinstance(payload, dict):
                _write_stdout_jsonl({"event": "warning", "text": "Invalid daemon command payload"})
                continue

            cmd = str(payload.get("cmd", "")).strip().lower()

            if cmd == "query":
                query_text = payload.get("text")
                if not isinstance(query_text, str) or not query_text.strip():
                    _emit_turn_error("query.text is required")
                    continue
                if _worker_is_running():
                    _emit_turn_error("A query is already running", category_hint=ERROR_CATEGORY_FATAL)
                    continue

                requested_provider = payload.get("provider")
                requested_tier = payload.get("tier")
                if isinstance(requested_tier, str) and requested_tier.strip():
                    try:
                        _apply_runtime_model_selection(
                            provider=requested_provider if isinstance(requested_provider, str) else None,
                            tier=requested_tier.strip().lower(),
                            allow_unavailable=True,
                        )
                    except Exception as e:
                        _emit_turn_error(str(e), category_hint=ERROR_CATEGORY_ESCALATABLE)
                        continue

                requested_mode = payload.get("mode")
                mode = requested_mode.strip().lower() if isinstance(requested_mode, str) else None
                raw_auto_approve = payload.get("auto_approve")
                turn_auto_approve = raw_auto_approve if isinstance(raw_auto_approve, bool) else auto_approve
                plan_context_payload = payload.get("plan_context")
                plan_context = dict(plan_context_payload) if isinstance(plan_context_payload, dict) else None
                approved_plan_payload = payload.get("approved_plan")
                approved_plan = pending_work_plan_from_payload(approved_plan_payload)
                last_query_text = query_text.strip()
                last_query_auto_approve = turn_auto_approve
                _start_query_worker_thread(
                    query_text=last_query_text,
                    turn_auto_approve=turn_auto_approve,
                    mode=mode,
                    approved_plan=approved_plan,
                    plan_context=plan_context,
                )
                continue

            if cmd == "consent_response":
                choice = payload.get("choice")
                _set_daemon_consent_response(choice.strip().lower() if isinstance(choice, str) else "")
                daemon_consent_event.set()
                continue

            if cmd == "set_context_sources":
                sources_payload = payload.get("sources", [])
                try:
                    _set_user_context_sources(sources_payload)
                    _mark_query_context_stale(reason="set_context_sources")
                except Exception as e:
                    _write_stdout_jsonl({"event": "warning", "text": f"Failed to set context sources: {e}"})
                continue

            if cmd == "compact":
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot compact while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                compact_result = _compact_context()
                _emit_tui_context_event_if_enabled(ctx.agent, settings=settings, model_manager=model_manager)
                _write_stdout_jsonl(
                    {
                        "event": "compact_complete",
                        "automatic": False,
                        "compacted": bool(compact_result.get("compacted", False)),
                        "before_tokens_est": compact_result.get("before_tokens_est"),
                        "after_tokens_est": compact_result.get("after_tokens_est"),
                        "budget_tokens": compact_result.get("budget_tokens"),
                        "summary_passes": compact_result.get("summary_passes", 0),
                        "trimmed_messages": compact_result.get("trimmed_messages", 0),
                        "compacted_read_results": compact_result.get("compacted_read_results", 0),
                        "compaction_headroom_tokens": compact_result.get("compaction_headroom_tokens"),
                        "warning": compact_result.get("warning"),
                        "fork_kind": compact_result.get("fork_kind"),
                        "fork_parent_message_count": compact_result.get("fork_parent_message_count"),
                        "fork_prefix_hash": compact_result.get("fork_prefix_hash"),
                        "fork_extra_prompt_chars": compact_result.get("fork_extra_prompt_chars"),
                        "fork_used_pending_reminder": compact_result.get("fork_used_pending_reminder"),
                    }
                )
                continue

            if cmd == "set_sop":
                raw_name = payload.get("name")
                if not isinstance(raw_name, str) or not raw_name.strip():
                    _write_stdout_jsonl(
                        _build_tui_error_event("set_sop.name is required", category_hint=ERROR_CATEGORY_FATAL)
                    )
                    continue
                raw_content = payload.get("content")
                if raw_content is not None and not isinstance(raw_content, str):
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "set_sop.content must be a string or null",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                _set_daemon_sop_override(raw_name, raw_content)
                ctx.refresh_system_prompt()
                _mark_query_context_stale(reason="set_sop")
                with contextlib.suppress(Exception):
                    session_store.save(active_session_id, meta=ctx.build_session_meta())
                continue

            if cmd == "set_profile":
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot apply profile while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                raw_profile = payload.get("profile")
                if not isinstance(raw_profile, dict):
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "set_profile.profile is required",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                before_model_info = _current_model_info_event()
                try:
                    applied_profile = _apply_profile(raw_profile)
                    _mark_query_context_stale(reason="set_profile")
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_FATAL))
                    continue

                with contextlib.suppress(Exception):
                    session_store.save(active_session_id, meta=ctx.build_session_meta())

                after_model_info = _current_model_info_event()
                if (
                    before_model_info.get("provider") != after_model_info.get("provider")
                    or before_model_info.get("tier") != after_model_info.get("tier")
                    or before_model_info.get("model_id") != after_model_info.get("model_id")
                ):
                    _write_stdout_jsonl(after_model_info)
                _write_stdout_jsonl({"event": "profile_applied", "profile": applied_profile})
                continue

            if cmd == "get_bundles":
                _write_stdout_jsonl(_get_bundles_payload())
                continue

            if cmd == "set_bundle":
                raw_bundle = payload.get("bundle")
                if not isinstance(raw_bundle, dict):
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "set_bundle.bundle is required",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                try:
                    catalog_event = _set_bundle(raw_bundle)
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_FATAL))
                    continue
                _write_stdout_jsonl(catalog_event)
                continue

            if cmd == "delete_bundle":
                raw_id = payload.get("id")
                if not isinstance(raw_id, str) or not raw_id.strip():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "delete_bundle.id is required",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                try:
                    catalog_event = _delete_bundle(raw_id.strip())
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_FATAL))
                    continue
                _write_stdout_jsonl(catalog_event)
                continue

            if cmd == "apply_bundle":
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot apply bundle while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                raw_id = payload.get("id")
                if not isinstance(raw_id, str) or not raw_id.strip():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "apply_bundle.id is required",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                before_model_info = _current_model_info_event()
                try:
                    applied_bundle = _apply_bundle(raw_id.strip())
                    _mark_query_context_stale(reason="apply_bundle")
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_FATAL))
                    continue
                with contextlib.suppress(Exception):
                    session_store.save(active_session_id, meta=ctx.build_session_meta())
                after_model_info = _current_model_info_event()
                if (
                    before_model_info.get("provider") != after_model_info.get("provider")
                    or before_model_info.get("tier") != after_model_info.get("tier")
                    or before_model_info.get("model_id") != after_model_info.get("model_id")
                ):
                    _write_stdout_jsonl(after_model_info)
                _write_stdout_jsonl({"event": "bundle_applied", "bundle": applied_bundle, "profile": applied_bundle})
                continue

            if cmd == "set_safety_overrides":
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot set safety overrides while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                payload_update: dict[str, Any] = {}
                raw_nested = payload.get("overrides")
                if isinstance(raw_nested, dict):
                    payload_update.update(raw_nested)
                for key in ("tool_consent", "tool_allowlist", "tool_blocklist"):
                    if key in payload:
                        payload_update[key] = payload.get(key)
                if not payload_update:
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "set_safety_overrides requires tool_consent/tool_allowlist/tool_blocklist payload",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                try:
                    applied = _apply_session_safety_overrides(payload_update)
                    _mark_query_context_stale(reason="set_safety_overrides")
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_FATAL))
                    continue
                _write_stdout_jsonl({"event": "safety_overrides", "overrides": applied})
                continue

            if cmd == "get_prompt_assets":
                _write_stdout_jsonl(_get_prompt_assets_payload())
                continue

            if cmd == "set_prompt_asset":
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot update prompt assets while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                try:
                    payload_event = _set_prompt_asset(payload.get("asset"))
                    _write_stdout_jsonl(payload_event)
                    _mark_query_context_stale(reason="set_prompt_asset")
                    _refresh_query_context_guarded(interactive=True, phase="set_prompt_asset")
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_FATAL))
                continue

            if cmd == "delete_prompt_asset":
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot delete prompt assets while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                try:
                    payload_event = _delete_prompt_asset(str(payload.get("id", "")))
                    _write_stdout_jsonl(payload_event)
                    _mark_query_context_stale(reason="delete_prompt_asset")
                    _refresh_query_context_guarded(interactive=True, phase="delete_prompt_asset")
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_FATAL))
                continue

            if cmd == "connect":
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot connect provider while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                provider = normalize_provider_name(payload.get("provider") or "github_copilot")
                method = str(payload.get("method", "device")).strip().lower()
                try:
                    if provider == "github_copilot":
                        if method == "api":
                            raw_api_key = payload.get("api_key")
                            api_key = str(raw_api_key).strip() if isinstance(raw_api_key, str) else ""
                            if not api_key:
                                _write_stdout_jsonl(
                                    _build_tui_error_event(
                                        "connect(api) requires api_key",
                                        category_hint=ERROR_CATEGORY_FATAL,
                                    )
                                )
                                continue
                            path = save_api_key(api_key)
                            _write_stdout_jsonl(
                                {"event": "warning", "text": f"Saved GitHub Copilot API key to: {path}"}
                            )
                        else:
                            open_browser = bool(payload.get("open_browser", True))
                            result = login_device_flow(
                                open_browser=open_browser,
                                status=lambda line: _write_stdout_jsonl({"event": "warning", "text": line}),
                            )
                            _write_stdout_jsonl(
                                {
                                    "event": "warning",
                                    "text": f"GitHub Copilot connected. Saved credentials to: {result.get('path')}",
                                }
                            )
                        _write_stdout_jsonl(_current_model_info_event())
                        continue

                    if provider == "bedrock":
                        requested_profile = payload.get("profile")
                        profile = str(requested_profile).strip() if isinstance(requested_profile, str) else ""
                        source = _connect_aws_credentials(
                            profile=profile or None,
                            emit=lambda line: _write_stdout_jsonl({"event": "warning", "text": line}),
                        )
                        if _active_runtime_provider_name() == "bedrock":
                            model_manager.set_tier(ctx.agent, model_manager.current_tier)
                            agent_kwargs["model"] = ctx.agent.model
                            _refresh_conversation_manager()
                            _mark_query_context_stale(reason="connect bedrock")
                            _refresh_query_context_guarded(interactive=True, phase="connect bedrock")
                        _write_stdout_jsonl(
                            {
                                "event": "warning",
                                "text": f"AWS auth source: {source}",
                                "auth_source": source,
                                "provider": "bedrock",
                            }
                        )
                        _write_stdout_jsonl(_current_model_info_event())
                        continue

                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            f"Unsupported connect provider: {provider}",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_AUTH_ERROR))
                continue

            if cmd == "auth":
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot inspect auth while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                action = str(payload.get("action", "list")).strip().lower()
                if action in {"list", "ls"}:
                    text = _render_auth_records_text()
                    _write_stdout_jsonl({"event": "warning", "text": text})
                    continue
                if action == "logout":
                    provider = normalize_provider_name(payload.get("provider") or "github_copilot")
                    deleted = delete_provider_record(provider)
                    if deleted:
                        _write_stdout_jsonl(
                            {"event": "warning", "text": f"Removed saved credentials for provider: {provider}"}
                        )
                    else:
                        _write_stdout_jsonl(
                            {"event": "warning", "text": f"No saved credentials found for provider: {provider}"}
                        )
                    continue
                _write_stdout_jsonl({"event": "warning", "text": "Unknown auth action"})
                continue

            if cmd == "restore_session":
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot restore session while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                raw_session_id = payload.get("session_id")
                if not isinstance(raw_session_id, str) or not raw_session_id.strip():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "restore_session.session_id is required",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                try:
                    _restore_session(raw_session_id)
                except Exception as e:
                    _write_stdout_jsonl({"event": "warning", "text": f"Failed to restore session: {e}"})
                continue

            if cmd == "interrupt":
                interrupt_event.set()
                _set_daemon_consent_waiting(False)
                continue

            if cmd == "set_tier":
                tier = payload.get("tier")
                if not isinstance(tier, str) or not tier.strip():
                    _write_stdout_jsonl(
                        _build_tui_error_event("set_tier.tier is required", category_hint=ERROR_CATEGORY_FATAL)
                    )
                    continue
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot set tier while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                try:
                    _apply_runtime_model_selection(
                        provider=_active_runtime_provider_name(),
                        tier=tier.strip().lower(),
                        allow_unavailable=True,
                    )
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_ESCALATABLE))
                continue

            if cmd == "set_model":
                provider = payload.get("provider")
                tier = payload.get("tier")
                if not isinstance(provider, str) or not provider.strip():
                    _write_stdout_jsonl(
                        _build_tui_error_event("set_model.provider is required", category_hint=ERROR_CATEGORY_FATAL)
                    )
                    continue
                if not isinstance(tier, str) or not tier.strip():
                    _write_stdout_jsonl(
                        _build_tui_error_event("set_model.tier is required", category_hint=ERROR_CATEGORY_FATAL)
                    )
                    continue
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot set model while a query is running",
                            category_hint=ERROR_CATEGORY_FATAL,
                        )
                    )
                    continue
                try:
                    _apply_runtime_model_selection(
                        provider=provider.strip().lower(),
                        tier=tier.strip().lower(),
                        allow_unavailable=True,
                    )
                except Exception as e:
                    _write_stdout_jsonl(_build_tui_error_event(str(e), category_hint=ERROR_CATEGORY_ESCALATABLE))
                continue

            if cmd == "retry_tool":
                tool_use_id = str(payload.get("tool_use_id", "")).strip()
                if not tool_use_id:
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "retry_tool.tool_use_id is required",
                            category_hint=ERROR_CATEGORY_TOOL_ERROR,
                        )
                    )
                    continue
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot retry tool while a query is running",
                            category_hint=ERROR_CATEGORY_TOOL_ERROR,
                            tool_use_id=tool_use_id,
                        )
                    )
                    continue
                if not last_query_text:
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "No previous query available for tool retry",
                            category_hint=ERROR_CATEGORY_TOOL_ERROR,
                            tool_use_id=tool_use_id,
                        )
                    )
                    continue
                recovery_query = (
                    f"Tool call {tool_use_id} failed previously. Retry that tool call now and continue the task."
                )
                _start_query_worker_thread(
                    query_text=recovery_query,
                    turn_auto_approve=last_query_auto_approve,
                    mode="execute",
                )
                continue

            if cmd == "skip_tool":
                tool_use_id = str(payload.get("tool_use_id", "")).strip()
                if not tool_use_id:
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "skip_tool.tool_use_id is required",
                            category_hint=ERROR_CATEGORY_TOOL_ERROR,
                        )
                    )
                    continue
                if _worker_is_running():
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "Cannot skip tool while a query is running",
                            category_hint=ERROR_CATEGORY_TOOL_ERROR,
                            tool_use_id=tool_use_id,
                        )
                    )
                    continue
                if not last_query_text:
                    _write_stdout_jsonl(
                        _build_tui_error_event(
                            "No previous query available for tool skip",
                            category_hint=ERROR_CATEGORY_TOOL_ERROR,
                            tool_use_id=tool_use_id,
                        )
                    )
                    continue
                recovery_query = (
                    f"Tool call {tool_use_id} failed previously. Continue the task without rerunning that tool."
                )
                _start_query_worker_thread(
                    query_text=recovery_query,
                    turn_auto_approve=last_query_auto_approve,
                    mode="execute",
                )
                continue

            if cmd == "shutdown":
                interrupt_event.set()
                _set_daemon_consent_waiting(False)
                break

            _write_stdout_jsonl({"event": "warning", "text": f"Unknown daemon cmd: {cmd}"})

        with worker_lock:
            active_worker = worker_thread
        if active_worker is not None and active_worker.is_alive():
            active_worker.join(timeout=2.0)

    def _seed_local_surface_branch_from_runtime(*, surface: str) -> None:
        nonlocal selected_provider, selection_is_auto
        if args.tui_daemon:
            return
        store = ctx.session_store
        if store is None:
            return

        discovery_path = runtime_discovery_path(cwd=Path.cwd())
        if not discovery_path.exists():
            return

        source_session_id = (os.getenv("SWARMEE_SESSION_ID") or "").strip() or None
        branch_session_id: str | None = None
        client: RuntimeServiceClient | None = None
        try:
            client = RuntimeServiceClient.from_discovery_file(discovery_path, timeout_s=1.0)
            client.connect()
            hello = client.hello(client_name=f"swarmee-{surface}", surface="cli")
            if not isinstance(hello, dict) or str(hello.get("event", "")).strip().lower() == "error":
                return
            if runtime_hello_supports_capability(hello, "fork_surface_session"):
                fork_event = client.fork_surface_session(
                    cwd=str(Path.cwd().resolve()),
                    surface=surface,
                    source_session_id=source_session_id,
                )
                if isinstance(fork_event, dict) and str(fork_event.get("event", "")).strip().lower() != "error":
                    branch_session_id = str(fork_event.get("session_id", "")).strip() or None
                elif isinstance(fork_event, dict):
                    code = str(fork_event.get("code", "")).strip().lower()
                    if code not in {"unknown_cmd", "no_active_parent_session"}:
                        return
            if not branch_session_id:
                branch_session_id = source_session_id or default_session_id_for_cwd(Path.cwd())
        except Exception:
            return
        finally:
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close()

        if not branch_session_id:
            return

        try:
            branch_meta = store.read_meta(branch_session_id)
            branch_messages = store.load_messages(branch_session_id, max_messages=_SESSION_MESSAGE_MAX_COUNT)
        except Exception:
            return

        provider = str(branch_meta.get("provider", "")).strip().lower() or None
        tier = str(branch_meta.get("tier", "")).strip().lower() or None
        if provider and tier:
            with contextlib.suppress(Exception):
                model_manager.set_selection(ctx.agent, provider_name=provider, tier_name=tier)
                selected_provider = provider
                selection_is_auto = False
                agent_kwargs["model"] = ctx.agent.model
                if callable(ctx.refresh_conversation_manager):
                    ctx.refresh_conversation_manager()

        state = getattr(ctx.agent, "state", None)
        with contextlib.suppress(Exception):
            ctx.swap_agent(branch_messages, state)
            agent_kwargs["model"] = ctx.agent.model
        ctx.current_session_id = branch_session_id
        os.environ["SWARMEE_SESSION_ID"] = branch_session_id

    if args.tui_daemon:
        _run_tui_daemon()
        return

    # Query mode (one-shot) or interactive REPL mode.
    _seed_local_surface_branch_from_runtime(surface="cli")

    if args.query:
        query = " ".join(args.query)
        # Retrieval is best-effort and should never block query execution.
        _best_effort_retrieve(query, warn_on_error=True)

        _refresh_query_context(interactive=False)

        try:
            plan, response, executed = _run_query_with_optional_plan(
                query_text=query,
                forced_mode=None,
                auto_approve=auto_approve,
                welcome_text="",
                generate_plan=_generate_plan,
                execute_with_plan=lambda req, p, welcome: _execute_with_plan(req, p, welcome_text_local=welcome),
                run_agent=lambda req: run_agent(req, invocation_state={"swarmee": {"mode": "execute"}}),
                classify_intent_fn=classify_intent,
                on_plan=lambda p: (
                    setattr(ctx, "last_plan", p.current_plan),
                    _emit_plan_event(p, write_jsonl=False, echo_console=True),
                ),
            )
            if plan is not None and not executed:
                if not _tui_events_enabled():
                    print(
                        "\nPlan generated. Re-run with --yes (or set `runtime.auto_approve=true` in "
                        "`.swarmee/settings.json`) to execute."
                    )
                return
        except AgentInterruptedError as e:
            _handle_agent_interrupt(e)
            return
        except MaxTokensReachedException:
            return

        kb_for_turn = _current_knowledge_base_id()
        if kb_for_turn and executed:
            # Persist the executed exchange when KB is configured.
            store_conversation_in_kb(ctx.agent, query, response, kb_for_turn)
    else:
        # Render welcome banner once at startup in interactive mode.
        try:
            welcome_text = read_welcome_text()
        except Exception:
            welcome_text = ""
        if welcome_text:
            ctx.welcome_text = welcome_text
            render_welcome_message(welcome_text)
        _refresh_query_context(interactive=True)
        run_repl(
            ctx=ctx,
            registry=registry,
            get_user_input=get_user_input,
            classify_intent=classify_intent,
            render_goodbye_message=render_goodbye_message,
        )


if __name__ == "__main__":
    main()
