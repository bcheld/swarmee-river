#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""
Swarmee - A minimal CLI interface for Swarmee River (built on Strands)
"""

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import threading
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

from swarmee_river.handlers.callback_handler import callback_handler, set_interrupt_event

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
    from swarmee_river.hooks.jsonl_logger import JSONLLoggerHooks as _JSONLLoggerHooks
    from swarmee_river.hooks.tui_metrics import TuiMetricsHooks as _TuiMetricsHooks
    from swarmee_river.hooks.tool_consent import ToolConsentHooks as _ToolConsentHooks
    from swarmee_river.hooks.tool_policy import ToolPolicyHooks as _ToolPolicyHooks
    from swarmee_river.hooks.tool_result_limiter import ToolResultLimiterHooks as _ToolResultLimiterHooks

    _HAS_STRANDS_HOOKS = True
except Exception:
    _JSONLLoggerHooks = None  # type: ignore[misc,assignment]
    _TuiMetricsHooks = None  # type: ignore[misc,assignment]
    _ToolConsentHooks = None  # type: ignore[misc,assignment]
    _ToolResultLimiterHooks = None  # type: ignore[misc,assignment]
    _ToolPolicyHooks = None  # type: ignore[misc,assignment]
    _HAS_STRANDS_HOOKS = False
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
from swarmee_river.agent_runner import invoke_agent
from swarmee_river.artifacts import ArtifactStore, tools_expected_from_plan
from swarmee_river.cli.builtin_commands import register_builtin_commands
from swarmee_river.cli.commands import CLIContext, CommandRegistry, handle_common_session_command, handle_pack_command
from swarmee_river.cli.diagnostics import (
    render_config_command_for_surface,
    render_diagnostic_command_for_surface,
)
from swarmee_river.cli.repl import run_repl
from swarmee_river.context.prompt_cache import PromptCacheState
from swarmee_river.interrupts import (
    AgentInterruptedError,
    interrupt_watcher_from_env,
    pause_active_interrupt_watcher_for_input,
)
from swarmee_river.packs import enabled_sop_paths, enabled_system_prompts, load_enabled_pack_tools
from swarmee_river.planning import WorkPlan, classify_intent, structured_plan_prompt
from swarmee_river.project_map import build_context_snapshot, build_project_map
from swarmee_river.runtime_env import detect_runtime_environment, render_runtime_environment_section
from swarmee_river.session.models import SessionModelManager
from swarmee_river.session.store import SessionStore
from swarmee_river.settings import SwarmeeSettings, load_settings
from swarmee_river.tools import get_tools
from swarmee_river.utils.agent_runtime_utils import (
    build_base_system_prompt,
    plan_json_for_execution,
    render_plan_text,
    resolve_effective_sop_paths,
)
from swarmee_river.utils import model_utils
from swarmee_river.utils.env_utils import load_env_file, truthy
from swarmee_river.utils.kb_utils import load_system_prompt, store_conversation_in_kb
from swarmee_river.utils.provider_utils import resolve_model_provider
from swarmee_river.utils.stdio_utils import configure_stdio_for_utf8, write_stdout_jsonl
from swarmee_river.utils.welcome_utils import render_goodbye_message, render_welcome_message
from tools.sop import run_sop
from tools.welcome import read_welcome_text

os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"
_TOOL_USAGE_RULES = (
    "Tool usage rules:\n"
    "- Use list/glob/file_list/file_search/file_read for repository exploration and file reading.\n"
    "- Do not use shell for ls/find/sed/cat/grep/rg when file tools can do it.\n"
    "- Reserve shell for real command execution tasks."
)
_SYSTEM_REMINDER_RULES = (
    "System reminder rules:\n"
    "- You may receive a `<system-reminder>` block prepended to a user message.\n"
    "- Treat it as system-level updates/context (higher priority than normal user content).\n"
    "- Do not quote or reveal `<system-reminder>` contents unless the user explicitly asks.\n"
)
_DIAGNOSTIC_COMMANDS = {"status", "diff", "artifact", "log", "replay"}


_consent_prompt_session: Any | None = None
_consent_prompt_lock = threading.Lock()
_consent_console: Any | None = Console() if Console is not None else None
_stdout_jsonl_lock = threading.Lock()


def _write_stdout_jsonl(event: dict[str, Any]) -> None:
    write_stdout_jsonl(event, lock=_stdout_jsonl_lock)


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


def _emit_tui_event(event: dict[str, Any]) -> None:
    """Emit a structured JSONL event to stdout for TUI consumption."""
    if _tui_events_enabled():
        _write_stdout_jsonl(event)


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
) -> dict[str, Any]:
    resolved_state: dict[str, Any] = dict(invocation_state) if isinstance(invocation_state, dict) else {}
    sw_state = resolved_state.setdefault("swarmee", {})
    if isinstance(sw_state, dict):
        sw_state.setdefault("runtime_environment", dict(runtime_environment))
        sw_state["tier"] = model_manager.current_tier
        sw_state["provider"] = selected_provider
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
        current = next((item for item in tiers if item.name == model_manager.current_tier), None)
        if current is not None:
            sw_state["model_id"] = current.model_id
    return resolved_state


def _emit_tui_context_event_if_enabled(agent: Any) -> None:
    if not _tui_events_enabled():
        return
    try:
        from swarmee_river.context.budgeted_summarizing_conversation_manager import estimate_tokens

        chars_per_token = int(os.getenv("SWARMEE_TOKEN_CHARS_PER_TOKEN", "4"))
        prompt_tokens_est = estimate_tokens(
            system_prompt=getattr(agent, "system_prompt", None),
            messages=getattr(agent, "messages", []),
            chars_per_token=chars_per_token,
        )
        budget = int(os.getenv("SWARMEE_CONTEXT_BUDGET_TOKENS", "20000"))
        _emit_tui_event({
            "event": "context",
            "prompt_tokens_est": prompt_tokens_est,
            "budget_tokens": budget,
            "chars_per_token": chars_per_token,
            "messages": len(getattr(agent, "messages", []) or []),
        })
    except Exception:
        pass


def _build_session_meta_payload(
    *,
    settings: SwarmeeSettings,
    selected_provider: str,
    current_tier: str,
    active_sop_name: str | None,
) -> dict[str, Any]:
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
        "active_sop": active_sop_name,
    }


def _build_model_info_event_payload(*, model_manager: SessionModelManager, selected_provider: str) -> dict[str, Any]:
    tiers = model_manager.list_tiers()
    current = next((item for item in tiers if item.name == model_manager.current_tier), None)
    return {
        "event": "model_info",
        "provider": current.provider if current is not None else selected_provider,
        "tier": model_manager.current_tier,
        "model_id": current.model_id if current is not None else None,
        "tiers": [
            {
                "name": item.name,
                "provider": item.provider,
                "model_id": item.model_id,
                "display_name": item.display_name,
                "description": item.description,
                "available": item.available,
                "reason": item.reason,
            }
            for item in tiers
        ],
    }


def _build_conversation_manager(
    *,
    window_size: int | None,
    per_turn: int | None,
    summarization_system_prompt: str | None = None,
) -> Any:
    manager_name = (os.getenv("SWARMEE_CONTEXT_MANAGER", "summarize") or "summarize").strip().lower()

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
            )
        except Exception:
            return None

        preserve_recent = int(os.getenv("SWARMEE_PRESERVE_RECENT_MESSAGES", "10"))
        summary_ratio = float(os.getenv("SWARMEE_SUMMARY_RATIO", "0.3"))
        max_prompt_tokens = int(os.getenv("SWARMEE_CONTEXT_BUDGET_TOKENS", "20000"))
        return BudgetedSummarizingConversationManager(
            max_prompt_tokens=max_prompt_tokens,
            summary_ratio=summary_ratio,
            preserve_recent_messages=preserve_recent,
            summarization_system_prompt=summarization_system_prompt,
        )

    # Default: sliding window
    try:
        from strands.agent.conversation_manager import SlidingWindowConversationManager
    except Exception:
        return None

    resolved_window_size = window_size if window_size is not None else int(os.getenv("SWARMEE_WINDOW_SIZE", "20"))
    resolved_per_turn = per_turn if per_turn is not None else int(os.getenv("SWARMEE_CONTEXT_PER_TURN", "1"))
    should_truncate_results = truthy(os.getenv("SWARMEE_TRUNCATE_RESULTS", "true"))

    return SlidingWindowConversationManager(
        window_size=resolved_window_size,
        should_truncate_results=should_truncate_results,
        per_turn=resolved_per_turn,
    )


def _build_runtime_hooks(
    *,
    args: argparse.Namespace,
    safety_settings: Any,
    auto_approve: bool,
    consent_prompt_fn: Callable[[str], str] | None,
) -> list[Any]:
    if not _HAS_STRANDS_HOOKS:
        return []

    def _consent_prompt(text: str) -> str:
        if callable(consent_prompt_fn):
            return str(consent_prompt_fn(text) or "")
        if _tui_events_enabled():
            _emit_tui_event({
                "event": "consent_prompt",
                "context": text,
                "options": ["y", "n", "a", "v"],
            })
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
        TuiMetricsHooks(),
        ToolPolicyHooks(safety_settings),
        ToolConsentHooks(
            safety_settings,
            interactive=not bool(args.query),
            auto_approve=auto_approve,
            prompt=_consent_prompt,
        ),
        ToolResultLimiterHooks(),
    ]
    if ToolMessageRepairHooks is not None:
        hooks.insert(2, ToolMessageRepairHooks())
    return hooks


def _build_runtime_tools(settings: SwarmeeSettings) -> tuple[dict[str, Any], list[Any]]:
    tools_dict = get_tools()
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
        env_provider=os.getenv("SWARMEE_MODEL_PROVIDER"),
        settings_provider=settings.models.provider,
    )
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
    base_system_prompt = build_base_system_prompt(
        raw_system_prompt=load_system_prompt(),
        runtime_environment_prompt_section=runtime_environment_prompt_section,
        pack_prompt_sections=pack_prompt_sections,
        tool_usage_rules=_TOOL_USAGE_RULES,
        system_reminder_rules=_SYSTEM_REMINDER_RULES,
    )
    effective_sop_paths = resolve_effective_sop_paths(cli_sop_paths=args.sop_paths, pack_sop_paths=pack_sop_paths)

    summarization_system_prompt = (
        base_system_prompt if truthy(os.getenv("SWARMEE_CACHE_SAFE_SUMMARY", "false")) else None
    )
    conversation_manager = _build_conversation_manager(
        window_size=args.window_size,
        per_turn=args.context_per_turn,
        summarization_system_prompt=summarization_system_prompt,
    )
    hooks = _build_runtime_hooks(
        args=args,
        safety_settings=settings.safety,
        auto_approve=auto_approve,
        consent_prompt_fn=consent_prompt_fn,
    )

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "system_prompt": base_system_prompt,
        "callback_handler": callback_handler,
        "load_tools_from_directory": not truthy(os.getenv("SWARMEE_FREEZE_TOOLS", "false")),
    }
    if conversation_manager is not None:
        agent_kwargs["conversation_manager"] = conversation_manager
    if hooks:
        agent_kwargs["hooks"] = hooks

    def create_agent(*, messages: Any | None = None, state: Any | None = None) -> Agent:
        kwargs = dict(agent_kwargs)
        if messages is not None:
            kwargs["messages"] = messages
        if state is not None:
            kwargs["state"] = state
        try:
            return Agent(**kwargs)
        except TypeError:
            kwargs.pop("conversation_manager", None)
            kwargs.pop("hooks", None)
            kwargs.pop("messages", None)
            kwargs.pop("state", None)
            return Agent(**kwargs)

    agent = create_agent()
    interrupt_event = threading.Event()

    preflight_prompt_section: str | None = None
    project_map_prompt_section: str | None = None
    artifact_store = ArtifactStore()
    prompt_cache = PromptCacheState()

    def refresh_system_prompt(welcome_text_local: str) -> None:
        prompt_cache.queue_if_changed("project_map", project_map_prompt_section)
        prompt_cache.queue_if_changed("preflight", preflight_prompt_section)
        if args.include_welcome_in_prompt and welcome_text_local:
            prompt_cache.queue_if_changed("welcome", f"Welcome Text Reference:\n{welcome_text_local}")

        if ctx.active_sop_name:
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
            if sop_text:
                prompt_cache.queue_if_changed("active_sop", f"Active SOP ({ctx.active_sop_name}):\n{sop_text}")

    def run_agent(
        query: str,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[Any] | None = None,
        structured_output_prompt: str | None = None,
    ) -> Any:
        nonlocal agent
        _emit_tui_context_event_if_enabled(agent)
        invocation_state = _build_resolved_invocation_state(
            invocation_state=invocation_state,
            runtime_environment=runtime_environment,
            model_manager=model_manager,
            selected_provider=selected_provider,
            settings=settings,
            structured_output_model=structured_output_model,
        )

        interrupt_event.clear()
        set_interrupt_event(interrupt_event)
        with interrupt_watcher_from_env(interrupt_event):
            try:
                system_reminder = prompt_cache.pop_reminder()
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
                configured = os.getenv("SWARMEE_MAX_TOKENS") or os.getenv("STRANDS_MAX_TOKENS") or "(unset)"
                error_msg = (
                    "Error: Response hit the max output token limit.\n"
                    f"- Current max: {configured}\n"
                    "- Fix: increase SWARMEE_MAX_TOKENS (or pass --max-output-tokens), or ask for a shorter response.\n"
                    "- Resetting agent loop so you can continue."
                )
                _emit_tui_event({"event": "error", "text": error_msg})
                if not _tui_events_enabled():
                    print(f"\n{error_msg}")
                agent = create_agent()
                raise
            except Exception as exc:
                should_retry_without_reminder = bool(system_reminder) and _is_context_window_overflow_error(exc)
                if should_retry_without_reminder:
                    warning_text = (
                        "Prompt overflow detected; retrying once without preflight reminder context for this turn."
                    )
                    _emit_tui_event({"event": "warning", "text": warning_text})
                    if not _tui_events_enabled():
                        print(f"\n[warn] {warning_text}")
                    return invoke_agent(
                        agent,
                        query,
                        callback_handler=callback_handler,
                        interrupt_event=interrupt_event,
                        invocation_state=invocation_state,
                        system_reminder=None,
                        structured_output_model=structured_output_model,
                        structured_output_prompt=structured_output_prompt,
                    )
                raise

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
                last_error = exc

            if not _maybe_escalate_tier(attempted=attempted):
                break

        if last_error is not None:
            raise last_error
        raise RuntimeError("Execution failed")

    def _generate_plan(user_request: str) -> WorkPlan:
        def _run_plan_once() -> WorkPlan:
            result = run_agent(
                user_request,
                invocation_state={"swarmee": {"mode": "plan"}},
                structured_output_model=WorkPlan,
                structured_output_prompt=structured_plan_prompt(),
            )
            plan = getattr(result, "structured_output", None)
            if not isinstance(plan, WorkPlan):
                raise ValueError("Structured plan parse failed")
            artifact_store.write_text(
                kind="plan",
                text=plan.model_dump_json(indent=2),
                suffix="json",
                metadata={"request": user_request},
            )
            return plan

        return _retry_with_escalation(
            run_once=_run_plan_once,
            retryable_exceptions=(MaxTokensReachedException, ValueError),
        )

    def _swap_agent(messages: Any | None, state: Any | None) -> None:
        nonlocal agent
        agent = create_agent(messages=messages, state=state)
        ctx.agent = agent

    def _build_session_meta() -> dict[str, Any]:
        return _build_session_meta_payload(
            settings=settings,
            selected_provider=selected_provider,
            current_tier=model_manager.current_tier,
            active_sop_name=ctx.active_sop_name,
        )

    def _execute_with_plan(user_request: str, plan: WorkPlan, *, welcome_text_local: str) -> Any:
        allowed_tools = sorted(tool_name for tool_name in tools_expected_from_plan(plan) if tool_name != "WorkPlan")
        invocation_state = {"swarmee": {"mode": "execute", "enforce_plan": True, "allowed_tools": allowed_tools}}
        plan_json_payload = plan_json_for_execution(plan)

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
            )
        finally:
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
    )
    ctx.active_sop_name = args.sop
    ctx.session_store = SessionStore()

    def _refresh_query_context(*, interactive: bool) -> None:
        nonlocal preflight_prompt_section, project_map_prompt_section
        profile = settings.harness.tier_profiles.get(model_manager.current_tier)
        snapshot = build_context_snapshot(
            artifact_store=artifact_store,
            interactive=interactive,
            default_preflight_level=profile.preflight_level if profile else None,
        )
        preflight_prompt_section = snapshot.preflight_prompt_section
        project_map_prompt_section = snapshot.project_map_prompt_section
        ctx.refresh_system_prompt()

    def _current_model_info_event() -> dict[str, Any]:
        return _build_model_info_event_payload(model_manager=model_manager, selected_provider=selected_provider)

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
        "refresh_query_context": _refresh_query_context,
        "current_model_info_event": _current_model_info_event,
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
    generate_plan: Callable[[str], WorkPlan],
    execute_with_plan: Callable[[str, WorkPlan, str], Any],
    run_agent: Callable[[str], Any],
    classify_intent_fn: Callable[[str], str],
    on_plan: Callable[[WorkPlan], None],
) -> tuple[WorkPlan | None, Any | None, bool]:
    mode = (forced_mode or "").strip().lower()
    if mode == "execute":
        return None, run_agent(query_text), True

    should_plan = mode == "plan" or classify_intent_fn(query_text) == "work"
    if not should_plan:
        return None, run_agent(query_text), True

    plan = generate_plan(query_text)
    on_plan(plan)
    if not auto_approve:
        return plan, None, False

    return plan, execute_with_plan(query_text, plan, welcome_text), True


def main() -> None:
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Swarmee - An enterprise analytics + coding assistant")
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
        help="Max output tokens for the model (sets SWARMEE_MAX_TOKENS, and STRANDS_MAX_TOKENS for Bedrock)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Conversation history window size (overrides SWARMEE_WINDOW_SIZE)",
    )
    parser.add_argument(
        "--context-per-turn",
        type=int,
        default=None,
        help="Max history items to drop per truncation pass (overrides SWARMEE_CONTEXT_PER_TURN)",
    )
    parser.add_argument(
        "--context-manager",
        type=str,
        default=None,
        choices=["summarize", "sliding", "none"],
        help="Context management strategy (overrides SWARMEE_CONTEXT_MANAGER)",
    )
    parser.add_argument(
        "--context-budget-tokens",
        type=int,
        default=None,
        help="Approximate max prompt tokens before summarization (overrides SWARMEE_CONTEXT_BUDGET_TOKENS)",
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
        help="OS-separated directories containing `*.sop.md` files (overrides SWARMEE_SOP_PATHS)",
    )
    args = parser.parse_args()

    # Load .env early for local dev credentials/config.
    load_env_file()

    daemon_session_id: str | None = None
    if args.tui_daemon:
        daemon_session_id = (os.getenv("SWARMEE_SESSION_ID") or "").strip() or uuid.uuid4().hex
        os.environ["SWARMEE_SESSION_ID"] = daemon_session_id

    if args.max_output_tokens is not None:
        os.environ["SWARMEE_MAX_TOKENS"] = str(args.max_output_tokens)
        os.environ["STRANDS_MAX_TOKENS"] = str(args.max_output_tokens)

    if args.context_manager:
        os.environ["SWARMEE_CONTEXT_MANAGER"] = args.context_manager
    if args.context_budget_tokens:
        os.environ["SWARMEE_CONTEXT_BUDGET_TOKENS"] = str(args.context_budget_tokens)

    # Resolve KB from CLI first, then environment.
    knowledge_base_id = (
        args.knowledge_base_id or os.getenv("SWARMEE_KNOWLEDGE_BASE_ID") or os.getenv("STRANDS_KNOWLEDGE_BASE_ID")
    )

    settings = load_settings()
    settings_path_for_project = Path.cwd() / ".swarmee" / "settings.json"
    auto_approve = args.yes or truthy(os.getenv("SWARMEE_AUTO_APPROVE", "false"))
    command = args.query[0].strip().lower() if args.query else ""
    sub = args.query[1:] if len(args.query) > 1 else []

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

    # Session management is project-local under `.swarmee/sessions`.
    if command == "session":
        store = SessionStore()

        def _build_cli_session_meta() -> dict[str, Any]:
            try:
                pm = build_project_map()
                git_root = pm.get("git_root")
            except Exception:
                git_root = None
            return {
                "cwd": str(Path.cwd()),
                "git_root": git_root,
                "provider": (os.getenv("SWARMEE_MODEL_PROVIDER") or settings.models.provider),
                "tier": os.getenv("SWARMEE_MODEL_TIER") or settings.models.default_tier,
            }

        subcmd = (sub[0].lower() if sub else "list").strip()
        usage_text = (
            "Usage: swarmee session list | swarmee session new | "
            "swarmee session info <id> | swarmee session rm <id>"
        )
        output_text, _created_session_id, _deleted_session_id = handle_common_session_command(
            store=store,
            subcmd=subcmd,
            args=sub,
            create_meta=_build_cli_session_meta,
            usage_text=usage_text,
            quote_ids=False,
            new_output_mode="id_only",
            require_info_arg=True,
            current_session_id=None,
        )
        if output_text:
            print(output_text)
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
            print(render_diagnostic_command_for_surface(cmd=cmd, args=sub, cwd=Path.cwd(), surface="cli"))
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

    def _daemon_consent_prompt(text: str) -> str:
        _write_stdout_jsonl({
            "event": "consent_prompt",
            "context": text,
            "options": ["y", "n", "a", "v"],
        })
        _set_daemon_consent_response("")
        daemon_consent_event.clear()
        daemon_consent_event.wait()
        return _consume_daemon_consent_response().strip()

    runtime = _build_agent_runtime(
        args,
        settings,
        auto_approve,
        _daemon_consent_prompt if args.tui_daemon else None,
        knowledge_base_id=knowledge_base_id,
        settings_path_for_project=settings_path_for_project,
    )

    selected_provider = runtime["selected_provider"]
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
    _refresh_query_context = runtime["refresh_query_context"]
    _current_model_info_event = runtime["current_model_info_event"]

    def _best_effort_retrieve(query_text: str, *, warn_on_error: bool) -> None:
        if not knowledge_base_id:
            return
        try:
            ctx.agent.tool.retrieve(text=query_text, knowledgeBaseId=knowledge_base_id)
        except Exception as e:
            if warn_on_error and not _tui_events_enabled():
                print(f"[warn] retrieve failed: {e}")

    def _emit_plan_event(plan: WorkPlan, *, write_jsonl: bool, echo_console: bool) -> str:
        rendered = _render_plan(plan)
        event = {"event": "plan", "plan_json": plan.model_dump(), "rendered": rendered}
        if write_jsonl:
            _write_stdout_jsonl(event)
        else:
            _emit_tui_event(event)
        if echo_console and not _tui_events_enabled():
            print(rendered)
        return rendered

    def _run_tui_daemon() -> None:
        _refresh_query_context(interactive=True)

        resolved_session_id = daemon_session_id or (os.getenv("SWARMEE_SESSION_ID") or "").strip() or uuid.uuid4().hex
        _write_stdout_jsonl({"event": "ready", "session_id": resolved_session_id})
        _write_stdout_jsonl(_current_model_info_event())

        worker_thread: threading.Thread | None = None
        worker_lock = threading.Lock()

        def _worker_is_running() -> bool:
            with worker_lock:
                return worker_thread is not None and worker_thread.is_alive()

        def _emit_turn_error(text: str) -> None:
            _write_stdout_jsonl({"event": "error", "text": text})
            _write_stdout_jsonl({"event": "turn_complete", "exit_status": "error"})

        def _set_daemon_consent_waiting(waiting: bool) -> None:
            _set_daemon_consent_response("")
            if waiting:
                daemon_consent_event.clear()
            else:
                daemon_consent_event.set()

        def _run_query_worker(query_text: str, *, turn_auto_approve: bool, mode: str | None = None) -> None:
            exit_status = "ok"
            response: Any | None = None
            executed = False
            forced_mode = (mode or "").strip().lower()
            if forced_mode not in {"", "plan", "execute"}:
                forced_mode = ""

            try:
                _refresh_query_context(interactive=False)
                _best_effort_retrieve(query_text, warn_on_error=False)

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
                        setattr(ctx, "last_plan", plan),
                        _emit_plan_event(plan, write_jsonl=True, echo_console=False),
                    ),
                )

                if executed and knowledge_base_id:
                    store_conversation_in_kb(ctx.agent, query_text, response, knowledge_base_id)
            except AgentInterruptedError:
                callback_handler(force_stop=True)
                exit_status = "interrupted"
            except MaxTokensReachedException:
                exit_status = "error"
            except Exception as e:
                callback_handler(force_stop=True)
                _write_stdout_jsonl({"event": "error", "text": str(e)})
                exit_status = "error"
            finally:
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
                    _emit_turn_error("A query is already running")
                    continue

                requested_tier = payload.get("tier")
                if isinstance(requested_tier, str) and requested_tier.strip():
                    try:
                        model_manager.set_tier(ctx.agent, requested_tier.strip().lower())
                        agent_kwargs["model"] = ctx.agent.model
                        _refresh_query_context(interactive=True)
                        _write_stdout_jsonl(_current_model_info_event())
                    except Exception as e:
                        _emit_turn_error(str(e))
                        continue

                requested_mode = payload.get("mode")
                mode = requested_mode.strip().lower() if isinstance(requested_mode, str) else None
                raw_auto_approve = payload.get("auto_approve")
                turn_auto_approve = raw_auto_approve if isinstance(raw_auto_approve, bool) else auto_approve

                interrupt_event.clear()
                _set_daemon_consent_waiting(True)

                with worker_lock:
                    worker_thread = threading.Thread(
                        target=_run_query_worker,
                        kwargs={
                            "query_text": query_text.strip(),
                            "turn_auto_approve": turn_auto_approve,
                            "mode": mode,
                        },
                        daemon=True,
                    )
                    worker_thread.start()
                continue

            if cmd == "consent_response":
                choice = payload.get("choice")
                _set_daemon_consent_response(choice.strip().lower() if isinstance(choice, str) else "")
                daemon_consent_event.set()
                continue

            if cmd == "interrupt":
                interrupt_event.set()
                _set_daemon_consent_waiting(False)
                continue

            if cmd == "set_tier":
                tier = payload.get("tier")
                if not isinstance(tier, str) or not tier.strip():
                    _write_stdout_jsonl({"event": "error", "text": "set_tier.tier is required"})
                    continue
                if _worker_is_running():
                    _write_stdout_jsonl({"event": "error", "text": "Cannot set tier while a query is running"})
                    continue
                try:
                    model_manager.set_tier(ctx.agent, tier.strip().lower())
                    agent_kwargs["model"] = ctx.agent.model
                    _refresh_query_context(interactive=True)
                    _write_stdout_jsonl(_current_model_info_event())
                except Exception as e:
                    _write_stdout_jsonl({"event": "error", "text": str(e)})
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

    if args.tui_daemon:
        _run_tui_daemon()
        return

    # Query mode (one-shot) or interactive REPL mode.
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
                    setattr(ctx, "last_plan", p),
                    _emit_plan_event(p, write_jsonl=False, echo_console=True),
                ),
            )
            if plan is not None and not executed:
                if not _tui_events_enabled():
                    print("\nPlan generated. Re-run with --yes (or set SWARMEE_AUTO_APPROVE=true) to execute.")
                return
        except AgentInterruptedError as e:
            _handle_agent_interrupt(e)
            return
        except MaxTokensReachedException:
            return

        if knowledge_base_id and executed:
            # Persist the executed exchange when KB is configured.
            store_conversation_in_kb(ctx.agent, query, response, knowledge_base_id)
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
