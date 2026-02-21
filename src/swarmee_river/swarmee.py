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
from pathlib import Path
from typing import Any, Callable, Optional

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
from swarmee_river.cli.commands import CLIContext, CommandRegistry
from swarmee_river.cli.diagnostics import (
    render_artifact_get,
    render_artifact_list,
    render_effective_config,
    render_git_diff,
    render_git_status,
    render_log_tail,
    render_replay_invocation,
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
from swarmee_river.settings import PackEntry, PacksConfig, SwarmeeSettings, load_settings, save_settings
from swarmee_river.tools import get_tools
from swarmee_river.utils import model_utils
from swarmee_river.utils.env_utils import load_env_file
from swarmee_river.utils.kb_utils import load_system_prompt, store_conversation_in_kb
from swarmee_river.utils.provider_utils import resolve_model_provider
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


_consent_prompt_session: Any | None = None
_consent_prompt_lock = threading.Lock()
_consent_console: Any | None = Console() if Console is not None else None
_stdout_jsonl_lock = threading.Lock()


def _configure_stdio_for_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")


def _write_stdout_jsonl(event: dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with _stdout_jsonl_lock:
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
            return
        except UnicodeEncodeError:
            pass

        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            with contextlib.suppress(Exception):
                buffer.write(line.encode("utf-8", errors="replace"))
                buffer.flush()
                return

        with contextlib.suppress(Exception):
            sys.stdout.write(line.encode("ascii", errors="replace").decode("ascii"))
            sys.stdout.flush()


_configure_stdio_for_utf8()


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _tui_events_enabled() -> bool:
    """True when running as a subprocess spawned by the TUI."""
    return _truthy(os.getenv("SWARMEE_TUI_EVENTS"))


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


def _plan_json_for_execution(plan: WorkPlan) -> str:
    payload = plan.model_dump(exclude={"confirmation_prompt"})
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _build_conversation_manager(
    *,
    window_size: Optional[int],
    per_turn: Optional[int],
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
    should_truncate_results = _truthy(os.getenv("SWARMEE_TRUNCATE_RESULTS", "true"))

    return SlidingWindowConversationManager(
        window_size=resolved_window_size,
        should_truncate_results=should_truncate_results,
        per_turn=resolved_per_turn,
    )


def _build_agent_runtime(
    args: argparse.Namespace,
    settings: SwarmeeSettings,
    auto_approve: bool,
    consent_prompt_fn: Callable[[str], str] | None,
    *,
    knowledge_base_id: str | None = None,
    settings_path_for_project: Path | None = None,
) -> dict[str, Any]:
    selected_provider, provider_notice = resolve_model_provider(
        cli_provider=args.model_provider.stem if args.model_provider is not None else None,
        env_provider=os.getenv("SWARMEE_MODEL_PROVIDER"),
        settings_provider=settings.models.provider,
    )

    model_manager = SessionModelManager(settings, fallback_provider=selected_provider)
    model_manager.set_fallback_config(args.model_config)
    model = model_manager.build_model()
    runtime_environment = detect_runtime_environment(cwd=Path.cwd())
    runtime_environment_prompt_section = render_runtime_environment_section(runtime_environment)

    tools_dict = get_tools()
    for name, tool_obj in load_enabled_pack_tools(settings).items():
        tools_dict.setdefault(name, tool_obj)
    tools = [tools_dict[name] for name in sorted(tools_dict)]

    pack_sop_paths = enabled_sop_paths(settings)
    pack_prompt_sections = enabled_system_prompts(settings)

    raw_system_prompt = load_system_prompt()
    base_prompt_parts: list[str] = [raw_system_prompt, _TOOL_USAGE_RULES, _SYSTEM_REMINDER_RULES]
    if runtime_environment_prompt_section:
        base_prompt_parts.append(runtime_environment_prompt_section)
    if pack_prompt_sections:
        base_prompt_parts.extend(pack_prompt_sections)
    base_system_prompt = "\n\n".join([p for p in base_prompt_parts if p]).strip()

    effective_sop_paths: str | None = args.sop_paths
    if pack_sop_paths:
        pack_paths_str = os.pathsep.join(str(p) for p in pack_sop_paths)
        effective_sop_paths = (
            pack_paths_str if not effective_sop_paths else os.pathsep.join([effective_sop_paths, pack_paths_str])
        )

    summarization_system_prompt = (
        base_system_prompt if _truthy(os.getenv("SWARMEE_CACHE_SAFE_SUMMARY", "false")) else None
    )
    conversation_manager = _build_conversation_manager(
        window_size=args.window_size,
        per_turn=args.context_per_turn,
        summarization_system_prompt=summarization_system_prompt,
    )

    hooks = []
    if _HAS_STRANDS_HOOKS:

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
            ToolPolicyHooks(settings.safety),
            ToolConsentHooks(
                settings.safety,
                interactive=not bool(args.query),
                auto_approve=auto_approve,
                prompt=_consent_prompt,
            ),
            ToolResultLimiterHooks(),
        ]
        if ToolMessageRepairHooks is not None:
            hooks.insert(2, ToolMessageRepairHooks())

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "system_prompt": base_system_prompt,
        "callback_handler": callback_handler,
        "load_tools_from_directory": not _truthy(os.getenv("SWARMEE_FREEZE_TOOLS", "false")),
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
        if _tui_events_enabled():
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
        invocation_state = resolved_state

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

    def _render_plan(plan: WorkPlan) -> str:
        lines: list[str] = ["\nProposed plan:", f"- Summary: {plan.summary}"]
        if plan.assumptions:
            lines.append("- Assumptions:")
            lines.extend([f"  - {a}" for a in plan.assumptions])
        if plan.questions:
            lines.append("- Questions:")
            lines.extend([f"  - {q}" for q in plan.questions])
        if plan.steps:
            lines.append("- Steps:")
            for i, step in enumerate(plan.steps, start=1):
                lines.append(f"  {i}. {step.description}")
                if step.files_to_read:
                    lines.append(f"     - read: {', '.join(step.files_to_read)}")
                if step.files_to_edit:
                    lines.append(f"     - edit: {', '.join(step.files_to_edit)}")
                if step.tools_expected:
                    lines.append(f"     - tools: {', '.join(step.tools_expected)}")
                if step.commands_expected:
                    lines.append(f"     - cmds: {', '.join(step.commands_expected)}")
                if step.risks:
                    lines.append(f"     - risks: {', '.join(step.risks)}")
        return "\n".join(lines).strip()

    def _generate_plan(user_request: str) -> WorkPlan:
        attempted: set[str] = set()
        max_attempts = max(1, model_manager.max_escalations_per_task + 1)
        last_error: Exception | None = None

        for _attempt in range(max_attempts):
            try:
                result = run_agent(
                    user_request,
                    invocation_state={"swarmee": {"mode": "plan"}},
                    structured_output_model=WorkPlan,
                    structured_output_prompt=structured_plan_prompt(),
                )
                plan = getattr(result, "structured_output", None)
                if isinstance(plan, WorkPlan):
                    artifact_store.write_text(
                        kind="plan",
                        text=plan.model_dump_json(indent=2),
                        suffix="json",
                        metadata={"request": user_request},
                    )
                    return plan
                last_error = ValueError("Structured plan parse failed")
            except MaxTokensReachedException as e:
                last_error = e

            prev = model_manager.current_tier
            if not model_manager.maybe_escalate(agent, attempted=attempted):
                break
            if model_manager.current_tier != prev and not _tui_events_enabled():
                print(f"\n[auto-escalation] tier: {prev} -> {model_manager.current_tier}")
            agent_kwargs["model"] = agent.model

        raise last_error or ValueError("Failed to generate plan")

    def _swap_agent(messages: Any | None, state: Any | None) -> None:
        nonlocal agent
        agent = create_agent(messages=messages, state=state)
        ctx.agent = agent

    def _build_session_meta() -> dict[str, Any]:
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
            "tier": model_manager.current_tier,
            "packs": enabled_packs,
            "active_sop": ctx.active_sop_name,
        }

    def _execute_with_plan(user_request: str, plan: WorkPlan, *, welcome_text_local: str) -> Any:
        allowed_tools = sorted(tool_name for tool_name in tools_expected_from_plan(plan) if tool_name != "WorkPlan")
        invocation_state = {"swarmee": {"mode": "execute", "enforce_plan": True, "allowed_tools": allowed_tools}}
        plan_json_for_execution = _plan_json_for_execution(plan)

        approved_plan_section = (
            "Approved Plan (execute ONLY this plan; if you need changes, ask to :replan):\n" + plan_json_for_execution
        )
        prompt_cache.queue_one_off(approved_plan_section)
        refresh_system_prompt(welcome_text_local)
        try:
            attempted: set[str] = set()
            max_attempts = max(1, model_manager.max_escalations_per_task + 1)
            last_error: Exception | None = None

            for _attempt in range(max_attempts):
                try:
                    result = run_agent(user_request, invocation_state=invocation_state)
                    if knowledge_base_id:
                        try:
                            agent.tool.store_in_kb(
                                content=(f"Approved plan for request:\n{user_request}\n\n{plan_json_for_execution}\n"),
                                title=f"Plan: {user_request[:50]}{'...' if len(user_request) > 50 else ''}",
                                knowledge_base_id=knowledge_base_id,
                                record_direct_tool_call=False,
                            )
                        except Exception:
                            pass
                    return result
                except AgentInterruptedError:
                    raise
                except MaxTokensReachedException as e:
                    last_error = e
                except Exception as e:
                    last_error = e

                prev = model_manager.current_tier
                if not model_manager.maybe_escalate(agent, attempted=attempted):
                    break
                if model_manager.current_tier != prev and not _tui_events_enabled():
                    print(f"\n[auto-escalation] tier: {prev} -> {model_manager.current_tier}")
                agent_kwargs["model"] = agent.model

            raise last_error or RuntimeError("Execution failed")
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
        render_plan=_render_plan,
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

    return {
        "selected_provider": selected_provider,
        "provider_notice": provider_notice,
        "model_manager": model_manager,
        "agent_kwargs": agent_kwargs,
        "interrupt_event": interrupt_event,
        "create_agent": create_agent,
        "run_agent": run_agent,
        "render_plan": _render_plan,
        "generate_plan": _generate_plan,
        "execute_with_plan": _execute_with_plan,
        "ctx": ctx,
        "registry": registry,
        "refresh_query_context": _refresh_query_context,
        "current_model_info_event": _current_model_info_event,
    }


def main() -> None:
    # Parse command line arguments
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

    # Load .env early (useful for OPENAI_API_KEY and similar local dev settings)
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

    # Get knowledge_base_id from args or environment variable
    knowledge_base_id = (
        args.knowledge_base_id or os.getenv("SWARMEE_KNOWLEDGE_BASE_ID") or os.getenv("STRANDS_KNOWLEDGE_BASE_ID")
    )

    settings = load_settings()
    settings_path_for_project = Path.cwd() / ".swarmee" / "settings.json"
    auto_approve = args.yes or _truthy(os.getenv("SWARMEE_AUTO_APPROVE", "false"))

    # Optional full-screen Textual UI (keep normal CLI path unchanged unless explicitly requested).
    if args.query and args.query[0].strip().lower() == "tui":
        from swarmee_river.tui.app import run_tui

        raise SystemExit(run_tui())

    # Pack management is intentionally CLI-first: treat `swarmee pack ...` as a command.
    if args.query and args.query[0].strip().lower() == "pack":
        sub = args.query[1:] if len(args.query) > 1 else []

        settings_path = Path.cwd() / ".swarmee" / "settings.json"

        def _persist_packs(installed: list[PackEntry]) -> None:
            new_settings = SwarmeeSettings(
                models=settings.models,
                safety=settings.safety,
                packs=PacksConfig(installed=installed),
                harness=settings.harness,
                raw=settings.raw,
            )
            save_settings(new_settings, settings_path)

        if not sub or sub[0] in {"list", "ls"}:
            lines = ["# Packs", ""]
            if not settings.packs.installed:
                lines.append("No packs installed.")
            else:
                for pack_entry in settings.packs.installed:
                    status = "enabled" if pack_entry.enabled else "disabled"
                    lines.append(f"- {pack_entry.name} ({status}) -> {pack_entry.path}")
            print("\n".join(lines))
            return

        if sub[0] == "install" and len(sub) >= 2:
            pack_path = sub[1]
            if pack_path.startswith("s3://"):
                print("S3 pack install is not implemented yet. Use a local path for now.")
                return
            pack_dir = Path(pack_path).expanduser()
            if not pack_dir.exists() or not pack_dir.is_dir():
                print(f"Pack path not found: {pack_dir}")
                return
            meta_path = pack_dir / "pack.json"
            name = pack_dir.name
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(meta.get("name"), str) and meta["name"].strip():
                        name = meta["name"].strip()
                except Exception:
                    pass
            installed_packs = [pack_entry for pack_entry in settings.packs.installed if pack_entry.name != name]
            installed_packs.append(PackEntry(name=name, path=str(pack_dir.resolve()), enabled=True))
            _persist_packs(installed_packs)
            print(f"Installed pack: {name} -> {pack_dir.resolve()}")
            return

        if sub[0] in {"enable", "disable"} and len(sub) >= 2:
            target = sub[1].strip()
            if not target:
                print("Pack name is required.")
                return
            updated: list[PackEntry] = []
            found = False
            for pack_entry in settings.packs.installed:
                if pack_entry.name == target:
                    updated.append(PackEntry(name=pack_entry.name, path=pack_entry.path, enabled=(sub[0] == "enable")))
                    found = True
                else:
                    updated.append(pack_entry)
            if not found:
                print(f"Pack not found: {target}")
                return
            _persist_packs(updated)
            print(f"{'Enabled' if sub[0] == 'enable' else 'Disabled'} pack: {target}")
            return

        print("Usage: swarmee pack list | swarmee pack install <path> | swarmee pack enable|disable <name>")
        return

    # Session management is CLI-first as well (project-local under .swarmee/sessions).
    if args.query and args.query[0].strip().lower() == "session":
        sub = args.query[1:] if len(args.query) > 1 else []
        store = SessionStore()

        if not sub or sub[0] in {"list", "ls"}:
            entries = store.list()
            if not entries:
                print("No sessions found.")
                return
            lines = ["# Sessions", ""]
            for entry in entries:
                sid = str(entry.get("id") or "")
                updated_at = str(entry.get("updated_at") or entry.get("created_at") or "")
                provider = str(entry.get("provider") or "")
                tier = str(entry.get("tier") or "")
                suffix = " ".join([part for part in [updated_at, provider, tier] if part]).strip()
                lines.append(f"- {sid}" + (f" ({suffix})" if suffix else ""))
            print("\n".join(lines))
            return

        if sub[0] == "new":
            try:
                pm = build_project_map()
                git_root = pm.get("git_root")
            except Exception:
                git_root = None
            meta = {
                "cwd": str(Path.cwd()),
                "git_root": git_root,
                "provider": (os.getenv("SWARMEE_MODEL_PROVIDER") or settings.models.provider),
                "tier": os.getenv("SWARMEE_MODEL_TIER") or settings.models.default_tier,
            }
            sid = store.create(meta=meta)
            print(sid)
            return

        if sub[0] in {"rm", "delete"} and len(sub) >= 2:
            sid = sub[1].strip()
            store.delete(sid)
            print(f"Deleted session: {sid}")
            return

        if sub[0] == "info" and len(sub) >= 2:
            sid = sub[1].strip()
            meta = store.read_meta(sid)
            print(json.dumps(meta, indent=2, ensure_ascii=False))
            return

        print("Usage: swarmee session list | swarmee session new | swarmee session info <id> | swarmee session rm <id>")
        return

    # Lightweight one-shot commands (do not require model invocation).
    if args.query and args.query[0].strip().lower() in {"config", "status", "diff", "artifact", "log", "replay"}:
        cmd = args.query[0].strip().lower()
        sub = args.query[1:] if len(args.query) > 1 else []

        if cmd == "config":
            subcmd = sub[0].strip().lower() if sub else "show"
            if subcmd != "show":
                print("Usage: swarmee config show")
                return

            selected_provider, provider_notice = resolve_model_provider(
                cli_provider=args.model_provider.stem if args.model_provider is not None else None,
                env_provider=os.getenv("SWARMEE_MODEL_PROVIDER"),
                settings_provider=settings.models.provider,
            )
            if provider_notice:
                print(f"[provider] {provider_notice}")

            model_manager = SessionModelManager(settings, fallback_provider=selected_provider)
            model_manager.set_fallback_config(args.model_config)

            print(
                render_effective_config(
                    cwd=Path.cwd(),
                    settings_path=settings_path_for_project,
                    settings=settings,
                    selected_provider=selected_provider,
                    model_manager=model_manager,
                    knowledge_base_id=knowledge_base_id,
                    effective_sop_paths=args.sop_paths,
                    auto_approve=auto_approve,
                )
            )
            return

        if cmd == "status":
            print(render_git_status(cwd=Path.cwd()))
            return

        if cmd == "diff":
            staged = False
            paths: list[str] = []
            for item in sub:
                if item in {"--staged", "--cached"}:
                    staged = True
                elif item.startswith("-"):
                    continue
                else:
                    paths.append(item)
            print(render_git_diff(cwd=Path.cwd(), staged=staged, paths=paths or None))
            return

        if cmd == "artifact":
            subcmd = sub[0].strip().lower() if sub else "list"
            if subcmd in {"list", "ls"}:
                print(render_artifact_list(cwd=Path.cwd()))
                return
            if subcmd == "get":
                artifact_id = sub[1].strip() if len(sub) >= 2 else None
                if not artifact_id:
                    print("Usage: swarmee artifact get <artifact_id>")
                    return
                print(render_artifact_get(cwd=Path.cwd(), artifact_id=artifact_id, path=None))
                return
            print("Usage: swarmee artifact list | swarmee artifact get <artifact_id>")
            return

        if cmd == "log":
            subcmd = sub[0].strip().lower() if sub else "tail"
            if subcmd != "tail":
                print("Usage: swarmee log tail [--lines N]")
                return
            n = 50
            if "--lines" in sub:
                try:
                    idx = sub.index("--lines")
                    n = int(sub[idx + 1])
                except Exception:
                    n = 50
            print(render_log_tail(cwd=Path.cwd(), lines=n))
            return

        if cmd == "replay":
            if not sub:
                print("Usage: swarmee replay <invocation_id>")
                return
            print(render_replay_invocation(cwd=Path.cwd(), invocation_id=sub[0].strip()))
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

        def _run_query_worker(query_text: str, *, turn_auto_approve: bool, mode: str | None = None) -> None:
            exit_status = "ok"
            response: Any | None = None
            executed = False
            forced_mode = (mode or "").strip().lower()
            if forced_mode not in {"", "plan", "execute"}:
                forced_mode = ""

            try:
                _refresh_query_context(interactive=False)

                if knowledge_base_id:
                    try:
                        ctx.agent.tool.retrieve(text=query_text, knowledgeBaseId=knowledge_base_id)
                    except Exception:
                        pass

                if forced_mode == "plan":
                    plan = _generate_plan(query_text)
                    ctx.last_plan = plan
                    _write_stdout_jsonl({
                        "event": "plan",
                        "plan_json": plan.model_dump(),
                        "rendered": _render_plan(plan),
                    })
                    if turn_auto_approve:
                        response = _execute_with_plan(query_text, plan, welcome_text_local=ctx.welcome_text)
                        executed = True
                elif forced_mode == "execute":
                    response = run_agent(query_text, invocation_state={"swarmee": {"mode": "execute"}})
                    executed = True
                else:
                    intent = classify_intent(query_text)
                    if intent == "work":
                        plan = _generate_plan(query_text)
                        ctx.last_plan = plan
                        _write_stdout_jsonl({
                            "event": "plan",
                            "plan_json": plan.model_dump(),
                            "rendered": _render_plan(plan),
                        })
                        if turn_auto_approve:
                            response = _execute_with_plan(query_text, plan, welcome_text_local=ctx.welcome_text)
                            executed = True
                    else:
                        response = run_agent(query_text, invocation_state={"swarmee": {"mode": "execute"}})
                        executed = True

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
                    _write_stdout_jsonl({"event": "error", "text": "query.text is required"})
                    _write_stdout_jsonl({"event": "turn_complete", "exit_status": "error"})
                    continue
                if _worker_is_running():
                    _write_stdout_jsonl({"event": "error", "text": "A query is already running"})
                    _write_stdout_jsonl({"event": "turn_complete", "exit_status": "error"})
                    continue

                requested_mode = payload.get("mode")
                mode = requested_mode.strip().lower() if isinstance(requested_mode, str) else None
                raw_auto_approve = payload.get("auto_approve")
                turn_auto_approve = raw_auto_approve if isinstance(raw_auto_approve, bool) else auto_approve

                interrupt_event.clear()
                _set_daemon_consent_response("")
                daemon_consent_event.clear()

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
                _set_daemon_consent_response("")
                daemon_consent_event.set()
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
                _set_daemon_consent_response("")
                daemon_consent_event.set()
                break

            _write_stdout_jsonl({"event": "warning", "text": f"Unknown daemon cmd: {cmd}"})

        with worker_lock:
            active_worker = worker_thread
        if active_worker is not None and active_worker.is_alive():
            active_worker.join(timeout=2.0)

    if args.tui_daemon:
        _run_tui_daemon()
        return

    # Process query or enter interactive mode
    if args.query:
        query = " ".join(args.query)
        # Use retrieve if knowledge_base_id is defined
        if knowledge_base_id:
            try:
                ctx.agent.tool.retrieve(text=query, knowledgeBaseId=knowledge_base_id)
            except Exception as e:
                # Retrieval is best-effort; missing tool / missing AWS creds shouldn't block the session.
                if not _tui_events_enabled():
                    print(f"[warn] retrieve failed: {e}")

        _refresh_query_context(interactive=False)

        intent = classify_intent(query)
        if intent == "work":
            try:
                plan = _generate_plan(query)
            except AgentInterruptedError as e:
                callback_handler(force_stop=True)
                if not _tui_events_enabled():
                    print(f"\n{str(e)}")
                return
            ctx.last_plan = plan
            rendered_plan = _render_plan(plan)
            _emit_tui_event({
                "event": "plan",
                "plan_json": plan.model_dump(),
                "rendered": rendered_plan,
            })
            if not _tui_events_enabled():
                print(rendered_plan)
            if not auto_approve:
                if not _tui_events_enabled():
                    print("\nPlan generated. Re-run with --yes (or set SWARMEE_AUTO_APPROVE=true) to execute.")
                return
            try:
                response = _execute_with_plan(query, plan, welcome_text_local="")
            except AgentInterruptedError as e:
                callback_handler(force_stop=True)
                if not _tui_events_enabled():
                    print(f"\n{str(e)}")
                return
            except MaxTokensReachedException:
                return
        else:
            try:
                response = run_agent(query, invocation_state={"swarmee": {"mode": "execute"}})
            except AgentInterruptedError as e:
                callback_handler(force_stop=True)
                if not _tui_events_enabled():
                    print(f"\n{str(e)}")
                return
            except MaxTokensReachedException:
                return

        if knowledge_base_id:
            # Store conversation in knowledge base
            store_conversation_in_kb(ctx.agent, query, response, knowledge_base_id)
    else:
        # Display welcome text at startup.
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
