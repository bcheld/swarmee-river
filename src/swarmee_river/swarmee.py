#!/usr/bin/env python3
"""
Swarmee - A minimal CLI interface for Swarmee River (built on Strands)
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

# Strands
from strands import Agent
from strands_tools.utils.user_input import get_user_input
from strands.types.exceptions import MaxTokensReachedException

from swarmee_river.handlers.callback_handler import callback_handler, set_interrupt_event
try:
    from swarmee_river.hooks.jsonl_logger import JSONLLoggerHooks
    from swarmee_river.hooks.tool_consent import ToolConsentHooks
    from swarmee_river.hooks.tool_result_limiter import ToolResultLimiterHooks
    from swarmee_river.hooks.tool_policy import ToolPolicyHooks

    _HAS_STRANDS_HOOKS = True
except Exception:
    JSONLLoggerHooks = None  # type: ignore[assignment]
    ToolConsentHooks = None  # type: ignore[assignment]
    ToolResultLimiterHooks = None  # type: ignore[assignment]
    ToolPolicyHooks = None  # type: ignore[assignment]
    _HAS_STRANDS_HOOKS = False
from swarmee_river.interrupts import AgentInterruptedError, interrupt_watcher_from_env
from swarmee_river.intent import classify_intent
from swarmee_river.planning import WorkPlan, structured_plan_prompt
from swarmee_river.project_map import build_project_map, render_project_map_summary, save_project_map
from swarmee_river.packs import enabled_sop_paths, enabled_system_prompts, load_enabled_pack_tools
from swarmee_river.session.models import SessionModelManager
from swarmee_river.settings import PackEntry, PacksConfig, SwarmeeSettings, load_settings, save_settings
from swarmee_river.tools import get_tools
from swarmee_river.artifacts import ArtifactStore, tools_expected_from_plan
from swarmee_river.utils import model_utils
from swarmee_river.utils.env_utils import load_env_file
from swarmee_river.utils.kb_utils import load_system_prompt, store_conversation_in_kb
from swarmee_river.utils.welcome_utils import render_goodbye_message, render_welcome_message

os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _build_conversation_manager(*, window_size: Optional[int], per_turn: Optional[int]) -> Any:
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


def main():
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

    if args.max_output_tokens is not None:
        os.environ["SWARMEE_MAX_TOKENS"] = str(args.max_output_tokens)
        os.environ["STRANDS_MAX_TOKENS"] = str(args.max_output_tokens)

    if args.context_manager:
        os.environ["SWARMEE_CONTEXT_MANAGER"] = args.context_manager
    if args.context_budget_tokens:
        os.environ["SWARMEE_CONTEXT_BUDGET_TOKENS"] = str(args.context_budget_tokens)

    # Get knowledge_base_id from args or environment variable
    knowledge_base_id = (
        args.knowledge_base_id
        or os.getenv("SWARMEE_KNOWLEDGE_BASE_ID")
        or os.getenv("STRANDS_KNOWLEDGE_BASE_ID")
    )

    settings = load_settings()
    auto_approve = args.yes or _truthy(os.getenv("SWARMEE_AUTO_APPROVE", "false"))

    # Pack management is intentionally CLI-first: treat `swarmee pack ...` as a command.
    if args.query and args.query[0].strip().lower() == "pack":
        sub = args.query[1:] if len(args.query) > 1 else []

        settings_path = Path.cwd() / ".swarmee" / "settings.json"

        def _persist_packs(installed: list[PackEntry]) -> None:
            new_settings = SwarmeeSettings(
                models=settings.models,
                safety=settings.safety,
                packs=PacksConfig(installed=installed),
                raw=settings.raw,
            )
            save_settings(new_settings, settings_path)
        if not sub or sub[0] in {"list", "ls"}:
            lines = ["# Packs", ""]
            if not settings.packs.installed:
                lines.append("No packs installed.")
            else:
                for p in settings.packs.installed:
                    status = "enabled" if p.enabled else "disabled"
                    lines.append(f"- {p.name} ({status}) -> {p.path}")
            print("\n".join(lines))
            return

        if sub[0] == "install" and len(sub) >= 2:
            pack_path = sub[1]
            if pack_path.startswith("s3://"):
                print("S3 pack install is not implemented yet. Use a local path for now.")
                return
            p = Path(pack_path).expanduser()
            if not p.exists() or not p.is_dir():
                print(f"Pack path not found: {p}")
                return
            meta_path = p / "pack.json"
            name = p.name
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(meta.get("name"), str) and meta["name"].strip():
                        name = meta["name"].strip()
                except Exception:
                    pass
            installed = [e for e in settings.packs.installed if e.name != name]
            installed.append(PackEntry(name=name, path=str(p.resolve()), enabled=True))
            _persist_packs(installed)
            print(f"Installed pack: {name} -> {p.resolve()}")
            return

        if sub[0] in {"enable", "disable"} and len(sub) >= 2:
            target = sub[1].strip()
            if not target:
                print("Pack name is required.")
                return
            updated: list[PackEntry] = []
            found = False
            for e in settings.packs.installed:
                if e.name == target:
                    updated.append(PackEntry(name=e.name, path=e.path, enabled=(sub[0] == "enable")))
                    found = True
                else:
                    updated.append(e)
            if not found:
                print(f"Pack not found: {target}")
                return
            _persist_packs(updated)
            print(f"{'Enabled' if sub[0] == 'enable' else 'Disabled'} pack: {target}")
            return

        print("Usage: swarmee pack list | swarmee pack install <path> | swarmee pack enable|disable <name>")
        return

    if args.model_provider is not None:
        selected_provider = args.model_provider.stem
    elif settings.models.provider:
        selected_provider = settings.models.provider
    elif os.getenv("OPENAI_API_KEY"):
        selected_provider = "openai"
    else:
        selected_provider = "bedrock"

    model_manager = SessionModelManager(settings, fallback_provider=selected_provider)
    model_manager.set_fallback_config(args.model_config)
    model = model_manager.build_model()

    # Load system prompt
    system_prompt = load_system_prompt()

    # Base tools + enabled pack tools (pack tools never override core tools)
    tools_dict = get_tools()
    for name, tool_obj in load_enabled_pack_tools(settings).items():
        tools_dict.setdefault(name, tool_obj)

    tools = tools_dict.values()

    pack_sop_paths = enabled_sop_paths(settings)
    pack_prompt_sections = enabled_system_prompts(settings)

    effective_sop_paths: str | None = args.sop_paths
    if pack_sop_paths:
        pack_paths_str = os.pathsep.join(str(p) for p in pack_sop_paths)
        effective_sop_paths = (
            pack_paths_str
            if not effective_sop_paths
            else os.pathsep.join([effective_sop_paths, pack_paths_str])
        )

    conversation_manager = _build_conversation_manager(window_size=args.window_size, per_turn=args.context_per_turn)

    hooks = []
    if _HAS_STRANDS_HOOKS:
        def _consent_prompt(text: str) -> str:
            return get_user_input(text, default="", keyboard_interrupt_return_default=True)

        hooks = [
            JSONLLoggerHooks(),  # type: ignore[misc]
            ToolPolicyHooks(),  # type: ignore[misc]
            ToolConsentHooks(  # type: ignore[misc]
                settings.safety,
                interactive=not bool(args.query),
                auto_approve=auto_approve,
                prompt=_consent_prompt,
            ),
            ToolResultLimiterHooks(),  # type: ignore[misc]
        ]

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "system_prompt": system_prompt,
        "callback_handler": callback_handler,
        "load_tools_from_directory": True,
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
            # Backwards compatibility for older Strands versions.
            kwargs.pop("conversation_manager", None)
            kwargs.pop("hooks", None)
            kwargs.pop("messages", None)
            kwargs.pop("state", None)
            return Agent(**kwargs)

    agent = create_agent()

    interrupt_event = threading.Event()

    active_sop_name: str | None = args.sop
    preflight_prompt_section: str | None = None
    project_map_prompt_section: str | None = None
    active_plan_prompt_section: str | None = None

    artifact_store = ArtifactStore()

    def refresh_system_prompt(welcome_text_local: str) -> None:
        base_system_prompt = load_system_prompt()
        parts: list[str] = [base_system_prompt]

        if pack_prompt_sections:
            parts.extend(pack_prompt_sections)

        if project_map_prompt_section:
            parts.append(project_map_prompt_section)

        if preflight_prompt_section:
            parts.append(preflight_prompt_section)

        if args.include_welcome_in_prompt and welcome_text_local:
            parts.append(f"Welcome Text Reference:\n{welcome_text_local}")

        if active_sop_name:
            sop_text = ""
            try:
                sop_result = agent.tool.sop(
                    action="get",
                    name=active_sop_name,
                    sop_paths=effective_sop_paths,
                    record_direct_tool_call=False,
                )
                if sop_result.get("status") == "success":
                    sop_text = sop_result.get("content", [{}])[0].get("text", "")
            except Exception:
                sop_text = ""
            if sop_text:
                parts.append(f"Active SOP:\n{sop_text}")

        if active_plan_prompt_section:
            parts.append(active_plan_prompt_section)

        agent.system_prompt = "\n\n".join(parts).strip()

    def run_agent(
        query: str,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[Any] | None = None,
        structured_output_prompt: str | None = None,
    ) -> Any:
        nonlocal agent
        interrupt_event.clear()
        set_interrupt_event(interrupt_event)
        with interrupt_watcher_from_env(interrupt_event):
            try:
                async def _invoke() -> Any:
                    loop = asyncio.get_running_loop()
                    previous_handler = loop.get_exception_handler()

                    def _exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
                        if interrupt_event.is_set():
                            message = str(context.get("message") or "")
                            exc = context.get("exception")
                            if message.startswith("an error occurred during closing of asynchronous generator"):
                                return
                            if isinstance(exc, RuntimeError) and "athrow(): asynchronous generator is already running" in str(
                                exc
                            ):
                                return
                        if previous_handler:
                            previous_handler(loop, context)
                        else:
                            loop.default_exception_handler(context)

                    loop.set_exception_handler(_exception_handler)

                    class _OtelDetachFilter(logging.Filter):
                        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
                            if interrupt_event.is_set() and str(record.getMessage()).startswith("Failed to detach context"):
                                return False
                            return True

                    otel_logger = logging.getLogger("opentelemetry.context")
                    otel_filter = _OtelDetachFilter()
                    otel_logger.addFilter(otel_filter)

                    current_task = asyncio.current_task()

                    async def _canceller() -> None:
                        while not interrupt_event.is_set():
                            await asyncio.sleep(0.05)
                        callback_handler(force_stop=True)
                        if current_task:
                            current_task.cancel()

                    canceller_task = asyncio.create_task(_canceller())
                    try:
                        return await agent.invoke_async(
                            query,
                            invocation_state=invocation_state,
                            structured_output_model=structured_output_model,
                            structured_output_prompt=structured_output_prompt,
                        )
                    except asyncio.CancelledError as e:
                        if interrupt_event.is_set():
                            raise AgentInterruptedError("Interrupted by user (Esc)") from e
                        raise
                    finally:
                        canceller_task.cancel()
                        with contextlib.suppress(BaseException):
                            await canceller_task
                        otel_logger.removeFilter(otel_filter)
                        loop.set_exception_handler(previous_handler)

                return asyncio.run(_invoke())
            except MaxTokensReachedException:
                callback_handler(force_stop=True)
                configured = os.getenv("SWARMEE_MAX_TOKENS") or os.getenv("STRANDS_MAX_TOKENS") or "(unset)"
                print(
                    "\nError: Response hit the max output token limit.\n"
                    f"- Current max: {configured}\n"
                    "- Fix: increase SWARMEE_MAX_TOKENS (or pass --max-output-tokens), or ask for a shorter response.\n"
                    "- Resetting agent loop so you can continue."
                )
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
            if model_manager.current_tier != prev:
                print(f"\n[auto-escalation] tier: {prev} -> {model_manager.current_tier}")
            agent_kwargs["model"] = agent.model

        raise last_error or ValueError("Failed to generate plan")

    def _execute_with_plan(user_request: str, plan: WorkPlan, *, welcome_text_local: str) -> Any:
        nonlocal active_plan_prompt_section

        allowed_tools = sorted(tools_expected_from_plan(plan))
        invocation_state = {"swarmee": {"mode": "execute", "enforce_plan": True, "allowed_tools": allowed_tools}}

        active_plan_prompt_section = (
            "Approved Plan (execute ONLY this plan; if you need changes, ask to :replan):\n"
            + plan.model_dump_json(indent=2)
        )
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
                                content=(
                                    f"Approved plan for request:\n{user_request}\n\n"
                                    f"{plan.model_dump_json(indent=2)}\n"
                                ),
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
                if model_manager.current_tier != prev:
                    print(f"\n[auto-escalation] tier: {prev} -> {model_manager.current_tier}")
                agent_kwargs["model"] = agent.model

            raise last_error or RuntimeError("Execution failed")
        finally:
            active_plan_prompt_section = None
            refresh_system_prompt(welcome_text_local)

    # Process query or enter interactive mode
    if args.query:
        query = " ".join(args.query)
        # Use retrieve if knowledge_base_id is defined
        if knowledge_base_id:
            agent.tool.retrieve(text=query, knowledgeBaseId=knowledge_base_id)

        intent = classify_intent(query)
        if intent == "work":
            plan = _generate_plan(query)
            print(_render_plan(plan))
            if not auto_approve:
                print("\nPlan generated. Re-run with --yes (or set SWARMEE_AUTO_APPROVE=true) to execute.")
                return
            try:
                response = _execute_with_plan(query, plan, welcome_text_local="")
            except AgentInterruptedError as e:
                callback_handler(force_stop=True)
                print(f"\n{str(e)}")
                return
            except MaxTokensReachedException:
                return
        else:
            try:
                response = run_agent(query, invocation_state={"swarmee": {"mode": "execute"}})
            except AgentInterruptedError as e:
                callback_handler(force_stop=True)
                print(f"\n{str(e)}")
                return
            except MaxTokensReachedException:
                return

        if knowledge_base_id:
            # Store conversation in knowledge base
            store_conversation_in_kb(agent, query, response, knowledge_base_id)
    else:
        # Display welcome text at startup
        welcome_result = agent.tool.welcome(action="view", record_direct_tool_call=False)
        welcome_text = ""
        if welcome_result["status"] == "success":
            welcome_text = welcome_result["content"][0]["text"]
            render_welcome_message(welcome_text)
        # Project preflight + cached project map
        if _truthy(os.getenv("SWARMEE_PREFLIGHT", "enabled")):
            level = (os.getenv("SWARMEE_PREFLIGHT_LEVEL", "summary") or "summary").strip().lower()
            max_chars = int(os.getenv("SWARMEE_PREFLIGHT_MAX_CHARS", "8000"))
            actions = ["summary"]
            if level == "summary+tree":
                actions.append("tree")
            elif level == "summary+files":
                actions.append("files")
            preflight_parts: list[str] = []
            for action in actions:
                try:
                    result = agent.tool.project_context(action=action, max_chars=max_chars, record_direct_tool_call=False)
                    if result.get("status") == "success":
                        preflight_parts.append(result.get("content", [{"text": ""}])[0].get("text", ""))
                except Exception:
                    continue
            preflight_text = "\n\n".join([p for p in preflight_parts if p]).strip()
            if preflight_text:
                artifact_store.write_text(kind="preflight", text=preflight_text, suffix="txt", metadata={"level": level})
                preflight_prompt_section = f"Project preflight:\n{preflight_text}"
                print("\n[preflight]\n" + preflight_text + "\n")

        if _truthy(os.getenv("SWARMEE_PROJECT_MAP", "enabled")):
            try:
                pm = build_project_map()
                pm_path = save_project_map(pm)
                project_map_prompt_section = render_project_map_summary(pm) + f"\n(project_map: {pm_path})"
            except Exception:
                project_map_prompt_section = None
        else:
            project_map_prompt_section = None

        refresh_system_prompt(welcome_text)
        pending_plan: WorkPlan | None = None
        pending_request: str | None = None
        force_plan_next = False
        while True:
            try:
                user_input = get_user_input("\n~ ", default="", keyboard_interrupt_return_default=False)
                if user_input.lower() in ["exit", "quit"]:
                    render_goodbye_message()
                    break
                if user_input.strip() == ":tools":
                    print("\n".join(sorted(tools_dict.keys())))
                    continue
                if user_input.startswith(":tier"):
                    parts = user_input.strip().split(maxsplit=2)
                    subcmd = parts[1].lower() if len(parts) >= 2 else "list"
                    if subcmd == "list":
                        current = model_manager.current_tier
                        lines = [f"# Model tiers (current: {current})", ""]
                        for t in model_manager.list_tiers():
                            mark = "*" if t.name == current else " "
                            status = "available" if t.available else f"unavailable ({t.reason})"
                            model_bits: list[str] = []
                            if t.display_name:
                                model_bits.append(t.display_name)
                            if t.model_id:
                                model_bits.append(f"model_id={t.model_id}")
                            model_str = f" ({', '.join(model_bits)})" if model_bits else ""
                            desc = f" - {t.description}" if t.description else ""
                            lines.append(f"{mark} {t.name}: provider={t.provider}{model_str} [{status}]{desc}")
                        print("\n".join(lines))
                    elif subcmd == "set" and len(parts) == 3:
                        tier = parts[2].strip().lower()
                        model_manager.set_tier(agent, tier)
                        agent_kwargs["model"] = agent.model
                        print(f"Active tier set to: {tier}")
                    elif subcmd == "auto" and len(parts) == 3:
                        val = parts[2].strip().lower()
                        enabled = val in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}
                        model_manager.set_auto_escalation(enabled)
                        print(f"Auto-escalation: {'on' if enabled else 'off'}")
                    else:
                        print("Usage: :tier list | :tier set <fast|balanced|deep|long> | :tier auto on|off")
                    continue
                if user_input.strip() in {":approve", ":y"}:
                    if pending_plan and pending_request:
                        try:
                            response = _execute_with_plan(pending_request, pending_plan, welcome_text_local=welcome_text)
                            if knowledge_base_id:
                                store_conversation_in_kb(agent, pending_request, response, knowledge_base_id)
                        except AgentInterruptedError as e:
                            callback_handler(force_stop=True)
                            print(f"\n{str(e)}")
                        except MaxTokensReachedException:
                            pass
                        finally:
                            pending_plan = None
                            pending_request = None
                    else:
                        print("No pending plan to approve.")
                    continue
                if user_input.strip() in {":n"}:
                    pending_plan = None
                    pending_request = None
                    print("Plan canceled.")
                    continue
                if user_input.strip() == ":replan":
                    if not pending_request:
                        print("No pending request to replan.")
                        continue
                    pending_plan = _generate_plan(pending_request)
                    print(_render_plan(pending_plan))
                    print(pending_plan.confirmation_prompt)
                    continue
                if user_input.strip() == ":plan":
                    force_plan_next = True
                    print("Next prompt will be planned before execution.")
                    continue
                if user_input.startswith(":sop"):
                    # Minimal interactive SOP management:
                    # :sop list
                    # :sop use <name>
                    # :sop clear
                    # :sop show
                    parts = user_input.strip().split(maxsplit=2)
                    subcmd = parts[1].lower() if len(parts) >= 2 else "list"

                    if subcmd == "list":
                        result = agent.tool.sop(
                            action="list", sop_paths=effective_sop_paths, record_direct_tool_call=False
                        )
                        print(result.get("content", [{"text": ""}])[0].get("text", ""))
                    elif subcmd == "use" and len(parts) == 3:
                        active_sop_name = parts[2].strip()
                        refresh_system_prompt(welcome_text)
                        print(f"Active SOP set to: {active_sop_name}")
                    elif subcmd == "clear":
                        active_sop_name = None
                        refresh_system_prompt(welcome_text)
                        print("Active SOP cleared.")
                    elif subcmd == "show":
                        if active_sop_name:
                            result = agent.tool.sop(
                                action="get",
                                name=active_sop_name,
                                sop_paths=effective_sop_paths,
                                record_direct_tool_call=False,
                            )
                            print(result.get("content", [{"text": ""}])[0].get("text", ""))
                        else:
                            print("No active SOP.")
                    else:
                        print("Usage: :sop list | :sop use <name> | :sop clear | :sop show")
                    continue
                if user_input.startswith("!"):
                    shell_command = user_input[1:]  # Remove the ! prefix
                    print(f"$ {shell_command}")

                    try:
                        # Execute shell command directly using the shell tool
                        agent.tool.shell(
                            command=shell_command,
                            user_message_override=user_input,
                            non_interactive_mode=True,
                            record_direct_tool_call=False,
                        )

                        print()  # new line after shell command execution
                    except Exception as e:
                        print(f"Shell command execution error: {str(e)}")
                    continue

                if user_input.strip():
                    # Use retrieve if knowledge_base_id is defined
                    if knowledge_base_id:
                        agent.tool.retrieve(text=user_input, knowledgeBaseId=knowledge_base_id)
                    refresh_system_prompt(welcome_text)

                    intent = "work" if force_plan_next else classify_intent(user_input)
                    force_plan_next = False

                    if intent == "work":
                        pending_request = user_input
                        pending_plan = _generate_plan(user_input)
                        print(_render_plan(pending_plan))
                        if auto_approve:
                            print("\nAuto-approving plan (--yes / SWARMEE_AUTO_APPROVE).")
                            response = _execute_with_plan(user_input, pending_plan, welcome_text_local=welcome_text)
                            pending_plan = None
                            pending_request = None
                            if knowledge_base_id:
                                store_conversation_in_kb(agent, user_input, response, knowledge_base_id)
                        else:
                            print(pending_plan.confirmation_prompt)
                        continue

                    try:
                        response = run_agent(user_input, invocation_state={"swarmee": {"mode": "execute"}})
                    except AgentInterruptedError as e:
                        callback_handler(force_stop=True)
                        print(f"\n{str(e)}")
                        continue
                    except MaxTokensReachedException:
                        # run_agent already printed guidance and reset agent
                        continue

                    if knowledge_base_id:
                        # Store conversation in knowledge base
                        store_conversation_in_kb(agent, user_input, response, knowledge_base_id)
            except (KeyboardInterrupt, EOFError):
                render_goodbye_message()
                break
            except Exception as e:
                callback_handler(force_stop=True)  # Stop spinners
                print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
