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

try:
    from swarmee_river.hooks.tool_message_repair import ToolMessageRepairHooks
except Exception:
    ToolMessageRepairHooks = None  # type: ignore[assignment]
from swarmee_river.interrupts import AgentInterruptedError, interrupt_watcher_from_env
from swarmee_river.intent import classify_intent
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
from swarmee_river.harness.context_snapshot import build_context_snapshot
from swarmee_river.planning import WorkPlan, structured_plan_prompt
from swarmee_river.project_map import build_project_map
from swarmee_river.packs import enabled_sop_paths, enabled_system_prompts, load_enabled_pack_tools
from swarmee_river.session.models import SessionModelManager
from swarmee_river.session.store import SessionStore
from swarmee_river.settings import PackEntry, PacksConfig, SwarmeeSettings, load_settings, save_settings
from swarmee_river.tools import get_tools
from swarmee_river.artifacts import ArtifactStore, tools_expected_from_plan
from swarmee_river.utils import model_utils
from swarmee_river.utils.env_utils import load_env_file
from swarmee_river.utils.kb_utils import load_system_prompt, store_conversation_in_kb
from swarmee_river.utils.provider_utils import resolve_model_provider
from swarmee_river.utils.welcome_utils import render_goodbye_message, render_welcome_message
from tools.sop import run_sop
from tools.welcome import read_welcome_text

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
    settings_path_for_project = Path.cwd() / ".swarmee" / "settings.json"
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
                harness=settings.harness,
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
            for e in entries:
                sid = str(e.get("id") or "")
                updated = str(e.get("updated_at") or e.get("created_at") or "")
                provider = str(e.get("provider") or "")
                tier = str(e.get("tier") or "")
                suffix = " ".join([p for p in [updated, provider, tier] if p]).strip()
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
            subcmd = (sub[0].strip().lower() if sub else "show")
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
            subcmd = (sub[0].strip().lower() if sub else "list")
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
            subcmd = (sub[0].strip().lower() if sub else "tail")
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

    selected_provider, provider_notice = resolve_model_provider(
        cli_provider=args.model_provider.stem if args.model_provider is not None else None,
        env_provider=os.getenv("SWARMEE_MODEL_PROVIDER"),
        settings_provider=settings.models.provider,
    )
    if provider_notice:
        print(f"[provider] {provider_notice}")

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
            # Keep consent prompts aligned with interactive UX:
            # 1) stop active spinners so prompt doesn't appear inline/garbled
            # 2) render consent context as an explicit line
            # 3) reuse the familiar input prompt style
            callback_handler(force_stop=True)
            prompt_text = (text or "").strip()
            if prompt_text:
                print(f"\n[tool consent] {prompt_text}")
            return get_user_input("\n~ consent> ", default="", keyboard_interrupt_return_default=True)

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
        if ToolMessageRepairHooks is not None:
            hooks.insert(2, ToolMessageRepairHooks())

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

    preflight_prompt_section: str | None = None
    project_map_prompt_section: str | None = None
    active_plan_prompt_section: str | None = None

    artifact_store = ArtifactStore()

    def refresh_system_prompt(welcome_text_local: str) -> None:
        parts: list[str] = [system_prompt]

        if pack_prompt_sections:
            parts.extend(pack_prompt_sections)

        if project_map_prompt_section:
            parts.append(project_map_prompt_section)

        if preflight_prompt_section:
            parts.append(preflight_prompt_section)

        if args.include_welcome_in_prompt and welcome_text_local:
            parts.append(f"Welcome Text Reference:\n{welcome_text_local}")

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
        resolved_state: dict[str, Any] = dict(invocation_state) if isinstance(invocation_state, dict) else {}
        sw_state = resolved_state.setdefault("swarmee", {})
        if isinstance(sw_state, dict):
            sw_state["tier"] = model_manager.current_tier
            profile = settings.harness.tier_profiles.get(model_manager.current_tier)
            if profile is not None:
                sw_state["tool_profile"] = profile.to_dict()
        invocation_state = resolved_state

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

    registry = CommandRegistry()
    register_builtin_commands(registry)

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
        settings_path=settings_path_for_project,
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

    # Process query or enter interactive mode
    if args.query:
        query = " ".join(args.query)
        # Use retrieve if knowledge_base_id is defined
        if knowledge_base_id:
            agent.tool.retrieve(text=query, knowledgeBaseId=knowledge_base_id)

        profile = settings.harness.tier_profiles.get(model_manager.current_tier)
        snapshot = build_context_snapshot(
            artifact_store=artifact_store,
            interactive=False,
            default_preflight_level=profile.preflight_level if profile else None,
        )
        preflight_prompt_section = snapshot.preflight_prompt_section
        project_map_prompt_section = snapshot.project_map_prompt_section
        ctx.refresh_system_prompt()

        intent = classify_intent(query)
        if intent == "work":
            plan = _generate_plan(query)
            ctx.last_plan = plan
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
        # Display welcome text at startup.
        try:
            welcome_text = read_welcome_text()
        except Exception:
            welcome_text = ""
        if welcome_text:
            ctx.welcome_text = welcome_text
            render_welcome_message(welcome_text)
        profile = settings.harness.tier_profiles.get(model_manager.current_tier)
        snapshot = build_context_snapshot(
            artifact_store=artifact_store,
            interactive=True,
            default_preflight_level=profile.preflight_level if profile else None,
        )
        preflight_prompt_section = snapshot.preflight_prompt_section
        project_map_prompt_section = snapshot.project_map_prompt_section

        ctx.refresh_system_prompt()
        run_repl(
            ctx=ctx,
            registry=registry,
            get_user_input=get_user_input,
            classify_intent=classify_intent,
            render_goodbye_message=render_goodbye_message,
        )


if __name__ == "__main__":
    main()
