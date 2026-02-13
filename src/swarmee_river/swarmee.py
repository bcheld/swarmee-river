#!/usr/bin/env python3
"""
Swarmee - A minimal CLI interface for Swarmee River (built on Strands)
"""

import argparse
import asyncio
import contextlib
import logging
import os
import threading
from typing import Any, Optional

# Strands
from strands import Agent
from strands_tools.utils.user_input import get_user_input
from strands.types.exceptions import MaxTokensReachedException

from swarmee_river.handlers.callback_handler import callback_handler, set_interrupt_event
try:
    from swarmee_river.hooks.jsonl_logger import JSONLLoggerHooks
    from swarmee_river.hooks.tool_result_limiter import ToolResultLimiterHooks
    from swarmee_river.hooks.tool_policy import ToolPolicyHooks

    _HAS_STRANDS_HOOKS = True
except Exception:
    JSONLLoggerHooks = None  # type: ignore[assignment]
    ToolResultLimiterHooks = None  # type: ignore[assignment]
    ToolPolicyHooks = None  # type: ignore[assignment]
    _HAS_STRANDS_HOOKS = False
from swarmee_river.interrupts import AgentInterruptedError, interrupt_watcher_from_env
from swarmee_river.tools import get_tools
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
        default="bedrock",
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

    model_config = args.model_config or model_utils.default_model_config(args.model_provider.stem)
    model = model_utils.load_model(args.model_provider, model_config)

    # Load system prompt
    system_prompt = load_system_prompt()

    tools_dict = get_tools()
    tools = tools_dict.values()

    conversation_manager = _build_conversation_manager(window_size=args.window_size, per_turn=args.context_per_turn)

    hooks = []
    if _HAS_STRANDS_HOOKS:
        hooks = [
            JSONLLoggerHooks(),  # type: ignore[misc]
            ToolPolicyHooks(),  # type: ignore[misc]
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

    def create_agent() -> Agent:
        try:
            return Agent(**agent_kwargs)
        except TypeError:
            # Backwards compatibility for older Strands versions.
            agent_kwargs.pop("conversation_manager", None)
            agent_kwargs.pop("hooks", None)
            return Agent(**agent_kwargs)

    agent = create_agent()

    interrupt_event = threading.Event()

    active_sop_name: str | None = args.sop

    def refresh_system_prompt(welcome_text_local: str) -> None:
        base_system_prompt = load_system_prompt()
        parts: list[str] = [base_system_prompt]

        if args.include_welcome_in_prompt and welcome_text_local:
            parts.append(f"Welcome Text Reference:\n{welcome_text_local}")

        if active_sop_name:
            sop_text = ""
            try:
                sop_result = agent.tool.sop(
                    action="get",
                    name=active_sop_name,
                    sop_paths=args.sop_paths,
                    record_direct_tool_call=False,
                )
                if sop_result.get("status") == "success":
                    sop_text = sop_result.get("content", [{}])[0].get("text", "")
            except Exception:
                sop_text = ""
            if sop_text:
                parts.append(f"Active SOP:\n{sop_text}")

        agent.system_prompt = "\n\n".join(parts).strip()

    def run_agent(query: str) -> Any:
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
                        return await agent.invoke_async(query)
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

    # Process query or enter interactive mode
    if args.query:
        query = " ".join(args.query)
        # Use retrieve if knowledge_base_id is defined
        if knowledge_base_id:
            agent.tool.retrieve(text=query, knowledgeBaseId=knowledge_base_id)

        try:
            response = run_agent(query)
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
        refresh_system_prompt(welcome_text)
        while True:
            try:
                user_input = get_user_input("\n~ ", default="", keyboard_interrupt_return_default=False)
                if user_input.lower() in ["exit", "quit"]:
                    render_goodbye_message()
                    break
                if user_input.strip() == ":tools":
                    print("\n".join(sorted(tools_dict.keys())))
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
                        result = agent.tool.sop(action="list", sop_paths=args.sop_paths, record_direct_tool_call=False)
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
                                sop_paths=args.sop_paths,
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

                    try:
                        response = run_agent(user_input)
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
