from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CommandInvocation:
    name: str
    args: list[str]
    raw: str


@dataclass(frozen=True)
class CommandDispatchResult:
    handled: bool
    should_exit: bool = False


CommandHandler = Callable[["CLIContext", CommandInvocation], CommandDispatchResult]


@dataclass
class CLIContext:
    """
    Mutable runtime context shared across REPL commands.

    This is intentionally lightweight and uses `Any` for agent/provider types so the CLI
    layer does not need to import Strands SDK internals.
    """

    agent: Any
    agent_kwargs: dict[str, Any]
    tools_dict: dict[str, Any]
    model_manager: Any
    knowledge_base_id: str | None
    effective_sop_paths: str | None
    welcome_text: str
    auto_approve: bool
    selected_provider: str | None
    settings: Any
    settings_path: Any

    # Callbacks / helpers wired by swarmee.py
    refresh_system_prompt: Callable[[], None]
    generate_plan: Callable[[str], Any]
    execute_with_plan: Callable[[str, Any], Any]
    render_plan: Callable[[Any], str]
    run_agent: Callable[..., Any]
    store_conversation: Callable[[str, Any], None]
    output: Callable[[str], None]
    stop_spinners: Callable[[], None]
    build_session_meta: Callable[[], dict[str, Any]]
    swap_agent: Callable[[Any | None, Any | None], None]

    # Interactive state
    pending_plan: Any | None = None
    pending_request: str | None = None
    force_plan_next: bool = False
    active_sop_name: str | None = None
    last_plan: Any | None = None

    # Optional: session persistence (wired later)
    session_store: Any | None = None
    current_session_id: str | None = None


@dataclass(frozen=True)
class CommandSpec:
    name: str
    handler: CommandHandler
    help: str
    usage: str | None = None


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, CommandSpec] = {}

    def register(self, name: str, handler: CommandHandler, *, help: str, usage: str | None = None) -> None:
        key = (name or "").strip().lstrip(":").lower()
        if not key:
            raise ValueError("Command name is required")
        self._commands[key] = CommandSpec(name=key, handler=handler, help=help, usage=usage)

    def list_commands(self) -> list[CommandSpec]:
        return sorted(self._commands.values(), key=lambda c: c.name)

    def parse(self, line: str) -> CommandInvocation | None:
        raw = (line or "").strip()
        if not raw.startswith(":"):
            return None

        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = raw.split()

        if not parts:
            return None

        name = parts[0].lstrip(":").strip().lower()
        args = [p for p in parts[1:] if p is not None]
        return CommandInvocation(name=name, args=args, raw=raw)

    def dispatch(self, ctx: CLIContext, line: str) -> CommandDispatchResult:
        invocation = self.parse(line)
        if invocation is None:
            return CommandDispatchResult(handled=False)

        if not invocation.name:
            return CommandDispatchResult(handled=True)

        spec = self._commands.get(invocation.name)
        if spec is None:
            ctx.output(f"Unknown command: :{invocation.name}. Type :help for options.")
            return CommandDispatchResult(handled=True)

        try:
            return spec.handler(ctx, invocation)
        except Exception as e:
            ctx.stop_spinners()
            ctx.output(f"\nError: {e}")
            return CommandDispatchResult(handled=True)
