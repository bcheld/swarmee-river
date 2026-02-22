from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from swarmee_river.settings import PackEntry, PacksConfig, SwarmeeSettings, save_settings


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
        args = parts[1:]
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


def _render_session_list(entries: list[dict[str, object]], *, quote_ids: bool) -> str:
    if not entries:
        return "No sessions found."

    lines = ["# Sessions", ""]
    for entry in entries:
        sid = str(entry.get("id") or "")
        updated = str(entry.get("updated_at") or entry.get("created_at") or "")
        provider = str(entry.get("provider") or "")
        tier = str(entry.get("tier") or "")
        suffix = " ".join([part for part in [updated, provider, tier] if part]).strip()
        sid_text = f"`{sid}`" if quote_ids else sid
        lines.append(f"- {sid_text}" + (f" ({suffix})" if suffix else ""))

    return "\n".join(lines)


def handle_common_session_command(
    *,
    store: Any,
    subcmd: str,
    args: list[str],
    create_meta: Callable[[], dict[str, object]],
    usage_text: str,
    quote_ids: bool,
    new_output_mode: Literal["id_only", "labeled"],
    require_info_arg: bool,
    current_session_id: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    cmd = (subcmd or "").strip().lower()

    if cmd in {"list", "ls"}:
        return _render_session_list(store.list(), quote_ids=quote_ids), None, None

    if cmd == "new":
        sid = store.create(meta=create_meta())
        output = sid if new_output_mode == "id_only" else f"New session: {sid}"
        return output, sid, None

    if cmd in {"rm", "delete"} and len(args) >= 2:
        sid = args[1].strip()
        store.delete(sid)
        return f"Deleted session: {sid}", None, sid

    if cmd == "info":
        sid = args[1].strip() if len(args) >= 2 else (current_session_id or "")
        if not sid:
            if require_info_arg:
                return usage_text, None, None
            return "No current session. Use :session new or :session load <id>.", None, None
        meta = store.read_meta(sid)
        return json.dumps(meta, indent=2, ensure_ascii=False), None, None

    return usage_text, None, None


def handle_pack_command(*, args: list[str], settings: SwarmeeSettings, settings_path: Path) -> str:
    sub = list(args or [])

    def _persist(installed: list[PackEntry]) -> None:
        save_settings(
            SwarmeeSettings(
                models=settings.models,
                safety=settings.safety,
                packs=PacksConfig(installed=installed),
                harness=settings.harness,
                raw=settings.raw,
            ),
            settings_path,
        )

    if not sub or sub[0] in {"list", "ls"}:
        lines = ["# Packs", ""]
        if not settings.packs.installed:
            lines.append("No packs installed.")
        else:
            for p in settings.packs.installed:
                lines.append(f"- {p.name} ({'enabled' if p.enabled else 'disabled'}) -> {p.path}")
        return "\n".join(lines)

    if sub[0] == "install" and len(sub) >= 2:
        pack_path = sub[1]
        if pack_path.startswith("s3://"):
            return "S3 pack install is not implemented yet. Use a local path for now."
        pack_dir = Path(pack_path).expanduser()
        if not pack_dir.exists() or not pack_dir.is_dir():
            return f"Pack path not found: {pack_dir}"
        name = pack_dir.name
        meta_path = pack_dir / "pack.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta.get("name"), str) and meta["name"].strip():
                    name = meta["name"].strip()
            except Exception:
                pass
        installed = [p for p in settings.packs.installed if p.name != name]
        installed.append(PackEntry(name=name, path=str(pack_dir.resolve()), enabled=True))
        _persist(installed)
        return f"Installed pack: {name} -> {pack_dir.resolve()}"

    if sub[0] in {"enable", "disable"} and len(sub) >= 2:
        target = sub[1].strip()
        if not target:
            return "Pack name is required."
        found = False
        updated: list[PackEntry] = []
        for p in settings.packs.installed:
            if p.name == target:
                updated.append(PackEntry(name=p.name, path=p.path, enabled=(sub[0] == "enable")))
                found = True
            else:
                updated.append(p)
        if not found:
            return f"Pack not found: {target}"
        _persist(updated)
        return f"{'Enabled' if sub[0] == 'enable' else 'Disabled'} pack: {target}"

    return "Usage: swarmee pack list | swarmee pack install <path> | swarmee pack enable|disable <name>"
