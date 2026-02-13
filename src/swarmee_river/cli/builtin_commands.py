from __future__ import annotations

import json
from pathlib import Path

from swarmee_river.cli.commands import CLIContext, CommandDispatchResult, CommandInvocation, CommandRegistry
from swarmee_river.cli.diagnostics import (
    render_artifact_get,
    render_artifact_list,
    render_effective_config,
    render_git_diff,
    render_git_status,
    render_log_tail,
    render_replay_invocation,
)
from tools.sop import run_sop


def register_builtin_commands(registry: CommandRegistry) -> None:
    def _help(ctx: CLIContext, _inv: CommandInvocation) -> CommandDispatchResult:
        lines: list[str] = ["# Commands", ""]
        for spec in registry.list_commands():
            usage = f":{spec.name} {spec.usage}".strip() if spec.usage else f":{spec.name}"
            lines.append(f"- {usage}: {spec.help}")
        ctx.output("\n".join(lines))
        return CommandDispatchResult(handled=True)

    def _tools(ctx: CLIContext, _inv: CommandInvocation) -> CommandDispatchResult:
        ctx.output("\n".join(sorted(ctx.tools_dict.keys())))
        return CommandDispatchResult(handled=True)

    def _exit(_ctx: CLIContext, _inv: CommandInvocation) -> CommandDispatchResult:
        return CommandDispatchResult(handled=True, should_exit=True)

    def _tier(ctx: CLIContext, inv: CommandInvocation) -> CommandDispatchResult:
        subcmd = (inv.args[0].lower() if inv.args else "list").strip()
        if subcmd == "list":
            current = ctx.model_manager.current_tier
            lines = [f"# Model tiers (current: {current})", ""]
            for t in ctx.model_manager.list_tiers():
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
            ctx.output("\n".join(lines))
            return CommandDispatchResult(handled=True)

        if subcmd == "set" and len(inv.args) >= 2:
            tier = inv.args[1].strip().lower()
            ctx.model_manager.set_tier(ctx.agent, tier)
            ctx.agent_kwargs["model"] = ctx.agent.model
            ctx.output(f"Active tier set to: {tier}")
            return CommandDispatchResult(handled=True)

        if subcmd == "auto" and len(inv.args) >= 2:
            val = inv.args[1].strip().lower()
            enabled = val in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}
            ctx.model_manager.set_auto_escalation(enabled)
            ctx.output(f"Auto-escalation: {'on' if enabled else 'off'}")
            return CommandDispatchResult(handled=True)

        ctx.output("Usage: :tier list | :tier set <fast|balanced|deep|long> | :tier auto on|off")
        return CommandDispatchResult(handled=True)

    def _approve(ctx: CLIContext, _inv: CommandInvocation) -> CommandDispatchResult:
        if ctx.pending_plan and ctx.pending_request:
            try:
                ctx.last_plan = ctx.pending_plan
                response = ctx.execute_with_plan(ctx.pending_request, ctx.pending_plan)
                if ctx.knowledge_base_id:
                    ctx.store_conversation(ctx.pending_request, response)
            finally:
                ctx.pending_plan = None
                ctx.pending_request = None
        else:
            ctx.output("No pending plan to approve.")
        return CommandDispatchResult(handled=True)

    def _cancel(ctx: CLIContext, _inv: CommandInvocation) -> CommandDispatchResult:
        ctx.pending_plan = None
        ctx.pending_request = None
        ctx.output("Plan canceled.")
        return CommandDispatchResult(handled=True)

    def _replan(ctx: CLIContext, _inv: CommandInvocation) -> CommandDispatchResult:
        if not ctx.pending_request:
            ctx.output("No pending request to replan.")
            return CommandDispatchResult(handled=True)
        ctx.pending_plan = ctx.generate_plan(ctx.pending_request)
        ctx.output(ctx.render_plan(ctx.pending_plan))
        try:
            ctx.output(ctx.pending_plan.confirmation_prompt)
        except Exception:
            pass
        return CommandDispatchResult(handled=True)

    def _plan(ctx: CLIContext, _inv: CommandInvocation) -> CommandDispatchResult:
        ctx.force_plan_next = True
        ctx.output("Next prompt will be planned before execution.")
        return CommandDispatchResult(handled=True)

    def _sop(ctx: CLIContext, inv: CommandInvocation) -> CommandDispatchResult:
        subcmd = (inv.args[0].lower() if inv.args else "list").strip()
        if subcmd == "list":
            result = run_sop(action="list", sop_paths=ctx.effective_sop_paths)
            ctx.output(result.get("content", [{"text": ""}])[0].get("text", ""))
            return CommandDispatchResult(handled=True)

        if subcmd == "use" and len(inv.args) >= 2:
            ctx.active_sop_name = inv.args[1].strip()
            ctx.refresh_system_prompt()
            ctx.output(f"Active SOP set to: {ctx.active_sop_name}")
            return CommandDispatchResult(handled=True)

        if subcmd == "clear":
            ctx.active_sop_name = None
            ctx.refresh_system_prompt()
            ctx.output("Active SOP cleared.")
            return CommandDispatchResult(handled=True)

        if subcmd == "show":
            if not ctx.active_sop_name:
                ctx.output("No active SOP.")
                return CommandDispatchResult(handled=True)
            result = run_sop(
                action="get",
                name=ctx.active_sop_name,
                sop_paths=ctx.effective_sop_paths,
            )
            ctx.output(result.get("content", [{"text": ""}])[0].get("text", ""))
            return CommandDispatchResult(handled=True)

        ctx.output("Usage: :sop list | :sop use <name> | :sop clear | :sop show")
        return CommandDispatchResult(handled=True)

    def _session(ctx: CLIContext, inv: CommandInvocation) -> CommandDispatchResult:
        store = ctx.session_store
        if store is None:
            ctx.output("Session store is not available.")
            return CommandDispatchResult(handled=True)

        subcmd = (inv.args[0].lower() if inv.args else "info").strip()

        if subcmd in {"list", "ls"}:
            entries = store.list()
            if not entries:
                ctx.output("No sessions found.")
                return CommandDispatchResult(handled=True)
            lines = ["# Sessions", ""]
            for e in entries:
                sid = str(e.get("id") or "")
                updated = str(e.get("updated_at") or e.get("created_at") or "")
                provider = str(e.get("provider") or "")
                tier = str(e.get("tier") or "")
                suffix = " ".join([p for p in [updated, provider, tier] if p]).strip()
                lines.append(f"- `{sid}`" + (f" ({suffix})" if suffix else ""))
            ctx.output("\n".join(lines))
            return CommandDispatchResult(handled=True)

        if subcmd == "new":
            meta = ctx.build_session_meta()
            sid = store.create(meta=meta)
            ctx.current_session_id = sid
            ctx.pending_plan = None
            ctx.pending_request = None
            ctx.last_plan = None
            ctx.swap_agent(None, None)
            ctx.refresh_system_prompt()
            ctx.output(f"New session: {sid}")
            return CommandDispatchResult(handled=True)

        if subcmd == "save":
            sid = inv.args[1].strip() if len(inv.args) >= 2 else (ctx.current_session_id or "")
            if not sid:
                meta = ctx.build_session_meta()
                sid = store.create(meta=meta)
                ctx.current_session_id = sid

            messages = getattr(ctx.agent, "messages", None)
            state = getattr(ctx.agent, "state", None)

            meta = ctx.build_session_meta()
            paths = store.save(sid, meta=meta, messages=messages, state=state, last_plan=ctx.last_plan)
            ctx.output(f"Saved session: {sid} -> {paths.dir}")
            return CommandDispatchResult(handled=True)

        if subcmd == "load" and len(inv.args) >= 2:
            sid = inv.args[1].strip()
            meta, messages, state, last_plan = store.load(sid)
            ctx.current_session_id = sid
            ctx.pending_plan = None
            ctx.pending_request = None
            ctx.last_plan = last_plan
            ctx.swap_agent(messages, state)
            ctx.refresh_system_prompt()
            ctx.output(f"Loaded session: {sid} ({meta.get('updated_at') or meta.get('created_at')})")
            return CommandDispatchResult(handled=True)

        if subcmd in {"rm", "delete"} and len(inv.args) >= 2:
            sid = inv.args[1].strip()
            store.delete(sid)
            if ctx.current_session_id == sid:
                ctx.current_session_id = None
            ctx.output(f"Deleted session: {sid}")
            return CommandDispatchResult(handled=True)

        if subcmd == "info":
            sid = inv.args[1].strip() if len(inv.args) >= 2 else (ctx.current_session_id or "")
            if not sid:
                ctx.output("No current session. Use :session new or :session load <id>.")
                return CommandDispatchResult(handled=True)
            meta = store.read_meta(sid)
            ctx.output(json.dumps(meta, indent=2, ensure_ascii=False))
            return CommandDispatchResult(handled=True)

        ctx.output("Usage: :session new | save [id] | load <id> | list | rm <id> | info [id]")
        return CommandDispatchResult(handled=True)

    def _config(ctx: CLIContext, inv: CommandInvocation) -> CommandDispatchResult:
        subcmd = (inv.args[0].lower() if inv.args else "show").strip()
        if subcmd != "show":
            ctx.output("Usage: :config show")
            return CommandDispatchResult(handled=True)

        text = render_effective_config(
            cwd=Path.cwd(),
            settings_path=Path(str(ctx.settings_path)),
            settings=ctx.settings,
            selected_provider=ctx.selected_provider,
            model_manager=ctx.model_manager,
            knowledge_base_id=ctx.knowledge_base_id,
            effective_sop_paths=ctx.effective_sop_paths,
            auto_approve=ctx.auto_approve,
        )
        ctx.output(text)
        return CommandDispatchResult(handled=True)

    def _status(ctx: CLIContext, _inv: CommandInvocation) -> CommandDispatchResult:
        ctx.output(render_git_status(cwd=Path.cwd()))
        return CommandDispatchResult(handled=True)

    def _diff(ctx: CLIContext, inv: CommandInvocation) -> CommandDispatchResult:
        staged = False
        paths: list[str] = []
        for a in inv.args:
            if a in {"--staged", "--cached"}:
                staged = True
            elif a.startswith("-"):
                continue
            else:
                paths.append(a)
        ctx.output(render_git_diff(cwd=Path.cwd(), staged=staged, paths=paths or None))
        return CommandDispatchResult(handled=True)

    def _artifact(ctx: CLIContext, inv: CommandInvocation) -> CommandDispatchResult:
        subcmd = (inv.args[0].lower() if inv.args else "list").strip()
        if subcmd in {"list", "ls"}:
            ctx.output(render_artifact_list(cwd=Path.cwd()))
            return CommandDispatchResult(handled=True)

        if subcmd == "get":
            artifact_id = inv.args[1].strip() if len(inv.args) >= 2 else None
            if not artifact_id:
                ctx.output("Usage: :artifact get <artifact_id>")
                return CommandDispatchResult(handled=True)
            ctx.output(render_artifact_get(cwd=Path.cwd(), artifact_id=artifact_id, path=None))
            return CommandDispatchResult(handled=True)

        ctx.output("Usage: :artifact list | :artifact get <artifact_id>")
        return CommandDispatchResult(handled=True)

    def _log(ctx: CLIContext, inv: CommandInvocation) -> CommandDispatchResult:
        subcmd = (inv.args[0].lower() if inv.args else "tail").strip()
        if subcmd != "tail":
            ctx.output("Usage: :log tail [--lines N]")
            return CommandDispatchResult(handled=True)
        n = 50
        if "--lines" in inv.args:
            try:
                idx = inv.args.index("--lines")
                n = int(inv.args[idx + 1])
            except Exception:
                n = 50
        ctx.output(render_log_tail(cwd=Path.cwd(), lines=n))
        return CommandDispatchResult(handled=True)

    def _replay(ctx: CLIContext, inv: CommandInvocation) -> CommandDispatchResult:
        if not inv.args:
            ctx.output("Usage: :replay <invocation_id>")
            return CommandDispatchResult(handled=True)
        ctx.output(render_replay_invocation(cwd=Path.cwd(), invocation_id=inv.args[0].strip()))
        return CommandDispatchResult(handled=True)

    registry.register("help", _help, help="Show available commands")
    registry.register("tools", _tools, help="List available tools")
    registry.register("tier", _tier, help="List/set model tiers", usage="list | set <tier> | auto on|off")
    registry.register("approve", _approve, help="Approve and execute the pending plan")
    registry.register("y", _approve, help="Alias for :approve")
    registry.register("n", _cancel, help="Cancel the pending plan")
    registry.register("replan", _replan, help="Regenerate the pending plan")
    registry.register("plan", _plan, help="Force plan-first for next prompt")
    registry.register("sop", _sop, help="Manage SOPs", usage="list | use <name> | clear | show")
    registry.register("session", _session, help="Manage sessions", usage="new|save|load|list|rm|info")
    registry.register("config", _config, help="Show effective config", usage="show")
    registry.register("status", _status, help="Show git status summary")
    registry.register("diff", _diff, help="Show git diff", usage="[--staged] [paths...]")
    registry.register("artifact", _artifact, help="List/get artifacts", usage="list|get")
    registry.register("log", _log, help="Tail Swarmee logs", usage="tail [--lines N]")
    registry.register("replay", _replay, help="Replay a logged invocation", usage="<invocation_id>")
    registry.register("exit", _exit, help="Exit the REPL")
    registry.register("quit", _exit, help="Exit the REPL")
