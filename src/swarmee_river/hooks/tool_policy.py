from __future__ import annotations

import os
import shlex
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

from swarmee_river.hooks._compat import register_hook_callback


def _truthy_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _csv_env(name: str) -> set[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


_WINDOWS_POSIX_BIASED_TOKENS = {
    "awk",
    "chmod",
    "chown",
    "grep",
    "sed",
    "source",
    "xargs",
}


def _first_command_token(command: str) -> str:
    text = (command or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text, posix=True)
    except ValueError:
        parts = text.split()
    if not parts:
        return ""
    return str(parts[0]).strip().lower()


def _looks_posix_only_shell_command(command: str) -> bool:
    text = (command or "").strip()
    if not text:
        return False
    lower = text.lower()
    if lower.startswith(("bash ", "bash -lc", "sh ", "zsh ")):
        return False
    token = _first_command_token(text)
    if token in _WINDOWS_POSIX_BIASED_TOKENS:
        return True
    if token.startswith("./") and token.endswith(".sh"):
        return True
    return False


class ToolPolicyHooks(HookProvider):
    """
    Simple guardrails for tool use in enterprise environments.

    Controls:
    - Disable selected tools via `SWARMEE_DISABLE_TOOLS` (comma-separated tool names)
    - Allow only selected tools via `SWARMEE_ENABLE_TOOLS` (comma-separated tool names)
    - Gate swarms via `SWARMEE_SWARM_ENABLED` (default: enabled)
    """

    def __init__(self) -> None:
        self.enabled_tools = _csv_env("SWARMEE_ENABLE_TOOLS")
        self.disabled_tools = _csv_env("SWARMEE_DISABLE_TOOLS")
        self.swarm_enabled = _truthy_env("SWARMEE_SWARM_ENABLED", True)
        # Plan mode should stay read-only to prevent the model from mutating the repo while planning.
        # We allow repo inspection tools so plans can be grounded in reality.
        self.plan_mode_allowed_tools = {"retrieve", "sop", "project_context", "file_read"}
        self.plan_mode_project_context_actions = {"summary", "files", "tree", "search", "read", "git_status"}

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        register_hook_callback(registry, BeforeToolCallEvent, self.before_tool_call)

    def _plan_mode_allowlist(self, sw_state: dict[str, Any]) -> set[str]:
        allowed = set(self.plan_mode_allowed_tools)
        extra = sw_state.get("plan_allowed_tools")
        if isinstance(extra, (list, tuple, set)):
            for item in extra:
                value = str(item).strip()
                if value:
                    allowed.add(value)
        return allowed

    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if event.cancel_tool:
            return

        tool_use = event.tool_use or {}
        name = tool_use.get("name")
        if not name:
            return

        sw = event.invocation_state.get("swarmee", {}) if isinstance(event.invocation_state, dict) else {}
        mode = sw.get("mode")
        runtime_env = sw.get("runtime_environment") if isinstance(sw, dict) else {}

        if name == "shell":
            tool_input = tool_use.get("input")
            command = tool_input.get("command") if isinstance(tool_input, dict) else None
            os_name = str(runtime_env.get("os") or "").strip().lower() if isinstance(runtime_env, dict) else ""
            shell_family = (
                str(runtime_env.get("shell_family") or "").strip().lower() if isinstance(runtime_env, dict) else ""
            )
            if (
                os_name == "windows"
                and shell_family in {"powershell", "cmd"}
                and isinstance(command, str)
                and _looks_posix_only_shell_command(command)
            ):
                event.cancel_tool = (
                    "Shell command appears POSIX-specific, but runtime shell is Windows "
                    f"{shell_family}. Use PowerShell/CMD syntax or invoke bash explicitly."
                )
                return

        if mode == "plan":
            plan_allowed_tools = self._plan_mode_allowlist(sw if isinstance(sw, dict) else {})
            if name not in plan_allowed_tools:
                event.cancel_tool = f"Tool '{name}' blocked in plan mode."
                return
        if mode == "plan" and name == "project_context":
            tool_input = tool_use.get("input")
            action = tool_input.get("action") if isinstance(tool_input, dict) else None
            if isinstance(action, str) and action.strip().lower() not in self.plan_mode_project_context_actions:
                event.cancel_tool = f"Tool 'project_context' action '{action}' blocked in plan mode."
                return

        # Structured-output planning tool should never run during execute mode.
        if mode == "execute" and name == "WorkPlan":
            event.cancel_tool = "Tool 'WorkPlan' is only allowed in plan mode."
            return

        if mode == "execute" and sw.get("enforce_plan"):
            allowed_tools = sw.get("allowed_tools")
            if isinstance(allowed_tools, (list, tuple, set)):
                allowed = {str(x) for x in allowed_tools if str(x).strip()}
                if name not in allowed:
                    event.cancel_tool = (
                        f"Tool '{name}' not in approved plan. Use :replan to update the plan "
                        "before using additional tools."
                    )
                    return

        if mode == "execute":
            profile = sw.get("tool_profile")
            tier = sw.get("tier")
            if isinstance(profile, dict):
                allow = profile.get("tool_allowlist")
                block = profile.get("tool_blocklist")
                allow_set = {str(x) for x in allow if str(x).strip()} if isinstance(allow, list) else set()
                block_set = {str(x) for x in block if str(x).strip()} if isinstance(block, list) else set()

                if allow_set and name not in allow_set:
                    suffix = f" (tier={tier})" if tier else ""
                    event.cancel_tool = f"Tool '{name}' blocked by tier tool_allowlist{suffix}."
                    return
                if block_set and name in block_set:
                    suffix = f" (tier={tier})" if tier else ""
                    event.cancel_tool = f"Tool '{name}' blocked by tier tool_blocklist{suffix}."
                    return

        if self.enabled_tools and name not in self.enabled_tools:
            event.cancel_tool = f"Tool '{name}' blocked by SWARMEE_ENABLE_TOOLS policy."
            return

        if name in self.disabled_tools:
            event.cancel_tool = f"Tool '{name}' blocked by SWARMEE_DISABLE_TOOLS policy."
            return

        if name == "swarm" and not self.swarm_enabled:
            event.cancel_tool = "Tool 'swarm' is disabled (set SWARMEE_SWARM_ENABLED=true to enable)."
