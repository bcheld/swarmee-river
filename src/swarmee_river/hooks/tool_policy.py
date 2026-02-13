from __future__ import annotations

import os
from typing import Any

from strands.hooks import HookRegistry, HookProvider
from strands.hooks.events import BeforeToolCallEvent


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
        self.plan_mode_allowed_tools = {"retrieve", "sop"}

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.register(BeforeToolCallEvent, self.before_tool_call)

    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if event.cancel_tool:
            return

        tool_use = event.tool_use or {}
        name = tool_use.get("name")
        if not name:
            return

        sw = event.invocation_state.get("swarmee", {}) if isinstance(event.invocation_state, dict) else {}
        mode = sw.get("mode")

        if mode == "plan" and name not in self.plan_mode_allowed_tools:
            event.cancel_tool = f"Tool '{name}' blocked in plan mode."
            return

        if mode == "execute" and sw.get("enforce_plan"):
            allowed_tools = sw.get("allowed_tools")
            if isinstance(allowed_tools, (list, tuple, set)):
                allowed = {str(x) for x in allowed_tools if str(x).strip()}
                if allowed and name not in allowed:
                    event.cancel_tool = (
                        f"Tool '{name}' not in approved plan. Use :replan to update the plan "
                        "before using additional tools."
                    )
                    return

        if self.enabled_tools and name not in self.enabled_tools:
            event.cancel_tool = f"Tool '{name}' blocked by SWARMEE_ENABLE_TOOLS policy."
            return

        if name in self.disabled_tools:
            event.cancel_tool = f"Tool '{name}' blocked by SWARMEE_DISABLE_TOOLS policy."
            return

        if name == "swarm" and not self.swarm_enabled:
            event.cancel_tool = "Tool 'swarm' is disabled (set SWARMEE_SWARM_ENABLED=true to enable)."
