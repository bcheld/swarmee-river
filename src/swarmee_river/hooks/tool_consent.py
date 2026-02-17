from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

from swarmee_river.hooks._compat import register_hook_callback
from swarmee_river.opencode_aliases import canonical_tool_name, equivalent_tool_names, normalize_tool_name
from swarmee_river.permissions import evaluate_permission_action
from swarmee_river.settings import SafetyConfig
from swarmee_river.utils.env_utils import truthy, truthy_env


def _truncate(value: str, limit: int = 240) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _tool_prompt_context(tool_name: str, tool_use: Any) -> str:
    tool_input = tool_use.get("input")
    if not isinstance(tool_input, dict):
        return ""

    if canonical_tool_name(tool_name) == "shell":
        command = _truncate(str(tool_input.get("command") or ""))
        cwd = _truncate(str(tool_input.get("cwd") or ""))
        lines = []
        if command:
            lines.append(f"  Command: {command}")
        if cwd:
            lines.append(f"  CWD: {cwd}")
        return "\n".join(lines)

    if tool_name == "git":
        action = _truncate(str(tool_input.get("action") or ""))
        ref = _truncate(str(tool_input.get("ref") or ""))
        message = _truncate(str(tool_input.get("message") or ""))
        lines = []
        if action:
            lines.append(f"  Action: {action}")
        if ref:
            lines.append(f"  Ref: {ref}")
        if message:
            lines.append(f"  Message: {message}")
        return "\n".join(lines)

    if tool_name == "project_context":
        action = _truncate(str(tool_input.get("action") or ""))
        query = _truncate(str(tool_input.get("query") or ""))
        path = _truncate(str(tool_input.get("path") or ""))
        lines = []
        if action:
            lines.append(f"  Action: {action}")
        if query:
            lines.append(f"  Query: {query}")
        if path:
            lines.append(f"  Path: {path}")
        return "\n".join(lines)

    try:
        payload = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
    except Exception:
        payload = str(tool_input)
    return f"  Input: {_truncate(payload)}"


@dataclass(frozen=True)
class ConsentDecision:
    allowed: bool
    remember: bool


class ToolConsentHooks(HookProvider):
    """
    Ask-for-consent tool hook (Claude Code–like trust model).

    High-risk tools default to requiring explicit consent:
    - shell
    - file_write
    - editor
    - http_request
    """

    def __init__(
        self,
        safety: SafetyConfig,
        *,
        interactive: bool,
        auto_approve: bool,
        prompt: Callable[[str], str],
    ) -> None:
        self._safety = safety
        self._interactive = interactive
        self._auto_approve = auto_approve
        self._prompt = prompt

        self._decisions: dict[str, bool] = {}
        self._lock = threading.Lock()

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        register_hook_callback(registry, BeforeToolCallEvent, self.before_tool_call)

    def _decision_key(self, tool_name: str) -> str:
        canonical = canonical_tool_name(tool_name)
        return canonical or tool_name

    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if event.cancel_tool:
            return

        tool_use = event.tool_use
        tool_name = normalize_tool_name(tool_use.get("name"))
        if not tool_name:
            return
        decision_key = self._decision_key(tool_name)

        sw = event.invocation_state.get("swarmee", {}) if isinstance(event.invocation_state, dict) else {}
        auto_approve = self._auto_approve
        if isinstance(sw, dict) and "auto_approve" in sw:
            auto_approve = truthy(sw.get("auto_approve"))
        if sw.get("mode") == "execute" and sw.get("enforce_plan"):
            allowed_tools = sw.get("allowed_tools")
            allowed = (
                {str(x).strip() for x in allowed_tools} if isinstance(allowed_tools, (list, tuple, set)) else set()
            )
            if allowed and equivalent_tool_names(tool_name).intersection(allowed):
                # Plan approval counts as consent for tools explicitly listed in the approved plan.
                return

        if truthy_env("BYPASS_TOOL_CONSENT", False):
            return

        action, remember_allowed = evaluate_permission_action(
            safety=self._safety,
            tool_name=tool_name,
            tool_use=tool_use,
        )
        action = (action or "ask").strip().lower()
        if action == "allow":
            return
        if action == "deny":
            event.cancel_tool = f"Tool '{tool_name}' blocked by safety policy."
            return

        # action == "ask"
        if auto_approve:
            return

        # Non-interactive sessions can't prompt; fail closed.
        if not self._interactive:
            event.cancel_tool = (
                f"Tool '{tool_name}' requires consent, but session is non-interactive. "
                "Re-run with --yes or SWARMEE_AUTO_APPROVE=true to allow."
            )
            return

        if remember_allowed and decision_key in self._decisions:
            if not self._decisions[decision_key]:
                event.cancel_tool = f"Tool '{tool_name}' denied for this session."
            return

        with self._lock:
            if remember_allowed and decision_key in self._decisions:
                if not self._decisions[decision_key]:
                    event.cancel_tool = f"Tool '{tool_name}' denied for this session."
                return

            if remember_allowed:
                prompt = (
                    f"Allow tool '{tool_name}'?\n"
                    f"{_tool_prompt_context(tool_name, tool_use)}\n"
                    "  [y] Yes   [n] No   [a] Always (session)   [v] Never (session): "
                )
            else:
                prompt = f"Allow tool '{tool_name}'?\n{_tool_prompt_context(tool_name, tool_use)}\n  [y] Yes   [n] No: "

            choice = (self._prompt(prompt) or "").strip().lower()
            if remember_allowed:
                if choice in {"a", "always"}:
                    self._decisions[decision_key] = True
                    return
                if choice in {"v", "never"}:
                    self._decisions[decision_key] = False
                    event.cancel_tool = f"Tool '{tool_name}' denied for this session."
                    return
            if choice in {"y", "yes"}:
                return

            event.cancel_tool = f"Tool '{tool_name}' denied by user."
