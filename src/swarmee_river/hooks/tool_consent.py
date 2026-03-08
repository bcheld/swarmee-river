from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

from swarmee_river.diff_review import preview_mutating_tool_change, summarize_diff_stats, truncate_diff_preview
from swarmee_river.hooks._compat import register_hook_callback
from swarmee_river.opencode_aliases import canonical_tool_name, equivalent_tool_names, normalize_tool_name
from swarmee_river.permissions import evaluate_permission_action
from swarmee_river.settings import SafetyConfig
from swarmee_river.utils.env_utils import truthy


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


def _format_non_text_change_summary(non_text_paths: list[dict[str, str]]) -> str:
    if not non_text_paths:
        return ""
    lines = ["  Non-text changes:"]
    for item in non_text_paths[:8]:
        path = str(item.get("path", "")).strip() or "(unknown)"
        before = str(item.get("before", "")).strip() or "unknown"
        after = str(item.get("after", "")).strip() or "unknown"
        lines.append(f"    - {path}: {before} -> {after}")
    remaining = len(non_text_paths) - 8
    if remaining > 0:
        lines.append(f"    - ... {remaining} more")
    return "\n".join(lines)


def _build_consent_prompt(
    tool_name: str,
    tool_use: Any,
) -> tuple[str, dict[str, Any] | None, str | None]:
    lines = [f"Allow tool '{tool_name}'?"]
    context = _tool_prompt_context(tool_name, tool_use)
    if context:
        lines.append(context)

    tool_input = tool_use.get("input") if isinstance(tool_use, dict) else None
    preview = preview_mutating_tool_change(tool_name, tool_input if isinstance(tool_input, dict) else None)
    if preview is None:
        return "\n".join(lines), None, None
    if not preview.trusted:
        reason = str(preview.reason or "").strip() or "missing deterministic file diff preview"
        blocked = (
            f"Tool '{tool_name}' blocked: unable to build a trustworthy pre-approval diff preview ({reason}). "
            "Use `editor` for single-file edits or `patch_apply` for structured multi-file patches."
        )
        return "\n".join(lines), None, blocked

    payload: dict[str, Any] = {
        "tool": tool_name,
        "changed_paths": list(preview.changed_paths),
        "touched_paths": list(preview.touched_paths),
    }

    if preview.changed_paths:
        lines.append(f"  Changed paths: {', '.join(preview.changed_paths)}")
    elif preview.touched_paths:
        lines.append(f"  Touched paths: {', '.join(preview.touched_paths)}")

    if preview.diff_text.strip():
        preview_text, hidden_lines = truncate_diff_preview(preview.diff_text)
        stats = summarize_diff_stats(preview.diff_text)
        stats["files_changed"] = max(len(preview.changed_paths), stats.get("files_changed", 0))
        payload["diff_preview"] = preview_text
        payload["diff_hidden_lines"] = hidden_lines
        payload["diff_stats"] = stats
        lines.append("  Diff preview:")
        lines.extend(f"    {line}" if line else "" for line in preview_text.splitlines())
        if hidden_lines > 0:
            lines.append(f"    ... {hidden_lines} more line(s) hidden")

    non_text_paths = list(preview.non_text_paths or [])
    if non_text_paths:
        summary = _format_non_text_change_summary(non_text_paths)
        payload["non_text_paths"] = non_text_paths
        payload["non_text_change_summary"] = summary.strip()
        if summary:
            lines.append(summary)

    if not preview.diff_text.strip() and not non_text_paths:
        lines.append("  No file content changes predicted.")

    return "\n".join(lines), payload, None


@dataclass(frozen=True)
class ConsentDecision:
    allowed: bool
    remember: bool


class ToolConsentHooks(HookProvider):
    """
    Ask-for-consent tool hook (Claude Code–like trust model).

    High-risk tools default to requiring explicit consent:
    - shell
    - editor
    - patch_apply
    - http_request
    """

    def __init__(
        self,
        safety: SafetyConfig,
        *,
        interactive: bool,
        auto_approve: bool,
        prompt: Callable[[str, dict[str, Any] | None], str],
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
        session_overrides = sw.get("session_safety_overrides", {}) if isinstance(sw, dict) else {}
        if isinstance(session_overrides, dict):
            override_consent = str(session_overrides.get("tool_consent", "")).strip().lower()
            if override_consent == "allow":
                return
            if override_consent == "deny":
                event.cancel_tool = f"Tool '{tool_name}' blocked by session safety override."
                return

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
                "Re-run with --yes or set `runtime.auto_approve=true` in `.swarmee/settings.json` to allow."
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
                prompt_text, prompt_payload, preview_block = _build_consent_prompt(tool_name, tool_use)
                if preview_block:
                    event.cancel_tool = preview_block
                    return
                prompt = (
                    f"{prompt_text}\n"
                    "  [y] Yes   [n] No   [a] Always (session)   [v] Never (session): "
                )
            else:
                prompt_text, prompt_payload, preview_block = _build_consent_prompt(tool_name, tool_use)
                if preview_block:
                    event.cancel_tool = preview_block
                    return
                prompt = f"{prompt_text}\n  [y] Yes   [n] No: "

            choice = (self._prompt(prompt, prompt_payload) or "").strip().lower()
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
