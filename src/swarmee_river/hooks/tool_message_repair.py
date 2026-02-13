from __future__ import annotations

import os
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeModelCallEvent

from swarmee_river.hooks._compat import event_messages, register_hook_callback


def _truthy_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _extract_tool_use_ids(message: Any) -> list[str]:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []

    out: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        tool_use = item.get("toolUse")
        if not isinstance(tool_use, dict):
            continue
        tool_use_id = tool_use.get("toolUseId")
        if isinstance(tool_use_id, str) and tool_use_id.strip():
            out.append(tool_use_id.strip())
    return out


def _extract_tool_result_ids(message: Any) -> list[str]:
    if not isinstance(message, dict) or message.get("role") != "user":
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []

    out: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        tool_result = item.get("toolResult")
        if not isinstance(tool_result, dict):
            continue
        tool_use_id = tool_result.get("toolUseId")
        if isinstance(tool_use_id, str) and tool_use_id.strip():
            out.append(tool_use_id.strip())
    return out


def _sanitize_unmatched_tool_uses(messages: list[Any]) -> tuple[list[Any], list[str]]:
    pending_counts: dict[str, int] = {}
    for message in messages:
        for tool_use_id in _extract_tool_use_ids(message):
            pending_counts[tool_use_id] = pending_counts.get(tool_use_id, 0) + 1
        for tool_use_id in _extract_tool_result_ids(message):
            count = pending_counts.get(tool_use_id, 0)
            if count <= 1:
                pending_counts.pop(tool_use_id, None)
            else:
                pending_counts[tool_use_id] = count - 1

    unresolved_ids = sorted([tool_use_id for tool_use_id, count in pending_counts.items() if count > 0])
    if not unresolved_ids:
        return messages, []

    unresolved = set(unresolved_ids)
    sanitized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        if message.get("role") != "assistant":
            sanitized.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            sanitized.append(message)
            continue

        new_content: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue
            tool_use = item.get("toolUse")
            if isinstance(tool_use, dict):
                tool_use_id = tool_use.get("toolUseId")
                if isinstance(tool_use_id, str) and tool_use_id.strip() in unresolved:
                    continue
            new_content.append(item)

        if new_content:
            new_message = dict(message)
            new_message["content"] = new_content
            sanitized.append(new_message)

    return sanitized, unresolved_ids


class ToolMessageRepairHooks(HookProvider):
    """
    Repairs invalid tool message chains before model invocation.

    Some SDK/runtime combinations can occasionally leave orphaned `toolUse` blocks
    in conversation history (without corresponding `toolResult` blocks), which is
    rejected by providers like Bedrock. This hook removes unresolved `toolUse`
    content blocks right before the next model call.
    """

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = _truthy_env("SWARMEE_REPAIR_TOOL_MESSAGES", True) if enabled is None else enabled

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        register_hook_callback(registry, BeforeModelCallEvent, self.before_model_call)

    def before_model_call(self, event: BeforeModelCallEvent) -> None:
        if not self.enabled:
            return

        messages = event_messages(event)
        if not isinstance(messages, list):
            return

        sanitized, unresolved_ids = _sanitize_unmatched_tool_uses(messages)
        if not unresolved_ids:
            return

        messages[:] = sanitized
        invocation_state = getattr(event, "invocation_state", None)
        if isinstance(invocation_state, dict):
            state_messages = invocation_state.get("messages")
            if isinstance(state_messages, list) and state_messages is not messages:
                state_messages[:] = sanitized
            sw = invocation_state.setdefault("swarmee", {})
            if isinstance(sw, dict):
                sw["repaired_tool_use_ids"] = unresolved_ids
