from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from strands import Agent
from strands._async import run_async
from strands.event_loop.streaming import process_stream
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

from swarmee_river.context.prompt_cache import PromptCacheState, inject_system_reminder
from swarmee_river.hooks._compat import register_hook_callback
from swarmee_river.opencode_aliases import equivalent_tool_names, normalize_tool_name


def _safe_deepcopy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _snapshot_prompt_cache(agent: Any) -> PromptCacheState | None:
    prompt_cache = getattr(agent, "_swarmee_prompt_cache", None)
    return prompt_cache if isinstance(prompt_cache, PromptCacheState) else None


def _snapshot_invocation_state(agent: Any) -> dict[str, Any]:
    invocation_state = getattr(agent, "_swarmee_current_invocation_state", None)
    if isinstance(invocation_state, dict):
        return _safe_deepcopy(invocation_state)
    return {}


def _ordered_tool_objects(agent: Any) -> list[Any]:
    registry = getattr(getattr(agent, "tool_registry", None), "registry", None)
    dynamic_tools = getattr(getattr(agent, "tool_registry", None), "dynamic_tools", None)
    if not isinstance(registry, dict):
        return []
    ordered_names: list[str] = []
    with_config = getattr(getattr(agent, "tool_registry", None), "get_all_tools_config", None)
    if callable(with_config):
        try:
            tool_config = with_config()
            if isinstance(tool_config, dict):
                ordered_names = [str(name).strip() for name in tool_config.keys() if str(name).strip()]
        except Exception:
            ordered_names = []
    if not ordered_names:
        ordered_names = [str(name).strip() for name in registry.keys() if str(name).strip()]
    ordered_tools: list[Any] = []
    for name in ordered_names:
        tool_obj = registry.get(name)
        if tool_obj is None and isinstance(dynamic_tools, dict):
            tool_obj = dynamic_tools.get(name)
        if tool_obj is not None:
            ordered_tools.append(tool_obj)
    return ordered_tools


def _tool_specs(agent: Any) -> list[dict[str, Any]]:
    getter = getattr(getattr(agent, "tool_registry", None), "get_all_tool_specs", None)
    if not callable(getter):
        return []
    try:
        specs = getter()
    except Exception:
        return []
    if not isinstance(specs, list):
        return []
    return _safe_deepcopy(specs)


def _system_prompt_input(agent: Any) -> Any:
    content = getattr(agent, "_system_prompt_content", None)
    if content is not None:
        cloned = _safe_deepcopy(content)
        if cloned is not None:
            return cloned
    return _safe_deepcopy(getattr(agent, "system_prompt", None))


def _extract_text_from_message(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts).strip()


def _message_uses_tools(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict) and "toolUse" in item:
            return True
    return False


def _prefix_hash_payload(
    *,
    system_prompt: Any,
    messages: list[dict[str, Any]],
    tool_specs: list[dict[str, Any]],
    pending_reminder: str,
) -> str:
    payload = {
        "system_prompt": system_prompt,
        "messages": messages,
        "tool_specs": tool_specs,
        "pending_reminder": pending_reminder,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class SharedPrefixForkSnapshot:
    kind: str
    model: Any
    system_prompt: str | None
    system_prompt_content: list[dict[str, Any]] | None
    system_prompt_input: Any
    messages: list[dict[str, Any]]
    tool_specs: list[dict[str, Any]]
    tool_objects: list[Any]
    pending_reminder: str
    parent_message_count: int
    prefix_hash: str
    base_invocation_state: dict[str, Any]


@dataclass(frozen=True)
class SharedPrefixTextForkResult:
    text: str
    stop_reason: str
    message: dict[str, Any] | None
    used_tool: bool
    diagnostics: dict[str, Any]


class StaticToolAllowlistHooks(HookProvider):
    def __init__(self, allowed_tool_names: list[str] | set[str] | tuple[str, ...]) -> None:
        self._allowed = {
            str(name).strip()
            for name in allowed_tool_names
            if str(name).strip()
        }

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        register_hook_callback(registry, BeforeToolCallEvent, self.before_tool_call)

    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if not self._allowed or event.cancel_tool:
            return
        tool_name = normalize_tool_name(event.tool_use.get("name"))
        if tool_name and equivalent_tool_names(tool_name).intersection(self._allowed):
            return
        event.cancel_tool = "Tool blocked by shared-prefix fork allowlist."


def capture_shared_prefix_fork(parent_agent: Any, *, kind: str) -> SharedPrefixForkSnapshot:
    prompt_cache = _snapshot_prompt_cache(parent_agent)
    pending_reminder = prompt_cache.peek_reminder() if prompt_cache is not None else ""
    messages = _safe_deepcopy(getattr(parent_agent, "messages", []) or [])
    tool_specs = _tool_specs(parent_agent)
    system_prompt = getattr(parent_agent, "system_prompt", None)
    system_prompt_content = _safe_deepcopy(getattr(parent_agent, "_system_prompt_content", None))
    return SharedPrefixForkSnapshot(
        kind=str(kind or "").strip() or "fork",
        model=getattr(parent_agent, "model", None),
        system_prompt=system_prompt if isinstance(system_prompt, str) else None,
        system_prompt_content=system_prompt_content if isinstance(system_prompt_content, list) else None,
        system_prompt_input=_system_prompt_input(parent_agent),
        messages=messages if isinstance(messages, list) else [],
        tool_specs=tool_specs,
        tool_objects=_ordered_tool_objects(parent_agent),
        pending_reminder=pending_reminder,
        parent_message_count=len(messages) if isinstance(messages, list) else 0,
        prefix_hash=_prefix_hash_payload(
            system_prompt=system_prompt_content if system_prompt_content is not None else system_prompt,
            messages=messages if isinstance(messages, list) else [],
            tool_specs=tool_specs,
            pending_reminder=pending_reminder,
        ),
        base_invocation_state=_snapshot_invocation_state(parent_agent),
    )


def build_fork_invocation_state(
    snapshot: SharedPrefixForkSnapshot,
    *,
    extra_prompt_chars: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    invocation_state = _safe_deepcopy(snapshot.base_invocation_state)
    if not isinstance(invocation_state, dict):
        invocation_state = {}
    sw = invocation_state.setdefault("swarmee", {})
    if not isinstance(sw, dict):
        sw = {}
        invocation_state["swarmee"] = sw
    sw.update(
        {
            "fork_kind": snapshot.kind,
            "fork_parent_message_count": snapshot.parent_message_count,
            "fork_prefix_hash": snapshot.prefix_hash,
            "fork_extra_prompt_chars": max(0, int(extra_prompt_chars or 0)),
            "fork_used_pending_reminder": bool(snapshot.pending_reminder),
        }
    )
    if isinstance(extra_fields, dict):
        for key, value in extra_fields.items():
            if value is not None:
                sw[key] = value
    return invocation_state


def inject_fork_prompt(snapshot: SharedPrefixForkSnapshot, prompt_text: str) -> str:
    return inject_system_reminder(user_query=prompt_text, reminder=snapshot.pending_reminder)


async def _run_text_fork_async(
    snapshot: SharedPrefixForkSnapshot,
    *,
    prompt_text: str,
    invocation_state: dict[str, Any],
) -> SharedPrefixTextForkResult:
    appended_text = inject_fork_prompt(snapshot, prompt_text)
    messages = _safe_deepcopy(snapshot.messages)
    messages.append({"role": "user", "content": [{"text": appended_text}]})
    stream = snapshot.model.stream(
        messages,
        tool_specs=_safe_deepcopy(snapshot.tool_specs),
        system_prompt=snapshot.system_prompt,
        system_prompt_content=_safe_deepcopy(snapshot.system_prompt_content),
        invocation_state=invocation_state,
    )
    result_message: dict[str, Any] | None = None
    stop_reason = ""
    async for event in process_stream(stream):
        if "stop" in event:
            stop_reason, result_message, *_rest = event["stop"]
    if result_message is None:
        raise RuntimeError("Shared-prefix fork returned no final message")
    return SharedPrefixTextForkResult(
        text=_extract_text_from_message(result_message),
        stop_reason=str(stop_reason or "").strip(),
        message=result_message,
        used_tool=_message_uses_tools(result_message),
        diagnostics={
            "fork_kind": snapshot.kind,
            "fork_parent_message_count": snapshot.parent_message_count,
            "fork_prefix_hash": snapshot.prefix_hash,
            "fork_extra_prompt_chars": max(0, len(str(prompt_text or "").strip())),
            "fork_used_pending_reminder": bool(snapshot.pending_reminder),
        },
    )


def run_shared_prefix_text_fork(
    parent_agent: Any,
    *,
    kind: str,
    prompt_text: str,
    extra_fields: dict[str, Any] | None = None,
) -> SharedPrefixTextForkResult:
    snapshot = capture_shared_prefix_fork(parent_agent, kind=kind)
    invocation_state = build_fork_invocation_state(
        snapshot,
        extra_prompt_chars=len(str(prompt_text or "").strip()),
        extra_fields=extra_fields,
    )
    return run_async(lambda: _run_text_fork_async(snapshot, prompt_text=prompt_text, invocation_state=invocation_state))


def create_shared_prefix_child_agent(
    *,
    parent_agent: Any,
    kind: str,
    seed_instruction: str | None = None,
    tool_allowlist: list[str] | None = None,
    callback_handler: Callable[..., Any] | None = None,
) -> tuple[Agent, SharedPrefixForkSnapshot]:
    snapshot = capture_shared_prefix_fork(parent_agent, kind=kind)
    seeded_messages = _safe_deepcopy(snapshot.messages)
    instruction = str(seed_instruction or "").strip()
    if instruction:
        seeded_messages.append({"role": "user", "content": [{"text": inject_fork_prompt(snapshot, instruction)}]})

    state_payload: dict[str, Any] | None = None
    state = getattr(parent_agent, "state", None)
    getter = getattr(state, "get", None)
    if callable(getter):
        try:
            raw_state = getter()
            if isinstance(raw_state, dict):
                state_payload = _safe_deepcopy(raw_state)
        except Exception:
            state_payload = None

    hooks: list[Any] = []
    if tool_allowlist:
        hooks.append(StaticToolAllowlistHooks(tool_allowlist))

    kwargs: dict[str, Any] = {
        "model": snapshot.model,
        "messages": seeded_messages,
        "tools": list(snapshot.tool_objects),
        "system_prompt": snapshot.system_prompt_input,
        "callback_handler": callback_handler,
        "load_tools_from_directory": False,
    }
    if state_payload is not None:
        kwargs["state"] = state_payload
    if hooks:
        kwargs["hooks"] = hooks
    trace_attributes = getattr(parent_agent, "trace_attributes", None)
    if isinstance(trace_attributes, dict) and trace_attributes:
        kwargs["trace_attributes"] = dict(trace_attributes)
    retry_strategy = getattr(parent_agent, "_retry_strategy", None)
    if retry_strategy is not None:
        kwargs["retry_strategy"] = retry_strategy
    try:
        child = Agent(**kwargs)
    except TypeError:
        kwargs.pop("hooks", None)
        kwargs.pop("retry_strategy", None)
        kwargs.pop("state", None)
        child = Agent(**kwargs)
        if hooks:
            for hook in hooks:
                child.hooks.add_hook(hook)

    child._swarmee_prompt_cache = _snapshot_prompt_cache(parent_agent)
    child._swarmee_current_invocation_state = _safe_deepcopy(snapshot.base_invocation_state)
    return child, snapshot
