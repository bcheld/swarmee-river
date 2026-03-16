from __future__ import annotations

import re
from typing import Any

from strands import tool

from swarmee_river.profiles.models import ORCHESTRATOR_AGENT_ID
from swarmee_river.prompt_assets import load_prompt_assets, resolve_agent_prompt_text
from swarmee_river.tool_permissions import set_permissions
from swarmee_river.utils.agent_utils import extract_text, run_coroutine
from swarmee_river.utils.fork_utils import build_fork_invocation_state, create_shared_prefix_child_agent

CALL_AGENT_TOOL_PREFIX = "call_agent_"
_CALL_AGENT_TOOL_TAG = "builder-agent"


def _sanitize_tool_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "builder_agent"


def build_call_agent_tool_name(agent_def: dict[str, Any]) -> str:
    agent_id = str((agent_def or {}).get("id", "")).strip()
    return f"{CALL_AGENT_TOOL_PREFIX}{_sanitize_tool_token(agent_id)}"


def build_call_agent_tool_description(agent_def: dict[str, Any]) -> str:
    name = str((agent_def or {}).get("name", "")).strip() or str((agent_def or {}).get("id", "")).strip() or "Agent"
    summary = str((agent_def or {}).get("summary", "")).strip()
    if summary:
        return f"Call activated Builder agent '{name}'. {summary}"
    return f"Call activated Builder agent '{name}'."


def is_activated_builder_agent(agent_def: Any) -> bool:
    if not isinstance(agent_def, dict):
        return False
    agent_id = str(agent_def.get("id", "")).strip().lower()
    return bool(agent_id) and agent_id != ORCHESTRATOR_AGENT_ID and bool(agent_def.get("activated"))


def build_runtime_agent_tool_metadata(agents: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for agent_def in agents or []:
        if not is_activated_builder_agent(agent_def):
            continue
        tool_name = build_call_agent_tool_name(agent_def)
        if tool_name in seen:
            continue
        seen.add(tool_name)
        entries.append(
            {
                "name": tool_name,
                "description": build_call_agent_tool_description(agent_def),
                "tags": [_CALL_AGENT_TOOL_TAG],
                "access_read": False,
                "access_write": False,
                "access_execute": True,
                "source": "runtime-generated",
            }
        )
    return entries


def _resolved_agent_prompt(agent_def: dict[str, Any]) -> str:
    assets = {str(item.id).strip().lower(): item for item in load_prompt_assets()}
    return resolve_agent_prompt_text(agent_def, assets).strip()


def _build_seed_instruction(agent_def: dict[str, Any]) -> str:
    name = str(agent_def.get("name", "")).strip() or str(agent_def.get("id", "")).strip() or "Agent"
    agent_id = str(agent_def.get("id", "")).strip()
    summary = str(agent_def.get("summary", "")).strip()
    resolved_prompt = _resolved_agent_prompt(agent_def)
    requested_tool_names = [str(name).strip() for name in (agent_def.get("tool_names") or []) if str(name).strip()]
    lines = [
        f"You are running as activated Builder agent '{name}' (id: {agent_id}).",
        "Answer the tool request directly and keep any tool use focused on the assigned task.",
    ]
    if summary:
        lines.append(f"Role: {summary}")
    if requested_tool_names:
        lines.append(f"Allowed tools: {', '.join(requested_tool_names)}")
    if resolved_prompt:
        lines.append(f"Additional agent instructions:\n{resolved_prompt}")
    return "\n\n".join(line for line in lines if line).strip()


def create_runtime_call_agent_tool(agent_def: dict[str, Any]) -> Any:
    tool_name = build_call_agent_tool_name(agent_def)
    description = build_call_agent_tool_description(agent_def)
    requested_tool_names = [str(name).strip() for name in (agent_def.get("tool_names") or []) if str(name).strip()]
    agent_id = str(agent_def.get("id", "")).strip()

    def _call_agent_tool(*, query: str, agent: Any | None = None) -> dict[str, Any]:
        if not str(query or "").strip():
            return {"status": "error", "content": [{"text": "query is required"}]}
        if agent is None or getattr(agent, "model", None) is None:
            return {
                "status": "error",
                "content": [{"text": "call_agent tool requires an agent context with a configured model"}],
            }

        try:
            child_agent, snapshot = create_shared_prefix_child_agent(
                parent_agent=agent,
                kind=f"call_agent:{agent_id}",
                seed_instruction=_build_seed_instruction(agent_def),
                tool_allowlist=requested_tool_names,
                callback_handler=None,
            )
            invocation_state = build_fork_invocation_state(
                snapshot,
                extra_prompt_chars=len(str(query or "").strip()),
                extra_fields={"builder_agent_id": agent_id, "builder_agent_tool": tool_name},
            )
            result = run_coroutine(child_agent.invoke_async(query, invocation_state=invocation_state))
        except Exception as exc:
            return {"status": "error", "content": [{"text": f"{tool_name} failed: {exc}"}]}

        result_text = extract_text(result) or str(result or "").strip()
        if not result_text:
            return {"status": "error", "content": [{"text": f"{tool_name} produced no text output"}]}
        return {"status": "success", "content": [{"text": result_text.strip()}]}

    _call_agent_tool.__name__ = tool_name
    _call_agent_tool.__doc__ = description
    wrapped = tool(_call_agent_tool)
    set_permissions(wrapped, "execute")
    return wrapped


def build_runtime_agent_tools(agents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for agent_def in agents or []:
        if not is_activated_builder_agent(agent_def):
            continue
        tool_name = build_call_agent_tool_name(agent_def)
        if tool_name in tools:
            continue
        tools[tool_name] = create_runtime_call_agent_tool(agent_def)
    return tools


__all__ = [
    "CALL_AGENT_TOOL_PREFIX",
    "build_call_agent_tool_description",
    "build_call_agent_tool_name",
    "build_runtime_agent_tool_metadata",
    "build_runtime_agent_tools",
    "create_runtime_call_agent_tool",
    "is_activated_builder_agent",
]
