from __future__ import annotations

from typing import Any, Optional, cast

from strands import Agent, tool
from strands.multiagent import Swarm


def _create_custom_agents(
    agent_specs: list[dict[str, Any]],
    *,
    parent_agent: Optional[Any] = None,
) -> list[Agent]:
    if not agent_specs:
        raise ValueError("At least one agent specification is required")

    agents: list[Agent] = []
    used_names: set[str] = set()

    for i, spec in enumerate(agent_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"Agent specification {i} must be a dictionary")

        agent_name = str(spec.get("name") or f"agent_{i + 1}")
        if agent_name in used_names:
            base = agent_name
            counter = 1
            while agent_name in used_names:
                agent_name = f"{base}_{counter}"
                counter += 1
        used_names.add(agent_name)

        system_prompt = spec.get("system_prompt")
        if not system_prompt:
            if parent_agent and getattr(parent_agent, "system_prompt", None):
                system_prompt = (
                    "You are a helpful AI assistant specializing in collaborative problem solving.\n\n"
                    f"Base Instructions:\n{parent_agent.system_prompt}"
                )
            else:
                system_prompt = "You are a helpful AI assistant specializing in collaborative problem solving."
        else:
            if (
                parent_agent
                and getattr(parent_agent, "system_prompt", None)
                and spec.get("inherit_parent_prompt", False)
            ):
                system_prompt = f"{system_prompt}\n\nBase Instructions:\n{parent_agent.system_prompt}"

        agent_tools: Any = spec.get("tools")
        if agent_tools and parent_agent and hasattr(parent_agent, "tool_registry"):
            available_tools = parent_agent.tool_registry.registry.keys()
            filtered_tool_names = [t for t in agent_tools if t in available_tools]
            agent_tools = [parent_agent.tool_registry.registry[name] for name in filtered_tool_names]

        swarm_agent = Agent(
            name=agent_name,
            system_prompt=system_prompt,
            tools=agent_tools,
            callback_handler=getattr(parent_agent, "callback_handler", None) if parent_agent else None,
            trace_attributes=getattr(parent_agent, "trace_attributes", None) if parent_agent else None,
        )
        swarm_agent_any = cast(Any, swarm_agent)

        # Provider/settings-based configuration (used by Strands Tools).
        model_provider = spec.get("model_provider")
        if model_provider:
            swarm_agent_any.model_provider = model_provider
        elif parent_agent and hasattr(parent_agent, "model_provider"):
            swarm_agent_any.model_provider = cast(Any, parent_agent).model_provider

        model_settings = spec.get("model_settings")
        if model_settings:
            swarm_agent_any.model_settings = model_settings
        elif parent_agent and hasattr(parent_agent, "model_settings"):
            swarm_agent_any.model_settings = cast(Any, parent_agent).model_settings

        agents.append(swarm_agent)

    return agents


@tool
async def swarm(
    task: str,
    agents: list[dict[str, Any]],
    max_handoffs: int = 20,
    max_iterations: int = 20,
    execution_timeout: float = 900.0,
    node_timeout: float = 300.0,
    repetitive_handoff_detection_window: int = 8,
    repetitive_handoff_min_unique_agents: int = 3,
    agent: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Cancellable Swarm tool wrapper.

    This intentionally uses `await Swarm.invoke_async(...)` (not `Swarm(task)`), so Esc cancellation can reliably
    stop execution without leaving background threads running.
    """
    if not task or not task.strip():
        return {"status": "error", "content": [{"text": "Task is required."}]}

    try:
        swarm_agents = _create_custom_agents(agents, parent_agent=agent)

        sdk_swarm = Swarm(
            nodes=swarm_agents,
            max_handoffs=max_handoffs,
            max_iterations=max_iterations,
            execution_timeout=execution_timeout,
            node_timeout=node_timeout,
            repetitive_handoff_detection_window=repetitive_handoff_detection_window,
            repetitive_handoff_min_unique_agents=repetitive_handoff_min_unique_agents,
        )

        result = await sdk_swarm.invoke_async(task)

        response_parts: list[str] = []
        response_parts.append("🎯 **Swarm Execution Complete**")
        response_parts.append(f"📊 **Status:** {result.status}")
        response_parts.append(f"⏱️ **Execution Time:** {result.execution_time}ms")
        response_parts.append(f"🤖 **Team Size:** {len(swarm_agents)} agents")
        response_parts.append(f"🔄 **Iterations:** {result.execution_count}")

        if getattr(result, "node_history", None):
            agent_chain = " → ".join([node.node_id for node in result.node_history])
            response_parts.append(f"🔗 **Collaboration Chain:** {agent_chain}")

        if getattr(result, "results", None):
            response_parts.append("\n**🤖 Individual Agent Contributions:**")
            for agent_name, node_result in result.results.items():
                node = getattr(node_result, "result", None)
                content = getattr(node, "content", None)
                if not content:
                    continue
                texts: list[str] = []
                for block in content:
                    text = getattr(block, "text", None)
                    if text:
                        texts.append(text)
                if texts:
                    response_parts.append(f"\n**{str(agent_name).upper().replace('_', ' ')}:**")
                    response_parts.extend(texts)

        if getattr(result, "accumulated_usage", None):
            usage = result.accumulated_usage
            response_parts.append("\n**📈 Team Resource Usage:**")
            response_parts.append(f"• Input tokens: {usage.get('inputTokens', 0):,}")
            response_parts.append(f"• Output tokens: {usage.get('outputTokens', 0):,}")
            response_parts.append(f"• Total tokens: {usage.get('totalTokens', 0):,}")

        return {"status": "success", "content": [{"text": "\n".join(response_parts)}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"⚠️ Swarm execution failed: {str(e)}"}]}
