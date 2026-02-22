from __future__ import annotations

from typing import Any

from strands import tool

from swarmee_river.utils.agent_utils import create_sub_agent, extract_text, run_coroutine


@tool
def use_agent(
    *,
    prompt: str | None = None,
    text: str | None = None,
    system_prompt: str | None = None,
    agent: Any | None = None,
) -> dict[str, Any]:
    """
    Cross-platform fallback for Strands Tools `use_agent` / deprecated `use_llm`.

    Safety model:
    - Creates a tool-less sub-agent (no tools) to avoid bypassing Swarmee policy/consent hooks.
    """
    effective_prompt = (prompt or text or "").strip()
    if not effective_prompt:
        return {"status": "error", "content": [{"text": "prompt is required"}]}

    if agent is None or getattr(agent, "model", None) is None:
        return {"status": "error", "content": [{"text": "use_agent requires an agent context with a configured model"}]}

    sub = create_sub_agent(
        parent_agent=agent,
        system_prompt=str(system_prompt or "").strip() or "You are a helpful assistant.",
    )

    try:
        result = run_coroutine(sub.invoke_async(effective_prompt))
    except Exception as exc:
        return {"status": "error", "content": [{"text": f"use_agent failed: {exc}"}]}

    return {"status": "success", "content": [{"text": extract_text(result)}]}


@tool
def use_llm(
    *,
    prompt: str,
    system_prompt: str | None = None,
    agent: Any | None = None,
) -> dict[str, Any]:
    """
    Back-compat alias for deprecated `use_llm` → `use_agent`.
    """
    return use_agent(prompt=prompt, system_prompt=system_prompt, agent=agent)
