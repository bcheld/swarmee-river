from __future__ import annotations

from typing import Any

from strands import tool

from swarmee_river.tool_permissions import set_permissions
from swarmee_river.utils.agent_utils import extract_text
from swarmee_river.utils.fork_utils import run_shared_prefix_text_fork


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

    try:
        instruction = str(system_prompt or "").strip()
        fork_prompt = effective_prompt
        if instruction:
            fork_prompt = (
                "Follow these additional subtask instructions while answering.\n"
                f"{instruction}\n\n"
                f"Subtask:\n{effective_prompt}"
            )
        result = run_shared_prefix_text_fork(
            agent,
            kind="use_agent",
            prompt_text=fork_prompt,
        )
    except Exception as exc:
        return {"status": "error", "content": [{"text": f"use_agent failed: {exc}"}]}

    if result.used_tool:
        return {"status": "error", "content": [{"text": "use_agent rejected a tool call from the text-only fork"}]}

    text = str(result.text or "").strip()
    if not text:
        return {"status": "error", "content": [{"text": "use_agent produced no text output"}]}

    return {"status": "success", "content": [{"text": extract_text({"content": [{"text": text}]})}]}


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


set_permissions(use_agent, "execute")
set_permissions(use_llm, "execute")
