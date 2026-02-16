from __future__ import annotations

import asyncio
import threading
from typing import Any

from strands import Agent, tool


def _null_callback_handler(**_kwargs: Any) -> None:
    return None


def _run_coroutine(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    out: dict[str, Any] = {}
    err: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            out["result"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            err["exc"] = e

    t = threading.Thread(target=_worker, daemon=True, name="use-agent-invoke")
    t.start()
    t.join()
    if "exc" in err:
        raise err["exc"]
    return out.get("result")


def _extract_text(result: Any) -> str:
    message = getattr(result, "message", None)
    if isinstance(message, list):
        for item in message:
            if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text").strip():
                return str(item.get("text")).strip()
    if isinstance(result, dict):
        msg = result.get("message")
        if isinstance(msg, list):
            for item in msg:
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text").strip():
                    return str(item.get("text")).strip()
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text").strip():
                    return str(item.get("text")).strip()
    return str(result or "").strip()


def _create_sub_agent(*, parent_agent: Any, system_prompt: str) -> Agent:
    kwargs: dict[str, Any] = {
        "model": getattr(parent_agent, "model", None),
        "tools": [],
        "system_prompt": system_prompt,
        "messages": [],
        "callback_handler": _null_callback_handler,
        "load_tools_from_directory": False,
    }
    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("load_tools_from_directory", None)
        return Agent(**kwargs)


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

    sub = _create_sub_agent(
        parent_agent=agent,
        system_prompt=str(system_prompt or "").strip() or "You are a helpful assistant.",
    )

    try:
        result = _run_coroutine(sub.invoke_async(effective_prompt))
    except Exception as exc:
        return {"status": "error", "content": [{"text": f"use_agent failed: {exc}"}]}

    return {"status": "success", "content": [{"text": _extract_text(result)}]}


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
