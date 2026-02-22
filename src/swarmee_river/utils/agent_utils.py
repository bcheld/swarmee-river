from __future__ import annotations

import asyncio
import threading
from typing import Any

from strands import Agent


def null_callback_handler(**_kwargs: Any) -> None:
    return None


def run_coroutine(coro: Any) -> Any:
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

    t = threading.Thread(target=_worker, daemon=True, name="agent-utils-invoke")
    t.start()
    t.join()
    if "exc" in err:
        raise err["exc"]
    return out.get("result")


def extract_text(result: Any) -> str:
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


def create_sub_agent(*, parent_agent: Any, system_prompt: str) -> Agent:
    kwargs: dict[str, Any] = {
        "model": getattr(parent_agent, "model", None),
        "tools": [],
        "system_prompt": system_prompt,
        "messages": [],
        "callback_handler": null_callback_handler,
        "load_tools_from_directory": False,
    }
    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("load_tools_from_directory", None)
        return Agent(**kwargs)
