from __future__ import annotations

import asyncio
import copy
import threading
from dataclasses import dataclass
from typing import Any

from strands import Agent

from swarmee_river.utils.fork_utils import create_shared_prefix_child_agent


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


@dataclass
class AgentTurnStateSnapshot:
    messages: Any
    state: Any
    prompt_cache_pending: list[str]
    prompt_cache_sent_hashes: dict[str, str]


def _safe_deepcopy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def capture_agent_turn_state(agent: Any, *, prompt_cache: Any | None = None) -> AgentTurnStateSnapshot:
    return AgentTurnStateSnapshot(
        messages=_safe_deepcopy(getattr(agent, "messages", [])),
        state=_safe_deepcopy(getattr(agent, "state", None)),
        prompt_cache_pending=list(getattr(prompt_cache, "pending", []) or []),
        prompt_cache_sent_hashes=dict(getattr(prompt_cache, "sent_hashes", {}) or {}),
    )


def restore_prompt_cache_turn_state(prompt_cache: Any | None, snapshot: AgentTurnStateSnapshot) -> None:
    if prompt_cache is None:
        return
    pending = getattr(prompt_cache, "pending", None)
    if isinstance(pending, list):
        pending[:] = list(snapshot.prompt_cache_pending)
    else:
        try:
            prompt_cache.pending = list(snapshot.prompt_cache_pending)
        except Exception:
            pass
    sent_hashes = getattr(prompt_cache, "sent_hashes", None)
    if isinstance(sent_hashes, dict):
        sent_hashes.clear()
        sent_hashes.update(dict(snapshot.prompt_cache_sent_hashes))
    else:
        try:
            prompt_cache.sent_hashes = dict(snapshot.prompt_cache_sent_hashes)
        except Exception:
            pass


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
    agent, _snapshot = create_shared_prefix_child_agent(
        parent_agent=parent_agent,
        kind="subagent",
        seed_instruction=str(system_prompt or "").strip() or None,
        tool_allowlist=["__swarmee_text_only__"],
        callback_handler=null_callback_handler,
    )
    return agent
