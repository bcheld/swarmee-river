from __future__ import annotations

import asyncio
from types import SimpleNamespace

from swarmee_river.runtime_agent_tools import (
    build_call_agent_tool_description,
    build_call_agent_tool_name,
    build_runtime_agent_tool_metadata,
    build_runtime_agent_tools,
    create_runtime_call_agent_tool,
)


def test_build_runtime_agent_tools_only_includes_activated_non_orchestrator_agents() -> None:
    tools = build_runtime_agent_tools(
        [
            {"id": "orchestrator", "name": "Orchestrator", "activated": True},
            {"id": "fast-agent", "name": "Fast Agent", "summary": "Quick repo work", "activated": True},
            {"id": "writer.bot", "name": "Writer", "activated": True},
            {"id": "inactive", "name": "Inactive", "activated": False},
        ]
    )

    assert sorted(tools) == ["call_agent_fast_agent", "call_agent_writer_bot"]


def test_runtime_agent_tool_metadata_marks_runtime_generated_entries() -> None:
    agent_def = {"id": "fast-agent", "name": "Fast Agent", "summary": "Quick repo work", "activated": True}

    metadata = build_runtime_agent_tool_metadata([agent_def])

    assert metadata == [
        {
            "name": "call_agent_fast_agent",
            "description": "Call activated Builder agent 'Fast Agent'. Quick repo work",
            "tags": ["builder-agent"],
            "access_read": False,
            "access_write": False,
            "access_execute": True,
            "source": "runtime-generated",
        }
    ]
    build_call_agent_name = build_call_agent_tool_name(agent_def)
    assert build_call_agent_name == "call_agent_fast_agent"
    assert "Fast Agent" in build_call_agent_tool_description(agent_def)


def test_create_runtime_call_agent_tool_delegates_via_shared_prefix(monkeypatch) -> None:
    import swarmee_river.runtime_agent_tools as runtime_agent_tools

    created: dict[str, object] = {}
    invoke_calls: list[dict[str, object]] = []

    class _ChildAgent:
        async def invoke_async(self, query, invocation_state=None):  # noqa: ANN001
            invoke_calls.append({"query": query, "invocation_state": invocation_state})
            return {"content": [{"text": "child agent output"}]}

    monkeypatch.setattr(runtime_agent_tools, "_resolved_agent_prompt", lambda _agent_def: "Use ripgrep first.")
    monkeypatch.setattr(
        runtime_agent_tools,
        "create_shared_prefix_child_agent",
        lambda **kwargs: (
            created.update(kwargs) or _ChildAgent(),
            SimpleNamespace(kind="call_agent:fast-agent", parent_message_count=4, prefix_hash="fork-hash"),
        ),
    )
    monkeypatch.setattr(
        runtime_agent_tools,
        "build_fork_invocation_state",
        lambda snapshot, *, extra_prompt_chars, extra_fields=None: {
            "swarmee": {
                "fork_kind": snapshot.kind,
                "fork_parent_message_count": snapshot.parent_message_count,
                "fork_prefix_hash": snapshot.prefix_hash,
                "fork_extra_prompt_chars": extra_prompt_chars,
                **(extra_fields or {}),
            }
        },
    )
    monkeypatch.setattr(runtime_agent_tools, "run_coroutine", lambda coro: asyncio.run(coro))

    tool_obj = create_runtime_call_agent_tool(
        {
            "id": "fast-agent",
            "name": "Fast Agent",
            "summary": "Quick repo work",
            "tool_names": ["file_search", "file_read"],
            "activated": True,
        }
    )

    result = tool_obj(query="search the repo", agent=SimpleNamespace(model=object()))

    assert result["status"] == "success"
    assert result["content"][0]["text"] == "child agent output"
    assert created["kind"] == "call_agent:fast-agent"
    assert created["tool_allowlist"] == ["file_search", "file_read"]
    assert "Fast Agent" in str(created["seed_instruction"])
    assert "Allowed tools: file_search, file_read" in str(created["seed_instruction"])
    assert "Use ripgrep first." in str(created["seed_instruction"])
    assert invoke_calls == [
        {
            "query": "search the repo",
            "invocation_state": {
                "swarmee": {
                    "fork_kind": "call_agent:fast-agent",
                    "fork_parent_message_count": 4,
                    "fork_prefix_hash": "fork-hash",
                    "fork_extra_prompt_chars": len("search the repo"),
                    "builder_agent_id": "fast-agent",
                    "builder_agent_tool": "call_agent_fast_agent",
                }
            },
        }
    ]
