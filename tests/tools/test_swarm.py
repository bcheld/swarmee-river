from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import tools.swarm as swarm_module


def test_create_custom_agents_uses_shared_prefix_children(monkeypatch) -> None:
    created: list[dict[str, Any]] = []

    class _ChildAgent(SimpleNamespace):
        pass

    monkeypatch.setattr(
        swarm_module,
        "create_shared_prefix_child_agent",
        lambda **kwargs: (
            created.append(kwargs) or _ChildAgent(name=None, description=None),
            SimpleNamespace(kind="swarm", parent_message_count=2, prefix_hash="prefix"),
        ),
    )

    agents = swarm_module._create_custom_agents(
        [
            {
                "name": "planner",
                "role": "Plan work",
                "system_prompt": "Focus on risk and sequencing.",
                "tools": ["shell", "file_read"],
            }
        ],
        parent_agent=SimpleNamespace(model=object()),
    )

    assert len(agents) == 1
    assert agents[0].name == "planner"
    assert agents[0].description == "Plan work"
    assert created[0]["kind"] == "swarm"
    assert created[0]["tool_allowlist"] == ["shell", "file_read"]
    assert "Focus on risk and sequencing." in str(created[0]["seed_instruction"])


@pytest.mark.asyncio
async def test_swarm_passes_shared_prefix_invocation_state(monkeypatch) -> None:
    invoke_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        swarm_module,
        "_create_custom_agents",
        lambda specs, *, parent_agent=None: [SimpleNamespace(name="planner")],
    )
    monkeypatch.setattr(
        swarm_module,
        "capture_shared_prefix_fork",
        lambda agent, *, kind: SimpleNamespace(
            kind=kind,
            parent_message_count=4,
            prefix_hash="fork-hash",
            pending_reminder="",
        ),
    )
    monkeypatch.setattr(
        swarm_module,
        "build_fork_invocation_state",
        lambda snapshot, *, extra_prompt_chars: {
            "swarmee": {
                "fork_kind": snapshot.kind,
                "fork_parent_message_count": snapshot.parent_message_count,
                "fork_prefix_hash": snapshot.prefix_hash,
                "fork_extra_prompt_chars": extra_prompt_chars,
            }
        },
    )

    class _FakeSwarm:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def invoke_async(self, task: str, invocation_state: dict[str, Any] | None = None):
            invoke_calls.append({"task": task, "invocation_state": invocation_state, "nodes": self.kwargs["nodes"]})
            return SimpleNamespace(
                status="success",
                execution_time=12,
                execution_count=1,
                node_history=[],
                results={},
                accumulated_usage=None,
            )

    monkeypatch.setattr(swarm_module, "Swarm", _FakeSwarm)

    result = await swarm_module.swarm(
        task="Investigate the cache miss",
        agents=[{"name": "planner"}],
        agent=SimpleNamespace(model=object()),
    )

    assert result["status"] == "success"
    assert invoke_calls[0]["task"] == "Investigate the cache miss"
    assert invoke_calls[0]["invocation_state"]["swarmee"]["fork_kind"] == "swarm"
    assert len(invoke_calls[0]["nodes"]) == 1
