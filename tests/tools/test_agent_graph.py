from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

from swarmee_river.utils.fork_utils import SharedPrefixTextForkResult


def test_agent_graph_imports_without_strands_tools(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if str(name).startswith("strands_tools"):
            raise ImportError("simulated missing strands_tools")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import tools.agent_graph as agent_graph

    importlib.reload(agent_graph)


def test_agent_graph_node_fork_uses_shared_prefix(monkeypatch) -> None:
    import tools.agent_graph as agent_graph

    calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        agent_graph,
        "run_shared_prefix_text_fork",
        lambda agent, *, kind, prompt_text, extra_fields=None: (
            calls.append({"kind": kind, "prompt_text": prompt_text}) or
            SharedPrefixTextForkResult(
                text="node output",
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "node output"}]},
                used_tool=False,
                diagnostics={},
            )
        ),
    )

    out = agent_graph._invoke_llm_text(
        parent_agent=SimpleNamespace(model=object()),
        system_prompt="Review diffs",
        prompt="Inspect the patch",
    )

    assert out == "node output"
    assert calls and calls[0]["kind"] == "agent_graph"
    assert "Review diffs" in calls[0]["prompt_text"]
