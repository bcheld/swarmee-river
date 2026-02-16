from __future__ import annotations

import builtins
import importlib


def test_agent_graph_imports_without_strands_tools(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if str(name).startswith("strands_tools"):
            raise ImportError("simulated missing strands_tools")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import tools.agent_graph as agent_graph

    importlib.reload(agent_graph)

