from __future__ import annotations

import importlib
from typing import Any, Optional

# Custom tools (packaged + hot-loaded from ./tools)
from tools import agent_graph, artifact, git, patch_apply, project_context, run_checks, sop, store_in_kb, strand, swarm, welcome

from tools.python_repl import python_repl as python_repl_fallback
from tools.shell import shell as shell_fallback


def _load_strands_tool(name: str) -> Optional[Any]:
    """
    Best-effort import of a tool from `strands_tools`.

    Some tools may be unavailable depending on platform and optional dependencies.
    """
    try:
        strands_tools = importlib.import_module("strands_tools")
        return getattr(strands_tools, name)
    except Exception:
        return None


def get_tools() -> dict[str, Any]:
    """
    Returns the collection of available agent tools.

    This function is intentionally defensive so the package can run across Windows/macOS/Linux
    even when some optional tool modules are not importable on a given platform.
    """
    tools: dict[str, Any] = {}

    # Core Strands tools (attempt to load individually)
    for tool_name in [
        "agent_graph",
        "calculator",
        "current_time",
        "editor",
        "environment",
        "file_read",
        "file_write",
        "generate_image",
        "http_request",
        "image_reader",
        "journal",
        "load_tool",
        "memory",
        "nova_reels",
        "retrieve",
        "slack",
        "speak",
        "stop",
        "swarm",
        "think",
        "use_aws",
        "use_llm",
        "workflow",
        # Optional / platform-dependent:
        "cron",
        "python_repl",
        "shell",
    ]:
        loaded = _load_strands_tool(tool_name)
        if loaded is not None:
            tools[tool_name] = loaded

    # Cross-platform fallbacks (only if the Strands Tools variant isn't available)
    tools.setdefault("shell", shell_fallback)
    tools.setdefault("python_repl", python_repl_fallback)

    # Packaged custom tools
    tools |= {
        "store_in_kb": store_in_kb,
        "strand": strand,
        "welcome": welcome,
        "sop": sop,
        "artifact": artifact,
        "project_context": project_context,
        "git": git,
        "patch_apply": patch_apply,
        "run_checks": run_checks,
        # Override any `strands_tools.agent_graph` with a cancellable implementation.
        "agent_graph": agent_graph,
        # Override any `strands_tools.swarm` with a cancellable implementation.
        "swarm": swarm,
    }

    return tools
