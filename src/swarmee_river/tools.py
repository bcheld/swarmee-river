from __future__ import annotations

import importlib
import os
from typing import Any

from swarmee_river.opencode_aliases import configure_alias_targets, opencode_alias_tools

# Custom tools (packaged + hot-loaded from ./tools)
from tools import (
    agent_graph,
    artifact,
    file_ops,
    git,
    patch_apply,
    path_ops,
    project_context,
    run_checks,
    sop,
    store_in_kb,
    strand,
    swarm,
    todo,
    welcome,
)
from tools.editor import editor as editor_fallback
from tools.file_write import file_write as file_write_fallback
from tools.python_repl import python_repl as python_repl_fallback
from tools.retrieve import retrieve as retrieve_fallback
from tools.shell import shell as shell_fallback


def _truthy_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _load_strands_tool(name: str) -> Any | None:
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
    tools.setdefault("file_write", file_write_fallback)
    tools.setdefault("editor", editor_fallback)
    tools.setdefault("retrieve", retrieve_fallback)

    # Packaged custom tools
    custom_tools: dict[str, Any] = {
        # Core repository navigation primitives (safe, non-shell).
        "file_list": file_ops.file_list,
        "file_search": file_ops.file_search,
        "file_read": file_ops.file_read,
        "glob": path_ops.glob,
        "list": path_ops.list,
        "store_in_kb": store_in_kb,
        "strand": strand,
        "welcome": welcome,
        "sop": sop,
        "artifact": artifact,
        "git": git,
        "patch_apply": patch_apply,
        "run_checks": run_checks,
        "todoread": todo.todoread,
        "todowrite": todo.todowrite,
        # Override any `strands_tools.agent_graph` with a cancellable implementation.
        "agent_graph": agent_graph,
        # Override any `strands_tools.swarm` with a cancellable implementation.
        "swarm": swarm,
    }
    if _truthy_env("SWARMEE_ENABLE_PROJECT_CONTEXT_TOOL", False):
        custom_tools["project_context"] = project_context
    tools |= custom_tools

    configure_alias_targets(tools)
    tools |= opencode_alias_tools()

    return tools
