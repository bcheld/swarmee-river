from __future__ import annotations

import importlib
from typing import Any

from swarmee_river.opencode_aliases import configure_alias_targets, opencode_alias_tools
from swarmee_river.utils.env_utils import truthy_env
from swarmee_river.utils.import_utils import load_optional_attr

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
from tools.calculator import calculator as calculator_fallback
from tools.current_time import current_time as current_time_fallback
from tools.editor import editor as editor_fallback
from tools.environment import environment as environment_fallback
from tools.file_write import file_write as file_write_fallback
from tools.http_request import http_request as http_request_fallback
from tools.python_repl import python_repl as python_repl_fallback
from tools.retrieve import retrieve as retrieve_fallback
from tools.shell import shell as shell_fallback
from tools.use_agent import use_agent as use_agent_fallback
from tools.use_agent import use_llm as use_llm_fallback

_OPTIONAL_STRANDS_TOOL_NAMES: tuple[str, ...] = (
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
    # `use_llm` was deprecated in some Strands Tools versions in favor of `use_agent`.
    "use_agent",
    "use_llm",
    "workflow",
    # Optional / platform-dependent:
    "cron",
    "python_repl",
    "shell",
)

_FALLBACK_TOOLS: dict[str, Any] = {
    "shell": shell_fallback,
    "python_repl": python_repl_fallback,
    "file_write": file_write_fallback,
    "editor": editor_fallback,
    "retrieve": retrieve_fallback,
    "http_request": http_request_fallback,
    "calculator": calculator_fallback,
    "current_time": current_time_fallback,
    "environment": environment_fallback,
    "use_agent": use_agent_fallback,
    "use_llm": use_llm_fallback,
}

_CUSTOM_TOOLS: dict[str, Any] = {
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


def _load_optional_strands_tools() -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for tool_name in _OPTIONAL_STRANDS_TOOL_NAMES:
        loaded = load_optional_attr("strands_tools", tool_name, import_module=importlib.import_module)
        if loaded is not None:
            tools[tool_name] = loaded
    return tools


def get_tools() -> dict[str, Any]:
    """
    Returns the collection of available agent tools.

    This function is intentionally defensive so the package can run across Windows/macOS/Linux
    even when some optional tool modules are not importable on a given platform.
    """
    tools = _load_optional_strands_tools()
    for name, fallback in _FALLBACK_TOOLS.items():
        tools.setdefault(name, fallback)

    custom_tools = dict(_CUSTOM_TOOLS)
    if truthy_env("SWARMEE_ENABLE_PROJECT_CONTEXT_TOOL", False):
        custom_tools["project_context"] = project_context
    tools |= custom_tools

    configure_alias_targets(tools)
    tools |= opencode_alias_tools()

    return tools
