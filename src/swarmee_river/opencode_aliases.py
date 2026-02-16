from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from typing import Any, Callable

from strands import tool

from tools import file_ops
from tools.patch_apply import patch_apply as patch_apply_fallback
from tools.shell import shell as shell_fallback

OPENCODE_TOOL_ALIASES: dict[str, str] = {
    "grep": "file_search",
    "read": "file_read",
    "bash": "shell",
    "patch": "patch_apply",
    "write": "file_write",
    "edit": "editor",
}

SAFE_OPENCODE_ALIASES: frozenset[str] = frozenset({"grep", "read"})
RISKY_OPENCODE_ALIASES: frozenset[str] = frozenset({"bash", "patch", "write", "edit"})

_TARGET_TO_ALIASES: dict[str, set[str]] = {}
for alias_name, target_name in OPENCODE_TOOL_ALIASES.items():
    _TARGET_TO_ALIASES.setdefault(target_name, set()).add(alias_name)

_ALIAS_TARGETS: dict[str, Callable[..., Any]] = {}

_FALLBACK_TARGETS: dict[str, Callable[..., Any]] = {
    "file_search": file_ops.file_search,
    "file_read": file_ops.file_read,
    "shell": shell_fallback,
    "patch_apply": patch_apply_fallback,
}


def normalize_tool_name(name: str | None) -> str:
    return str(name or "").strip()


def canonical_tool_name(name: str | None) -> str:
    normalized = normalize_tool_name(name)
    if not normalized:
        return ""
    return OPENCODE_TOOL_ALIASES.get(normalized, normalized)


def equivalent_tool_names(name: str | None) -> set[str]:
    canonical = canonical_tool_name(name)
    if not canonical:
        return set()
    names = {canonical}
    names.update(_TARGET_TO_ALIASES.get(canonical, set()))
    return names


def configure_alias_targets(tools: Mapping[str, Any]) -> None:
    resolved_targets: dict[str, Callable[..., Any]] = {}
    for alias_name, target_name in OPENCODE_TOOL_ALIASES.items():
        candidate = tools.get(target_name)
        if callable(candidate):
            resolved_targets[alias_name] = candidate
    _ALIAS_TARGETS.clear()
    _ALIAS_TARGETS.update(resolved_targets)


def _load_strands_tool(name: str) -> Callable[..., Any] | None:
    try:
        strands_tools = importlib.import_module("strands_tools")
        loaded = getattr(strands_tools, name)
    except Exception:
        return None
    return loaded if callable(loaded) else None


def _resolve_alias_target(alias_name: str) -> tuple[str, Callable[..., Any] | None]:
    target_name = OPENCODE_TOOL_ALIASES.get(alias_name, "")
    if not target_name:
        return "", None

    configured = _ALIAS_TARGETS.get(alias_name)
    if callable(configured):
        return target_name, configured

    loaded = _load_strands_tool(target_name)
    if callable(loaded):
        return target_name, loaded

    fallback = _FALLBACK_TARGETS.get(target_name)
    if callable(fallback):
        return target_name, fallback
    return target_name, None


def _invoke_with_supported_kwargs(target: Callable[..., Any], payload: dict[str, Any]) -> Any:
    try:
        return target(**payload)
    except TypeError:
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            raise
        parameters = signature.parameters.values()
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
            raise
        accepted_keys = {
            param.name
            for param in signature.parameters.values()
            if param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        filtered_payload = {key: value for key, value in payload.items() if key in accepted_keys}
        if filtered_payload == payload:
            raise
        return target(**filtered_payload)


def _invoke_alias(alias_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    target_name, target = _resolve_alias_target(alias_name)
    if target is None:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"Alias '{alias_name}' unavailable: underlying tool '{target_name}' "
                        "is not installed in this environment."
                    )
                }
            ],
        }

    try:
        result = _invoke_with_supported_kwargs(target, payload)
    except Exception as exc:
        return {
            "status": "error",
            "content": [{"text": f"Alias '{alias_name}' failed via '{target_name}': {exc}"}],
        }

    if isinstance(result, dict):
        return result
    return {"status": "success", "content": [{"text": str(result)}]}


@tool
def grep(
    query: str,
    *,
    cwd: str | None = None,
    max_matches: int = 200,
    max_chars: int = 12000,
) -> dict[str, Any]:
    return _invoke_alias(
        "grep",
        {"query": query, "cwd": cwd, "max_matches": max_matches, "max_chars": max_chars},
    )


@tool
def read(
    path: str,
    *,
    cwd: str | None = None,
    start_line: int = 1,
    max_lines: int = 300,
    max_chars: int = 12000,
) -> dict[str, Any]:
    return _invoke_alias(
        "read",
        {
            "path": path,
            "cwd": cwd,
            "start_line": start_line,
            "max_lines": max_lines,
            "max_chars": max_chars,
        },
    )


@tool
def bash(
    command: str,
    *,
    cwd: str | None = None,
    timeout_s: int = 600,
    env: dict[str, str] | None = None,
    non_interactive_mode: bool = True,
    user_message_override: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "command": command,
        "cwd": cwd,
        "timeout_s": timeout_s,
        "env": env,
        "non_interactive_mode": non_interactive_mode,
    }
    if user_message_override is not None:
        payload["user_message_override"] = user_message_override
    return _invoke_alias("bash", payload)


@tool
def patch(
    patch: str,
    *,
    cwd: str | None = None,
    dry_run: bool = False,
    timeout_s: int = 60,
    max_chars: int = 12000,
    max_backup_bytes: int = 200000,
) -> dict[str, Any]:
    return _invoke_alias(
        "patch",
        {
            "patch": patch,
            "cwd": cwd,
            "dry_run": dry_run,
            "timeout_s": timeout_s,
            "max_chars": max_chars,
            "max_backup_bytes": max_backup_bytes,
        },
    )


@tool
def write(
    path: str,
    content: str,
    *,
    cwd: str | None = None,
) -> dict[str, Any]:
    return _invoke_alias(
        "write",
        {
            "path": path,
            "content": content,
            "cwd": cwd,
        },
    )


@tool
def edit(
    command: str,
    path: str,
    *,
    old_str: str | None = None,
    new_str: str | None = None,
    file_text: str | None = None,
    insert_line: int | None = None,
    view_range: list[int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "command": command,
        "path": path,
        "old_str": old_str,
        "new_str": new_str,
        "file_text": file_text,
        "insert_line": insert_line,
        "view_range": view_range,
    }
    return _invoke_alias("edit", payload)


def opencode_alias_tools() -> dict[str, Callable[..., Any]]:
    return {
        "grep": grep,
        "read": read,
        "bash": bash,
        "patch": patch,
        "write": write,
        "edit": edit,
    }
