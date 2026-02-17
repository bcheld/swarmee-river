from __future__ import annotations

import fnmatch
import re
from typing import Any
from urllib.parse import urlparse

from swarmee_river.opencode_aliases import equivalent_tool_names, normalize_tool_name
from swarmee_river.settings import PermissionRule, SafetyConfig, ToolRule

_VALID_ACTIONS = {"allow", "ask", "deny"}
_CONDITION_KEYS = {"command_regex", "command_glob", "path_regex", "path_glob", "host_regex", "host_glob", "method"}
_HIGH_RISK_TOOLS = {
    "shell",
    "bash",
    "file_write",
    "write",
    "editor",
    "edit",
    "patch_apply",
    "patch",
    "http_request",
}


def _normalize_action(value: str | None) -> str:
    action = str(value or "ask").strip().lower()
    return action if action in _VALID_ACTIONS else "ask"


def _equivalent_names(name: str | None) -> set[str]:
    normalized = normalize_tool_name(name)
    if not normalized:
        return set()
    names = equivalent_tool_names(normalized)
    if names:
        return names
    return {normalized}


def _tool_matches(rule_tool: str | None, tool_name: str) -> bool:
    rule_names = _equivalent_names(rule_tool)
    tool_names = _equivalent_names(tool_name)
    if not rule_names or not tool_names:
        return False
    return bool(rule_names.intersection(tool_names))


def _find_tool_rule(safety: SafetyConfig, tool_name: str) -> ToolRule | None:
    for rule in safety.tool_rules:
        if rule.tool == tool_name:
            return rule

    equivalent = _equivalent_names(tool_name) - {tool_name}
    for rule in safety.tool_rules:
        if rule.tool in equivalent:
            return rule
    return None


def _matches_regex(value: str, pattern: Any) -> bool:
    if not isinstance(pattern, str):
        return False
    try:
        return bool(re.search(pattern, value))
    except re.error:
        return False


def _matches_glob(value: str, pattern: Any) -> bool:
    if not isinstance(pattern, str):
        return False
    return fnmatch.fnmatchcase(value, pattern)


def _rule_when_matches(*, tool_use: Any, when: dict[str, Any]) -> bool:
    if not when:
        return True

    if any(key not in _CONDITION_KEYS for key in when):
        return False

    tool_input = tool_use.get("input") if isinstance(tool_use, dict) else None
    if not isinstance(tool_input, dict):
        return False

    if "command_regex" in when:
        command = tool_input.get("command")
        if not isinstance(command, str) or not _matches_regex(command, when.get("command_regex")):
            return False

    if "command_glob" in when:
        command = tool_input.get("command")
        if not isinstance(command, str) or not _matches_glob(command, when.get("command_glob")):
            return False

    if "path_regex" in when:
        path = tool_input.get("path")
        if not isinstance(path, str) or not _matches_regex(path, when.get("path_regex")):
            return False

    if "path_glob" in when:
        path = tool_input.get("path")
        if not isinstance(path, str) or not _matches_glob(path, when.get("path_glob")):
            return False

    if "host_regex" in when or "host_glob" in when:
        url = tool_input.get("url")
        if not isinstance(url, str):
            return False
        host = urlparse(url).hostname
        if not isinstance(host, str) or not host:
            return False
        if "host_regex" in when and not _matches_regex(host, when.get("host_regex")):
            return False
        if "host_glob" in when and not _matches_glob(host, when.get("host_glob")):
            return False

    if "method" in when:
        method = tool_input.get("method")
        expected_method = when.get("method")
        if not isinstance(method, str) or not isinstance(expected_method, str):
            return False
        if method.strip().upper() != expected_method.strip().upper():
            return False

    return True


def _find_matching_permission_rule(*, safety: SafetyConfig, tool_name: str, tool_use: Any) -> PermissionRule | None:
    for rule in safety.permission_rules:
        if not _tool_matches(rule.tool, tool_name):
            continue
        if _rule_when_matches(tool_use=tool_use, when=rule.when):
            return rule
    return None


def evaluate_declarative_rule_action(*, safety: SafetyConfig, tool_name: str, tool_use: Any) -> str | None:
    normalized_tool = normalize_tool_name(tool_name)
    if not normalized_tool:
        return None

    rule = _find_matching_permission_rule(safety=safety, tool_name=normalized_tool, tool_use=tool_use)
    if rule is None:
        return None
    return _normalize_action(rule.action)


def evaluate_permission_action(*, safety: SafetyConfig, tool_name: str, tool_use: Any) -> tuple[str, bool]:
    normalized_tool = normalize_tool_name(tool_name)
    if not normalized_tool:
        return "allow", True

    permission_rule = _find_matching_permission_rule(safety=safety, tool_name=normalized_tool, tool_use=tool_use)
    if permission_rule is not None:
        return _normalize_action(permission_rule.action), bool(permission_rule.remember)

    tool_rule = _find_tool_rule(safety, normalized_tool)
    if tool_rule is not None:
        return _normalize_action(tool_rule.default), bool(tool_rule.remember)

    if normalized_tool in _HIGH_RISK_TOOLS:
        return _normalize_action(safety.tool_consent), True

    return "allow", True
