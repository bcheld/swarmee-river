"""Tests for the per-tool permission metadata system."""

from __future__ import annotations

import pytest

from swarmee_river.tool_permissions import (
    STRANDS_TOOL_PERMISSIONS,
    get_permissions,
    set_permissions,
)


def test_set_permissions_attaches_frozenset() -> None:
    class _FakeTool:
        pass

    obj = _FakeTool()
    set_permissions(obj, "read", "write")
    perms = get_permissions(obj)
    assert perms == frozenset({"read", "write"})
    assert isinstance(perms, frozenset)


def test_set_permissions_rejects_invalid() -> None:
    class _FakeTool:
        pass

    obj = _FakeTool()
    with pytest.raises(ValueError, match="Invalid permission"):
        set_permissions(obj, "rea")


def test_get_permissions_returns_none_for_unannotated() -> None:
    class _FakeTool:
        pass

    obj = _FakeTool()
    assert get_permissions(obj) is None


def test_all_project_tools_have_permissions() -> None:
    """Every tool returned by get_tools() should have declared permissions
    or be present in the STRANDS_TOOL_PERMISSIONS fallback map."""
    from swarmee_river.tools import get_tools

    tools = get_tools()
    missing: list[str] = []
    for name, tool_obj in tools.items():
        declared = get_permissions(tool_obj)
        if declared is not None:
            continue
        if name in STRANDS_TOOL_PERMISSIONS:
            continue
        missing.append(name)

    assert not missing, f"Tools without permission metadata: {sorted(missing)}"


def test_high_risk_tools_consistent_with_permissions() -> None:
    """Every tool in _HIGH_RISK_TOOLS should have 'write' or 'execute' permission."""
    from swarmee_river.permissions import _HIGH_RISK_TOOLS
    from swarmee_river.tools import get_tools

    tools = get_tools()
    inconsistent: list[str] = []
    for name in _HIGH_RISK_TOOLS:
        tool_obj = tools.get(name)
        if tool_obj is None:
            # Alias tools may not be in the tools dict directly.
            continue
        declared = get_permissions(tool_obj)
        if declared is None:
            declared = STRANDS_TOOL_PERMISSIONS.get(name)
        if declared is None:
            inconsistent.append(f"{name}: no permissions declared")
            continue
        if "write" not in declared and "execute" not in declared:
            inconsistent.append(f"{name}: has {declared}, expected write or execute")

    assert not inconsistent, f"HIGH_RISK_TOOLS inconsistencies: {inconsistent}"


def test_plan_mode_only_allows_read_tools() -> None:
    """The permission-derived plan-mode allowlist should contain no
    write/execute tools."""
    from swarmee_river.hooks.tool_policy import _build_plan_mode_allowlist
    from swarmee_river.tools import get_tools

    tools = get_tools()
    allowed = _build_plan_mode_allowlist(tools)

    for name in allowed:
        tool_obj = tools.get(name)
        if tool_obj is None:
            continue
        declared = get_permissions(tool_obj)
        if declared is None:
            declared = STRANDS_TOOL_PERMISSIONS.get(name)
        if declared is None:
            continue
        assert "write" not in declared, f"{name} has 'write' but is in plan-mode allowlist"
        assert "execute" not in declared, f"{name} has 'execute' but is in plan-mode allowlist"
