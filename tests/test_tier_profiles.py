from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.hooks.tool_policy import ToolPolicyHooks


def test_tier_tool_profile_allowlist_blocks_tools() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"mode": "execute", "tier": "fast", "tool_profile": {"tool_allowlist": ["file_read"]}}},
        cancel_tool=False,
    )
    hook.before_tool_call(event)
    assert event.cancel_tool
    assert "tool_allowlist" in str(event.cancel_tool)


def test_tier_tool_profile_blocklist_blocks_tools() -> None:
    hook = ToolPolicyHooks()
    event = SimpleNamespace(
        tool_use={"name": "shell"},
        invocation_state={"swarmee": {"mode": "execute", "tier": "fast", "tool_profile": {"tool_blocklist": ["shell"]}}},
        cancel_tool=False,
    )
    hook.before_tool_call(event)
    assert event.cancel_tool
    assert "tool_blocklist" in str(event.cancel_tool)

