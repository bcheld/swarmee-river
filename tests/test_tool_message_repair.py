from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.hooks.tool_message_repair import ToolMessageRepairHooks


def test_tool_message_repair_removes_orphaned_tool_uses() -> None:
    hook = ToolMessageRepairHooks(enabled=True)

    messages = [
        {
            "role": "assistant",
            "content": [
                {"text": "I will run a tool."},
                {"toolUse": {"toolUseId": "tool-1", "name": "shell", "input": {"command": "echo hi"}}},
            ],
        },
        {"role": "user", "content": [{"text": "continue"}]},
    ]
    event = SimpleNamespace(messages=messages, invocation_state={"messages": list(messages), "swarmee": {}})

    hook.before_model_call(event)

    repaired_messages = event.messages
    assert isinstance(repaired_messages, list)
    assert repaired_messages[0]["content"] == [{"text": "I will run a tool."}]
    assert event.invocation_state["swarmee"]["repaired_tool_use_ids"] == ["tool-1"]


def test_tool_message_repair_leaves_valid_tool_chains_untouched() -> None:
    hook = ToolMessageRepairHooks(enabled=True)

    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "tool-1", "name": "shell", "input": {"command": "echo hi"}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "tool-1", "status": "success", "content": [{"text": "hi"}]}}],
        },
    ]
    event = SimpleNamespace(messages=messages, invocation_state={"messages": messages, "swarmee": {}})

    hook.before_model_call(event)

    assert event.messages == messages
    assert "repaired_tool_use_ids" not in event.invocation_state["swarmee"]
