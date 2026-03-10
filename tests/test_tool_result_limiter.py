from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.hooks.tool_result_limiter import ToolResultLimiterHooks


def test_tool_result_limiter_caps_file_read_results_and_uses_compact_artifact_reference(tmp_path):
    hooks = ToolResultLimiterHooks(artifacts_dir=tmp_path, max_text_chars=8000)
    event = SimpleNamespace(
        tool_use={"toolUseId": "tool-1", "name": "file_read"},
        result={"content": [{"text": "x" * 5000}]},
    )

    hooks.after_tool_call(event)

    content = event.result["content"][0]["text"]
    assert len(content) < 4500
    assert "artifact=" in content
    assert str(tmp_path) not in content


def test_tool_result_limiter_keeps_generic_tool_limit_for_non_read_tools(tmp_path):
    hooks = ToolResultLimiterHooks(artifacts_dir=tmp_path, max_text_chars=8000)
    event = SimpleNamespace(
        tool_use={"toolUseId": "tool-2", "name": "shell"},
        result={"content": [{"text": "x" * 8200}]},
    )

    hooks.after_tool_call(event)

    content = event.result["content"][0]["text"]
    assert "kept 8000 chars" in content
