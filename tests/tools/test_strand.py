#!/usr/bin/env python3
"""Unit tests for the strand tool."""

import asyncio
import os
from io import StringIO
from types import SimpleNamespace
from unittest import mock

from tools.strand import strand


class TestStrandTool:
    """Test cases for the strand tool"""

    def test_strand_with_query(self):
        """Test basic query processing"""
        with mock.patch("tools.strand.Agent") as mock_agent_class:
            # Setup mock agent
            mock_agent_instance = mock.MagicMock()
            mock_agent_class.return_value = mock_agent_instance
            mock_agent_instance.return_value = {"status": "success", "content": [{"text": "Agent response"}]}

            # Call the strand tool
            tool_use = {"toolUseId": "test_id", "input": {"query": "test query"}}
            # Store result to validate return value
            result = strand(tool_use)
            assert result["status"] == "success"

            # Verify agent was created and called with the query
            mock_agent_class.assert_called_once()
            mock_agent_instance.assert_called_once()

            # Verify success response
            assert result["status"] == "success"

    def test_strand_empty_query(self):
        """Test handling of empty query"""
        # Call the strand tool with empty query
        result = strand(query="")

        # Verify error response
        assert result["status"] == "error"
        assert "No query provided" in result["content"][0]["text"]

    def test_strand_custom_system_prompt(self):
        """Test using custom system prompt"""
        with mock.patch("tools.strand.Agent") as mock_agent_class:
            # Setup mock agent
            mock_agent_instance = mock.MagicMock()
            mock_agent_class.return_value = mock_agent_instance
            mock_agent_instance.return_value = {"status": "success", "content": [{"text": "Agent response"}]}

            # Call the strand tool with custom prompt
            # Store result to validate return value
            result = strand(query="test query", system_prompt="Custom system prompt")
            assert result["status"] == "success"

            # Verify agent was created with custom prompt
            mock_agent_class.assert_called_once()
            kwargs = mock_agent_class.call_args.kwargs
            assert kwargs["system_prompt"] == "Custom system prompt"

    def test_strand_specific_tools(self):
        """Test specifying particular tools"""
        with mock.patch("tools.strand.Agent") as mock_agent_class:
            # Setup mock agent
            mock_agent_instance = mock.MagicMock()
            mock_agent_class.return_value = mock_agent_instance
            mock_agent_instance.return_value = {"status": "success", "content": [{"text": "Agent response"}]}

            # Call the strand tool with specific tools
            tool_use = {"toolUseId": "test_id", "input": {"query": "test query", "tool_names": ["tool1", "tool2"]}}
            # Store result to validate return value
            result = strand(tool_use)
            assert result["status"] == "success"

            # Check that the tools were passed
            called_args = mock_agent_class.call_args.kwargs
            assert "tools" in called_args


def test_strand_uses_shared_prefix_child_agent_when_parent_present(monkeypatch):
    import tools.strand as strand_module

    created: dict[str, object] = {}
    invoke_calls: list[dict[str, object]] = []

    class _ChildAgent:
        async def invoke_async(self, query, invocation_state=None):  # noqa: ANN001
            invoke_calls.append({"query": query, "invocation_state": invocation_state})
            return {"content": [{"text": "shared-prefix output"}]}

    monkeypatch.setattr(
        strand_module,
        "create_shared_prefix_child_agent",
        lambda **kwargs: (
            created.update(kwargs) or _ChildAgent(),
            SimpleNamespace(kind="strand", parent_message_count=3, prefix_hash="fork-hash", pending_reminder=""),
        ),
    )
    monkeypatch.setattr(
        strand_module,
        "build_fork_invocation_state",
        lambda snapshot, *, extra_prompt_chars: {
            "swarmee": {
                "fork_kind": snapshot.kind,
                "fork_parent_message_count": snapshot.parent_message_count,
                "fork_prefix_hash": snapshot.prefix_hash,
                "fork_extra_prompt_chars": extra_prompt_chars,
            }
        },
    )
    monkeypatch.setattr(strand_module, "run_coroutine", lambda coro: asyncio.run(coro))

    result = strand(
        query="inspect this",
        system_prompt="Focus on diffs",
        tool_names=["shell", "file_read"],
        agent=SimpleNamespace(model=object()),
    )

    assert result["status"] == "success"
    assert "shared-prefix output" in result["content"][0]["text"]
    assert created["kind"] == "strand"
    assert "Focus on diffs" in str(created["seed_instruction"])
    assert created["tool_allowlist"] == ["shell", "file_read"]
    assert invoke_calls[0]["query"] == "inspect this"
    assert invoke_calls[0]["invocation_state"]["swarmee"]["fork_kind"] == "strand"
            # Specific test frameworks may need to be updated for exact tool count

    def test_strand_resolved_system_prompt(self):
        """Test loading system prompt from prompt asset resolver."""
        with (
            mock.patch("tools.strand.load_system_prompt", return_value="Prompt from assets"),
            mock.patch("tools.strand.Agent") as mock_agent_class,
        ):
            # Setup mock agent
            mock_agent_instance = mock.MagicMock()
            mock_agent_class.return_value = mock_agent_instance
            mock_agent_instance.return_value = {"status": "success", "content": [{"text": "Agent response"}]}

            # Call the strand tool
            tool_use = {"toolUseId": "test_id", "input": {"query": "test query"}}
            # Store result to validate return value
            result = strand(tool_use)
            assert result["status"] == "success"

            # Verify agent was created with resolved prompt
            mock_agent_class.assert_called_once()
            kwargs = mock_agent_class.call_args.kwargs
            assert kwargs["system_prompt"] == "Prompt from assets"

    def test_strand_default_system_prompt(self):
        """Test using default system prompt when no others available"""
        with (
            mock.patch("pathlib.Path.exists", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("tools.strand.Agent") as mock_agent_class,
        ):
            # Setup mock agent
            mock_agent_instance = mock.MagicMock()
            mock_agent_class.return_value = mock_agent_instance
            mock_agent_instance.return_value = {"status": "success", "content": [{"text": "Agent response"}]}

            # Call the strand tool
            tool_use = {"toolUseId": "test_id", "input": {"query": "test query"}}
            # Store result to validate return value
            result = strand(tool_use)
            assert result["status"] == "success"

            # Verify agent was created with default prompt
            mock_agent_class.assert_called_once()
            kwargs = mock_agent_class.call_args.kwargs
            assert kwargs["system_prompt"] == "You are a helpful assistant."

    def test_strand_exception_handling(self):
        """Test handling of exceptions"""
        with mock.patch("tools.strand.Agent") as mock_agent_class:
            # Make agent raise an exception
            mock_agent_class.side_effect = Exception("Test error")

            # Call the strand tool
            tool_use = {"toolUseId": "test_id", "input": {"query": "test query"}}
            # Store result to validate return value
            result = strand(tool_use)

            # Verify error response
            assert result["status"] == "error"
            assert "Error" in result["content"][0]["text"]

    def test_strand_tool_with_resolved_prompt(self):
        """Test strand tool using resolved orchestrator prompt asset."""
        with (
            mock.patch("tools.strand.load_system_prompt", return_value="Test prompt from assets"),
            mock.patch("sys.stdout", new_callable=StringIO),
            mock.patch("tools.strand.Agent") as mock_agent_class,
        ):
            # Mock the agent instance
            mock_agent_instance = mock.MagicMock()
            mock_agent_class.return_value = mock_agent_instance
            mock_agent_instance.return_value = {"status": "success", "content": [{"text": "Agent response"}]}

            # Call the strand tool
            tool_use = {"toolUseId": "test_id", "input": {"query": "test query"}}
            # Store result to validate return value
            result = strand(tool_use)
            assert result["status"] == "success"

            # Verify the system prompt was loaded from resolver
            mock_agent_class.assert_called_once()
            called_kwargs = mock_agent_class.call_args.kwargs
            assert called_kwargs["system_prompt"] == "Test prompt from assets"

    def test_strand_tool_with_specific_tools(self):
        """Test strand tool with specific tool selection"""

        with (
            mock.patch("sys.stdout", new_callable=StringIO),
            mock.patch("tools.strand.Agent") as mock_agent_class,
        ):
            # Mock the agent instance
            mock_agent_instance = mock.MagicMock()
            mock_agent_class.return_value = mock_agent_instance
            mock_agent_instance.return_value = {"status": "success", "content": [{"text": "Agent response"}]}

            # Call the strand tool with specific tools
            tool_use = {"toolUseId": "test_id", "input": {"query": "test query", "tool_names": ["shell", "editor"]}}
            _ = strand(tool_use)

            # Verify agent was initialized with the right tools
            mock_agent_class.assert_called_once()
            # Check that only the specified tools were passed
            called_args = mock_agent_class.call_args.kwargs
            assert "tools" in called_args
