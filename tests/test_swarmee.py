#!/usr/bin/env python3
"""
Unit tests for the swarmee.py module using pytest
"""

import os
import sys
import asyncio
from unittest import mock

import pytest

from swarmee_river import swarmee


class TestInteractiveMode:
    """Test cases for interactive mode functionality"""

    def test_interactive_mode(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Test the interactive mode of swarmee"""
        # Setup mocks
        mock_user_input.side_effect = ["test query", "exit"]

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        swarmee.main()

        # Verify welcome message was rendered
        mock_welcome_message.assert_called_once()

        # Verify user input was called with the correct parameters
        mock_user_input.assert_called_with("\n~ ", default="", keyboard_interrupt_return_default=False)

        # Verify user input was processed
        call = mock_agent.invoke_async.call_args
        assert call.args[0] == "test query"
        assert call.kwargs["invocation_state"]["swarmee"]["mode"] == "execute"
        assert "structured_output_model" not in call.kwargs
        assert "structured_output_prompt" not in call.kwargs

        # Verify goodbye message was rendered
        mock_goodbye_message.assert_called_once()

    def test_shell_command_shortcut(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Test the shell command shortcut with ! prefix"""
        # Setup mocks
        mock_user_input.side_effect = ["!ls -la", "exit"]

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        swarmee.main()

        # Verify shell was called with the command
        mock_agent.tool.shell.assert_called_with(
            command="ls -la",
            user_message_override="!ls -la",
            non_interactive_mode=True,
            record_direct_tool_call=False,
        )

    def test_keyboard_interrupt(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Test handling of keyboard interrupt (Ctrl+C)"""
        # Setup mocks - simulate keyboard interrupt
        mock_user_input.side_effect = KeyboardInterrupt()

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function - should exit gracefully
        swarmee.main()

        # Verify goodbye message was rendered
        mock_goodbye_message.assert_called_once()

    def test_empty_input(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Test handling of empty input"""
        # Setup mocks - empty input followed by exit
        mock_user_input.side_effect = ["", "   ", "\t", "exit"]

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        swarmee.main()

        # Verify agent's methods were not called for the empty input
        mock_agent.assert_not_called()

    def test_invoke_async_compat_without_structured_output_prompt(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Ensure compatibility with SDK variants lacking structured_output_prompt kwarg."""

        async def invoke_async_no_prompt(
            prompt: str,
            *,
            invocation_state: dict[str, object],
            structured_output_model: type[object] | None = None,
        ):
            del invocation_state
            del structured_output_model
            return mock.MagicMock(structured_output=None, message=[{"role": "assistant", "content": [{"text": "ok"}]}])

        mock_agent.invoke_async = invoke_async_no_prompt
        mock_user_input.side_effect = ["test query", "exit"]
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        swarmee.main()

    def test_plan_generation_sets_structured_output_tool_allowlist(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        from swarmee_river.planning import PlanStep, WorkPlan

        plan = WorkPlan(summary="Fix a bug", steps=[PlanStep(description="Inspect failing test", tools_expected=["file_read"])])
        mock_agent.invoke_async = mock.AsyncMock(return_value=mock.MagicMock(structured_output=plan, message=[]))

        monkeypatch.setattr(sys, "argv", ["swarmee", "fix", "the", "bug"])

        swarmee.main()

        call = mock_agent.invoke_async.call_args
        sw_state = call.kwargs["invocation_state"]["swarmee"]
        assert sw_state["mode"] == "plan"
        assert "WorkPlan" in sw_state.get("plan_allowed_tools", [])

    @mock.patch.object(swarmee, "get_user_input")
    @mock.patch.object(swarmee, "Agent")
    @mock.patch.object(swarmee, "render_goodbye_message")
    def test_keyboard_interrupt_exception(self, mock_goodbye, mock_agent, mock_input):
        """Test handling of KeyboardInterrupt exception in interactive mode"""
        # Setup mocks
        mock_agent_instance = mock.MagicMock()
        mock_agent.return_value = mock_agent_instance

        # Simulate KeyboardInterrupt when getting input
        mock_input.side_effect = KeyboardInterrupt()

        # Run main
        with mock.patch.object(sys, "argv", ["swarmee"]):
            swarmee.main()

        # Verify goodbye message was called
        mock_goodbye.assert_called_once()

    @mock.patch.object(swarmee, "get_user_input")
    @mock.patch.object(swarmee, "Agent")
    @mock.patch.object(swarmee, "render_goodbye_message")
    def test_eof_error_exception(self, mock_goodbye, mock_agent, mock_input):
        """Test handling of EOFError exception in interactive mode"""
        # Setup mocks
        mock_agent_instance = mock.MagicMock()
        mock_agent.return_value = mock_agent_instance

        # Simulate EOFError when getting input
        mock_input.side_effect = EOFError()

        # Run main
        with mock.patch.object(sys, "argv", ["swarmee"]):
            swarmee.main()

        # Verify goodbye message was called
        mock_goodbye.assert_called_once()

    @mock.patch.object(swarmee, "get_user_input")
    @mock.patch.object(swarmee, "Agent")
    @mock.patch.object(swarmee, "print")
    @mock.patch.object(swarmee, "callback_handler")
    def test_general_exception_handling(self, mock_callback_handler, mock_print, mock_agent, mock_input):
        """Test handling of general exceptions in interactive mode"""
        # Setup mocks
        mock_agent_instance = mock.MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke_async = mock.AsyncMock(
            return_value=mock.MagicMock(structured_output=None, message=[])
        )

        # First return valid input, then cause exception, then exit
        mock_input.side_effect = ["test input", Exception("Test error"), "exit"]

        # Run main
        with mock.patch.object(sys, "argv", ["swarmee"]), mock.patch.object(swarmee, "render_goodbye_message"):
            swarmee.main()

        # Verify error was printed
        mock_print.assert_any_call("\nError: Test error")

        # Verify callback_handler was called to stop spinners
        mock_callback_handler.assert_called_once_with(force_stop=True)


class TestCommandLine:
    """Test cases for command line mode functionality"""

    def test_command_line_query(self, mock_agent, mock_bedrock, mock_load_prompt, monkeypatch):
        """Test processing a query from command line arguments"""
        # Mock sys.argv with a test query
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        # Call the main function
        swarmee.main()

        # Verify agent was called with the query
        call = mock_agent.invoke_async.call_args
        assert call.args[0] == "test query"
        assert call.kwargs["invocation_state"]["swarmee"]["mode"] == "execute"
        assert "structured_output_model" not in call.kwargs
        assert "structured_output_prompt" not in call.kwargs

    def test_command_line_query_with_kb(
        self, mock_agent, mock_bedrock, mock_load_prompt, mock_store_conversation, monkeypatch
    ):
        """Test processing a query with knowledge base from command line"""
        # Mock sys.argv with a test query and KB ID
        monkeypatch.setattr(sys, "argv", ["swarmee", "--kb", "test-kb-id", "test", "query"])

        # Call the main function
        swarmee.main()

        # Verify retrieve was called
        mock_agent.tool.retrieve.assert_called_with(text="test query", knowledgeBaseId="test-kb-id")

        # Verify conversation was stored
        mock_store_conversation.assert_called_with(mock_agent, "test query", mock.ANY, "test-kb-id")

    @mock.patch.object(swarmee, "Agent")
    @mock.patch.object(swarmee, "store_conversation_in_kb")
    def test_command_line_with_kb_environment(self, mock_store, mock_agent):
        """Test command line mode with KB from environment variable"""
        # Setup mocks
        mock_agent_instance = mock.MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke_async = mock.AsyncMock(
            return_value=mock.MagicMock(structured_output=None, message=[])
        )

        # Run main with test query and environment variable
        with (
            mock.patch.object(sys, "argv", ["swarmee", "test", "query"]),
            mock.patch.dict(os.environ, {"SWARMEE_KNOWLEDGE_BASE_ID": "env-kb-id"}),
        ):
            swarmee.main()

        # Verify retrieve was called with the right KB ID
        mock_agent_instance.tool.retrieve.assert_called_once_with(text="test query", knowledgeBaseId="env-kb-id")

        # Verify store_conversation_in_kb was called
        mock_store.assert_called_once_with(mock_agent_instance, "test query", mock.ANY, "env-kb-id")


class TestConfiguration:
    """Test cases for configuration handling"""

    def test_environment_variables(self, mock_agent, mock_bedrock, mock_load_prompt, monkeypatch):
        """Test handling of environment variables"""
        # Set environment variables
        monkeypatch.setenv("STRANDS_SYSTEM_PROMPT", "Custom prompt from env")

        # Mock sys.argv with a test query
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        # Call the main function
        swarmee.main()

        # Verify load_system_prompt was called
        mock_load_prompt.assert_called_once()

        # Verify agent was called with the correct prompt
        call = mock_agent.invoke_async.call_args
        assert call.args[0] == "test query"
        assert call.kwargs["invocation_state"]["swarmee"]["mode"] == "execute"
        assert "structured_output_model" not in call.kwargs
        assert "structured_output_prompt" not in call.kwargs

    def test_kb_environment_variable(
        self, mock_agent, mock_bedrock, mock_load_prompt, mock_store_conversation, monkeypatch
    ):
        """Test handling of knowledge base environment variable"""
        # Set environment variables
        monkeypatch.setenv("SWARMEE_KNOWLEDGE_BASE_ID", "env-kb-id")

        # Mock sys.argv with a test query
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        # Call the main function
        swarmee.main()

        # Verify retrieve was called with the right KB ID
        mock_agent.tool.retrieve.assert_called_with(text="test query", knowledgeBaseId="env-kb-id")

        # Verify conversation was stored
        mock_store_conversation.assert_called_with(mock_agent, "test query", mock.ANY, "env-kb-id")


class TestErrorHandling:
    """Test cases for error handling"""

    def test_general_exception(self, mock_agent, mock_bedrock, mock_load_prompt, monkeypatch, capfd):
        """Test handling of general exceptions"""
        # Make agent raise an exception
        mock_agent.invoke_async.side_effect = Exception("Test error")

        # Mock sys.argv with a test query
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        # Call the main function
        with pytest.raises(Exception, match="Test error"):
            swarmee.main()

        # Ensure the test passes without checking stderr
        assert True


class TestShellCommandError:
    """Test shell command error handling"""

    @mock.patch("builtins.print")
    def test_shell_command_exception(
        self, mock_print, mock_agent, mock_bedrock, mock_load_prompt, mock_user_input, mock_welcome_message, monkeypatch
    ):
        """Test handling exceptions when executing shell commands"""
        # Setup mocks
        mock_user_input.side_effect = ["!failing-command", "exit"]

        # Configure shell command to raise an exception
        mock_agent_instance = mock_agent
        mock_agent_instance.tool.shell.side_effect = Exception("Shell command failed")

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        with mock.patch.object(swarmee, "render_goodbye_message"):
            swarmee.main()

        # Verify error was printed
        mock_print.assert_any_call("Shell command execution error: Shell command failed")


class TestKnowledgeBaseIntegration:
    """Test cases for knowledge base integration"""

    def test_interactive_mode_with_kb(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        mock_store_conversation,
        monkeypatch,
    ):
        """Test interactive mode with knowledge base"""
        # Setup mocks
        mock_user_input.side_effect = ["test query", "exit"]

        # Configure environment
        monkeypatch.setenv("SWARMEE_KNOWLEDGE_BASE_ID", "test-kb-id")

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        swarmee.main()

        # Verify retrieve was called with knowledge base ID
        mock_agent.tool.retrieve.assert_called_with(text="test query", knowledgeBaseId="test-kb-id")

        # Verify store_conversation_in_kb was called
        mock_store_conversation.assert_called_once()

    def test_welcome_message_with_kb(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        monkeypatch,
    ):
        """Test that welcome text is included in system prompt when enabled"""
        # Setup mocks
        mock_user_input.side_effect = ["test query", "exit"]

        monkeypatch.setattr(swarmee, "read_welcome_text", lambda: "Custom welcome text")

        # Mock load_system_prompt
        base_system_prompt = "Base system prompt"
        mock_load_prompt.return_value = base_system_prompt

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee", "--include-welcome-in-prompt"])

        # Call the main function
        with mock.patch.object(swarmee, "render_welcome_message"), mock.patch.object(swarmee, "render_goodbye_message"):
            swarmee.main()

        # Verify system prompt includes both base prompt and welcome text
        assert base_system_prompt in mock_agent.system_prompt
        assert "Custom welcome text" in mock_agent.system_prompt

    def test_welcome_message_failure(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        monkeypatch,
    ):
        """Test handling of welcome message retrieval failure"""
        # Setup mocks
        mock_user_input.side_effect = ["test query", "exit"]

        monkeypatch.setattr(swarmee, "read_welcome_text", lambda: "")

        # Mock load_system_prompt
        base_system_prompt = "Base system prompt"
        mock_load_prompt.return_value = base_system_prompt

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee", "--include-welcome-in-prompt"])

        # Call the main function
        with mock.patch.object(swarmee, "render_welcome_message"), mock.patch.object(swarmee, "render_goodbye_message"):
            swarmee.main()

        # Verify agent was called with system prompt that excludes welcome text reference
        assert mock_agent.system_prompt == base_system_prompt


class TestToolConsentPrompt:
    """Tests for consent prompt UX wiring."""

    def test_consent_prompt_uses_repl_style_input(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        class FakeToolConsentHooks:
            def __init__(self, *args, **kwargs):
                captured["prompt"] = kwargs["prompt"]

        monkeypatch.setattr(swarmee, "ToolConsentHooks", FakeToolConsentHooks)
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        swarmee.main()

        prompt_fn = captured.get("prompt")
        assert callable(prompt_fn)

        consent_text = "Allow tool 'shell'? [y]es/[n]o/[a]lways/[v]never:"
        with (
            mock.patch.object(swarmee, "callback_handler") as mock_callback_handler,
            mock.patch.object(swarmee, "get_user_input", return_value="y") as mock_user_input,
            mock.patch.object(swarmee, "print") as mock_print,
        ):
            response = prompt_fn(consent_text)  # type: ignore[operator]

        assert response == "y"
        mock_callback_handler.assert_called_once_with(force_stop=True)
        mock_print.assert_any_call(f"\n[tool consent] {consent_text}")
        mock_user_input.assert_called_once_with("\n~ consent> ", default="", keyboard_interrupt_return_default=True)

    def test_get_user_input_compat_uses_prompt_toolkit_when_event_loop_running(self):
        async def _run() -> str:
            with (
                mock.patch.object(swarmee, "_prompt_input_with_prompt_toolkit", return_value="y") as mock_prompt,
                mock.patch.object(swarmee, "get_user_input", side_effect=AssertionError("should not be called")),
            ):
                result = swarmee._get_user_input_compat(
                    "\n~ consent> ", default="", keyboard_interrupt_return_default=True
                )

            mock_prompt.assert_called_once_with(
                "\n~ consent> ",
                default="",
                keyboard_interrupt_return_default=True,
            )
            return result

        assert asyncio.run(_run()) == "y"
