from unittest import mock

from swarmee_river.utils.kb_utils import load_system_prompt, store_conversation_in_kb


def test_load_system_prompt_from_prompt_assets():
    """load_system_prompt should resolve prompt via default prompt asset."""
    with (
        mock.patch("swarmee_river.prompt_assets.resolve_orchestrator_prompt_from_agent") as mock_resolve,
    ):
        mock_resolve.return_value = "Prompt from asset"
        prompt = load_system_prompt()
        assert prompt == "Prompt from asset"
        mock_resolve.assert_called_once_with(None)


def test_load_system_prompt_defaults_when_resolver_fails():
    """load_system_prompt should fall back to static default on resolver failures."""
    with mock.patch(
        "swarmee_river.prompt_assets.resolve_orchestrator_prompt_from_agent",
        side_effect=RuntimeError("boom"),
    ):
        prompt = load_system_prompt()
        assert prompt == "You are a helpful assistant."


def test_store_conversation_in_kb():
    """Test storing conversation in knowledge base with reasoning"""
    # Create mock agent
    mock_agent = mock.MagicMock()

    # Create mock response with reasoning and text
    mock_response = mock.MagicMock()
    mock_response.message = [
        {"reasoningContent": {"reasoningText": {"text": "Test reasoning"}}},
        {"text": "Test response"},
    ]

    # Call the function
    store_conversation_in_kb(mock_agent, "Test query", mock_response, "test-kb-id")

    # Verify agent called store_in_kb with correct parameters
    expected_content = "User: Test query\n\nAssistant Reasoning: Test reasoning\n\nAssistant Response: Test response"
    mock_agent.tool.store_in_kb.assert_called_with(
        content=expected_content,
        title="Conversation: Test query",
        knowledge_base_id="test-kb-id",
        record_direct_tool_call=False,
    )


def test_store_conversation_without_reasoning():
    """Test storing conversation without reasoning content"""
    # Create mock agent
    mock_agent = mock.MagicMock()

    # Create mock response with only text
    mock_response = mock.MagicMock()
    mock_response.message = [{"text": "Test response"}]

    # Call the function
    store_conversation_in_kb(mock_agent, "Test query", mock_response, "test-kb-id")

    # Verify agent called store_in_kb with correct parameters
    expected_content = "User: Test query\n\nAssistant: Test response"
    mock_agent.tool.store_in_kb.assert_called_with(
        content=expected_content,
        title="Conversation: Test query",
        knowledge_base_id="test-kb-id",
        record_direct_tool_call=False,
    )


def test_store_conversation_empty_response():
    """Test storing conversation with empty response"""
    # Create mock agent
    mock_agent = mock.MagicMock()

    # Create mock response with empty message that won't raise exception on str()
    class MockResponse:
        def __init__(self):
            self.message = []

        def __str__(self):
            return ""

    mock_response = MockResponse()

    # Call the function
    store_conversation_in_kb(mock_agent, "Test query", mock_response, "test-kb-id")

    # Verify agent called store_in_kb with correct parameters
    mock_agent.tool.store_in_kb.assert_called_once()
    # Check that the content parameter contains the user query
    call_args = mock_agent.tool.store_in_kb.call_args
    called_content = call_args.kwargs["content"]
    assert "User: Test query" in called_content
    assert "knowledge_base_id" in call_args.kwargs
    assert call_args.kwargs["knowledge_base_id"] == "test-kb-id"


def test_store_conversation_direct_mode():
    """Test storing conversation in direct mode (without response)"""
    # Create mock agent
    mock_agent = mock.MagicMock()

    # Call the function with only query and KB ID
    store_conversation_in_kb(mock_agent, "Test query", knowledge_base_id="test-kb-id")

    # Verify agent called store_in_kb with just the user query
    expected_content = "User: Test query"
    mock_agent.tool.store_in_kb.assert_called_with(
        content=expected_content,
        title="Conversation: Test query",
        knowledge_base_id="test-kb-id",
        record_direct_tool_call=False,
    )


def test_store_conversation_no_kb():
    """Test store_conversation_in_kb when no knowledge_base_id is provided"""
    # This should just return without error
    agent_mock = mock.MagicMock()
    result = store_conversation_in_kb(agent_mock, "test input", None, None)
    assert result is None
    agent_mock.assert_not_called()  # Agent should not be used when no KB ID


def test_store_conversation_exception_handling():
    """Test exception handling in store_conversation_in_kb"""
    agent_mock = mock.MagicMock()
    agent_mock.tool.store_in_kb.side_effect = Exception("Test exception")

    with mock.patch("builtins.print") as mock_print:
        store_conversation_in_kb(agent_mock, "test input", None, "test-kb")

        # Verify error was printed
        mock_print.assert_called_once()
        assert "Error storing conversation" in mock_print.call_args[0][0]


def test_store_conversation_response_parsing_error():
    """Test error handling during response parsing in store_conversation_in_kb"""
    agent_mock = mock.MagicMock()

    # Create a response that will cause a parsing error
    complex_response = mock.MagicMock()
    complex_response.message = [{"invalid_structure": True}]

    with mock.patch("builtins.print"):
        store_conversation_in_kb(agent_mock, "test input", complex_response, "test-kb")

        # Verify the function handled the error and continued
        agent_mock.tool.store_in_kb.assert_called_once()

        # The content should include the user input
        call_kwargs = agent_mock.tool.store_in_kb.call_args.kwargs
        assert "test input" in call_kwargs["content"]


def test_store_conversation_exception_in_response_str():
    """Test storing conversation when response string conversion fails"""
    # Create mock agent
    mock_agent = mock.MagicMock()

    # Create mock response that raises exception when stringified
    class ExceptionResponse:
        def __init__(self):
            self.message = [{"complex": "structure"}]

        def __str__(self):
            # This is the case we need to test - when str(response) raises an exception
            raise Exception("Cannot convert to string")

    mock_response = ExceptionResponse()

    # Mock print function to capture errors
    with mock.patch("builtins.print"):
        # Call the function with knowledge_base_id
        store_conversation_in_kb(mock_agent, "Test query", mock_response, "test-kb-id")

        # The function should handle the exception and call store_in_kb
        # The knowledge_base_id parameter should get passed through
        mock_agent.tool.store_in_kb.assert_called_with(
            content="User: Test query\n\nAssistant: ",
            title="Conversation: Test query",
            knowledge_base_id="test-kb-id",
            record_direct_tool_call=False,
        )
