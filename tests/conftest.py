"""
Test configuration and fixtures for pytest
"""

import os
from unittest import mock

import pytest

from swarmee_river import swarmee


@pytest.fixture(autouse=True)
def _disable_frontier_startup(monkeypatch):
    """Keep unit tests deterministic and side-effect free."""
    monkeypatch.setenv("SWARMEE_PREFLIGHT", "disabled")
    monkeypatch.setenv("SWARMEE_PROJECT_MAP", "disabled")
    monkeypatch.setattr(swarmee, "read_welcome_text", lambda: "Test welcome message")


@pytest.fixture
def mock_agent():
    """
    Fixture to mock the Agent class and its instance
    """
    with mock.patch.object(swarmee, "Agent") as mock_agent_class:
        mock_agent_instance = mock.MagicMock()
        mock_agent_class.return_value = mock_agent_instance
        mock_agent_instance.invoke_async = mock.AsyncMock(
            return_value=mock.MagicMock(structured_output=None, message=[])
        )
        yield mock_agent_instance


@pytest.fixture
def mock_bedrock():
    """
    Fixture to mock model loading
    """
    with mock.patch.object(swarmee.model_utils, "load_model") as mock_load_model:
        mock_load_model.return_value = mock.MagicMock()
        yield mock_load_model


@pytest.fixture
def mock_load_prompt():
    """
    Fixture to mock the load_system_prompt function
    """
    with mock.patch.object(swarmee, "load_system_prompt") as mock_load_prompt:
        mock_load_prompt.return_value = "Test system prompt"
        yield mock_load_prompt


@pytest.fixture
def mock_user_input():
    """
    Fixture to mock the get_user_input function
    """
    with mock.patch.object(swarmee, "get_user_input") as mock_input:
        yield mock_input


@pytest.fixture
def mock_welcome_message():
    """
    Fixture to mock the render_welcome_message function
    """
    with mock.patch.object(swarmee, "render_welcome_message") as mock_welcome:
        yield mock_welcome


@pytest.fixture
def mock_goodbye_message():
    """
    Fixture to mock the render_goodbye_message function
    """
    with mock.patch.object(swarmee, "render_goodbye_message") as mock_goodbye:
        yield mock_goodbye


@pytest.fixture
def mock_store_conversation():
    """
    Fixture to mock the store_conversation_in_kb function
    """
    with mock.patch.object(swarmee, "store_conversation_in_kb") as mock_store:
        yield mock_store


@pytest.fixture
def temp_env():
    """
    Fixture to create a clean environment for testing
    """
    old_env = os.environ.copy()
    os.environ.clear()
    yield os.environ
    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture
def temp_file(tmp_path):
    """
    Fixture to create a temporary file
    """

    def _create_file(content, name=".prompt"):
        file_path = tmp_path / name
        file_path.write_text(content)
        return file_path

    return _create_file
