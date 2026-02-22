from __future__ import annotations

from swarmee_river.error_classification import (
    ERROR_CATEGORY_AUTH_ERROR,
    ERROR_CATEGORY_ESCALATABLE,
    ERROR_CATEGORY_FATAL,
    ERROR_CATEGORY_TOOL_ERROR,
    ERROR_CATEGORY_TRANSIENT,
    classify_error_message,
)


def test_classify_bedrock_throttling_as_transient() -> None:
    result = classify_error_message("ThrottlingException: Rate exceeded for model")
    assert result["category"] == ERROR_CATEGORY_TRANSIENT
    assert result["retryable"] is True


def test_classify_openai_context_length_as_escalatable() -> None:
    result = classify_error_message("OpenAI error: context_length_exceeded")
    assert result["category"] == ERROR_CATEGORY_ESCALATABLE
    assert result["retryable"] is True


def test_classify_ollama_connection_refused_as_transient() -> None:
    result = classify_error_message("Ollama request failed: connection refused")
    assert result["category"] == ERROR_CATEGORY_TRANSIENT
    assert result["retryable"] is True


def test_classify_tool_error_extracts_tool_use_id() -> None:
    result = classify_error_message("tool execution failed; tool_use_id=t-123")
    assert result["category"] == ERROR_CATEGORY_TOOL_ERROR
    assert result["retryable"] is False
    assert result["tool_use_id"] == "t-123"


def test_classify_auth_error() -> None:
    result = classify_error_message("Unauthorized: invalid API key")
    assert result["category"] == ERROR_CATEGORY_AUTH_ERROR
    assert result["retryable"] is False


def test_classify_unmatched_as_fatal() -> None:
    result = classify_error_message("unexpected fatal parser panic")
    assert result["category"] == ERROR_CATEGORY_FATAL
    assert result["retryable"] is False

