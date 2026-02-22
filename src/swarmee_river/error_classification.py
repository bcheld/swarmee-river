from __future__ import annotations

import re
from typing import Any

ERROR_CATEGORY_TRANSIENT = "transient"
ERROR_CATEGORY_ESCALATABLE = "escalatable"
ERROR_CATEGORY_TOOL_ERROR = "tool_error"
ERROR_CATEGORY_AUTH_ERROR = "auth_error"
ERROR_CATEGORY_FATAL = "fatal"

_ERROR_CATEGORIES = {
    ERROR_CATEGORY_TRANSIENT,
    ERROR_CATEGORY_ESCALATABLE,
    ERROR_CATEGORY_TOOL_ERROR,
    ERROR_CATEGORY_AUTH_ERROR,
    ERROR_CATEGORY_FATAL,
}

_TOOL_USE_ID_RE = re.compile(r"(?:tool_use_id|tool-id|tool id)\s*[:=]\s*(?P<id>[A-Za-z0-9._-]+)", re.IGNORECASE)

_TRANSIENT_MARKERS = (
    # Bedrock / OpenAI / generic HTTP transient failures.
    "throttlingexception",
    "modelnotreadyexception",
    "rate limit",
    "rate_limit_error",
    "429",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "try again",
    "service unavailable",
    "503",
    "502",
    "500",
    # Ollama availability/network shape.
    "connection refused",
    "econnrefused",
)

_ESCALATABLE_MARKERS = (
    "context_length_exceeded",
    "context length",
    "maximum context",
    "too many tokens",
    "token limit exceeded",
    "prompt is too long",
    "model capacity",
    "insufficient capacity",
    "max output token limit",
)

_AUTH_MARKERS = (
    "invalid api key",
    "unauthorized",
    "authentication failed",
    "forbidden",
    "access denied",
    "credential",
    "credentials",
    "expired token",
    "signaturedoesnotmatch",
    "no aws credentials",
)

_TOOL_MARKERS = (
    "tool execution failed",
    "tool failed",
    "tool error",
    "failed tool call",
)


def normalize_error_category(category: Any) -> str | None:
    value = str(category or "").strip().lower()
    if value in _ERROR_CATEGORIES:
        return value
    return None


def extract_tool_use_id(message: Any) -> str | None:
    text = str(message or "")
    if not text.strip():
        return None
    match = _TOOL_USE_ID_RE.search(text)
    if match:
        token = match.group("id").strip()
        if token:
            return token
    return None


def classify_error_message(
    message: Any,
    *,
    category_hint: str | None = None,
    tool_use_id: str | None = None,
) -> dict[str, Any]:
    text = str(message or "").strip()
    lowered = text.lower()

    normalized_hint = normalize_error_category(category_hint)
    resolved_tool_id = (tool_use_id or "").strip() or extract_tool_use_id(text)

    category = normalized_hint
    retryable = False

    if category is None:
        if any(marker in lowered for marker in _TOOL_MARKERS) or (
            "tool" in lowered and any(token in lowered for token in (" failed", " error", " exception"))
        ):
            category = ERROR_CATEGORY_TOOL_ERROR
        elif any(marker in lowered for marker in _AUTH_MARKERS) or (
            "permission denied" in lowered and "tool" not in lowered
        ):
            category = ERROR_CATEGORY_AUTH_ERROR
        elif any(marker in lowered for marker in _ESCALATABLE_MARKERS):
            category = ERROR_CATEGORY_ESCALATABLE
            retryable = True
        elif any(marker in lowered for marker in _TRANSIENT_MARKERS):
            category = ERROR_CATEGORY_TRANSIENT
            retryable = True
        else:
            category = ERROR_CATEGORY_FATAL

    if category == ERROR_CATEGORY_TRANSIENT:
        retryable = True
    elif category == ERROR_CATEGORY_ESCALATABLE:
        retryable = True
    else:
        retryable = False

    result: dict[str, Any] = {
        "category": category,
        "retryable": retryable,
    }
    if resolved_tool_id:
        result["tool_use_id"] = resolved_tool_id
    return result

