from __future__ import annotations

from typing import Any

from strands.hooks import HookRegistry


def register_hook_callback(registry: HookRegistry, event_type: type[Any], callback: Any) -> None:
    register = getattr(registry, "register", None)
    if callable(register):
        register(event_type, callback)
        return

    add_callback = getattr(registry, "add_callback", None)
    if callable(add_callback):
        add_callback(event_type, callback)
        return

    raise AttributeError(
        "HookRegistry does not support callback registration. "
        "Expected either 'register' or 'add_callback'."
    )


def event_messages(event: Any) -> list[Any] | None:
    """
    Extract message-like payloads across Strands event API versions.

    Newer versions expose `messages`; older variants may use `input`.
    Some events only keep messages in `invocation_state`.
    """
    messages = getattr(event, "messages", None)
    if isinstance(messages, list):
        return messages

    legacy_input = getattr(event, "input", None)
    if isinstance(legacy_input, list):
        return legacy_input

    invocation_state = getattr(event, "invocation_state", None)
    if isinstance(invocation_state, dict):
        state_messages = invocation_state.get("messages")
        if isinstance(state_messages, list):
            return state_messages

    return None


def model_response_payload(event: Any) -> Any:
    """
    Extract model response across Strands event API versions.

    Newer versions expose `stop_response.message`; older variants exposed
    a direct `response` attribute.
    """
    response = getattr(event, "response", None)
    if response is not None:
        return response

    stop_response = getattr(event, "stop_response", None)
    if stop_response is None:
        return None

    return getattr(stop_response, "message", stop_response)
