from __future__ import annotations

from typing import Any


def get_user_input(
    prompt: str,
    *,
    default: str = "",
    keyboard_interrupt_return_default: bool = True,
    **_kwargs: Any,
) -> str:
    """
    Minimal `get_user_input` implementation.

    Swarmee historically used `strands_tools.utils.user_input.get_user_input`, but we provide this local
    version so the CLI can run even when `strands-agents-tools` is not installed.
    """
    try:
        response = input(str(prompt))
    except (KeyboardInterrupt, EOFError):
        if keyboard_interrupt_return_default:
            return str(default)
        raise
    response = str(response or "")
    if not response.strip() and default:
        return str(default)
    return response
