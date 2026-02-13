"""Create instances of Swarmee model providers.

Module must expose an `instance` function that returns a `strands.types.models.Model` implementation.
"""

from . import bedrock, ollama

try:
    from . import openai  # noqa: F401
except Exception:  # pragma: no cover
    openai = None  # type: ignore[assignment]

__all__ = ["bedrock", "ollama", "openai"]
