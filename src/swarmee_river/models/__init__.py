"""Create instances of Swarmee model providers.

Module must expose an `instance` function that returns a `strands.types.models.Model` implementation.
"""

from __future__ import annotations

import importlib

__all__ = ["bedrock", "ollama", "openai", "github_copilot"]

# Provider modules load lazily: each pulls a heavy SDK (boto3, openai, ...)
# and most invocations only ever touch one provider. `openai` and
# `github_copilot` stay optional — attribute access yields None when their
# dependencies are unavailable, matching the previous eager behavior.
_PROVIDERS = {"bedrock", "ollama", "openai", "github_copilot"}
_OPTIONAL_PROVIDERS = {"openai", "github_copilot"}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in _PROVIDERS:
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except Exception:
            if name in _OPTIONAL_PROVIDERS:
                globals()[name] = None
                return None
            raise
        globals()[name] = module
        return module
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PROVIDERS)
