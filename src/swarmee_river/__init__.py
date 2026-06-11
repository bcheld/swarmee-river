"""Swarmee River package."""

from __future__ import annotations

import importlib

__all__ = ["handlers", "jupyter", "models", "utils"]

# Submodules load lazily (PEP 562): `models` alone pulls boto3 + openai +
# strands (~1s), and eager package-level imports made every CLI entrypoint
# pay that cost before doing anything.
_LAZY_SUBMODULES = {"handlers", "jupyter", "models", "utils", "swarmee"}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)
