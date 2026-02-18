"""Swarmee River package."""

from __future__ import annotations

import importlib

from . import handlers, jupyter, models, utils

__all__ = ["handlers", "jupyter", "models", "utils"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    # Avoid importing `swarmee_river.swarmee` at package import time to prevent `-m swarmee_river.swarmee`
    # from triggering runpy's "found in sys.modules" RuntimeWarning.
    if name == "swarmee":
        module = importlib.import_module(f"{__name__}.swarmee")
        globals()["swarmee"] = module
        return module
    raise AttributeError(name)
