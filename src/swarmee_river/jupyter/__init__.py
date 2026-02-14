"""Jupyter/IPython integrations for Swarmee River.

Usage (in a notebook):
    %load_ext swarmee_river.jupyter
    %%swarmee
    What should we do next?
"""

from __future__ import annotations

from typing import Any


def load_ipython_extension(ipython: Any) -> None:
    from .magic import load_ipython_extension as _load

    _load(ipython)


def unload_ipython_extension(ipython: Any) -> None:
    from .magic import unload_ipython_extension as _unload

    _unload(ipython)
