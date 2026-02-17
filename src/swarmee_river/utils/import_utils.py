from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def load_optional_attr(
    module_name: str,
    attr: str,
    *,
    import_module: Callable[[str], Any] | None = None,
) -> Any | None:
    try:
        import_module_fn = import_module or importlib.import_module
        module = import_module_fn(module_name)
        return getattr(module, attr)
    except Exception:
        return None
