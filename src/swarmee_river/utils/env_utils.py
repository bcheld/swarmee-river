from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> bool:
    """
    Load a dotenv-style file into the environment.

    - Lines beginning with `#` are ignored.
    - Supports simple `KEY=VALUE` assignments (no multiline, no export keyword).
    - When `override=False`, existing environment variables are preserved.

    Returns:
        True if a file was found and parsed, False otherwise.
    """
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return False

    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value

    return True

