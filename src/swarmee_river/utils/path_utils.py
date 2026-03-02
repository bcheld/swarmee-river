from __future__ import annotations

import os
from pathlib import Path


def safe_cwd(cwd: str | None) -> Path:
    """
    Resolve a working directory path safely.
    """
    return Path(cwd or os.getcwd()).expanduser().resolve()


def resolve_target(path: str, *, cwd: str | None) -> tuple[Path, Path]:
    """
    Resolve a target path relative to a working directory, ensuring it doesn't escape the cwd.
    """
    rel_path = (path or "").strip()
    if not rel_path:
        raise ValueError("path is required")

    base = safe_cwd(cwd)
    target = (base / rel_path).expanduser().resolve()
    if base not in target.parents and target != base:
        raise ValueError("Refusing to operate outside cwd")
    return base, target


SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".swarmee",
    "node_modules",
}
