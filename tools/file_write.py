from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from strands import tool


def _safe_cwd(cwd: str | None) -> Path:
    return Path(cwd or os.getcwd()).expanduser().resolve()


def _resolve_target(path: str, *, cwd: str | None) -> tuple[Path, Path]:
    rel_path = (path or "").strip()
    if not rel_path:
        raise ValueError("path is required")

    base = _safe_cwd(cwd)
    target = (base / rel_path).expanduser().resolve()
    if base not in target.parents and target != base:
        raise ValueError("Refusing to write outside cwd")
    return base, target


@tool
def file_write(
    path: str,
    content: str,
    *,
    cwd: str | None = None,
    append: bool = False,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """
    Cross-platform fallback for `strands_tools.file_write`.

    Supports overwrite (default) and append writes under the current working directory.
    """
    if content is None:
        return {"status": "error", "content": [{"text": "content is required"}]}

    try:
        base, target = _resolve_target(path, cwd=cwd)
    except ValueError as exc:
        return {"status": "error", "content": [{"text": str(exc)}]}

    if target.exists() and target.is_dir():
        return {"status": "error", "content": [{"text": "Refusing to write to a directory"}]}

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding=encoding, errors="replace") as handle:
            handle.write(content)
    except Exception as exc:
        return {"status": "error", "content": [{"text": f"Failed to write file: {exc}"}]}

    rel = os.path.relpath(target, base)
    action = "Appended" if append else "Wrote"
    return {
        "status": "success",
        "content": [{"text": f"{action} {len(content)} chars to {rel}"}],
    }
