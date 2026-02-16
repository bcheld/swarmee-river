from __future__ import annotations

import os
from pathlib import Path


def state_dir(*, cwd: Path | None = None) -> Path:
    """
    Return the base directory for Swarmee runtime state (artifacts/logs/sessions/project map).

    Default: `<cwd>/.swarmee`
    Override: `SWARMEE_STATE_DIR`

    Notes:
    - If SWARMEE_STATE_DIR is relative, it is interpreted relative to `cwd` (or Path.cwd()).
    - This does not create directories; callers should mkdir as needed.
    """
    base = (cwd or Path.cwd()).expanduser().resolve()

    raw = os.getenv("SWARMEE_STATE_DIR")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw.strip()).expanduser()
        if not p.is_absolute():
            p = (base / p).expanduser()
        return p.resolve()

    return base / ".swarmee"


def artifacts_dir(*, cwd: Path | None = None) -> Path:
    return state_dir(cwd=cwd) / "artifacts"


def logs_dir(*, cwd: Path | None = None) -> Path:
    return state_dir(cwd=cwd) / "logs"


def sessions_dir(*, cwd: Path | None = None) -> Path:
    return state_dir(cwd=cwd) / "sessions"


def project_map_path(*, cwd: Path | None = None) -> Path:
    return state_dir(cwd=cwd) / "project_map.json"


def todo_path(*, cwd: Path | None = None) -> Path:
    return state_dir(cwd=cwd) / "todo.md"
