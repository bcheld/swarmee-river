from __future__ import annotations

from pathlib import Path

_STATE_DIR_OVERRIDE: Path | None = None


def set_state_dir_override(path: str | Path | None, *, cwd: Path | None = None) -> None:
    """
    Override the resolved state directory for this process.

    This replaces the legacy SWARMEE_STATE_DIR env override. Callers should set
    this once at startup (CLI/TUI/daemon entrypoints).
    """
    global _STATE_DIR_OVERRIDE
    if path is None:
        _STATE_DIR_OVERRIDE = None
        return
    raw = str(path).strip()
    if not raw:
        _STATE_DIR_OVERRIDE = None
        return
    relative_base = (cwd or Path.cwd()).expanduser().resolve()
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = (relative_base / resolved).expanduser()
    _STATE_DIR_OVERRIDE = resolved.resolve()


def scope_root(*, cwd: Path | None = None) -> Path:
    """
    Resolve the default Swarmee scope root.

    Priority:
    1) Nearest ancestor containing `.swarmee`
    2) Git repository root (nearest ancestor containing `.git`)
    3) User home directory when no repository is detected
    """
    base = (cwd or Path.cwd()).expanduser().resolve()

    # 1) Check for explicit .swarmee directory
    for candidate in (base, *base.parents):
        swarmee_marker = candidate / ".swarmee"
        if swarmee_marker.is_dir():
            return candidate

    # 2) Check for git repository
    for candidate in (base, *base.parents):
        git_marker = candidate / ".git"
        if git_marker.exists():
            return candidate

    # 3) Fallback to home directory
    return Path.home().expanduser().resolve()


def state_dir(*, cwd: Path | None = None) -> Path:
    """
    Return the base directory for Swarmee runtime state (artifacts/logs/sessions/project map).

    Default:
    - `<git-repo-root>/.swarmee` when running inside a git repository
    - `~/.swarmee` when no repository is detected
    Override: `set_state_dir_override(...)`

    Notes:
    - If the override path is relative, it is interpreted relative to `cwd` (or Path.cwd()).
    - This does not create directories; callers should mkdir as needed.
    """
    base = scope_root(cwd=cwd)
    if _STATE_DIR_OVERRIDE is not None:
        return _STATE_DIR_OVERRIDE

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
