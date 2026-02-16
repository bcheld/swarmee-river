from __future__ import annotations

from pathlib import Path
from typing import Any

from strands import tool

from swarmee_river.state_paths import todo_path


def _safe_cwd(cwd: str | None) -> Path:
    return Path(cwd).expanduser().resolve() if cwd else Path.cwd().expanduser().resolve()


@tool
def todoread(
    *,
    cwd: str | None = None,
) -> dict[str, Any]:
    """
    Read the project-local todo list from <state_dir>/todo.md.
    """
    path = todo_path(cwd=_safe_cwd(cwd))
    if not path.exists():
        return {"status": "success", "content": [{"text": "(no todos)"}]}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"status": "error", "content": [{"text": f"Failed to read todo list: {exc}"}]}
    return {"status": "success", "content": [{"text": text if text.strip() else "(no todos)"}]}


@tool
def todowrite(
    todos: str,
    *,
    append: bool = False,
    cwd: str | None = None,
) -> dict[str, Any]:
    """
    Write the project-local todo list to <state_dir>/todo.md.
    """
    if not isinstance(todos, str):
        return {"status": "error", "content": [{"text": "todos must be a string"}]}

    path = todo_path(cwd=_safe_cwd(cwd))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(todos)
        else:
            path.write_text(todos, encoding="utf-8")
    except Exception as exc:
        return {"status": "error", "content": [{"text": f"Failed to write todo list: {exc}"}]}

    mode = "appended" if append else "updated"
    return {"status": "success", "content": [{"text": f"Todo list {mode}: {path}"}]}
