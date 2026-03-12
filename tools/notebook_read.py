from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from strands import tool

from swarmee_river.tool_permissions import set_permissions
from swarmee_river.utils.notebook_utils import load_notebook_text
from swarmee_river.utils.path_utils import safe_cwd
from swarmee_river.utils.text_utils import truncate


@tool
def notebook_read(
    path: str,
    *,
    cwd: Optional[str] = None,
    cell_types: list[str] | None = None,
    start_cell: int | None = None,
    end_cell: int | None = None,
    include_outputs: bool = False,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """Read a Jupyter notebook as compact text without executing code."""
    rel_path = str(path or "").strip()
    if not rel_path:
        return {"status": "error", "content": [{"text": "path is required"}]}

    base = safe_cwd(cwd)
    target = (base / rel_path).resolve()
    if base not in target.parents and target != base:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"Refusing to read outside the current scope: {target}. "
                        "Cross-scope reads are blocked; change cwd/scope to a parent directory "
                        "if you want to inspect this notebook."
                    )
                }
            ],
        }
    if not target.exists() or not target.is_file():
        return {"status": "error", "content": [{"text": f"Notebook not found: {rel_path}"}]}
    if target.suffix.lower() != ".ipynb":
        return {"status": "error", "content": [{"text": f"Expected a .ipynb file, got: {rel_path}"}]}

    text = load_notebook_text(
        Path(target),
        cell_types=cell_types,
        start_cell=start_cell,
        end_cell=end_cell,
        include_outputs=include_outputs,
    )
    if text is None:
        return {
            "status": "error",
            "content": [{"text": "Failed to parse notebook. Ensure nbformat is installed and the file is valid."}],
        }
    if not text.strip():
        return {"status": "success", "content": [{"text": "(no notebook content in selected range)"}]}
    return {"status": "success", "content": [{"text": truncate(text, max_chars)}]}


set_permissions(notebook_read, "read")
