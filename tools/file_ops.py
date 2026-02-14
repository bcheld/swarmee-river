from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from strands import tool


def _safe_cwd(cwd: str | None) -> Path:
    return Path(cwd or os.getcwd()).expanduser().resolve()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated to {max_chars} chars) ..."


def _run_rg(args: list[str], *, cwd: Path, timeout_s: int = 15) -> tuple[int | None, str, str]:
    try:
        completed = subprocess.run(
            ["rg", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except FileNotFoundError:
        return None, "", "rg not found"
    except Exception as e:
        return 1, "", str(e)


@tool
def file_list(
    *,
    cwd: Optional[str] = None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    List repository files (prefers `rg --files`).
    """
    base = _safe_cwd(cwd)
    code, out, err = _run_rg(["--files"], cwd=base, timeout_s=10)
    if code == 0 and out.strip():
        return {"status": "success", "content": [{"text": _truncate(out.strip(), max_chars)}]}

    files: list[str] = []
    for root, dirs, filenames in os.walk(base):
        rel_root = os.path.relpath(root, base)
        if rel_root.startswith(".git") or rel_root.startswith(".venv") or rel_root.startswith("dist"):
            dirs[:] = []
            continue
        for fn in filenames:
            files.append(os.path.join(rel_root, fn) if rel_root != "." else fn)
        if len(files) > 10000:
            break
    text = "\n".join(files).strip()
    if not text:
        text = err.strip() if err.strip() else "(no files found)"
    return {"status": "success", "content": [{"text": _truncate(text, max_chars)}]}


@tool
def file_search(
    query: str,
    *,
    cwd: Optional[str] = None,
    max_matches: int = 200,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Search text in repository files (prefers `rg -n --no-heading`).
    """
    q = (query or "").strip()
    if not q:
        return {"status": "error", "content": [{"text": "query is required"}]}

    base = _safe_cwd(cwd)
    code, out, err = _run_rg(
        ["-n", "--no-heading", "--max-count", str(max(1, int(max_matches))), q],
        cwd=base,
        timeout_s=20,
    )
    if code in (0, 1):
        text = out.strip() if out.strip() else "(no matches)"
        return {"status": "success", "content": [{"text": _truncate(text, max_chars)}]}

    return {"status": "error", "content": [{"text": _truncate(err.strip() or "(search failed)", max_chars)}]}


@tool
def file_read(
    path: str,
    *,
    cwd: Optional[str] = None,
    start_line: int = 1,
    max_lines: int = 300,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Read a text file with line numbers.
    """
    rel_path = (path or "").strip()
    if not rel_path:
        return {"status": "error", "content": [{"text": "path is required"}]}

    base = _safe_cwd(cwd)
    target = (base / rel_path).resolve()
    if base not in target.parents and target != base:
        return {"status": "error", "content": [{"text": "Refusing to read outside cwd"}]}
    if not target.exists() or not target.is_file():
        return {"status": "error", "content": [{"text": f"File not found: {rel_path}"}]}

    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Failed to read file: {e}"}]}

    start = max(1, int(start_line))
    count = max(1, int(max_lines))
    selected = lines[start - 1 : start - 1 + count]
    numbered = "\n".join(f"{start + idx:>6} | {line}" for idx, line in enumerate(selected))
    if not numbered:
        numbered = "(no content in selected range)"
    return {"status": "success", "content": [{"text": _truncate(numbered, max_chars)}]}
