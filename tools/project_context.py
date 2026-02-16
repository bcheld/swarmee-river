from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from strands import tool

from tools.file_ops import file_list as _file_list
from tools.file_ops import file_search as _file_search


def _safe_cwd(cwd: str | None) -> Path:
    base = Path(cwd or os.getcwd()).expanduser().resolve()
    return base


def _run(cmd: list[str], *, cwd: Path, timeout_s: int = 15) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except Exception as e:
        return 1, "", str(e)
    return int(p.returncode), p.stdout or "", p.stderr or ""


def _truncate(s: str, limit: int) -> str:
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + f"\n… (truncated to {limit} chars) …"


_SKIP_DIRS = {
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


def _top_level_listing(base: Path, *, max_entries: int = 200) -> str:
    try:
        entries = list(base.iterdir())
    except Exception:
        return ""

    names: list[str] = []
    for p in sorted(entries, key=lambda x: x.name.lower()):
        try:
            suffix = "/" if p.is_dir() else ""
        except Exception:
            suffix = ""
        names.append(p.name + suffix)
        if len(names) >= max(1, int(max_entries)):
            break

    if not names:
        return ""
    return "\n".join(names)


def _shallow_tree(base: Path, *, max_depth: int = 2, max_files: int = 5000) -> str:
    files: list[str] = []
    for root, dirnames, filenames in os.walk(base):
        rel_root = os.path.relpath(root, base)
        rel_parts = () if rel_root == "." else Path(rel_root).parts

        dirnames[:] = sorted([d for d in dirnames if d not in _SKIP_DIRS])

        # `find . -maxdepth 2 -type f` includes:
        # - ./file (depth=1)
        # - ./dir/file (depth=2)
        # but not ./dir/sub/file (depth=3)
        if len(rel_parts) >= max(1, int(max_depth)):
            dirnames[:] = []
            continue

        for fn in sorted(filenames):
            if fn in {".DS_Store"}:
                continue
            rel_path = os.path.join(rel_root, fn) if rel_root != "." else fn
            files.append(rel_path)
            if len(files) >= max(1, int(max_files)):
                return "\n".join(files)

    return "\n".join(files)


@tool
def project_context(
    action: str = "summary",
    query: str | None = None,
    path: str | None = None,
    cwd: str | None = None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    return run_project_context(action=action, query=query, path=path, cwd=cwd, max_chars=max_chars)


def run_project_context(
    *,
    action: str = "summary",
    query: str | None = None,
    path: str | None = None,
    cwd: str | None = None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Explore the current project context from the working directory.

    Actions:
    - summary: basic repo summary (pwd, git status, top-level files)
    - files: list files (uses `rg --files` when available)
    - tree: shallow tree (depth=2)
    - search: ripgrep search for `query`
    - read: read a text file at `path` (first ~400 lines)
    - git_status: `git status --porcelain -b`
    """
    action = (action or "").strip().lower()
    base = _safe_cwd(cwd)

    if action == "git_status":
        code, out, err = _run(["git", "status", "--porcelain=v1", "-b"], cwd=base, timeout_s=10)
        text = out.strip() if out.strip() else err.strip()
        return {"status": "success" if code == 0 else "error", "content": [{"text": _truncate(text, max_chars)}]}

    if action == "files":
        return _file_list(cwd=str(base), max_chars=max_chars)

    if action == "tree":
        text = _shallow_tree(base)
        return {"status": "success", "content": [{"text": _truncate(text.strip() or "(no files found)", max_chars)}]}

    if action == "search":
        if not query or not str(query).strip():
            return {"status": "error", "content": [{"text": "query is required for action=search"}]}
        return _file_search(str(query), cwd=str(base), max_matches=200, max_chars=max_chars)

    if action == "read":
        if not path or not str(path).strip():
            return {"status": "error", "content": [{"text": "path is required for action=read"}]}
        p = (base / path).resolve()
        if base not in p.parents and p != base:
            return {"status": "error", "content": [{"text": "Refusing to read outside cwd"}]}
        if not p.exists() or not p.is_file():
            return {"status": "error", "content": [{"text": f"File not found: {path}"}]}
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Failed to read: {e}"}]}
        head = "\n".join(lines[:400])
        return {"status": "success", "content": [{"text": _truncate(head, max_chars)}]}

    if action == "summary":
        parts: list[str] = [f"cwd: {base}"]
        code, out, _err = _run(["git", "rev-parse", "--show-toplevel"], cwd=base, timeout_s=5)
        if code == 0 and out.strip():
            parts.append(f"git_root: {out.strip()}")
        code, out, err = _run(["git", "status", "--porcelain=v1", "-b"], cwd=base, timeout_s=10)
        if out.strip() or err.strip():
            parts.append("git_status:\n" + (out.strip() if out.strip() else err.strip()))
        listing = _top_level_listing(base)
        if listing:
            parts.append("top_level:\n" + listing)
        return {"status": "success", "content": [{"text": _truncate("\n\n".join(parts), max_chars)}]}

    return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}
