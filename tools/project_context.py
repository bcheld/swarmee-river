from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from strands import tool


def _safe_cwd(cwd: str | None) -> Path:
    base = Path(cwd or os.getcwd()).expanduser().resolve()
    return base


def _run(cmd: list[str], *, cwd: Path, timeout_s: int = 15) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout_s,
        check=False,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def _truncate(s: str, limit: int) -> str:
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + f"\n… (truncated to {limit} chars) …"


@tool
def project_context(
    action: str = "summary",
    query: str | None = None,
    path: str | None = None,
    cwd: Optional[str] = None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    return run_project_context(action=action, query=query, path=path, cwd=cwd, max_chars=max_chars)


def run_project_context(
    *,
    action: str = "summary",
    query: str | None = None,
    path: str | None = None,
    cwd: Optional[str] = None,
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
        code, out, err = _run(["rg", "--files"], cwd=base, timeout_s=10)
        if code != 0:
            # Fallback: basic os.walk, but keep it short.
            files: list[str] = []
            for root, dirs, filenames in os.walk(base):
                rel_root = os.path.relpath(root, base)
                if rel_root.startswith(".git") or rel_root.startswith(".venv") or rel_root.startswith("dist"):
                    dirs[:] = []
                    continue
                for fn in filenames:
                    files.append(os.path.join(rel_root, fn) if rel_root != "." else fn)
                if len(files) > 5000:
                    break
            return {"status": "success", "content": [{"text": _truncate("\n".join(files), max_chars)}]}
        return {"status": "success", "content": [{"text": _truncate(out.strip(), max_chars)}]}

    if action == "tree":
        code, out, err = _run(["find", ".", "-maxdepth", "2", "-type", "f"], cwd=base, timeout_s=10)
        if code == 0 and out.strip():
            return {"status": "success", "content": [{"text": _truncate(out.strip(), max_chars)}]}

        # Fallback for platforms without `find`.
        files: list[str] = []
        max_depth = 2
        base_parts = len(base.parts)
        for root, dirs, filenames in os.walk(base):
            rel_parts = Path(root).parts[base_parts:]
            if len(rel_parts) > max_depth:
                dirs[:] = []
                continue
            rel_root = os.path.relpath(root, base)
            if rel_root.startswith(".git") or rel_root.startswith(".venv") or rel_root.startswith("dist"):
                dirs[:] = []
                continue
            for fn in filenames:
                files.append(os.path.join(rel_root, fn) if rel_root != "." else fn)
            if len(files) > 5000:
                break
        return {"status": "success", "content": [{"text": _truncate("\n".join(files), max_chars)}]}

    if action == "search":
        if not query or not str(query).strip():
            return {"status": "error", "content": [{"text": "query is required for action=search"}]}
        code, out, err = _run(
            ["rg", "-n", "--no-heading", "--max-count", "200", str(query)],
            cwd=base,
            timeout_s=15,
        )
        text = out.strip() if out.strip() else err.strip()
        status = "success" if code in (0, 1) else "error"  # 1 == no matches
        return {"status": status, "content": [{"text": _truncate(text, max_chars)}]}

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
        code, out, _err = _run(["ls"], cwd=base, timeout_s=5)
        if code == 0 and out.strip():
            parts.append("top_level:\n" + out.strip())
        return {"status": "success", "content": [{"text": _truncate("\n\n".join(parts), max_chars)}]}

    return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}
