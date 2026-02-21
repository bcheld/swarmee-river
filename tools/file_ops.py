from __future__ import annotations

import os
import re
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
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except FileNotFoundError:
        return None, "", "rg not found"
    except Exception as e:
        return 1, "", str(e)


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


def _is_binary_file(path: Path, *, sniff_bytes: int = 2048) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(max(1, int(sniff_bytes)))
    except Exception:
        return True
    return b"\x00" in chunk


def _iter_text_files(base: Path, *, max_files: int = 10_000, max_file_bytes: int = 2_000_000) -> list[Path]:
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted([d for d in dirnames if d not in _SKIP_DIRS])
        for fn in sorted(filenames):
            p = Path(root) / fn
            try:
                if p.is_symlink():
                    continue
                if not p.is_file():
                    continue
                if p.stat().st_size > max_file_bytes:
                    continue
            except Exception:
                continue
            if _is_binary_file(p):
                continue
            files.append(p)
            if len(files) >= max_files:
                return files
    return files


def _python_search(
    query: str,
    *,
    cwd: Path,
    max_matches: int,
    max_chars: int,
) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"status": "error", "content": [{"text": "query is required"}]}

    try:
        pattern = re.compile(q)
    except re.error as e:
        return {"status": "error", "content": [{"text": f"Invalid regex: {e}"}]}

    matches: list[str] = []
    hit_count = 0

    for path in _iter_text_files(cwd):
        rel = os.path.relpath(path, cwd)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, start=1):
                    if pattern.search(line):
                        matches.append(f"{rel}:{lineno}:{line.rstrip()}")
                        hit_count += 1
                        if hit_count >= max(1, int(max_matches)):
                            break
        except Exception:
            continue
        if hit_count >= max(1, int(max_matches)):
            break

    text = "\n".join(matches).strip() if matches else "(no matches)"
    return {"status": "success", "content": [{"text": _truncate(text, max_chars)}]}


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
    max_files = 10_000
    for root, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted([d for d in dirnames if d not in _SKIP_DIRS])
        rel_root = os.path.relpath(root, base)
        for fn in sorted(filenames):
            if fn in {".DS_Store"}:
                continue
            files.append(os.path.join(rel_root, fn) if rel_root != "." else fn)
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
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
        ["-n", "--no-heading", "--max-count", str(max(1, int(max_matches))), q, "."],
        cwd=base,
        timeout_s=20,
    )
    if code in (0, 1):
        if code == 1 and (err or "").strip():
            return {"status": "error", "content": [{"text": _truncate(err.strip(), max_chars)}]}
        text = out.strip() if out.strip() else "(no matches)"
        return {"status": "success", "content": [{"text": _truncate(text, max_chars)}]}

    if code is None:
        return _python_search(q, cwd=base, max_matches=max_matches, max_chars=max_chars)

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
