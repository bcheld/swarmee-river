from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, List, Optional

from strands import tool


def _safe_cwd(cwd: str | None) -> Path:
    return Path(cwd or os.getcwd()).expanduser().resolve()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… (truncated to {max_chars} chars) …"


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


def _contains_parent_traversal(pattern: str) -> bool:
    # Treat both path separators as separators regardless of platform.
    normalized = (pattern or "").replace("\\", "/")
    parts = [p for p in normalized.split("/") if p not in {"", "."}]
    return any(p == ".." for p in parts)


def _normalize_glob_pattern(pattern: str) -> str:
    p = (pattern or "").strip()
    if not p:
        return ""
    p = p.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    while p.endswith("/"):
        p = p[:-1]
    return p


def _split_posix_path(text: str) -> tuple[str, ...]:
    value = (text or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    while value.startswith("/"):
        value = value[1:]
    while value.endswith("/"):
        value = value[:-1]
    if not value:
        return tuple()
    return tuple(part for part in value.split("/") if part and part != ".")


def _component_match_norm(name_norm: str, pattern_norm: str) -> bool:
    return fnmatch.fnmatchcase(name_norm, pattern_norm)


@tool
def list(
    path: str = ".",
    *,
    cwd: Optional[str] = None,
    include_hidden: bool = False,
    max_entries: int = 500,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    List directory contents (cross-platform, no shell).

    Notes:
    - Hidden entries (starting with `.`) are excluded by default.
    - Output uses `/` suffix for directories.
    """
    base = _safe_cwd(cwd)
    raw = (path or ".").strip() or "."

    target = (base / raw).expanduser().resolve()
    if base not in target.parents and target != base:
        return {"status": "error", "content": [{"text": "Refusing to list outside cwd"}]}
    if not target.exists():
        return {"status": "error", "content": [{"text": f"Path not found: {raw}"}]}
    if not target.is_dir():
        return {"status": "error", "content": [{"text": f"Not a directory: {raw}"}]}

    try:
        entries = sorted(target.iterdir(), key=lambda p: p.name.lower())
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Failed to list directory: {e}"}]}

    lines: List[str] = []
    limit = max(0, int(max_entries))
    for p in entries:
        if limit and len(lines) >= limit:
            break
        name = p.name
        if not include_hidden and name.startswith("."):
            continue
        suffix = ""
        try:
            suffix = "/" if p.is_dir() else ""
        except Exception:
            suffix = ""
        lines.append(name + suffix)

    text = "\n".join(lines).strip() if lines else "(no entries)"
    return {"status": "success", "content": [{"text": _truncate(text, max_chars)}]}


@tool
def glob(
    pattern: str,
    *,
    cwd: Optional[str] = None,
    include_dirs: bool = False,
    max_results: int = 2000,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Find paths matching a glob pattern under `cwd` (cross-platform, no shell).

    Pattern notes:
    - Uses `/` separators; `\\` is accepted and normalized.
    - Use `**/` to match recursively (e.g., `**/*.py`).
    - Parent traversal (`..`) and absolute patterns are rejected.
    """
    raw = (pattern or "").strip()
    if not raw:
        return {"status": "error", "content": [{"text": "pattern is required"}]}
    if os.path.isabs(raw):
        return {"status": "error", "content": [{"text": "Absolute patterns are not allowed"}]}
    if _contains_parent_traversal(raw):
        return {"status": "error", "content": [{"text": "Parent traversal ('..') is not allowed in pattern"}]}

    base = _safe_cwd(cwd)
    normalized = _normalize_glob_pattern(raw)
    if not normalized:
        return {"status": "error", "content": [{"text": "pattern is required"}]}

    pat_parts = _split_posix_path(normalized)
    pat_parts_norm = tuple(part if part == "**" else os.path.normcase(part) for part in pat_parts)

    def matches(rel_posix: str) -> bool:
        path_parts = _split_posix_path(rel_posix)
        path_parts_norm = tuple(os.path.normcase(part) for part in path_parts)

        memo: dict[tuple[int, int], bool] = {}

        def _match(i: int, j: int) -> bool:
            key = (i, j)
            cached = memo.get(key)
            if cached is not None:
                return cached

            if i >= len(pat_parts_norm):
                ok = j >= len(path_parts_norm)
                memo[key] = ok
                return ok

            part = pat_parts_norm[i]
            if part == "**":
                ok = _match(i + 1, j) or (j < len(path_parts_norm) and _match(i, j + 1))
                memo[key] = ok
                return ok

            if j >= len(path_parts_norm):
                memo[key] = False
                return False

            if not _component_match_norm(path_parts_norm[j], part):
                memo[key] = False
                return False

            ok = _match(i + 1, j + 1)
            memo[key] = ok
            return ok

        return _match(0, 0)

    want_dirs = bool(include_dirs)
    limit = max(0, int(max_results))

    results: List[str] = []

    for root, dirnames, filenames in os.walk(base):
        # Skip noisy/build dirs; also avoid following symlinked dirs.
        kept_dirnames: List[str] = []
        for d in sorted(dirnames):
            if d in _SKIP_DIRS:
                continue
            p = Path(root) / d
            try:
                if p.is_symlink():
                    continue
            except Exception:
                continue
            kept_dirnames.append(d)
        dirnames[:] = kept_dirnames

        rel_root = Path(os.path.relpath(root, base))
        if want_dirs:
            for d in dirnames:
                if limit and len(results) >= limit:
                    break
                rel = (rel_root / d).as_posix()
                if matches(rel):
                    results.append(rel + "/")

        for fn in sorted(filenames):
            if limit and len(results) >= limit:
                break
            p = Path(root) / fn
            try:
                if p.is_symlink() or not p.is_file():
                    continue
            except Exception:
                continue
            rel = (rel_root / fn).as_posix()
            if matches(rel):
                results.append(rel)

        if limit and len(results) >= limit:
            break

    text = "\n".join(results).strip() if results else "(no matches)"
    return {"status": "success", "content": [{"text": _truncate(text, max_chars)}]}
