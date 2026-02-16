from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from strands import tool


def _safe_cwd(cwd: str | None) -> Path:
    return Path(cwd or os.getcwd()).expanduser().resolve()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated to {max_chars} chars) ..."


def _resolve_target(path: str, *, cwd: str | None) -> tuple[Path, Path]:
    rel_path = (path or "").strip()
    if not rel_path:
        raise ValueError("path is required")

    base = _safe_cwd(cwd)
    target = (base / rel_path).expanduser().resolve()
    if base not in target.parents and target != base:
        raise ValueError("Refusing to edit outside cwd")
    return base, target


def _atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(text, encoding=encoding, errors="replace")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _read_text(path: Path, *, encoding: str) -> str:
    return path.read_text(encoding=encoding, errors="replace")


def _render_view(text: str, *, view_range: list[int] | None) -> str:
    if not view_range:
        return text or "(empty file)"
    if len(view_range) != 2:
        raise ValueError("view_range must be [start_line, end_line]")

    start = max(1, int(view_range[0]))
    end = max(start, int(view_range[1]))

    lines = text.splitlines()
    selected = lines[start - 1 : end]
    if not selected:
        return "(no content in selected range)"
    return "\n".join(f"{start + idx:>6} | {line}" for idx, line in enumerate(selected))


@tool
def editor(
    command: str,
    path: str,
    *,
    old_str: str | None = None,
    new_str: str | None = None,
    file_text: str | None = None,
    insert_line: int | None = None,
    view_range: list[int] | None = None,
    cwd: str | None = None,
    max_chars: int = 12000,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """
    Cross-platform fallback for `strands_tools.editor`.

    Supported commands: view, replace, insert, write.
    """
    cmd = (command or "").strip().lower()
    if not cmd:
        return {"status": "error", "content": [{"text": "command is required"}]}

    try:
        base, target = _resolve_target(path, cwd=cwd)
    except ValueError as exc:
        return {"status": "error", "content": [{"text": str(exc)}]}

    try:
        if cmd == "view":
            if not target.exists() or not target.is_file():
                return {"status": "error", "content": [{"text": f"File not found: {path}"}]}
            text = _read_text(target, encoding=encoding)
            rendered = _render_view(text, view_range=view_range)
            return {"status": "success", "content": [{"text": _truncate(rendered, max_chars)}]}

        if cmd == "replace":
            if not target.exists() or not target.is_file():
                return {"status": "error", "content": [{"text": f"File not found: {path}"}]}
            if old_str is None or old_str == "":
                return {"status": "error", "content": [{"text": "old_str is required for replace"}]}
            if new_str is None:
                return {"status": "error", "content": [{"text": "new_str is required for replace"}]}

            current = _read_text(target, encoding=encoding)
            if old_str not in current:
                return {"status": "error", "content": [{"text": "old_str was not found"}]}

            updated = current.replace(old_str, new_str, 1)
            _atomic_write_text(target, updated, encoding=encoding)
            rel = os.path.relpath(target, base)
            return {"status": "success", "content": [{"text": f"Replaced first occurrence in {rel}"}]}

        if cmd == "insert":
            if not target.exists() or not target.is_file():
                return {"status": "error", "content": [{"text": f"File not found: {path}"}]}
            if file_text is None:
                return {"status": "error", "content": [{"text": "file_text is required for insert"}]}
            if insert_line is None:
                return {"status": "error", "content": [{"text": "insert_line is required for insert"}]}

            current = _read_text(target, encoding=encoding)
            lines = current.splitlines(keepends=True)
            line_idx = max(1, int(insert_line))
            line_idx = min(line_idx, len(lines) + 1)

            lines.insert(line_idx - 1, file_text)
            updated = "".join(lines)
            _atomic_write_text(target, updated, encoding=encoding)
            rel = os.path.relpath(target, base)
            return {
                "status": "success",
                "content": [{"text": f"Inserted text at line {line_idx} in {rel}"}],
            }

        if cmd == "write":
            if file_text is None:
                return {"status": "error", "content": [{"text": "file_text is required for write"}]}
            if target.exists() and target.is_dir():
                return {"status": "error", "content": [{"text": "Refusing to write to a directory"}]}

            _atomic_write_text(target, file_text, encoding=encoding)
            rel = os.path.relpath(target, base)
            return {"status": "success", "content": [{"text": f"Wrote {len(file_text)} chars to {rel}"}]}

        return {
            "status": "error",
            "content": [{"text": "Unsupported command. Use one of: view, replace, insert, write"}],
        }
    except Exception as exc:
        return {"status": "error", "content": [{"text": f"Editor command failed: {exc}"}]}
