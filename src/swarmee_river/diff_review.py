from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from swarmee_river.utils.path_utils import resolve_target, safe_cwd

_TEXT_KIND = "text"
_MISSING_KIND = "missing"
_BINARY_KIND = "binary"
_UNREADABLE_KIND = "unreadable"
_DIRECTORY_KIND = "directory"


@dataclass(frozen=True)
class FileSnapshot:
    display_path: str
    path: str
    kind: str
    exists: bool
    text: str | None = None
    digest: str | None = None
    size: int | None = None


@dataclass(frozen=True)
class DiffPreview:
    tool_name: str
    trusted: bool
    changed_paths: list[str]
    touched_paths: list[str]
    diff_text: str = ""
    non_text_paths: list[dict[str, str]] | None = None
    reason: str | None = None


def extract_patch_target_files(patch_text: str) -> list[str]:
    targets: list[str] = []
    for line in (patch_text or "").splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        raw = line[4:].strip()
        if not raw or raw == "/dev/null":
            continue
        raw = raw.split("\t", 1)[0].split(" ", 1)[0].strip()
        if raw.startswith("a/") or raw.startswith("b/"):
            raw = raw[2:]
        if raw and raw not in targets:
            targets.append(raw)
    return targets


def resolve_mutating_tool_paths(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
) -> tuple[Path, list[tuple[str, Path]]]:
    payload = tool_input if isinstance(tool_input, Mapping) else {}
    from swarmee_river.opencode_aliases import canonical_tool_name

    canonical_name = canonical_tool_name(tool_name)
    cwd = str(payload.get("cwd", "") or "").strip() or None

    if canonical_name == "editor":
        command = str(payload.get("command", "")).strip().lower()
        if command not in {"replace", "insert", "write"}:
            return safe_cwd(cwd), []
        base, target = resolve_target(str(payload.get("path", "")), cwd=cwd)
        return base, [(str(payload.get("path", "")).strip(), target)]

    if canonical_name == "file_write":
        base, target = resolve_target(str(payload.get("path", "")), cwd=cwd)
        return base, [(str(payload.get("path", "")).strip(), target)]

    if canonical_name == "patch_apply":
        base = safe_cwd(cwd)
        paths: list[tuple[str, Path]] = []
        for rel in extract_patch_target_files(str(payload.get("patch", ""))):
            target = (base / rel).expanduser().resolve()
            if base not in target.parents and target != base:
                continue
            paths.append((rel, target))
        return base, paths

    return safe_cwd(cwd), []


def _text_snapshot_from_value(*, display_path: str, path: Path, text: str | None, exists: bool) -> FileSnapshot:
    encoded = (text or "").encode("utf-8", errors="replace")
    return FileSnapshot(
        display_path=display_path,
        path=str(path),
        kind=_TEXT_KIND if exists else _MISSING_KIND,
        exists=exists,
        text=text if exists else None,
        digest=hashlib.sha256(encoded).hexdigest() if exists else None,
        size=len(encoded) if exists else None,
    )


def _simulate_editor_after_snapshot(
    *,
    command: str,
    display_path: str,
    path: Path,
    before: FileSnapshot,
    payload: Mapping[str, Any],
) -> FileSnapshot:
    if command == "write":
        file_text = payload.get("file_text")
        if not isinstance(file_text, str):
            raise ValueError("editor write requires file_text")
        return _text_snapshot_from_value(display_path=display_path, path=path, text=file_text, exists=True)

    if before.kind != _TEXT_KIND or not isinstance(before.text, str):
        raise ValueError(f"editor {command} requires a readable text file")

    current = before.text
    if command == "replace":
        old_str = payload.get("old_str")
        new_str = payload.get("new_str")
        if not isinstance(old_str, str) or not old_str:
            raise ValueError("editor replace requires old_str")
        if not isinstance(new_str, str):
            raise ValueError("editor replace requires new_str")
        if old_str not in current:
            raise ValueError("editor replace old_str was not found")
        updated = current.replace(old_str, new_str, 1)
        return _text_snapshot_from_value(display_path=display_path, path=path, text=updated, exists=True)

    if command == "insert":
        file_text = payload.get("file_text")
        insert_line = payload.get("insert_line")
        if not isinstance(file_text, str):
            raise ValueError("editor insert requires file_text")
        if insert_line is None:
            raise ValueError("editor insert requires insert_line")
        line_idx = max(1, int(insert_line))
        lines = current.splitlines(keepends=True)
        line_idx = min(line_idx, len(lines) + 1)
        lines.insert(line_idx - 1, file_text)
        updated = "".join(lines)
        return _text_snapshot_from_value(display_path=display_path, path=path, text=updated, exists=True)

    raise ValueError(f"unsupported editor command: {command}")


def _simulate_file_write_after_snapshot(
    *,
    display_path: str,
    path: Path,
    before: FileSnapshot,
    payload: Mapping[str, Any],
) -> FileSnapshot:
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("file_write requires content")
    append = bool(payload.get("append"))
    if append:
        if before.kind == _MISSING_KIND:
            updated = content
        elif before.kind == _TEXT_KIND and isinstance(before.text, str):
            updated = before.text + content
        else:
            raise ValueError("file_write append requires a readable text file")
    else:
        updated = content
    return _text_snapshot_from_value(display_path=display_path, path=path, text=updated, exists=True)


def preview_mutating_tool_change(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
) -> DiffPreview | None:
    payload = tool_input if isinstance(tool_input, Mapping) else {}
    from swarmee_river.opencode_aliases import canonical_tool_name

    canonical_name = canonical_tool_name(tool_name)
    if canonical_name == "patch_apply":
        if bool(payload.get("dry_run")):
            return None
        patch_text = str(payload.get("patch", "") or "").strip()
        touched_paths = extract_patch_target_files(patch_text)
        if not patch_text:
            return DiffPreview(
                tool_name=tool_name,
                trusted=False,
                changed_paths=[],
                touched_paths=[],
                reason="patch_apply requires a unified diff previewable patch payload.",
            )
        if "GIT binary patch" in patch_text or "Binary files " in patch_text:
            non_text_paths = [{"path": path, "before": "unknown", "after": "unknown"} for path in touched_paths]
            return DiffPreview(
                tool_name=tool_name,
                trusted=True,
                changed_paths=list(touched_paths),
                touched_paths=list(touched_paths),
                non_text_paths=non_text_paths,
            )
        if not touched_paths:
            return DiffPreview(
                tool_name=tool_name,
                trusted=False,
                changed_paths=[],
                touched_paths=[],
                reason="patch_apply preview requires unified diff headers (--- / +++).",
            )
        return DiffPreview(
            tool_name=tool_name,
            trusted=True,
            changed_paths=list(touched_paths),
            touched_paths=list(touched_paths),
            diff_text=patch_text,
        )

    base, paths = resolve_mutating_tool_paths(tool_name, payload)
    if not paths:
        return None

    changed_paths: list[str] = []
    touched_paths = [display_path for display_path, _target in paths]
    textual_diffs: list[str] = []
    non_text_paths: list[dict[str, str]] = []

    for display_path, target in paths:
        before = snapshot_file(target, display_path=display_path)
        try:
            if canonical_name == "editor":
                command = str(payload.get("command", "")).strip().lower()
                after = _simulate_editor_after_snapshot(
                    command=command,
                    display_path=display_path,
                    path=target,
                    before=before,
                    payload=payload,
                )
            elif canonical_name == "file_write":
                after = _simulate_file_write_after_snapshot(
                    display_path=display_path,
                    path=target,
                    before=before,
                    payload=payload,
                )
            else:
                return DiffPreview(
                    tool_name=tool_name,
                    trusted=False,
                    changed_paths=[],
                    touched_paths=touched_paths,
                    reason="Pre-approval diff previews are only supported for editor and patch_apply.",
                )
        except Exception as exc:
            return DiffPreview(
                tool_name=tool_name,
                trusted=False,
                changed_paths=[],
                touched_paths=touched_paths,
                reason=str(exc),
            )

        before_kind = before.kind
        after_kind = after.kind
        if before_kind in {_TEXT_KIND, _MISSING_KIND} and after_kind in {_TEXT_KIND, _MISSING_KIND}:
            diff_text = build_unified_diff(before, after)
            if diff_text:
                changed_paths.append(display_path)
                textual_diffs.append(diff_text)
            continue

        changed_paths.append(display_path)
        non_text_paths.append(
            {
                "path": display_path,
                "before": before_kind,
                "after": after_kind,
            }
        )

    if not changed_paths and not non_text_paths:
        return DiffPreview(
            tool_name=tool_name,
            trusted=True,
            changed_paths=[],
            touched_paths=touched_paths,
        )

    return DiffPreview(
        tool_name=tool_name,
        trusted=True,
        changed_paths=changed_paths or [item["path"] for item in non_text_paths],
        touched_paths=touched_paths,
        diff_text="\n\n".join(chunk for chunk in textual_diffs if chunk.strip()).strip(),
        non_text_paths=non_text_paths or None,
    )


def snapshot_file(path: Path, *, display_path: str) -> FileSnapshot:
    if not path.exists():
        return FileSnapshot(display_path=display_path, path=str(path), kind=_MISSING_KIND, exists=False)
    if path.is_dir():
        return FileSnapshot(display_path=display_path, path=str(path), kind=_DIRECTORY_KIND, exists=True)
    if not path.is_file():
        return FileSnapshot(display_path=display_path, path=str(path), kind=_UNREADABLE_KIND, exists=True)

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    size = len(raw)
    if b"\x00" in raw:
        return FileSnapshot(
            display_path=display_path,
            path=str(path),
            kind=_BINARY_KIND,
            exists=True,
            digest=digest,
            size=size,
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return FileSnapshot(
            display_path=display_path,
            path=str(path),
            kind=_UNREADABLE_KIND,
            exists=True,
            digest=digest,
            size=size,
        )
    return FileSnapshot(
        display_path=display_path,
        path=str(path),
        kind=_TEXT_KIND,
        exists=True,
        text=text,
        digest=digest,
        size=size,
    )


def build_unified_diff(before: FileSnapshot, after: FileSnapshot) -> str:
    before_text = before.text if before.kind == _TEXT_KIND and isinstance(before.text, str) else ""
    after_text = after.text if after.kind == _TEXT_KIND and isinstance(after.text, str) else ""
    if before_text == after_text:
        return ""
    lines = list(
        difflib.unified_diff(
            before_text.splitlines(),
            after_text.splitlines(),
            fromfile=f"a/{before.display_path}",
            tofile=f"b/{after.display_path}",
            lineterm="",
        )
    )
    return "\n".join(lines).strip()


def summarize_diff_stats(diff_text: str) -> dict[str, int]:
    added = 0
    removed = 0
    files_changed = 0
    for line in (diff_text or "").splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
        elif line.startswith("diff --git ") or line.startswith("Index: "):
            files_changed += 1
    if files_changed == 0 and diff_text.strip():
        files_changed = max(1, diff_text.count("\n--- "))
        if files_changed == 0:
            files_changed = 1
    return {"files_changed": files_changed, "added_lines": added, "removed_lines": removed}


def truncate_diff_preview(
    diff_text: str,
    *,
    max_lines: int = 80,
    max_chars: int = 6000,
) -> tuple[str, int]:
    text = str(diff_text or "").strip()
    if not text:
        return "", 0

    lines = text.splitlines()
    truncated_lines = list(lines)
    hidden_lines = 0
    if max_lines > 0 and len(truncated_lines) > max_lines:
        hidden_lines = len(truncated_lines) - max_lines
        truncated_lines = truncated_lines[:max_lines]

    preview = "\n".join(truncated_lines)
    if max_chars > 0 and len(preview) > max_chars:
        preview = preview[:max_chars].rstrip()
        hidden_lines = max(hidden_lines, len(lines) - len(preview.splitlines()))
    return preview, hidden_lines
