from __future__ import annotations

import contextlib
import json as _json
from pathlib import Path
from typing import Any

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.diff_review import truncate_diff_preview
from swarmee_river.tui.sidebar_artifacts import (
    add_recent_artifacts,
    build_artifact_table_rows,
    normalize_artifact_index_entry,
)


class ArtifactsMixin:
    def _active_session_id(self) -> str | None:
        value = str(self.state.daemon.session_id or "").strip()
        return value or None

    def _sync_artifact_session_scope(self) -> None:
        active_session_id = self._active_session_id()
        if self.state.artifacts.session_id == active_session_id:
            return
        self.state.artifacts.session_id = active_session_id
        self.state.artifacts.recent_paths = []
        self.state.artifacts.entries = []
        self.state.artifacts.selected_item_id = None

    def _load_indexed_artifact_entries(self, *, limit: int = 200) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        active_session_id = self._active_session_id()
        try:
            store = ArtifactStore()
            indexed = store.list(limit=limit)
        except Exception:
            indexed = []

        for raw in indexed:
            if active_session_id:
                if not isinstance(raw, dict):
                    continue
                meta_raw = raw.get("meta")
                meta = meta_raw if isinstance(meta_raw, dict) else {}
                raw_session_id = str(meta.get("session_id", "")).strip()
                # Session panel should only show artifacts tagged with the active daemon session.
                if raw_session_id != active_session_id:
                    continue
            normalized = normalize_artifact_index_entry(raw if isinstance(raw, dict) else {})
            if normalized is None:
                continue
            path = str(normalized.get("path", "")).strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            entries.append(normalized)

        # Keep compatibility with legacy in-memory artifact paths that may not
        # have an index record (e.g., session logs).
        for raw_path in self.state.artifacts.recent_paths:
            path = str(raw_path or "").strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            entries.append(
                {
                    "item_id": path,
                    "id": Path(path).name,
                    "name": Path(path).name,
                    "kind": "path",
                    "created_at": "",
                    "path": path,
                    "bytes": None,
                    "chars": None,
                    "meta": {},
                }
            )
        return entries

    def _artifact_entry_by_item_id(self, item_id: str | None) -> dict[str, Any] | None:
        target = str(item_id or "").strip()
        if not target:
            return None
        for entry in self.state.artifacts.entries:
            if str(entry.get("item_id", "")).strip() == target:
                return entry
        return None

    def _artifact_looks_textual(self, entry: dict[str, Any]) -> bool:
        path = str(entry.get("path", "")).strip()
        kind = str(entry.get("kind", "")).strip().lower()
        if kind in {"tui_transcript", "tool_result", "diagnostic", "project_map", "file_diff"}:
            return True
        suffix = Path(path).suffix.lower()
        if suffix in {".txt", ".md", ".json", ".jsonl", ".log", ".yaml", ".yml", ".csv", ".patch", ".diff"}:
            return True
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".tar", ".bz2"}:
            return False
        return True

    def _artifact_metadata_preview(self, entry: dict[str, Any]) -> str:
        lines = [
            f"Kind: {entry.get('kind', 'unknown')}",
            f"ID: {entry.get('id', '(none)')}",
            f"Name: {entry.get('name', '(none)')}",
            f"Created: {entry.get('created_at', '(unknown)')}",
            f"Path: {entry.get('path', '(none)')}",
        ]
        bytes_value = entry.get("bytes")
        chars_value = entry.get("chars")
        if isinstance(bytes_value, int):
            lines.append(f"Bytes: {bytes_value}")
        if isinstance(chars_value, int):
            lines.append(f"Chars: {chars_value}")
        meta = entry.get("meta")
        if isinstance(meta, dict) and meta:
            try:
                lines.append("")
                lines.append("Meta:")
                lines.append(_json.dumps(meta, indent=2, ensure_ascii=False))
            except Exception:
                pass
        return "\n".join(lines)

    def _artifact_preview_text(self, entry: dict[str, Any]) -> str:
        path = str(entry.get("path", "")).strip()
        if not path:
            return self._artifact_metadata_preview(entry)
        artifact_path = Path(path).expanduser()
        if not artifact_path.exists() or not artifact_path.is_file():
            return self._artifact_metadata_preview(entry) + "\n\nFile not found."
        if not self._artifact_looks_textual(entry):
            return self._artifact_metadata_preview(entry)
        try:
            store = ArtifactStore()
            if str(entry.get("kind", "")).strip().lower() == "file_diff":
                body = store.read_text(artifact_path)
            else:
                body = store.read_text(artifact_path, max_chars=5000)
        except Exception as exc:
            return self._artifact_metadata_preview(entry) + f"\n\nFailed to read artifact: {exc}"
        header = self._artifact_metadata_preview(entry)
        return f"{header}\n\nPreview:\n{body}"

    def _artifact_preview_renderable(self, entry: dict[str, Any]) -> Any:
        kind = str(entry.get("kind", "")).strip().lower()
        if kind != "file_diff":
            return self._artifact_preview_text(entry)

        path = str(entry.get("path", "")).strip()
        if not path:
            return self._artifact_metadata_preview(entry)
        artifact_path = Path(path).expanduser()
        if not artifact_path.exists() or not artifact_path.is_file():
            return self._artifact_metadata_preview(entry) + "\n\nFile not found."

        try:
            store = ArtifactStore()
            diff_text = store.read_text(artifact_path)
        except Exception as exc:
            return self._artifact_metadata_preview(entry) + f"\n\nFailed to read artifact: {exc}"

        from swarmee_river.tui.widgets import render_diff_review_panel

        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        paths = meta.get("changed_paths")
        if not isinstance(paths, list) or not paths:
            paths = meta.get("touched_paths") if isinstance(meta.get("touched_paths"), list) else []
        preview_text, hidden_lines = truncate_diff_preview(diff_text, max_lines=400, max_chars=40000)
        return render_diff_review_panel(
            tool_name=str(meta.get("tool", "file_diff")).strip() or "file_diff",
            paths=[str(item).strip() for item in paths if str(item).strip()],
            diff_text=preview_text or "(empty diff artifact)",
            stats=meta.get("stats") if isinstance(meta, dict) else None,
            artifact_path=path,
            hidden_lines=hidden_lines,
        )

    def _set_artifact_selection(self, entry: dict[str, Any] | None) -> None:
        detail = self._artifacts_detail
        if detail is None:
            return
        if entry is None:
            self.state.artifacts.selected_item_id = None
            detail.set_preview("(no artifacts yet)")
            detail.set_actions([])
            return
        self.state.artifacts.selected_item_id = str(entry.get("item_id", "")).strip() or None
        detail.set_preview(self._artifact_preview_renderable(entry))
        detail.set_actions(
            [
                {"id": "artifact_action_open", "label": "Open", "variant": "default"},
                {"id": "artifact_action_copy_path", "label": "Copy path", "variant": "default"},
                {"id": "artifact_action_add_context", "label": "Add context", "variant": "default"},
            ]
        )

    def _render_artifacts_panel(self) -> None:
        self._sync_artifact_session_scope()
        self.state.artifacts.entries = self._load_indexed_artifact_entries(limit=200)
        if self._artifacts_header is not None:
            badge_count = len(self.state.artifacts.entries)
            self._artifacts_header.set_badges([f"{badge_count} item{'s' if badge_count != 1 else ''}"])
        table = self._artifacts_table
        if table is None:
            return
        if not table.columns:
            table.add_column("Name", key="name")
            table.add_column("Kind", key="kind", width=14)
            table.add_column("Created", key="created", width=20)
            table.add_column("Path", key="path")
        table.clear()
        rows = build_artifact_table_rows(self.state.artifacts.entries)
        for item_id, name, kind, created_at, path in rows:
            table.add_row(name, kind, created_at, path, key=item_id)
        selected_id = self.state.artifacts.selected_item_id
        if not selected_id and self.state.artifacts.entries:
            selected_id = str(self.state.artifacts.entries[0].get("item_id", "")).strip()
        if selected_id and rows:
            for idx, (item_id, _name, _kind, _created, _path) in enumerate(rows):
                if item_id == selected_id:
                    with contextlib.suppress(Exception):
                        table.move_cursor(row=idx)
                    break
        elif rows:
            with contextlib.suppress(Exception):
                table.move_cursor(row=0)
        cursor_coordinate = getattr(table, "cursor_coordinate", None)
        row_index = int(getattr(cursor_coordinate, "row", -1) or -1)
        selected_item_id = rows[row_index][0] if 0 <= row_index < len(rows) else selected_id
        selected_entry = self._artifact_entry_by_item_id(selected_item_id)
        if selected_entry is None and self.state.artifacts.entries:
            selected_entry = self.state.artifacts.entries[0]
        self._set_artifact_selection(selected_entry)

    def _get_artifacts_text(self) -> str:
        entries = self.state.artifacts.entries or self._load_indexed_artifact_entries(limit=200)
        if not entries:
            return "(no artifacts)\n"
        lines: list[str] = []
        for index, entry in enumerate(entries, start=1):
            kind = str(entry.get("kind", "unknown")).strip() or "unknown"
            artifact_id = str(entry.get("id", "")).strip() or "(no-id)"
            name = str(entry.get("name", "")).strip() or artifact_id
            created_at = str(entry.get("created_at", "")).strip() or "unknown time"
            path = str(entry.get("path", "")).strip() or "(no-path)"
            lines.append(f"{index}. {kind} | {name} | {created_at} | {path}")
        return "\n".join(lines).rstrip() + "\n"

    def _reset_artifacts_panel(self) -> None:
        self.state.artifacts.session_id = self._active_session_id()
        self.state.artifacts.recent_paths = []
        self.state.artifacts.entries = []
        self.state.artifacts.selected_item_id = None
        self._render_artifacts_panel()

    def _add_artifact_paths(self, paths: list[str]) -> None:
        self._sync_artifact_session_scope()
        updated = add_recent_artifacts(self.state.artifacts.recent_paths, paths, max_items=20)
        if updated != self.state.artifacts.recent_paths:
            self.state.artifacts.recent_paths = updated
        self._render_artifacts_panel()

    def action_copy_artifacts(self) -> None:
        self._copy_text(self._get_artifacts_text(), label="artifacts")

    def _open_artifact_path(self, path: str) -> None:
        import os
        import shutil
        import subprocess
        import sys

        resolved = str(path or "").strip()
        if not resolved:
            self._write_transcript_line("[open] invalid artifact path.")
            return
        editor = os.environ.get("EDITOR", "")
        try:
            if editor:
                subprocess.Popen([editor, resolved])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", resolved])
            elif shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", resolved])
            else:
                self._write_transcript_line(f"[open] set $EDITOR. Path: {resolved}")
                return
            self._write_transcript_line(f"[open] opened: {resolved}")
        except Exception as exc:
            self._write_transcript_line(f"[open] failed: {exc}")

    def _open_artifact(self, index_str: str) -> None:
        try:
            index = int(index_str.strip()) - 1
        except ValueError:
            self._write_transcript_line("Usage: /open <number>")
            return
        self._render_artifacts_panel()
        entries = self.state.artifacts.entries
        if index < 0 or index >= len(entries):
            self._write_transcript_line(f"[open] invalid index. {len(entries)} artifacts available.")
            return
        path = str(entries[index].get("path", "")).strip()
        self._open_artifact_path(path)

    def _copy_selected_artifact_path(self) -> None:
        selected = self._artifact_entry_by_item_id(self.state.artifacts.selected_item_id)
        if selected is None:
            self._notify("Select an artifact first.", severity="warning")
            return
        path = str(selected.get("path", "")).strip()
        if not path:
            self._notify("Selected artifact has no path.", severity="warning")
            return
        self._copy_text(path, label="artifact path")

    def _add_selected_artifact_as_context(self) -> None:
        from swarmee_river.tui.mixins.context_sources import _sanitize_context_source_id

        selected = self._artifact_entry_by_item_id(self.state.artifacts.selected_item_id)
        if selected is None:
            self._notify("Select an artifact first.", severity="warning")
            return
        path = str(selected.get("path", "")).strip()
        if not path:
            self._notify("Selected artifact has no path.", severity="warning")
            return
        payload = {"type": "file", "path": path, "id": _sanitize_context_source_id(f"artifact-{path}")}
        self._add_context_source(payload)
        self._write_transcript_line(f"[context] added file source from artifact: {path}")
