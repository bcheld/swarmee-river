"""Pure artifact-sidebar domain helpers for the TUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def add_recent_artifacts(existing: list[str], new_paths: list[str], *, max_items: int = 20) -> list[str]:
    """Dedupe and cap artifact paths while preserving recency."""
    updated = list(existing)
    for item in new_paths:
        path = item.strip()
        if not path:
            continue
        if path in updated:
            updated.remove(path)
        updated.append(path)
    if len(updated) > max_items:
        updated = updated[-max_items:]
    return updated


def normalize_artifact_index_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize an artifact index record for TUI list/detail rendering."""
    if not isinstance(entry, dict):
        return None
    raw_path = str(entry.get("path", "")).strip()
    if not raw_path:
        return None

    meta_raw = entry.get("meta")
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    artifact_id = str(entry.get("id", "")).strip() or Path(raw_path).name
    name = str(meta.get("name", meta.get("title", ""))).strip()
    if not name:
        name = artifact_id

    kind = str(entry.get("kind", "")).strip() or "unknown"
    created_at = str(entry.get("created_at", "")).strip()
    bytes_value = entry.get("bytes")
    chars_value = entry.get("chars")

    return {
        "item_id": raw_path,
        "id": artifact_id,
        "name": name,
        "kind": kind,
        "created_at": created_at,
        "path": raw_path,
        "bytes": int(bytes_value) if isinstance(bytes_value, int) else None,
        "chars": int(chars_value) if isinstance(chars_value, int) else None,
        "meta": meta,
    }


def build_artifact_sidebar_items(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build SidebarList item payloads from normalized artifact entries."""
    items: list[dict[str, str]] = []
    for entry in entries:
        normalized = normalize_artifact_index_entry(entry)
        if normalized is None:
            continue
        kind = str(normalized.get("kind", "unknown")).strip() or "unknown"
        name = str(normalized.get("name", normalized.get("id", ""))).strip() or "(unnamed)"
        path = str(normalized.get("path", "")).strip()
        items.append(
            {
                "id": str(normalized.get("item_id", path)).strip() or path,
                "title": name,
                "subtitle": f"{kind} · {path}" if path else kind,
                "state": "default",
            }
        )
    return items


def build_artifact_table_rows(entries: list[dict[str, Any]]) -> list[tuple[str, str, str, str, str]]:
    """Build DataTable rows for artifacts.

    Returns tuples of: (item_id, name, kind, created_at, path).
    """
    rows: list[tuple[str, str, str, str, str]] = []
    for entry in entries:
        normalized = normalize_artifact_index_entry(entry)
        if normalized is None:
            continue
        item_id = str(normalized.get("item_id", "")).strip()
        name = str(normalized.get("name", "")).strip() or "(unnamed)"
        kind = str(normalized.get("kind", "unknown")).strip() or "unknown"
        created_at = str(normalized.get("created_at", "")).strip()
        path = str(normalized.get("path", "")).strip()
        if len(path) > 96:
            path = "..." + path[-93:]
        rows.append((item_id, name, kind, created_at, path))
    return rows


def artifact_context_source_payload(path: str, *, source_id: str) -> dict[str, str]:
    """Build a context-source payload from an artifact path."""
    return {
        "type": "file",
        "path": str(path).strip(),
        "id": str(source_id).strip(),
    }


__all__ = [
    "add_recent_artifacts",
    "artifact_context_source_payload",
    "build_artifact_table_rows",
    "build_artifact_sidebar_items",
    "normalize_artifact_index_entry",
]
