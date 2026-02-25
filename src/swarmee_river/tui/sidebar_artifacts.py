"""Pure artifact-sidebar domain helpers for the TUI."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any


def _sanitize_context_source_id(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


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
        artifact_id = str(normalized.get("id", "")).strip() or "(no-id)"
        name = str(normalized.get("name", normalized.get("id", ""))).strip() or "(unnamed)"
        created_at = str(normalized.get("created_at", "")).strip() or ""
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


def artifact_context_source_payload(path: str, *, source_id: str | None = None) -> dict[str, str]:
    """Build a file context-source payload for an artifact path."""
    normalized_path = str(path or "").strip()
    if not normalized_path:
        raise ValueError("artifact path is required")
    return {
        "type": "file",
        "path": normalized_path,
        "id": _sanitize_context_source_id(source_id or uuid.uuid4().hex),
    }


__all__ = [
    "add_recent_artifacts",
    "artifact_context_source_payload",
    "build_artifact_sidebar_items",
    "normalize_artifact_index_entry",
]
