"""
SOP Tool - Load and inspect Standard Operating Procedures (SOPs).

Supports:
- Built-in SOPs from the optional `strands_agents_sops` package
- Local SOPs from one or more directories containing `*.sop.md` files

Local SOPs override built-ins with the same name.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from strands import tool

SOPRecord = dict[str, Any]


def _parse_sop_paths(sop_paths: str | None) -> list[Path]:
    raw = sop_paths or os.getenv("SWARMEE_SOP_PATHS") or ""
    paths: list[Path] = []

    if raw.strip():
        for part in raw.split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            paths.append(Path(part).expanduser())

    default_dir = Path.cwd() / "sops"
    if default_dir.exists() and default_dir.is_dir():
        paths.append(default_dir)

    # De-dup while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in paths:
        rp = p.resolve() if p.exists() else p
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def _parse_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    """
    Parse a minimal YAML-like frontmatter block:

    ---
    name: my-sop
    version: 1.2.3
    description: ...
    ---
    # SOP body...
    """
    text = markdown.lstrip("\ufeff")
    if not text.startswith("---\n"):
        return {}, markdown.strip()

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, markdown.strip()

    header = text[4:end].strip()
    body = text[end + len("\n---\n") :].strip()

    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            meta[key] = value
    return meta, body


def _csv_env(name: str) -> set[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return set()
    return {p.strip() for p in raw.split(",") if p.strip()}


def _is_sop_allowed(sop_name: str) -> tuple[bool, str | None]:
    enabled = _csv_env("SWARMEE_ENABLE_SOPS")
    disabled = _csv_env("SWARMEE_DISABLE_SOPS")

    if enabled and sop_name not in enabled:
        return False, "Blocked by SWARMEE_ENABLE_SOPS"
    if sop_name in disabled:
        return False, "Blocked by SWARMEE_DISABLE_SOPS"
    return True, None


def _load_builtin_sops() -> dict[str, SOPRecord]:
    try:
        import strands_agents_sops as sops
    except Exception:
        return {}

    builtin: dict[str, SOPRecord] = {}
    for name in dir(sops):
        if name.startswith("_"):
            continue
        value = getattr(sops, name, None)
        if isinstance(value, str) and len(value.strip()) >= 100:
            meta, body = _parse_frontmatter(value)
            builtin[name] = {
                "name": meta.get("name", name),
                "version": meta.get("version", "builtin"),
                "description": meta.get("description", ""),
                "source": "builtin",
                "path": None,
                "body": body,
            }
    return builtin


def _load_local_sops(paths: list[Path]) -> dict[str, SOPRecord]:
    local: dict[str, SOPRecord] = {}
    for base in paths:
        if not base.exists() or not base.is_dir():
            continue
        for file_path in sorted(base.glob("*.sop.md")):
            try:
                raw = file_path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            meta, body = _parse_frontmatter(raw)
            derived_name = file_path.name[: -len(".sop.md")]
            sop_name = meta.get("name", derived_name)
            local[sop_name] = {
                "name": sop_name,
                "version": meta.get("version", "0.0.0"),
                "description": meta.get("description", ""),
                "source": "local",
                "path": str(file_path),
                "body": body,
            }
    return local


@tool
def sop(
    action: str = "list",
    name: str | None = None,
    sop_paths: str | None = None,
    include_builtin: bool = True,
) -> dict[str, Any]:
    return run_sop(action=action, name=name, sop_paths=sop_paths, include_builtin=include_builtin)


def run_sop(
    *,
    action: str = "list",
    name: str | None = None,
    sop_paths: str | None = None,
    include_builtin: bool = True,
) -> dict[str, Any]:
    """
    Inspect and load SOPs (Standard Operating Procedures).

    Args:
        action: One of: "list", "get".
        name: SOP name (required for action="get"). Supports `name@version` for exact version match.
        sop_paths: OS-separated directories containing `*.sop.md` files.
        include_builtin: Whether to include SOPs from `strands_agents_sops` if installed.

    Returns:
        Tool result dict with SOP information.
    """
    action = (action or "").strip().lower()
    if action not in {"list", "get"}:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use 'list' or 'get'."}]}

    paths = _parse_sop_paths(sop_paths)
    builtin = _load_builtin_sops() if include_builtin else {}
    local = _load_local_sops(paths)

    # Local overrides builtin
    merged = dict(builtin)
    merged.update(local)

    if action == "list":
        lines: list[str] = ["# SOPs", ""]
        if paths:
            lines.append("## Search Paths")
            lines.extend([f"- {p}" for p in paths])
            lines.append("")

        if not merged:
            lines.append("No SOPs found. Install `strands-agents-sops` or add `./sops/*.sop.md` files.")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        lines.append("## Available")
        for sop_name in sorted(merged.keys()):
            record = merged[sop_name]
            allowed, reason = _is_sop_allowed(sop_name)
            source = record.get("source", "unknown")
            version = record.get("version", "")
            suffix = f"{source} {version}".strip()
            if not allowed:
                suffix = f"{suffix} (blocked: {reason})"
            lines.append(f"- `{sop_name}` ({suffix})")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # action == "get"
    if not name:
        return {"status": "error", "content": [{"text": "name is required for action='get'."}]}

    requested_name = name.strip()
    requested_version: str | None = None
    if "@" in requested_name:
        requested_name, requested_version = requested_name.split("@", 1)
        requested_name = requested_name.strip()
        requested_version = requested_version.strip()

    if requested_name not in merged:
        return {
            "status": "error",
            "content": [{"text": f"Unknown SOP: {requested_name}. Use action='list' to see options."}],
        }

    allowed, reason = _is_sop_allowed(requested_name)
    if not allowed:
        return {"status": "error", "content": [{"text": f"SOP '{requested_name}' is blocked: {reason}"}]}

    record = merged[requested_name]
    if requested_version and str(record.get("version")) != requested_version:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"SOP '{requested_name}' version mismatch: requested '{requested_version}', "
                        f"available '{record.get('version')}'."
                    )
                }
            ],
        }

    header = (
        f"# SOP: {record.get('name')}\n"
        f"(source: {record.get('source')}, version: {record.get('version')})\n"
        f"(path: {record.get('path')})\n"
    )
    body = record.get("body", "")
    return {"status": "success", "content": [{"text": header + "\n" + body}]}
