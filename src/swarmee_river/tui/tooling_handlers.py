"""Pure rendering helpers for the Tooling tab's subtabs.

Keeps app.py manageable by extracting list-item and detail-rendering
logic into standalone functions.
"""

from __future__ import annotations

from typing import Any


def render_tool_catalog_items(catalog: list[Any]) -> list[dict[str, str]]:
    """Build SidebarList item payloads from a list of ToolMeta objects/dicts.

    Each item shows: ``name [R][W][X]``  description  ``Tags: ...``
    """
    items: list[dict[str, str]] = []
    for tool in catalog:
        if isinstance(tool, dict):
            name = str(tool.get("name", ""))
            desc = str(tool.get("description", ""))
            tags = tool.get("tags", [])
            r = tool.get("access_read", False)
            w = tool.get("access_write", False)
            x = tool.get("access_execute", False)
        else:
            name = getattr(tool, "name", "")
            desc = getattr(tool, "description", "")
            tags = getattr(tool, "tags", [])
            r = getattr(tool, "access_read", False)
            w = getattr(tool, "access_write", False)
            x = getattr(tool, "access_execute", False)

        badges = ""
        if r:
            badges += "[R]"
        if w:
            badges += "[W]"
        if x:
            badges += "[X]"
        title = f"{name} {badges}".strip() if badges else name

        parts: list[str] = []
        if desc:
            parts.append(desc[:80])
        tag_str = ", ".join(str(t) for t in tags if str(t).strip())
        if tag_str:
            parts.append(f"Tags: {tag_str}")
        subtitle = "  ".join(parts) if parts else ""

        items.append(
            {
                "id": name,
                "title": title,
                "subtitle": subtitle,
                "state": "default",
            }
        )
    return items


def render_tool_detail(tool: Any) -> str:
    """Format full detail text for a single ToolMeta."""
    if isinstance(tool, dict):
        name = str(tool.get("name", ""))
        desc = str(tool.get("description", ""))
        tags = tool.get("tags", [])
        r = tool.get("access_read", False)
        w = tool.get("access_write", False)
        x = tool.get("access_execute", False)
        source = str(tool.get("source", ""))
    else:
        name = getattr(tool, "name", "")
        desc = getattr(tool, "description", "")
        tags = getattr(tool, "tags", [])
        r = getattr(tool, "access_read", False)
        w = getattr(tool, "access_write", False)
        x = getattr(tool, "access_execute", False)
        source = getattr(tool, "source", "")

    lines = [f"# {name}"]
    if desc:
        lines.append(f"\n{desc}")

    access_parts: list[str] = []
    if r:
        access_parts.append("Read")
    if w:
        access_parts.append("Write")
    if x:
        access_parts.append("Execute")
    if access_parts:
        lines.append(f"\nAccess: {', '.join(access_parts)}")

    tag_str = ", ".join(str(t) for t in tags if str(t).strip())
    if tag_str:
        lines.append(f"Tags: {tag_str}")

    if source:
        lines.append(f"Source: {source}")

    return "\n".join(lines)


def render_prompt_list_items(templates: list[Any]) -> list[dict[str, str]]:
    """Build SidebarList item payloads from a list of PromptTemplate objects/dicts."""
    items: list[dict[str, str]] = []
    for tmpl in templates:
        if isinstance(tmpl, dict):
            tid = str(tmpl.get("id", ""))
            name = str(tmpl.get("name", ""))
            content = str(tmpl.get("content", ""))
            tags = tmpl.get("tags", [])
            source = str(tmpl.get("source", ""))
        else:
            tid = getattr(tmpl, "id", "")
            name = getattr(tmpl, "name", "")
            content = getattr(tmpl, "content", "")
            tags = getattr(tmpl, "tags", [])
            source = getattr(tmpl, "source", "")

        subtitle_parts: list[str] = []
        if content:
            preview = content.replace("\n", " ")[:80]
            subtitle_parts.append(preview)
        tag_str = ", ".join(str(t) for t in tags if str(t).strip())
        if tag_str:
            subtitle_parts.append(f"Tags: {tag_str}")
        if source and source != "local":
            subtitle_parts.append(f"({source})")

        items.append(
            {
                "id": tid,
                "title": name or tid,
                "subtitle": "  ".join(subtitle_parts),
                "state": "default",
            }
        )
    return items


def render_prompt_detail(tmpl: Any) -> str:
    """Format full detail text for a single prompt asset."""
    if isinstance(tmpl, dict):
        prompt_id = str(tmpl.get("id", ""))
        name = str(tmpl.get("name", ""))
        content = str(tmpl.get("content", ""))
        tags = tmpl.get("tags", [])
        source = str(tmpl.get("source", ""))
        used_by = tmpl.get("used_by", [])
        is_orchestrator = bool(tmpl.get("is_orchestrator", False))
    else:
        prompt_id = str(getattr(tmpl, "id", ""))
        name = getattr(tmpl, "name", "")
        content = getattr(tmpl, "content", "")
        tags = getattr(tmpl, "tags", [])
        source = getattr(tmpl, "source", "")
        used_by = getattr(tmpl, "used_by", [])
        is_orchestrator = bool(getattr(tmpl, "is_orchestrator", False))

    lines = [f"# {name or prompt_id}"]
    if prompt_id:
        lines.append(f"ID: {prompt_id}")
    tag_str = ", ".join(str(t) for t in tags if str(t).strip())
    if tag_str:
        lines.append(f"Tags: {tag_str}")
    if source:
        lines.append(f"Source: {source}")
    if is_orchestrator:
        lines.append("Role: Orchestrator base prompt")
    if isinstance(used_by, list) and used_by:
        lines.append(f"Used by: {', '.join(str(item).strip() for item in used_by if str(item).strip())}")
    if content:
        lines.append(f"\n{content}")
    return "\n".join(lines)


def render_sop_detail(sop: Any, *, active: bool = False) -> str:
    """Format full detail text for a single SOP record."""
    if isinstance(sop, dict):
        name = str(sop.get("name", ""))
        source = str(sop.get("source", ""))
        path = str(sop.get("path", ""))
        preview = str(sop.get("first_paragraph_preview", ""))
        content = str(sop.get("content", ""))
    else:
        name = getattr(sop, "name", "")
        source = getattr(sop, "source", "")
        path = getattr(sop, "path", "")
        preview = getattr(sop, "first_paragraph_preview", "")
        content = getattr(sop, "content", "")

    lines = [f"# {name}"]
    lines.append(f"Status: {'active' if active else 'inactive'}")
    lines.append("Press Enter to activate/deactivate.")
    if source:
        lines.append(f"Source: {source}")
    if path:
        lines.append(f"Path: {path}")
    if preview:
        lines.append(f"\nPreview:\n{preview}")
    if content:
        lines.append(f"\nContent:\n{content}")
    return "\n".join(lines)


def render_kb_detail(entry: Any) -> str:
    """Format full detail text for a single knowledge-base entry."""
    if isinstance(entry, dict):
        kb_id = str(entry.get("id", ""))
        name = str(entry.get("name", ""))
        description = str(entry.get("description", ""))
    else:
        kb_id = getattr(entry, "id", "")
        name = getattr(entry, "name", "")
        description = getattr(entry, "description", "")

    resolved_id = kb_id.strip() or "(none)"
    resolved_name = name.strip() or resolved_id
    lines = [f"# {resolved_name}", f"ID: {resolved_id}"]
    if description.strip():
        lines.append(f"\n{description.strip()}")
    else:
        lines.append("\n(no description)")
    return "\n".join(lines)


def build_tool_table_rows(catalog: list[Any]) -> list[tuple[str, str, str, str]]:
    """Build DataTable row tuples from a tool catalog.

    Returns list of (name, access_badge, source, tags_str) tuples.
    """
    rows: list[tuple[str, str, str, str]] = []
    for tool in catalog:
        if isinstance(tool, dict):
            name = str(tool.get("name", ""))
            tags = tool.get("tags", [])
            r = tool.get("access_read", False)
            w = tool.get("access_write", False)
            x = tool.get("access_execute", False)
            source = str(tool.get("source", ""))
        else:
            name = getattr(tool, "name", "")
            tags = getattr(tool, "tags", [])
            r = getattr(tool, "access_read", False)
            w = getattr(tool, "access_write", False)
            x = getattr(tool, "access_execute", False)
            source = getattr(tool, "source", "")

        badge_parts: list[str] = []
        if r:
            badge_parts.append("[R]")
        if w:
            badge_parts.append("[W]")
        if x:
            badge_parts.append("[X]")
        badge = "".join(badge_parts)

        tag_str = ", ".join(str(t) for t in tags if str(t).strip())
        rows.append((name, badge, source, tag_str))
    return rows


def build_prompt_table_rows(assets: list[Any]) -> list[tuple[str, str, str, str, str, str]]:
    """Build DataTable row tuples from prompt assets."""
    rows: list[tuple[str, str, str, str, str, str]] = []
    for asset in assets:
        if isinstance(asset, dict):
            prompt_id = str(asset.get("id", ""))
            name = str(asset.get("name", ""))
            content = str(asset.get("content", ""))
            tags = asset.get("tags", [])
            used_by = asset.get("used_by", [])
        else:
            prompt_id = str(getattr(asset, "id", ""))
            name = str(getattr(asset, "name", ""))
            content = str(getattr(asset, "content", ""))
            tags = getattr(asset, "tags", [])
            used_by = getattr(asset, "used_by", [])
        tag_str = ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
        used_by_text = ", ".join(str(item).strip() for item in (used_by or []) if str(item).strip()) or "-"
        preview = content.replace("\n", " ").strip()
        if len(preview) > 80:
            preview = preview[:79].rstrip() + "…"
        rows.append((prompt_id, name or prompt_id, prompt_id, tag_str, used_by_text, preview))
    return rows


def build_sop_table_rows(sops: list[Any], active_names: set[str] | None = None) -> list[tuple[str, str, str, str]]:
    """Build DataTable row tuples from SOP catalog records."""
    active = {str(name).strip() for name in (active_names or set()) if str(name).strip()}
    rows: list[tuple[str, str, str, str]] = []
    for sop in sops:
        if isinstance(sop, dict):
            name = str(sop.get("name", ""))
            source = str(sop.get("source", ""))
            preview = str(sop.get("first_paragraph_preview", ""))
        else:
            name = str(getattr(sop, "name", ""))
            source = str(getattr(sop, "source", ""))
            preview = str(getattr(sop, "first_paragraph_preview", ""))
        if not name.strip():
            continue
        preview = preview.strip() or "(no preview available)"
        if len(preview) > 100:
            preview = preview[:99].rstrip() + "…"
        rows.append((name, "yes" if name in active else "no", source or "unknown", preview))
    return rows


def build_kb_table_rows(entries: list[Any]) -> list[tuple[str, str, str]]:
    """Build DataTable row tuples from knowledge-base records."""
    rows: list[tuple[str, str, str]] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            kb_id = str(entry.get("id", entry.get("name", f"kb-{index + 1}"))).strip() or f"kb-{index + 1}"
            name = str(entry.get("name", entry.get("id", f"KB {index + 1}"))).strip() or f"KB {index + 1}"
            description = str(entry.get("description", "")).strip()
        else:
            kb_id = str(getattr(entry, "id", f"kb-{index + 1}")).strip() or f"kb-{index + 1}"
            name = str(getattr(entry, "name", kb_id)).strip() or kb_id
            description = str(getattr(entry, "description", "")).strip()
        if len(description) > 100:
            description = description[:99].rstrip() + "…"
        rows.append((kb_id, name, description))
    return rows


__all__ = [
    "build_kb_table_rows",
    "build_prompt_table_rows",
    "build_sop_table_rows",
    "build_tool_table_rows",
    "render_kb_detail",
    "render_prompt_detail",
    "render_prompt_list_items",
    "render_sop_detail",
    "render_tool_catalog_items",
    "render_tool_detail",
]
