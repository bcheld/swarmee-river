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

        items.append({
            "id": name,
            "title": title,
            "subtitle": subtitle,
            "state": "default",
        })
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

        items.append({
            "id": tid,
            "title": name or tid,
            "subtitle": "  ".join(subtitle_parts),
            "state": "default",
        })
    return items


def render_prompt_detail(tmpl: Any) -> str:
    """Format full detail text for a single PromptTemplate."""
    if isinstance(tmpl, dict):
        name = str(tmpl.get("name", ""))
        content = str(tmpl.get("content", ""))
        tags = tmpl.get("tags", [])
        source = str(tmpl.get("source", ""))
    else:
        name = getattr(tmpl, "name", "")
        content = getattr(tmpl, "content", "")
        tags = getattr(tmpl, "tags", [])
        source = getattr(tmpl, "source", "")

    lines = [f"# {name}"]
    tag_str = ", ".join(str(t) for t in tags if str(t).strip())
    if tag_str:
        lines.append(f"Tags: {tag_str}")
    if source:
        lines.append(f"Source: {source}")
    if content:
        lines.append(f"\n{content}")
    return "\n".join(lines)


__all__ = [
    "render_prompt_detail",
    "render_prompt_list_items",
    "render_tool_catalog_items",
    "render_tool_detail",
]
