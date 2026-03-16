"""Tool metadata discovery, persistence, and tag query helpers for the TUI."""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from swarmee_river.packs import load_enabled_pack_tools
from swarmee_river.settings import load_settings
from swarmee_river.state_paths import state_dir
from swarmee_river.tool_permissions import STRANDS_TOOL_PERMISSIONS, get_permissions

_TOOL_OVERRIDES_FILENAME = "tool_metadata.json"

_CONNECTOR_TOOLS = frozenset(
    {
        "athena_query",
        "snowflake_query",
        "s3_browser",
        "session_s3",
        "retrieve",
        "http_request",
        "slack",
        "use_aws",
        "store_in_kb",
    }
)


@dataclass
class ToolMeta:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    access_read: bool = False
    access_write: bool = False
    access_execute: bool = False
    source: str = "core"  # "core", "pack", "native", "connector-backed", "runtime-generated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolMeta:
        return cls(
            name=str(data.get("name", "")).strip(),
            description=str(data.get("description", "")).strip(),
            tags=[str(t).strip() for t in (data.get("tags") or []) if str(t).strip()],
            access_read=bool(data.get("access_read", False)),
            access_write=bool(data.get("access_write", False)),
            access_execute=bool(data.get("access_execute", False)),
            source=str(data.get("source", "builtin")).strip(),
        )


def _resolve_permissions(name: str, tool_obj: Any) -> tuple[bool, bool, bool]:
    """Two-tier permission resolution: declared > SDK fallback.

    Unknown tools default to (False, False, False) — informational / no permissions.
    """
    # 1) Declared .permissions attribute on the tool object.
    declared = get_permissions(tool_obj)
    if declared is not None:
        return ("read" in declared, "write" in declared, "execute" in declared)
    # 2) SDK fallback map for tools we cannot annotate directly.
    sdk_perms = STRANDS_TOOL_PERMISSIONS.get(name)
    if sdk_perms is not None:
        return ("read" in sdk_perms, "write" in sdk_perms, "execute" in sdk_perms)
    # Unknown tool — treat as informational (no permissions).
    return (False, False, False)


def _overrides_path() -> Path:
    return state_dir() / _TOOL_OVERRIDES_FILENAME


def load_tool_metadata_overrides() -> dict[str, dict[str, Any]]:
    """Load user-edited tool metadata from .swarmee/tool_metadata.json."""
    path = _overrides_path()
    if not path.exists():
        return {}
    with contextlib.suppress(Exception):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def save_tool_metadata_overrides(overrides: dict[str, dict[str, Any]]) -> None:
    """Persist user-edited tool metadata to .swarmee/tool_metadata.json."""
    path = _overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, indent=2, default=str), encoding="utf-8")


def discover_tools_with_metadata(tools_dict: dict[str, Any] | None = None) -> list[ToolMeta]:
    """Discover all available tools, apply heuristic defaults, and merge user overrides.

    Parameters
    ----------
    tools_dict:
        Pre-loaded tools dict (from ``get_tools()``).  When *None* the function
        imports and calls ``get_tools()`` itself.
    """
    if tools_dict is None:
        from swarmee_river.tools import get_tools

        settings = load_settings()
        tools_dict = get_tools(settings)
        for name, tool_obj in load_enabled_pack_tools(settings).items():
            tools_dict.setdefault(name, tool_obj)
    else:
        settings = None

    overrides = load_tool_metadata_overrides()
    catalog: list[ToolMeta] = []
    pack_tool_names = set(load_enabled_pack_tools(settings).keys()) if settings is not None else set()

    for name, tool_obj in sorted(tools_dict.items()):
        # Extract docstring.
        docstring = ""
        with contextlib.suppress(Exception):
            raw_doc = getattr(tool_obj, "__doc__", None) or ""
            # Take the first non-empty line as a short description.
            for line in str(raw_doc).strip().splitlines():
                stripped = line.strip()
                if stripped:
                    docstring = stripped
                    break

        r, w, x = _resolve_permissions(name, tool_obj)
        if name in pack_tool_names:
            source = "pack"
        elif name in _CONNECTOR_TOOLS:
            source = "connector-backed"
        elif name in _CUSTOM_TOOL_NAMES:
            source = "core"
        else:
            source = "native"

        meta = ToolMeta(
            name=name,
            description=docstring,
            tags=[],
            access_read=r,
            access_write=w,
            access_execute=x,
            source=source,
        )

        # Merge user overrides (tags, access flags, description).
        user = overrides.get(name)
        if isinstance(user, dict):
            if "tags" in user:
                meta.tags = [str(t).strip() for t in (user["tags"] or []) if str(t).strip()]
            if "access_read" in user:
                meta.access_read = bool(user["access_read"])
            if "access_write" in user:
                meta.access_write = bool(user["access_write"])
            if "access_execute" in user:
                meta.access_execute = bool(user["access_execute"])
            if "description" in user and str(user["description"]).strip():
                meta.description = str(user["description"]).strip()

        catalog.append(meta)

    return catalog


# Names from _CUSTOM_TOOLS in tools.py for source classification.
# Keep in sync with ``_CUSTOM_TOOLS.keys()`` in ``tools.py``.
# ``project_context`` is conditionally added at runtime.
_CUSTOM_TOOL_NAMES = frozenset(
    {
        "file_list",
        "file_search",
        "file_read",
        "notebook_read",
        "office",
        "s3_browser",
        "session_s3",
        "snowflake_query",
        "athena_query",
        "glob",
        "list",
        "store_in_kb",
        "strand",
        "welcome",
        "rich_interface",
        "sop",
        "artifact",
        "git",
        "patch_apply",
        "plan_progress",
        "run_checks",
        "todoread",
        "todowrite",
        "agent_graph",
        "swarm",
        "project_context",
    }
)


# ── Tag query helpers ──────────────────────────────────────────────────────


def tools_by_tag(tag: str, catalog: list[ToolMeta]) -> list[str]:
    """Return tool names whose tags include *tag* (case-insensitive)."""
    tag_lower = tag.strip().lower()
    return [t.name for t in catalog if tag_lower in (tg.lower() for tg in t.tags)]


def all_tags(catalog: list[ToolMeta]) -> list[str]:
    """Return sorted unique tags across the full catalog."""
    tags: set[str] = set()
    for tool in catalog:
        tags.update(t.strip() for t in tool.tags if t.strip())
    return sorted(tags, key=str.lower)


__all__ = [
    "ToolMeta",
    "all_tags",
    "discover_tools_with_metadata",
    "load_tool_metadata_overrides",
    "save_tool_metadata_overrides",
    "tools_by_tag",
]
