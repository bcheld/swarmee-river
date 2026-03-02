"""Per-tool permission metadata protocol.

Tools declare their permissions by setting a ``permissions`` attribute on the
decorated function object.  The canonical values are ``"read"``, ``"write"``,
and ``"execute"``.

Example::

    from strands import tool
    from swarmee_river.tool_permissions import set_permissions

    @tool
    def file_read(path: str, ...) -> dict:
        ...

    set_permissions(file_read, "read")
"""

from __future__ import annotations

import types
from enum import Enum
from typing import Any

PERMISSIONS_ATTR = "permissions"


class ToolPermission(str, Enum):
    """Canonical permission values for tool annotations."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


_VALID_PERMISSIONS = frozenset({p.value for p in ToolPermission})


def set_permissions(tool_obj: Any, *perms: str) -> None:
    """Attach a frozenset of permission strings to a tool object.

    Validates that all values are recognised permission strings.
    Raises ``ValueError`` at module-load time for typos like ``"rea"``.
    """
    invalid = set(perms) - _VALID_PERMISSIONS
    if invalid:
        raise ValueError(f"Invalid permission(s): {invalid}. Valid: {_VALID_PERMISSIONS}")
    setattr(tool_obj, PERMISSIONS_ATTR, frozenset(perms))


def get_permissions(tool_obj: Any) -> frozenset[str] | None:
    """Read permissions from a tool object, or *None* if not declared.

    When *tool_obj* is a module (some tools are registered as their containing
    module rather than the decorated function), the function probes for a
    same-named callable inside the module and reads its permissions attribute.
    """
    value = getattr(tool_obj, PERMISSIONS_ATTR, None)
    if value is None and isinstance(tool_obj, types.ModuleType):
        # Probe for a same-named tool function inside the module.
        mod_name = getattr(tool_obj, "__name__", "")
        short_name = mod_name.rsplit(".", 1)[-1] if mod_name else ""
        if short_name:
            inner = getattr(tool_obj, short_name, None)
            if inner is not None:
                value = getattr(inner, PERMISSIONS_ATTR, None)
    if value is None:
        return None
    if isinstance(value, frozenset):
        return value
    # Tolerate set / tuple / list being set directly.
    return frozenset(str(v) for v in value)


# ---------------------------------------------------------------------------
# Fallback permissions for Strands SDK tools that we cannot annotate directly.
# When a project fallback tool overrides an SDK tool, the project tool's own
# annotation takes precedence; this map is only consulted when the tool object
# has no ``.permissions`` attribute.
# ---------------------------------------------------------------------------
STRANDS_TOOL_PERMISSIONS: dict[str, frozenset[str]] = {
    # Read-only
    "image_reader": frozenset({"read"}),
    "memory": frozenset({"read"}),
    "journal": frozenset({"read"}),
    # Informational (no permissions)
    "think": frozenset(),
    "stop": frozenset(),
    # Execute
    "use_aws": frozenset({"execute"}),
    "load_tool": frozenset({"execute"}),
    "workflow": frozenset({"execute"}),
    "cron": frozenset({"execute"}),
    "slack": frozenset({"execute"}),
    "speak": frozenset({"execute"}),
    "generate_image": frozenset({"execute"}),
    "nova_reels": frozenset({"execute"}),
}
