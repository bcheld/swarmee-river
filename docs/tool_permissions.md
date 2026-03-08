# Tool Permissions

Every tool in Swarmee River declares its **permissions** — `read`, `write`, and/or `execute` — directly on the tool function object. These annotations drive plan-mode gating, TUI metadata badges, and safety policy enforcement.

## Overview

| Permission | Meaning | Examples |
|------------|---------|----------|
| `read` | Inspects files, queries data, fetches metadata — no side effects | `file_read`, `glob`, `retrieve` |
| `write` | Creates, modifies, or deletes files or persistent state | `editor`, `patch_apply`, `artifact` |
| `execute` | Runs commands, spawns processes, or makes network calls | `shell`, `python_repl`, `http_request` |

A tool can have multiple permissions (e.g. `athena_query` has both `read` and `execute`). Informational tools like `calculator` and `plan_progress` declare no permissions at all.

### Why permissions exist

- **Plan mode**: only tools with `read` permission (or no permissions) are allowed during planning. All `write`/`execute` tools are blocked until the plan is approved.
- **TUI Tools tab**: the permission flags surface as access-class badges (R/W/X) in the tool catalog.
- **Safety policy**: the `_HIGH_RISK_TOOLS` set in `permissions.py` captures the subset of `write`/`execute` tools that require interactive user consent by default.
- **User overrides**: operators can override individual tool permissions via `.swarmee/tool_metadata.json`.

## Quick start

```python
from strands import tool
from swarmee_river.tool_permissions import set_permissions

@tool
def my_tool(query: str) -> dict:
    """Search for something."""
    ...

set_permissions(my_tool, "read")
```

That's it. The `set_permissions()` call attaches a validated `frozenset({"read"})` to the function object. All downstream consumers pick it up automatically.

## API reference

All exports live in `src/swarmee_river/tool_permissions.py`.

### `set_permissions(tool_obj, *perms)`

Attach a frozenset of permission strings to a tool object.

```python
set_permissions(shell, "execute")
set_permissions(athena_query, "read", "execute")
set_permissions(calculator)  # informational — no permissions
```

- **Validation**: raises `ValueError` at module-load time if any permission string is not in `{"read", "write", "execute"}`. This catches typos like `"rea"` immediately.
- **Attribute**: sets `tool_obj.permissions = frozenset(perms)`.

### `get_permissions(tool_obj) -> frozenset[str] | None`

Read the permissions frozenset from a tool object.

- Returns `None` if the tool has no `.permissions` attribute.
- **Module auto-probe**: when the tool object is a Python module (some tools are registered as their containing module in `tools.py`), the function extracts the module's short name and probes for a same-named function inside it. For example, if `tool_obj` is the `tools.git` module, it finds `tools.git.git` and reads its permissions.

### `ToolPermission` enum

```python
class ToolPermission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
```

Provided for programmatic use. The `set_permissions()` function accepts plain strings.

### `STRANDS_TOOL_PERMISSIONS` dict

A fallback map for Strands SDK tools that are loaded from `strands_tools` and cannot be annotated directly with `set_permissions()`:

```python
STRANDS_TOOL_PERMISSIONS = {
    "image_reader": frozenset({"read"}),
    "memory":       frozenset({"read"}),
    "journal":      frozenset({"read"}),
    "think":        frozenset(),
    "stop":         frozenset(),
    "use_aws":      frozenset({"execute"}),
    "load_tool":    frozenset({"execute"}),
    "workflow":     frozenset({"execute"}),
    "cron":         frozenset({"execute"}),
    "slack":        frozenset({"execute"}),
    "speak":        frozenset({"execute"}),
    "generate_image": frozenset({"execute"}),
    "nova_reels":   frozenset({"execute"}),
}
```

This map is only consulted when `get_permissions()` returns `None` for a given tool object.

## Permission resolution order

When the system needs a tool's permissions (for plan-mode gating, TUI display, etc.), it follows a three-tier resolution implemented in `tui/tool_metadata.py:_resolve_permissions()`:

1. **Declared attribute** — `get_permissions(tool_obj)`. If the tool function (or its containing module) has a `.permissions` attribute, use it.
2. **SDK fallback map** — `STRANDS_TOOL_PERMISSIONS.get(name)`. For `strands_tools` that can't be annotated.
3. **Legacy heuristic sets** — `_heuristic_access(name)` in `tool_metadata.py`. Hardcoded `_READ_TOOLS` / `_WRITE_TOOLS` / `_EXECUTE_TOOLS` frozensets, kept as a final safety net.

In practice, all built-in tools now hit tier 1 or tier 2. Tier 3 exists for backwards compatibility only.

## Built-in tool permission table

### File operations

| Tool | File | Permission(s) |
|------|------|---------------|
| `file_list` | `tools/file_ops.py` | `read` |
| `file_search` | `tools/file_ops.py` | `read` |
| `file_read` | `tools/file_ops.py` | `read` |
| `editor` | `tools/editor.py` | `write` |
| `glob` | `tools/path_ops.py` | `read` |
| `list` | `tools/path_ops.py` | `read` |
| `office` | `tools/office.py` | `read` |
| `patch_apply` | `tools/patch_apply.py` | `write` |

### Shell and code execution

| Tool | File | Permission(s) |
|------|------|---------------|
| `shell` | `tools/shell.py` | `execute` |
| `python_repl` | `tools/python_repl.py` | `execute` |
| `git` | `tools/git.py` | `execute` |
| `run_checks` | `tools/run_checks.py` | `execute` |

### Network and queries

| Tool | File | Permission(s) |
|------|------|---------------|
| `http_request` | `tools/http_request.py` | `execute` |
| `snowflake_query` | `tools/snowflake_query.py` | `read`, `execute` |
| `athena_query` | `tools/athena_query.py` | `read`, `execute` |

### Artifacts, storage, and knowledge base

| Tool | File | Permission(s) |
|------|------|---------------|
| `artifact` | `tools/artifact.py` | `write` |
| `store_in_kb` | `tools/store_in_kb.py` | `write` |
| `retrieve` | `tools/retrieve.py` | `read` |
| `session_s3` | `tools/session_s3.py` | `write` |
| `s3_browser` | `tools/s3_browser.py` | `read` |

### Multi-agent delegation

| Tool | File | Permission(s) |
|------|------|---------------|
| `strand` | `tools/strand.py` | `execute` |
| `swarm` | `tools/swarm.py` | `execute` |
| `agent_graph` | `tools/agent_graph.py` | `execute` |
| `use_agent` | `tools/use_agent.py` | `execute` |
| `use_llm` | `tools/use_agent.py` | `execute` |

### Context and utilities

| Tool | File | Permission(s) |
|------|------|---------------|
| `project_context` | `tools/project_context.py` | `read` |
| `sop` | `tools/sop.py` | `read` |
| `current_time` | `tools/current_time.py` | `read` |
| `environment` | `tools/environment.py` | `read` |
| `todoread` | `tools/todo.py` | `read` |
| `todowrite` | `tools/todo.py` | `write` |
| `welcome` | `tools/welcome.py` | `write` |

### Informational (no permissions)

| Tool | File | Notes |
|------|------|-------|
| `calculator` | `tools/calculator.py` | Pure compute, no side effects |
| `plan_progress` | `tools/plan_progress.py` | Status reporting only |

### Canonical coding workflow

- Read/search: `file_list`, `file_search`, `file_read`
- Edit one file: `editor`
- Apply structured or multi-file changes: `patch_apply`
- Verify: `run_checks` first, `shell` when you need a command outside the standard check runner

### Strands SDK tools (fallback map)

These are loaded from `strands_tools` when available and cannot be annotated directly. Their permissions are declared in the `STRANDS_TOOL_PERMISSIONS` map.

| Tool | Permission(s) |
|------|---------------|
| `image_reader` | `read` |
| `memory` | `read` |
| `journal` | `read` |
| `think` | *(none — informational)* |
| `stop` | *(none — informational)* |
| `use_aws` | `execute` |
| `load_tool` | `execute` |
| `workflow` | `execute` |
| `cron` | `execute` |
| `slack` | `execute` |
| `speak` | `execute` |
| `generate_image` | `execute` |
| `nova_reels` | `execute` |

## How permissions are consumed

### Plan-mode allowlist

`hooks/tool_policy.py` derives the plan-mode tool allowlist at runtime via `_build_plan_mode_allowlist()`. This function iterates all registered tools and includes only those with:

- Only `read` permission, or
- No permissions at all (informational tools like `think`, `calculator`).

Tools with `write` or `execute` are excluded. `project_context` is always included regardless of permissions because plan mode restricts it separately via an action allowlist.

A hardcoded `_FALLBACK_PLAN_MODE_ALLOWED_TOOLS` set is used if the permission-based derivation fails (e.g. during early bootstrap before tools are loaded).

### TUI Tools tab metadata

`tui/tool_metadata.py` calls `_resolve_permissions(name, tool_obj)` to populate three boolean flags on each `ToolMeta` dataclass: `access_read`, `access_write`, `access_execute`. These surface as R/W/X badges in the TUI tool catalog.

Users can override these flags per-tool via `.swarmee/tool_metadata.json`:

```json
{
  "my_tool": {
    "access_read": true,
    "access_write": false,
    "access_execute": false
  }
}
```

### `_HIGH_RISK_TOOLS` (consent gating)

`permissions.py` defines a static `_HIGH_RISK_TOOLS` set listing tools that require interactive user consent before execution. Every tool in this set should have `write` or `execute` permission, but not all `write`/`execute` tools are high-risk. The set is kept static (not derived at runtime) to avoid circular imports in the consent hot path.

Current high-risk tools: `shell`, `editor`, `patch_apply`, `http_request`.

## Adding permissions to a new tool

1. **Create your tool** in `tools/<name>.py` with the `@tool` decorator.

2. **Import and call `set_permissions()`** after the function definition:

   ```python
   from swarmee_river.tool_permissions import set_permissions

   @tool
   def my_tool(...) -> dict[str, Any]:
       ...

   set_permissions(my_tool, "write")
   ```

3. **Choose the right permissions**:
   - `"read"` — tool only inspects data (files, APIs, metadata). No mutations.
   - `"write"` — tool creates, modifies, or deletes files or state.
   - `"execute"` — tool runs commands, spawns processes, or makes network calls.
   - Combine them when needed: `set_permissions(my_tool, "read", "execute")`.
   - Leave empty for purely informational tools: `set_permissions(my_tool)`.

4. **Register the tool** in `src/swarmee_river/tools.py` (see `_CUSTOM_TOOLS` or `_FALLBACK_TOOLS`).

5. **Verify** — the existing test `test_all_project_tools_have_permissions` in `tests/test_tool_permissions.py` will automatically catch any tool that's missing permission metadata. Run:

   ```bash
   pytest tests/test_tool_permissions.py -v
   ```

### Multi-tool modules

When a single file defines multiple tools (e.g. `tools/todo.py`), annotate each function separately:

```python
@tool
def todoread(...) -> dict[str, Any]:
    ...

@tool
def todowrite(...) -> dict[str, Any]:
    ...

set_permissions(todoread, "read")
set_permissions(todowrite, "write")
```

### Module-registered tools

Some tools in `_CUSTOM_TOOLS` are registered as Python modules rather than function objects (e.g. `"git": git` where `git` is the `tools.git` module). The `get_permissions()` function handles this transparently — it detects module objects and probes for a same-named function inside them. No special action is required from tool authors.

### SDK tools you can't annotate

If you're integrating a Strands SDK tool where you don't control the source, add an entry to `STRANDS_TOOL_PERMISSIONS` in `src/swarmee_river/tool_permissions.py`:

```python
STRANDS_TOOL_PERMISSIONS["my_sdk_tool"] = frozenset({"execute"})
```
