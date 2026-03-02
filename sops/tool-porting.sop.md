---
name: tool-porting
version: 0.1.0
description: Port external tools (Strands SDK, MCP, pip packages, hand-written) into Swarmee River with permissions, safety, and tests.
---

# SOP: Tool porting (external tools into Swarmee River)

## Goal

Integrate a single external tool into Swarmee River so that it:

- Follows the standard `@tool` contract (input kwargs, output shape, error handling)
- Declares `read` / `write` / `execute` permissions via `set_permissions()`
- Respects plan-mode gating, consent rules, and policy hooks
- Has test coverage that proves it works when its upstream dependency is absent
- Is documented in the permission table

This SOP covers tools from **any** source: Strands SDK, MCP servers, pip packages, or hand-written.

## When to use

| Source | Use this SOP when... |
|--------|---------------------|
| **Strands SDK** (`strands_tools`) | Migrating an SDK tool to a local fallback, or adding a new SDK tool to the optional loader |
| **MCP server** | Wrapping an MCP tool so it can be used by the agent without external server management |
| **pip package** | Exposing a library (e.g. `boto3`, `requests`, `pandas`) as a callable agent tool |
| **Hand-written** | Building a new tool from scratch |

For narrow "replace a single `strands_tools` fallback" work, the older `tool-migration` SOP is still applicable but this SOP supersedes it for broader scope.

## Definitions

- **Canonical tool**: the primary tool name registered in `get_tools()` (e.g. `shell`, `file_write`).
- **Alias tool**: an OpenCode-compatibility name that delegates to a canonical tool (e.g. `bash` -> `shell`, `write` -> `file_write`). Defined in `opencode_aliases.py`.
- **Fallback tool**: a local implementation in `./tools/` used only when the upstream Strands SDK tool is unavailable. Registered via `_FALLBACK_TOOLS` + `setdefault`.
- **Custom tool**: a local implementation that **always** overrides any SDK tool of the same name. Registered via `_CUSTOM_TOOLS`.
- **SDK tool**: a tool loaded from the `strands_tools` package at runtime.
- **MCP tool**: a tool exposed by an external MCP (Model Context Protocol) server.
- **Pack tool**: a tool bundled into a Swarmee pack (`swarmee pack install ...`).

## Workflow

### Phase 1: Choose the integration pattern

| Pattern | When to use | Registration method |
|---------|-------------|-------------------|
| **A) Strands SDK + fallback** | Tool exists in `strands_tools`; you want a local fallback for environments without it | Add to `_OPTIONAL_STRANDS_TOOL_NAMES` + `_FALLBACK_TOOLS` (uses `setdefault` — never overwrites SDK) |
| **B) Custom tool (always loaded)** | Tool needs custom behavior, safety guardrails, or doesn't exist in SDK | Add to `_CUSTOM_TOOLS` in `tools.py` |
| **C) Pack-provided** | Tool is part of a reusable bundle (SOPs + tools + prompts) | Package as a Swarmee pack; install with `swarmee pack install` |
| **D) MCP server** | Tool runs as an external process; you want to connect it at runtime | Configure via MCP protocol settings; no local `tools/` file needed |

**Decision guide:**

- Does the tool already exist in `strands_tools`? -> Pattern A (fallback + optional import)
- Do you need to always override behavior for safety? -> Pattern B (custom override)
- Is the tool part of a larger capability set? -> Pattern C (pack)
- Does the tool require a long-running external process? -> Pattern D (MCP)
- None of the above? -> Pattern B (custom tool in `./tools/`)

### Phase 2: Implement the tool

Create `tools/<name>.py`. Every tool follows this template:

```python
from __future__ import annotations

from typing import Any

from strands import tool

from swarmee_river.tool_permissions import set_permissions


@tool
def my_tool(
    action: str = "default",
    *,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    One-line description of what this tool does.

    Longer description, listing actions/parameters if needed.
    """
    # --- Input validation ---
    action = (action or "").strip().lower()
    if not action:
        return {"status": "error", "content": [{"text": "action is required."}]}

    # --- Core logic ---
    result = "..."

    # --- Output ---
    return {"status": "success", "content": [{"text": result}]}


set_permissions(my_tool, "read")  # or "write", "execute", or a combination
```

**Output shape**: always return `{"status": "success"|"error", "content": [{"text": "..."}]}`. This is the standard contract consumed by the agent framework.

**Safety guardrails** (apply as appropriate):

| Concern | Guideline |
|---------|-----------|
| File paths | Resolve under `cwd`; reject `..` or out-of-tree targets |
| Shell commands | Set timeouts; avoid interactive stdin; consider POSIX vs Windows |
| Network calls | Allow only `http`/`https`; block `file://`; set connection timeouts |
| Secrets | Redact sensitive values by default; make reveal explicit |
| Large outputs | Truncate with `max_chars` parameter; persist full output to artifacts |
| External packages | Import inside the function body or guard with `try/except ImportError` |

### Phase 3: Declare permissions

Add a `set_permissions()` call **after** the function definition:

```python
from swarmee_river.tool_permissions import set_permissions

set_permissions(my_tool, "read")          # read-only
set_permissions(my_tool, "write")         # creates/modifies files or state
set_permissions(my_tool, "execute")       # runs commands or makes network calls
set_permissions(my_tool, "read", "execute")  # queries that also run commands
set_permissions(my_tool)                  # informational, no permissions
```

**How to choose:**

| If the tool... | Permission |
|---------------|------------|
| Only reads files, queries data, fetches metadata | `"read"` |
| Creates, modifies, or deletes files or persistent state | `"write"` |
| Runs shell commands, spawns processes, or calls external APIs | `"execute"` |
| Queries a database that requires running SQL | `"read", "execute"` |
| Does pure computation with no I/O | *(none)* |

**Multi-tool modules**: when a file defines multiple tools, annotate each separately:

```python
set_permissions(todoread, "read")
set_permissions(todowrite, "write")
```

**SDK tools you can't annotate**: if you're integrating a Strands SDK tool where you don't control the source code, add an entry to `STRANDS_TOOL_PERMISSIONS` in `src/swarmee_river/tool_permissions.py` instead:

```python
STRANDS_TOOL_PERMISSIONS["my_sdk_tool"] = frozenset({"execute"})
```

### Phase 4: Register the tool

Update `src/swarmee_river/tools.py`:

**Pattern A — Strands SDK + fallback:**

```python
# In _OPTIONAL_STRANDS_TOOL_NAMES, add:
"my_tool",

# In _FALLBACK_TOOLS, add:
"my_tool": my_tool_fallback,
```

The `setdefault` logic ensures your fallback is only used when `strands_tools.my_tool` is unavailable.

**Pattern B — Custom tool (always loaded):**

```python
# In _CUSTOM_TOOLS, add:
"my_tool": my_tool_module,           # if registering the module
"my_tool": my_tool_module.my_tool,   # if registering the function directly
```

**Adding aliases**: if the tool needs an OpenCode-compatible alias, update `src/swarmee_river/opencode_aliases.py`:

1. Add the alias mapping to `OPENCODE_TOOL_ALIASES`.
2. Create the alias function with the `@tool` decorator.
3. Call `set_permissions()` on the alias function.
4. Add the alias to the `opencode_alias_tools()` return dict.

### Phase 5: Enforce safety parity

After registration, review the safety implications:

1. **High-risk tools**: if the tool can mutate the filesystem or execute arbitrary commands, consider adding it to `_HIGH_RISK_TOOLS` in `src/swarmee_river/permissions.py`. This triggers interactive consent prompts.

2. **Consent rules**: if the tool needs a default consent rule (ask/allow/deny), add a `ToolRule` in `src/swarmee_river/settings.py`.

3. **Alias canonicalization**: ensure `opencode_aliases.py` maps any aliases so that policy hooks, consent rules, and plan-mode gating treat aliased and canonical names equivalently. Verify with:
   ```python
   from swarmee_river.opencode_aliases import equivalent_tool_names
   assert "my_alias" in equivalent_tool_names("my_tool")
   ```

4. **Plan-mode gating**: tools with `write` or `execute` permissions are automatically blocked in plan mode. No manual action needed — this is derived from the permission annotation.

### Phase 6: Test

**Required tests:**

1. **Registry test** — monkeypatch `strands_tools` import to fail, then verify the tool is still present in `get_tools()`. Add to the existing pattern in `tests/test_tools.py`.

2. **Tool-level test** — create `tests/tools/test_<name>.py` with focused behavior checks:
   ```python
   def test_my_tool_default_action():
       result = my_tool(action="list")
       assert result["status"] == "success"

   def test_my_tool_invalid_action():
       result = my_tool(action="unknown")
       assert result["status"] == "error"
   ```

3. **Permission enforcement** — no manual test needed. The existing `test_all_project_tools_have_permissions` in `tests/test_tool_permissions.py` automatically verifies that every tool in `get_tools()` has declared permissions or is covered by `STRANDS_TOOL_PERMISSIONS`.

**Run the full suite:**

```bash
pytest tests/ -v
```

### Phase 7: Document

1. **Permission table** — add the tool to the appropriate category in `docs/tool_permissions.md`.

2. **README** — if the tool introduces a new category, update the tool list in `README.md` under "Integrated Tools".

3. **Tool catalog** — if relevant to OpenCode parity, update `docs/opencode_port/tool_catalog.md`.

## Porting from specific sources

### Strands SDK tools

The standard pattern for optional Strands SDK tools:

1. Add the tool name to `_OPTIONAL_STRANDS_TOOL_NAMES` in `tools.py`.
2. The `_load_optional_strands_tools()` function attempts `from strands_tools import <name>` at runtime.
3. Create a local fallback in `tools/<name>.py` and add it to `_FALLBACK_TOOLS`.
4. If you can't annotate the SDK tool directly (it's loaded dynamically), add its permissions to `STRANDS_TOOL_PERMISSIONS` in `tool_permissions.py`.
5. Otherwise, call `set_permissions()` on the fallback function — the fallback's annotation is used when the SDK tool is absent.

### MCP tools

MCP tools run on external servers and connect via the Model Context Protocol:

1. Configure the MCP server connection in `.swarmee/settings.json` or environment variables.
2. MCP tools are registered at runtime by the MCP client layer, not via `tools.py`.
3. For permissions: MCP tools without annotations fall through to the heuristic tier in `_resolve_permissions()`. To set explicit permissions, add entries to `STRANDS_TOOL_PERMISSIONS` keyed by the MCP tool's registered name.
4. Plan-mode: MCP tools without declared permissions are conservatively excluded from the plan-mode allowlist.

### pip packages

When wrapping a pip-installable library:

1. Create `tools/<name>.py`.
2. Import the package inside the tool function body (not at module level) so the tool file loads even when the package isn't installed:
   ```python
   @tool
   def my_tool(...) -> dict[str, Any]:
       try:
           import some_package
       except ImportError:
           return {"status": "error", "content": [{"text": "some_package is required. Install with: pip install some_package"}]}
       ...
   ```
3. Add the package to the appropriate extras group in `pyproject.toml` if it should be an optional dependency.
4. Follow Phases 3-7 as normal.

### Hand-written tools

Start from scratch using the template in Phase 2. Good reference implementations:

- **Simple read tool**: `tools/current_time.py` (22 lines)
- **File ops with safety**: `tools/file_write.py` (cwd boundary checks, atomic writes)
- **Multi-action tool**: `tools/artifact.py` (list/get/upload/store_in_kb actions)
- **Shell with guardrails**: `tools/shell.py` (timeouts, POSIX/Windows, truncation)
- **Multi-tool module**: `tools/todo.py` (todoread + todowrite in one file)

## Review checklist

Before committing, verify:

- [ ] Tool follows the standard output shape: `{"status": ..., "content": [{"text": ...}]}`
- [ ] `set_permissions()` is called with the correct permission(s)
- [ ] Tool is registered in `tools.py` (via `_CUSTOM_TOOLS` or `_FALLBACK_TOOLS`)
- [ ] Plan-mode gating works correctly (write/execute tools blocked in plan mode)
- [ ] If high-risk: added to `_HIGH_RISK_TOOLS` in `permissions.py`
- [ ] If aliased: alias canonicalizes correctly for policy/consent
- [ ] Cross-platform: works on macOS, Linux, and Windows (or fails gracefully)
- [ ] Path safety: no out-of-tree writes, resolves under `cwd`
- [ ] External packages imported inside function body, not at module level
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Permission table updated in `docs/tool_permissions.md`

## Lessons learned

- **Never overwrite SDK tools by accident.** Use `setdefault` for fallbacks. Only use `_CUSTOM_TOOLS` when you intentionally want to override.
- **Aliases must not bypass policy.** Treat OpenCode aliases as "different spellings of the same tool" — `shell` and `bash` share consent decisions and plan-mode gating.
- **Module vs function registration.** If you register a module object (not a function) in `_CUSTOM_TOOLS`, `get_permissions()` auto-probes for the same-named function. No special action needed, but be aware of the mechanism.
- **SDK import timing.** Don't import heavy packages (`boto3`, `snowflake-connector-python`) at module level. Import inside the function body to keep startup fast and allow the tool file to load without the dependency.
- **Test portability.** Tests that bind sockets can fail in sandboxed runners. Guard server startup with `try/except PermissionError` and `pytest.skip()`.
- **Typos are caught early.** `set_permissions(tool, "rea")` raises `ValueError` at module-load time, not at runtime. This is intentional.
- **The permission test is your safety net.** `test_all_project_tools_have_permissions` automatically catches any new tool that forgot to declare permissions. Run it before committing.
