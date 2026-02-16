---
name: tool-migration
version: 0.1.0
description: Migrate tools off `strands-agents-tools` with cross-platform fallbacks, safety parity, and tests.
---

# SOP: Tool migration (Strands Tools → local fallbacks)

## Goal
Remove (or reduce) the runtime dependency on `strands-agents-tools` by implementing compatible **local fallback tools**
under `./tools/`, while preserving:

- Cross-platform behavior (Windows/macOS/Linux; SageMaker images without Node)
- Safety parity (ToolPolicy + ToolConsent)
- Consistent response shapes (tool outputs)
- Test coverage that proves the app works when `strands_tools` is missing

This SOP is optimized for **incremental migration**: a tool can be migrated and shipped without migrating everything at once.

## When to use
Use this SOP when:
- A runtime environment can’t install `strands-agents-tools` (e.g. locked-down SageMaker images)
- A Strands tool is platform-dependent (Windows gaps, missing binaries)
- You want to stabilize an interface and de-risk future upgrades of Strands Tools

## Definitions
- **Canonical tool**: the “real” tool name (`shell`, `file_write`, …).
- **Alias tool**: OpenCode compatibility name (`bash`, `write`, `edit`, …) that must not bypass policy/consent.
- **Fallback tool**: local implementation in `./tools/<name>.py` registered only if the Strands variant is missing.

## Workflow

### 1) Inventory: find what to migrate next
1) Identify tools that:
   - are used by the CLI/REPL hot path, or
   - are referenced by prompts / built-in SOPs / example workflows, or
   - are required for parity (OpenCode / Claude Code patterns).
2) Grep for direct imports and tool usage:
   - `rg -n "\\bstrands_tools\\b" -S .`
   - `rg -n "agent\\.tool\\." -S src tools`
3) Prioritize by impact:
   - “Boot critical”: must run without `strands_tools` (CLI startup, REPL, basic file ops)
   - “Safety critical”: anything that can mutate repo or execute commands
   - “Nice-to-have”: nonessential integrations (Slack, image/video generation, etc.)

### 2) Define the compatibility contract (API + output)
For each tool:
- Confirm the **expected input kwargs** (names and types) used by the app and common prompts.
- Confirm the **expected output shape**:
  - Standard is `{"status": "success"|"error", "content": [{"text": "..."}]}`.
- Decide “best-effort” vs “fail-closed” behavior.

Guideline:
- Prefer **best-effort** for optional tools (e.g. retrieve/memory).
- Prefer **fail-closed** for high-risk tools in non-interactive mode (consent hook already does this).

### 3) Implement the fallback tool in `./tools/`
1) Create `tools/<tool_name>.py`.
2) Implement a `@tool` function with:
   - strict input validation
   - bounded outputs (`max_chars` truncation where needed)
   - clear error messages
3) Add guardrails appropriate to the tool:
   - File write/edit: resolve under `cwd`; block `..`/out-of-tree; atomic writes.
   - Shell: timeouts; avoid interactive hangs; consider Windows/POSIX differences.
   - Environment: redact sensitive values by default; make “reveal” explicit.
   - Network: allow only http/https; block file://; timeouts.

### 4) Register the fallback (without breaking Strands Tools installs)
Update `/Users/bcheld/dev/vscode/swarmee-river/src/swarmee_river/tools.py`:
- Keep the best-effort loader for `strands_tools.<name>`.
- Register fallbacks with `tools.setdefault("<name>", <fallback>)`.

Rule:
- **Never overwrite** the Strands tool if it exists (use `setdefault`).
- If you must replace behavior for safety (e.g. cancellable variants), do it intentionally via the custom tool override map.

### 5) Enforce safety parity (aliases must not bypass)
Ensure ToolPolicy/ToolConsent treat aliases equivalently:
- Canonicalize tool names (alias → canonical).
- Ensure allowlists/blocklists and plan-mode allowlists handle aliases.
- Ensure remembered consent decisions and “plan approved” consent treat alias/canonical as equivalent.

If adding a new tool:
- Decide whether it belongs in the “high risk” defaults.
- If it’s risky, add a `ToolRule` default in `/Users/bcheld/dev/vscode/swarmee-river/src/swarmee_river/settings.py`.

### 6) Tests: prove it works without `strands_tools`
Add two layers of tests:

1) **Registry-level tests**:
   - In `/Users/bcheld/dev/vscode/swarmee-river/tests/test_tools.py`, monkeypatch imports so `import strands_tools` fails.
   - Assert the tool name is still present from `get_tools()`.

2) **Tool-level tests**:
   - Add `tests/tools/test_<tool>.py` with focused behavior checks.
   - Avoid environment assumptions:
     - Some sandboxes disallow binding local sockets → skip those tests gracefully.

### 7) Docs + matrix updates (optional but recommended)
- Update tool catalog / capability matrix if this is part of parity work.
- Call out fallbacks explicitly for constrained environments.

### 8) Review checklist (before committing)
- Does the fallback behave on Windows/macOS/Linux?
- Does it avoid bypassing policy/consent?
- Does it preserve path safety (no out-of-tree writes)?
- Does non-interactive execution fail safely for risky actions?
- Are tests stable in restricted CI/sandbox environments?

## Lessons learned (project-specific)
- Prefer `tools.setdefault(...)` for fallbacks so installs with `strands_tools` still use the upstream implementation.
- Treat OpenCode aliases as “different spellings of the same tool” in policy/consent; otherwise approvals can be bypassed.
- Any “sub-agent” helper tool must be **tool-less** by default (prevents recursion and bypassing hooks).
- Tests that bind sockets can fail in sandboxed runners; wrap server startup and `pytest.skip(...)` on `PermissionError`.
- For file mutation tools, enforce:
  - `cwd` boundary checks
  - atomic writes (`os.replace`)
  - directory target rejection
- For environment tools, redact secrets by default; make value reveal intentional.

## Output expectations
When applying this SOP, the deliverables are:
- `tools/<tool>.py` fallback implementation
- registration in `/Users/bcheld/dev/vscode/swarmee-river/src/swarmee_river/tools.py`
- at least 1 tool-level test + updated registry-level test
- (optional) settings/docs updates if behavior or risk profile changes

