# Writing Custom Tools

This guide covers everything you need to write, annotate, test, and deploy a custom tool in Swarmee River.

---

## Tool Anatomy

A tool is a Python function decorated with `@tool` from the Strands SDK. The function signature and docstring define everything the model sees.

```python
from strands import tool
from swarmee_river.tool_permissions import set_permissions


@tool
def my_tool(query: str, max_results: int = 10) -> str:
    """Search for something and return results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
    """
    # ... implementation ...
    return f"Found results for: {query}"


set_permissions(my_tool, "read")
```

Key rules:
- **Docstring** = the tool description the model sees. Make it clear and accurate.
- **Typed parameters** = the tool's input schema. Use standard Python types (`str`, `int`, `bool`, `list`, `dict`, `Optional[str]`).
- **Return type** = either `str` (plain text) or `dict` (structured, typically `{"content": [{"text": "..."}]}`). Plain strings are simplest.
- **`set_permissions()`** = required for every tool. See [Permission Tiers](#permission-tiers) below.

---

## The `tools/` Directory

Drop your tool file in `tools/` at the project root:

```
your-project/
  tools/
    my_tool.py       ← your new tool
    shell.py         ← existing built-in tools
    git.py
    ...
```

Swarmee River hot-loads everything in `tools/` at startup. No registration step needed. Your tool will appear immediately in:
- The TUI's **Tooling > Tools** tab
- The agent's available tool list
- The tool permission table

**Naming conventions:**
- File name = tool name (e.g., `tools/github_pr.py` exposes a tool named `github_pr`)
- Use snake_case
- One primary `@tool` function per file (helper functions are fine)

---

## Permission Tiers

You must call `set_permissions()` on every tool you write. The permission tells Swarmee River how the tool should be treated:

```python
set_permissions(my_tool, "read")        # safe, non-mutating
set_permissions(my_tool, "write")       # modifies files or state
set_permissions(my_tool, "execute")     # runs processes or calls external APIs
set_permissions(my_tool, "read", "write")   # multiple permissions
```

| Permission | Use when the tool... | Effect |
|-----------|---------------------|--------|
| `read` | Only reads data, makes no changes | Allowed during plan mode |
| `write` | Modifies files, databases, or local state | Blocked during plan mode; may prompt for consent |
| `execute` | Runs shell commands, calls external APIs, sends network requests | Blocked during plan mode; may prompt for consent |

When in doubt:
- HTTP GET requests → `read`
- HTTP POST/PUT/DELETE → `execute`
- Writing to disk → `write`
- Shell commands → `execute`

Full reference: [../tool_permissions.md](../tool_permissions.md)

---

## Return Values

**Simple string** (recommended for most tools):

```python
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # ...
    return f"Weather in {city}: 72°F, sunny"
```

**Structured dict** (for richer output or error handling):

```python
@tool
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    try:
        result = fetch_weather(city)
        return {"content": [{"text": f"Weather in {city}: {result}"}]}
    except Exception as e:
        return {"content": [{"text": f"Error: {e}"}], "status": "error"}
```

**Long results:** If your tool returns more than ~12,000 characters, Swarmee River automatically truncates the result and stores the full content as an **artifact** in `.swarmee/artifacts/`. The agent receives an artifact ID and can reference the full content later. You don't need to handle this yourself.

---

## Testing Your Tool

Write a standard pytest test:

```python
# tests/tools/test_my_tool.py
from tools.my_tool import my_tool


def test_my_tool_basic():
    result = my_tool(query="hello")
    assert "hello" in result


def test_my_tool_empty_query():
    result = my_tool(query="")
    assert result  # should return something, not crash
```

Run:

```bash
pytest tests/tools/test_my_tool.py -v
```

To verify the tool appears in the TUI:

```bash
swarmee tui
# Navigate to Tooling > Tools tab
# Find your tool in the list, check the Access column for your permissions
```

---

## Pack Tools

If you want to share a tool across multiple projects or distribute it to a team, package it as a **pack**:

```
packs/
  my-pack/
    tools/
      my_tool.py      ← same file as above
    sops/
      my-workflow.sop.md   ← optional
    README.md
```

Install and enable:

```bash
swarmee pack install ./packs/my-pack
swarmee pack enable my-pack
```

Pack tools are loaded identically to project tools. The only difference is source attribution (shown as `pack` in the TUI Tools tab instead of `custom`).

---

## Full Example: `github_pr_summary`

A tool that fetches a GitHub pull request and returns a structured summary. Uses the GitHub API with a token from the environment.

```python
# tools/github_pr_summary.py
from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

from strands import tool

from swarmee_river.tool_permissions import set_permissions


@tool
def github_pr_summary(
    owner: str,
    repo: str,
    pr_number: int,
    token: Optional[str] = None,
) -> str:
    """Fetch a GitHub pull request and return a structured summary.

    Args:
        owner: GitHub repository owner (username or org).
        repo: Repository name.
        pr_number: Pull request number.
        token: GitHub personal access token. Falls back to GITHUB_TOKEN env var.
    """
    auth_token = token or os.environ.get("GITHUB_TOKEN", "")
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"

    headers = {"Accept": "application/vnd.github.v3+json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            pr = json.loads(response.read().decode())
    except Exception as e:
        return f"Error fetching PR #{pr_number}: {e}"

    title = pr.get("title", "(no title)")
    state = pr.get("state", "unknown")
    author = (pr.get("user") or {}).get("login", "unknown")
    additions = pr.get("additions", 0)
    deletions = pr.get("deletions", 0)
    changed_files = pr.get("changed_files", 0)
    body = (pr.get("body") or "").strip() or "(no description)"

    return (
        f"PR #{pr_number}: {title}\n"
        f"State: {state} | Author: {author}\n"
        f"Changes: +{additions} -{deletions} across {changed_files} files\n\n"
        f"Description:\n{body}"
    )


set_permissions(github_pr_summary, "execute")
```

This uses `execute` because it makes an outbound network request.

Test it:

```python
# tests/tools/test_github_pr_summary.py
from unittest.mock import patch, MagicMock
import json
from tools.github_pr_summary import github_pr_summary


def test_github_pr_summary_parses_response():
    fake_pr = {
        "title": "Fix auth bug",
        "state": "open",
        "user": {"login": "alice"},
        "additions": 12,
        "deletions": 3,
        "changed_files": 2,
        "body": "Fixes the login flow.",
    }

    with patch("urllib.request.urlopen") as mock_open:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(fake_pr).encode()
        mock_open.return_value.__enter__ = lambda s: mock_response
        mock_open.return_value.__exit__ = MagicMock(return_value=False)

        result = github_pr_summary(owner="myorg", repo="myrepo", pr_number=42)

    assert "Fix auth bug" in result
    assert "alice" in result
    assert "+12" in result
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgot `set_permissions()` | Tool won't appear in TUI; add the call |
| Tool crashes on import | Check for missing dependencies; imports run at startup |
| Docstring is vague | The model uses it to decide when to call the tool — be specific |
| Returning a non-serializable object | Return `str` or a simple `dict`; avoid custom classes |
| Using `write` for a read-only tool | Use `read` — this gates plan-mode access correctly |

---

## Next Steps

- [sops-and-profiles.md](sops-and-profiles.md) — bind your new tool to an agent profile and SOP
- [../tool_permissions.md](../tool_permissions.md) — full permission model reference and table of all built-in tools
- [../testing/tui_e2e_testing.md](../testing/tui_e2e_testing.md) — test tools end-to-end with the TUI harness
