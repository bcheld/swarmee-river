---
name: repo-onboarding
version: 0.1.0
description: Quickly understand a repository and how to work in it.
---

# SOP: Repo onboarding

## Workflow
1) **Preflight**
   - Use `project_context(summary)` and `project_context(tree)` to identify entrypoints and structure.

2) **Identify entrypoints**
   - Find the CLI/app entry (`__main__`, scripts, `pyproject.toml` scripts, etc.).

3) **Find tests**
   - Determine how to run tests (pytest/hatch/nox/npm/etc.).

4) **Map the architecture**
   - Identify: key packages/modules, IO boundaries (filesystem/network), and config surfaces.

5) **Propose next steps**
   - Recommend 2–4 concrete tasks to make progress safely.

