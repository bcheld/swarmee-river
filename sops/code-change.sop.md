---
name: code-change
version: 0.1.0
description: Plan/approve/execute loop for making safe code changes.
---

# SOP: Code change

## Goal
Make a code change safely and reproducibly, with artifacts and a clear summary.

## Workflow
1) **Preflight context**
   - Use `project_context(summary|tree|search|read)` for quick repo discovery.
   - Keep context targeted (read only what you need).

2) **Plan**
   - Produce a short plan (3–8 steps).
   - Include: files to read, files to edit, tools/commands expected, and risks.

3) **Confirm**
   - Ask for user approval before running risky tools (shell/editor/file_write/http_request).

4) **Execute**
   - Apply changes step-by-step.
   - Run the smallest relevant tests/commands first.

5) **Verify + report**
   - Summarize what changed, how it was validated, and where artifacts/logs were written.

