---
name: bugfix
version: 0.1.0
description: Reproduce → isolate → fix → prevent regressions.
---

# SOP: Bugfix

## Workflow
1) **Reproduce**
   - Capture exact error messages, inputs, and environment assumptions.
   - Add a minimal failing test when feasible.

2) **Isolate**
   - Identify the smallest failing unit (file/function) and the root cause.
   - Prefer reading code + targeted search over broad scans.

3) **Fix**
   - Implement the minimal change that resolves the root cause.
   - Avoid drive-by refactors.

4) **Regression**
   - Add/adjust tests to prevent recurrence.
   - Run relevant tests locally (prefer fast unit tests).

5) **Document**
   - Summarize: root cause, fix, and verification steps.

