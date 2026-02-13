---
name: security-review
version: 0.1.0
description: Lightweight security review checklist for code changes.
---

# SOP: Security review

## Checklist
1) **Secrets hygiene**
   - Ensure `.env`/keys/tokens are not committed.
   - Validate `.gitignore` excludes secrets and local artifacts.

2) **Shell safety**
   - Avoid destructive commands by default.
   - Prefer dry-runs and explicit paths.

3) **Network calls**
   - Gate `http_request` usage; prefer allowlisted domains when possible.

4) **Data exfiltration**
   - Avoid uploading local code/data without explicit user consent.

5) **Dependencies**
   - Note any new dependencies; keep scope minimal.

