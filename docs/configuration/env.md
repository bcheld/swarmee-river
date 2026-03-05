# Environment Variables

## Supported End-User Env Vars (Secrets Only)

Swarmee only supports environment variables for secrets and credentials.

- `OPENAI_API_KEY` (OpenAI)
- `SWARMEE_GITHUB_COPILOT_API_KEY` (GitHub Copilot, preferred)
- `GITHUB_TOKEN` (GitHub token, legacy alias for Copilot auth)
- `GH_TOKEN` (GitHub token, legacy alias for Copilot auth)

## External Provider Env (SDK-Managed)

Swarmee does not read these directly, but upstream SDKs do.

- AWS SDK credentials (for Bedrock):
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_SESSION_TOKEN` (optional)

## Internal Env Vars (Not Supported User Config)

These are internal wiring variables used for subprocess/runtime-broker transport. They are intentionally not documented as supported configuration knobs and may change without notice.

See `src/swarmee_river/config/env_policy.py` for the current internal allowlist.

## Removed Env Vars (What To Use Instead)

All non-secret configuration that was previously controlled by `SWARMEE_*`/`STRANDS_*` env vars has moved to:

- `.swarmee/settings.json` structured fields
- CLI flags
- TUI settings

If your project has a legacy `.swarmee/settings.json` `env` section, use:

```bash
swarmee settings migrate
```

This rewrites legacy `env.*` overrides into structured settings fields and drops unsupported keys (secrets are never persisted into settings).

