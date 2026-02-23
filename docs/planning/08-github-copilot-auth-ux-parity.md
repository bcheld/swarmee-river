# GitHub Copilot Auth UX Parity Plan

## Objective

Deliver OpenCode-style authentication UX for GitHub Copilot so users can connect once and run without
manually managing environment variables.

## Implemented Direction

1. Add persisted auth storage with OpenCode compatibility:
   - Swarmee store: `~/.local/share/swarmee/auth.json` (override via `SWARMEE_AUTH_PATH`)
   - OpenCode import fallback: `~/.local/share/opencode/auth.json` (override via `SWARMEE_OPENCODE_AUTH_PATH`)

2. Add GitHub Copilot device-code login flow:
   - `POST https://github.com/login/device/code`
   - Poll `POST https://github.com/login/oauth/access_token`
   - Exchange to Copilot runtime token via:
     `GET https://api.github.com/copilot_internal/v2/token`

3. Store OAuth credentials and refresh runtime token automatically:
   - Save `refresh`, `access`, `expires`, and endpoint metadata
   - Refresh expired runtime access token from refresh token

4. Add user commands for parity:
   - CLI: `swarmee connect`, `swarmee auth list|login|logout`
   - REPL: `:connect`, `:auth list|login|logout`
   - TUI: `/connect`, `/auth list`, `/auth logout`

5. Wire provider runtime to auth store:
   - `has_github_copilot_token()` checks env + stored credentials
   - `github_copilot` model config resolves token/base URL from auth store when env vars are absent

## Safety and UX Notes

- No secrets are logged in plaintext by default command outputs.
- Auth files are written with user-only permissions where supported (`chmod 600` best-effort).
- Device flow opens browser by default but still prints verification URI/code for remote/headless use.
