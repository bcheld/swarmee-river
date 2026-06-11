# 18 — Daemon Containerization: Clean TUI ↔ Runtime Separation

**Date:** 2026-06-11
**Status:** Proposed
**Theme:** Prepare for migrating the backend daemon (LLM layer: agent loop, tools, context management) into a separate container, with the user application (TUI/CLI) staying local. Goal: a boundary where the only things crossing it are the wire protocol and explicitly declared mounts/secrets.

---

## Current Architecture (verified)

```
TUI (local) ──spawn or socket──► Runtime broker ──subprocess(stdio pipes)──► Session daemon
                                  (server.py)        (swarmee.py --tui-daemon)
```

- Two transports exist: direct subprocess (`tui/transport.py:39-72`) and a TCP
  socket to a broker, discovered via a `runtime.json` file (host, port, token;
  `runtime_service/client.py:62-64, 156-232`).
- The broker spawns each session daemon as a child process with **stdio pipes**
  and the **broker's entire environment** plus per-session overrides
  (`runtime_service/server.py:747-773`):
  ```python
  env = dict(os.environ)            # full host env inherited
  env.update(session.env_overrides)
  ... cwd=session.cwd, stdin/stdout=PIPE
  ```
- The daemon's command loop reads JSONL from `sys.stdin`
  (`swarmee.py:4537`) and emits events on stdout; the broker bridges those to
  the socket protocol.

**What's already in good shape** (credit list — these survive the split):
the JSONL protocol is transport-agnostic; token auth is in the discovery
payload, not the transport; consent and interrupt already round-trip through
the broker as protocol messages (`consent_prompt`/`consent_response`,
`{"cmd": "interrupt"}`); the TUI runs no agent logic and imports no
hooks/tools/agent modules; sessions are already isolated per process.

The remaining coupling is: stdio as the daemon transport, environment
inheritance as the secrets channel, and the filesystem as a shared database.

---

## Findings — What Breaks Across a Container Boundary

### F1 (Blocker) — stdio is the daemon's command/event transport

The broker↔daemon leg is stdin/stdout pipes (`server.py:757-764`,
`swarmee.py:4537`). A containerized daemon has no parent-process pipes to the
local broker. This is the foundational change: the daemon must speak the same
socket protocol the broker already speaks to the TUI.

**Approach:** Make the session daemon a network service. Either (a) move the
broker *into* the container so the TUI's existing socket attach works
unchanged and stdio remains container-internal, or (b) teach the daemon to
serve the JSONL protocol on a socket directly and slim the broker into a local
connection manager. Option (a) is the smaller diff and is recommended:
the TUI-facing protocol already exists; only discovery changes.

### F2 (Blocker, security) — secrets travel by environment inheritance and home-dir files

Two implicit channels exist today:

- `env = dict(os.environ)` (`server.py:749`) forwards *everything* — AWS
  credentials, `OPENAI_API_KEY`, plus unrelated host variables — to every
  session daemon.
- The daemon reads the auth store directly from the user's XDG data home
  (`auth/store.py:40-41` → `~/.local/share/swarmee/auth.json`), including
  provider tokens written by TUI-side auth flows (`auth/github_copilot.py`).

In a container, inheritance silently stops working (daemon gets the
*container's* env) and mounting the host auth store into the runtime container
would hand the LLM/tool layer durable access to all user credentials.

**Approach:** Make credentials an explicit, narrow contract:
1. An allowlisted env contract documented per provider (`AWS_*`,
   `OPENAI_API_KEY`, …) injected at container start — never `dict(os.environ)`.
2. For TUI-mediated auth (Copilot device flow), add a
   `set_credentials` protocol command so tokens are delivered in-memory over
   the authenticated socket; remove the daemon's direct `auth.json` read.
3. Refuse to start a containerized session if a secret arrives via an
   unexpected channel (defense against config drift).

### F3 (Blocker) — the filesystem is a shared database

Both sides read/write the same trees, with the daemon as writer and the TUI as
reader (or both as readers):

| Path | Writer → Reader | Code |
|------|-----------------|------|
| `.swarmee/settings.json` | both read; TUI writes | `settings.py:31`, daemon load at startup |
| state dir `sessions/` | daemon writes, TUI reads | `session/store.py:12`, `utils/state_paths.py:57-74` |
| state dir `artifacts/` (+ `index.jsonl`) | daemon writes, TUI reads | `artifacts.py:45-82` |
| `diagnostics/sessions/*.jsonl`, `broker.log` | daemon/broker write | `server.py:151-158`, `hooks/jsonl_logger.py:30-31` |
| `runtime.json` discovery file | broker writes, TUI reads | `client.py:62-64` |
| `sops/`, `.prompt/`, project files | daemon + tools read/write | prompt assets, tools |

With the daemon in a container, every one of these is a different filesystem
unless mounted; even when mounted, host and container *paths differ*, breaking
anything that stores absolute paths.

**Approach — declare two explicit zones:**
- **Workspace zone** (project files, `sops/`, `.prompt/`, `.swarmee/`):
  bind-mounted read-write into the container at a fixed path (e.g.
  `/workspace`). This is unavoidable — file/git/shell tools must operate on
  the real project (see F5).
- **State zone** (sessions, artifacts, diagnostics): stop sharing via disk.
  The daemon emits session/artifact/diagnostic records as protocol events; the
  TUI persists them locally. This keeps user-visible state on the user's
  machine (survives container teardown) and removes the path-translation
  problem. Interim step: mount the state dir too, but treat that as scaffolding.
- All persisted/protocol paths become **workspace-relative**, never absolute.

### F4 (High) — implicit `cwd` resolution

There are **56** `Path.cwd()` call sites in `src/`. Today the broker makes
this "work" by spawning the daemon with `cwd=session.cwd` (`server.py:765`).
A containerized daemon's cwd is whatever the image entrypoint sets; scope
resolution (`utils/state_paths.py:30-54`), settings loading, prompt/SOP
discovery, and tool target resolution all silently bind to the wrong root.

**Approach:** Make workspace root an explicit value: carried in the `attach`
command, stored on the session, and threaded through a single
`WorkspaceContext` object that settings/state-path/tool code consult instead
of `Path.cwd()`. Add a lint/CI grep forbidding new `Path.cwd()` in daemon-side
modules. This refactor is worth doing *before* containers — it also fixes
latent bugs where the TUI is launched from a subdirectory.

### F5 (High) — tool execution locality must become a first-class concept

All tools run in-process in the daemon (`tools.py:134-152`): `editor`, `git`,
`file_ops`, `patch_apply`, `shell`, `run_checks` mutate the workspace and
spawn subprocesses. In a container these operate on the container image +
mounted workspace, which is mostly *desired* (isolation!) but has
consequences to design for now:

- The container image must carry the toolchain (git, build tools, language
  runtimes) — define and version a runtime image.
- Some capabilities are inherently host-side: clipboard, opening a browser for
  auth flows, anything touching host services on `localhost`. Tag each tool
  with an execution locality (`runtime` | `client`) in tool metadata, and add a
  protocol affordance for client-executed tools (request/response mirroring
  the existing consent round-trip). Today the only confirmed client-side needs
  are auth flows and clipboard; the tagging prevents accidental future leaks.

### F6 (Medium) — boundary leak: the TUI re-implements daemon logic locally

The cleanest-boundary audit found the TUI generally free of daemon logic, with
one concrete exception: the usage-cost fallback in
`tui/event_router.py:94-122` loads settings and runs `resolve_pricing()`
locally when an event lacks `cost_usd`. That is LLM-layer accounting living in
the client — and it will *diverge* from the daemon's own pricing when versions
skew across the container boundary.

**Approach:** The daemon always computes and sends `cost_usd` (also wanted by
doc 12 F1); delete the TUI fallback. Adopt the general rule: **derived values
are computed runtime-side and shipped in events; the TUI only formats.**

### F7 (Medium) — discovery, versioning, and lifecycle for a remote runtime

- Discovery is a local file (`runtime.json`); a containerized runtime needs an
  endpoint config (env/CLI/setting) and per-session tokens, with the file
  becoming one possible *source* of an endpoint rather than the mechanism.
- The protocol carries `schema_version` (currently "2") only in the discovery
  payload. Once the two sides ship separately (local app vs container image),
  add a version handshake to `hello` with an explicit compatibility window and
  a clear TUI error for mismatches.
- Process lifetime: subprocess fallback and `SIGTERM`-based cleanup don't
  apply to remote runtimes. Lifecycle (start/health/stop of the container)
  becomes a deployment concern; the protocol needs a heartbeat/health command
  so the TUI can distinguish "runtime busy" from "runtime gone" instead of
  inferring from pipe EOF.

### F8 (Low) — TTY assumptions in the daemon

Interactive fallbacks remain in daemon code: consent via `input()` when
`SWARMEE_TUI_EVENTS` is unset (`swarmee.py:1212-1239`), and the Esc-key
interrupt watcher (`interrupts.py`) that monitors a TTY. In a container these
silently hang or no-op. **Approach:** in service mode, hard-disable all TTY
paths — consent/interrupt are protocol-only, and a missing TUI-events flag is
a startup error, not a fallback to interactive mode.

---

## Proposed Plan

Sequenced so every phase is shippable on the current (non-container)
architecture and reduces coupling immediately:

| Phase | Work | Findings |
|-------|------|----------|
| 1 | `WorkspaceContext`: explicit workspace root threaded through daemon code; ban new `Path.cwd()`; workspace-relative paths in protocol/persisted data | F4, F3 |
| 2 | Protocol hardening: daemon always ships derived values (cost, estimates); delete TUI pricing fallback; version handshake + heartbeat in `hello` | F6, F7 |
| 3 | State-over-protocol: session/artifact/diagnostic records emitted as events, persisted by the TUI; disk sharing limited to the workspace mount | F3 |
| 4 | Credential contract: allowlisted env injection, `set_credentials` command, remove daemon auth-store read; service-mode startup refuses TTY fallbacks | F2, F8 |
| 5 | Broker-in-container packaging: runtime image (toolchain + package), endpoint-based discovery, socket-only attach; tool locality tags + client-tool round-trip | F1, F5, F7 |

Phases 1–4 are pure decoupling with standalone value (they also fix latent
bugs like subdirectory launches and version-skew pricing). Phase 5 is the
actual migration and stays small because everything it needs exists by then.

## Test Strategy

- **Boundary contract tests:** run TUI and daemon in the same CI job but with
  *different* working directories, separate temp HOMEs, and a scrubbed daemon
  environment (no inherited host env). Everything that passes here survives a
  container split — this harness is the cheap proxy for containerization and
  should gate from Phase 1 onward.
- Protocol golden tests: command/event schemas with version-skew cases
  (old TUI ↔ new daemon and vice versa).
- Security test: assert the daemon process env contains only allowlisted keys;
  assert no daemon code path opens the auth store in service mode.
- Integration (Phase 5): docker-compose harness running the runtime image with
  a mounted fixture workspace; the existing E2E suite pointed at it.
