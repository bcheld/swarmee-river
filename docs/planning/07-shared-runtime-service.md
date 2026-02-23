# Shared Agent Runtime Service (MVP)

> Living document for promoting `--tui-daemon` into a shared, multi-client runtime.

## Context

Today, the TUI owns a single daemon subprocess (`python -m swarmee_river.swarmee --tui-daemon`) over stdin/stdout JSONL. That model blocks multi-surface attach (TUI + CLI + notebooks) because process ownership, consent routing, and lifecycle are tied to one parent process.

This document defines an MVP runtime broker: `swarmee serve`, a localhost TCP service that multiplexes clients to per-session daemon subprocesses while preserving existing daemon JSONL commands/events.

## Goals

1. Support multiple simultaneous clients attached to the same `session_id`.
2. Keep one daemon subprocess per `session_id` (same behavior/state model as today).
3. Preserve JSONL framing end-to-end (one JSON object per line).
4. Route consent safely: only the active query controller can answer consent prompts.
5. Work on macOS/Linux/Windows using localhost TCP only.
6. Add lightweight local auth via shared secret token in `.swarmee/runtime.json`.

## Non-Goals (MVP)

1. No remote/network exposure beyond `127.0.0.1`.
2. No Unix domain sockets.
3. No multi-repo/global runtime registry.
4. No redesign of daemon event schema; broker forwards daemon events mostly unchanged.
5. No HA/failover; single broker process is the control plane.

## Process Model

## Components

1. **Broker process** (`swarmee serve`)
2. **Session runtime** (`session_id -> one swarmee --tui-daemon subprocess`)
3. **Clients** (TUI first, then CLI/notebook adapters)

## Session routing

1. Broker maintains `sessions: dict[session_id, SessionRuntime]`.
2. `SessionRuntime` owns:
   - daemon subprocess (`stdin`, `stdout`)
   - attached authenticated clients
   - active controller metadata (`controller_client_id`, `query_active`)
3. All daemon stdout JSONL events are broadcast to every attached client for that session.

## Why this MVP shape

1. Reuses current daemon command loop (`query`, `consent_response`, `set_context_sources`, `set_sop`, `set_tier`, `restore_session`, `interrupt`, `shutdown`).
2. Limits change surface to a broker + client transport layer.
3. Keeps session persistence in existing `SessionStore` paths under `.swarmee/sessions`.

## Transport and Protocol

## Transport

1. TCP listener bound to `127.0.0.1:<port>` (default port configurable).
2. UTF-8 JSONL framing only.
3. Each inbound line is a complete JSON command object.
4. Each outbound line is a complete JSON event/response object.

## Envelope conventions

Client -> broker:

```json
{"cmd":"query","session_id":"abc123","text":"hello","request_id":"r1"}
```

Broker -> client:

```json
{"event":"turn_complete","session_id":"abc123","exit_status":"ok","request_id":"r1"}
```

Notes:
1. `request_id` is optional but echoed when possible for client correlation.
2. `session_id` is required for session-scoped commands unless already attached.

## Handshake and auth

1. `hello`
   - Purpose: protocol negotiation.
   - Payload: `{cmd:"hello", protocol_version:1, client:{name,version}}`
   - Response: `{event:"hello_ack", protocol_version:1, auth_required:true}`
2. `auth`
   - Purpose: authenticate connection with shared secret token.
   - Payload: `{cmd:"auth", token:"<secret>"}`
   - Response success: `{event:"auth_ok"}`
   - Response failure: `{event:"auth_error", message:"invalid token"}` then close connection.

## Session attach

`attach_session`

Payload:

```json
{"cmd":"attach_session","session_id":"abc123","spawn_if_missing":true}
```

Behavior:
1. If runtime exists: attach client.
2. If missing and `spawn_if_missing=true`: broker starts new daemon subprocess for this `session_id` and attaches.
3. If missing and `spawn_if_missing=false`: return `{event:"attach_missing"}`.
4. Response on success: `{event:"attached","session_id":"abc123","clients":N,"query_active":false}`.

## Command set (MVP)

All commands below are broker-level and then forwarded/handled as noted:

1. `query`
   - Forward to daemon as existing `{"cmd":"query",...}`.
   - Sets controller to calling client for this run.
2. `consent_response`
   - Allowed only from controller while query is active.
   - Forwarded to daemon unchanged.
3. `set_context_sources`
   - Forward unchanged to daemon.
4. `set_sop`
   - Forward unchanged to daemon.
5. `set_tier`
   - Forward unchanged to daemon.
6. `interrupt`
   - Allowed from any attached client in MVP.
   - Forward unchanged; clears pending controller wait states broker-side.
7. `restore_session`
   - Forward unchanged to daemon.
8. `shutdown`
   - If session-scoped: shuts down that session daemon.
   - If broker-scoped admin variant (`target:"broker"`): graceful broker shutdown.

## Event broadcast semantics

1. Daemon events are rebroadcast to all clients attached to the same `session_id`.
2. Existing daemon event names stay intact (`ready`, `text_delta`, `tool_start`, `tool_result`, `consent_prompt`, `turn_complete`, `error`, etc.).
3. Broker may add metadata fields:
   - `session_id`
   - `controller_client_id` (only on broker-side status events)
4. Broker-only events:
   - `hello_ack`, `auth_ok`, `auth_error`
   - `attached`, `detached`, `attach_missing`
   - `broker_warning`, `broker_error`

## Controller and consent rules

1. On accepted `query`, broker sets `controller_client_id` to caller.
2. While query is active:
   - only controller may send `consent_response`
   - non-controller attempts get sender-only error:
     - `{event:"broker_error","code":"not_controller","message":"Only active controller may answer consent"}`
3. `consent_prompt` is still broadcast to all attached clients for visibility.
4. Controller clears on `turn_complete` or forced interrupt.
5. If controller disconnects during active run, broker sends `interrupt` to daemon (safe default) and emits `broker_warning` to remaining clients.

## Lifecycle and discovery

## Discovery file

Broker writes `.swarmee/runtime.json` (via `state_dir()` conventions) on startup.

Proposed shape:

```json
{
  "version": 1,
  "host": "127.0.0.1",
  "port": 7342,
  "pid": 12345,
  "token": "<random-32-byte-hex>",
  "created_at": "2026-02-23T12:34:56Z",
  "cwd": "/path/to/repo",
  "protocol_version": 1
}
```

Security notes:
1. Token is random per broker start.
2. Best-effort file permissions: owner read/write only.
3. TCP bind is localhost-only; auth token is still required.

## PID and stale runtime handling (best effort)

1. On broker start:
   - if `runtime.json` exists, try connect+hello+auth using stored token.
   - if healthy, new process exits with “already running”.
   - if unhealthy, treat as stale and replace file.
2. On graceful shutdown:
   - stop listener, signal session daemons, remove `runtime.json`.
3. On crash:
   - stale file may remain; next start performs cleanup check above.

## Session cleanup strategy

1. If a session has no attached clients and no active query, start idle timer (default 10 minutes).
2. On idle timeout, send daemon `shutdown` and remove session runtime from memory.
3. On broker shutdown, terminate all session daemons (graceful then forced with platform-safe fallbacks).

## Windows compatibility constraints

1. TCP-only IPC for MVP (no Unix socket assumptions).
2. Subprocess termination path must avoid POSIX-only calls unless guarded.
3. Use portable `subprocess.Popen` flags; gate `os.killpg`/POSIX signals behind `os.name == "posix"`.

## Test Plan

## Unit tests

1. JSONL framing parser handles partial/malformed lines safely.
2. Auth flow: valid/invalid token behavior.
3. Controller enforcement for `consent_response`.
4. Broadcast fanout to N attached clients.
5. Runtime discovery read/write and stale-file replacement logic.

## Integration tests

1. Start broker, attach two clients to same `session_id`, run query, verify both receive streamed events.
2. Verify only controller can answer consent.
3. Verify `set_context_sources`, `set_sop`, `set_tier`, `restore_session`, `interrupt` pass through.
4. Disconnect controller mid-consent and verify broker interrupts run.
5. Multi-session isolation: separate events for separate `session_id`s.
6. Windows CI path for spawn/terminate and TCP attach.

## Failure-mode tests

1. Daemon subprocess crashes: broker emits `broker_warning`, marks session unhealthy.
2. Broker restart with stale `runtime.json`.
3. Invalid command payloads return structured errors, no broker crash.

## Incremental Rollout

## Phase 1: Broker introduction

1. Add `swarmee serve` command.
2. Support handshake/auth + attach + command pass-through.
3. Keep TUI default path unchanged initially (still direct daemon spawn).

## Phase 2: TUI attach-first, spawn-fallback

1. TUI startup flow:
   - read `.swarmee/runtime.json`
   - attempt connect + `hello` + `auth` + `attach_session(spawn_if_missing=false)`
2. If attach fails:
   - start broker (`swarmee serve`) and retry attach with `spawn_if_missing=true`.
3. If broker path fails entirely:
   - fallback to current direct `spawn_swarmee_daemon()` behavior (temporary compatibility path).

## Phase 3: Additional clients

1. Add CLI/notebook lightweight clients reusing same JSONL protocol.
2. Make broker-backed runtime default; keep direct daemon path behind compatibility flag until stable.

## Open Questions

1. Should non-controller `interrupt` remain allowed, or be controller-only?
2. Should controller handoff be explicit instead of interrupt-on-disconnect?
3. Do we need broker persistence of client display cursor/state for richer late-join replay?
4. When should direct daemon fallback be removed after rollout?
