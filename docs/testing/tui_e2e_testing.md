# TUI end-to-end testing with MockTransport

This document explains how to write **end-to-end tests for the Swarmee River TUI** using the MockTransport harness. Tests drive the real `SwarmeeTUI` Textual app via Textual's Pilot API, with all daemon I/O replaced by an in-process mock. No subprocess is spawned, no network is used, and test execution is fully deterministic.

**Audience:** coding agents and contributors adding TUI features, bug fixes, or regression tests.

**Prerequisites:** `pytest`, `pytest-asyncio`, `textual` (all included in `hatch test` environment).

---

## 1) Architecture

```
┌─────────────────────────┐      emit() / emit_event()      ┌───────────────────────────────────┐
│  Test code              │ ─────────────────────────────►   │  MockTransport._q  (queue.Queue)  │
│  (test_tui_e2e.py)      │                                  └──────────────┬────────────────────┘
│                         │                                                 │ read_line()
│  assert app.state.*     │                                                 ▼
│  transport.sent_commands│◄─── send_command() ───────────   ┌───────────────────────────────────┐
└─────────────────────────┘                                  │  DaemonMixin._stream_daemon_output│
                                                             │  (background thread)              │
                                                             └──────────────┬────────────────────┘
                                                                            │ _handle_output_line()
                                                                            ▼
                                                             ┌───────────────────────────────────┐
                                                             │  event_router.handle_daemon_event()│
                                                             │  → domain handlers update          │
                                                             │    app.state (DaemonState,         │
                                                             │    SessionState, PlanState, ...)   │
                                                             └───────────────────────────────────┘
```

**Two directions of I/O:**

| Direction | Real transport | Test substitute |
|-----------|---------------|-----------------|
| Daemon &rarr; TUI (events) | `_SubprocessTransport.read_line()` reads from subprocess stdout | `MockTransport.emit()` / `emit_event()` pushes lines into an in-process queue; the streaming thread reads them via `read_line()` |
| TUI &rarr; Daemon (commands) | `send_daemon_command()` writes JSONL to subprocess stdin | `MockTransport.send_command()` appends to an in-memory list; tests read `transport.sent_commands` |

**How it connects:**

1. `DaemonMixin` (in `src/swarmee_river/tui/mixins/daemon.py`) has a class-level hook `_test_transport_factory`. When set to a callable, `_spawn_daemon()` calls it instead of connecting to a runtime broker or spawning a subprocess.
2. The test fixture creates a subclass of `SwarmeeTUI` with `_test_transport_factory = staticmethod(lambda: transport)`, where `transport` is a `MockTransport` instance.
3. `app.run_test(size=...)` starts the Textual app in headless mode and returns a `Pilot` for programmatic input.

---

## 2) Key files

| File | Purpose |
|------|---------|
| `tests/tui_harness.py` | `MockTransport` class and `tui_app_factory` pytest fixture |
| `tests/test_tui_e2e.py` | Existing scenario tests (7 scenarios) &mdash; use as templates |
| `src/swarmee_river/tui/transport.py` | `_DaemonTransport` base class defining the transport interface |
| `src/swarmee_river/tui/mixins/daemon.py` | `_test_transport_factory` injection hook, `_stream_daemon_output()` thread |
| `src/swarmee_river/tui/event_router.py` | `handle_daemon_event()` dispatcher and all domain handlers |
| `src/swarmee_river/tui/state.py` | `AppState` dataclass tree (`DaemonState`, `SessionState`, `PlanState`, ...) |

---

## 3) MockTransport API

Defined in `tests/tui_harness.py`. Implements the `_DaemonTransport` interface.

### Transport interface methods (called by the app)

| Method | Signature | Behaviour |
|--------|-----------|-----------|
| `pid` | `@property -> int` | Returns `0` (fake PID) |
| `poll()` | `-> int \| None` | `None` while alive; `0` after `close()` |
| `wait(timeout)` | `-> int` | Blocks until `close()` is called |
| `read_line()` | `-> str` | Blocks on internal queue; returns `""` (EOF) after `close()` |
| `send_command(cmd_dict)` | `-> bool` | Appends `cmd_dict` to `_commands`; always returns `True` |
| `close()` | `-> None` | Signals EOF; unblocks `read_line()` |

### Test helper methods (called by your test)

| Method | Signature | Purpose |
|--------|-----------|---------|
| `emit(line)` | `str -> None` | Feed a raw line (auto-appends `\n` if missing) into the streaming thread |
| `emit_event(event)` | `dict -> None` | Serialize a Python dict as JSON and call `emit()` |
| `emit_ready(*, session_id="test-session")` | `-> None` | Shorthand for `emit_event({"event": "ready", "session_id": session_id})` |
| `sent_commands` | `@property -> list[dict]` | Copy of all commands the TUI sent to the daemon |

### Design

`MockTransport` uses a `queue.Queue` as a producer/consumer bridge. The TUI's background streaming thread (`_stream_daemon_output` in `DaemonMixin`) calls `read_line()`, which blocks on `queue.get(timeout=0.05)`. When your test calls `emit()`, it pushes a line onto the queue, unblocking the streaming thread. The thread then calls `_handle_output_line()` which parses the JSON and dispatches it through `handle_daemon_event()`.

---

## 4) The `tui_app_factory` fixture

Defined in `tests/tui_harness.py`. This is the primary entry point for all e2e tests.

```python
@pytest.fixture
def tui_app_factory(tmp_path, monkeypatch):
    # Isolate filesystem state — tests never touch real .swarmee/
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))
    monkeypatch.setenv("SWARMEE_PREFLIGHT", "disabled")
    monkeypatch.setenv("SWARMEE_PROJECT_MAP", "disabled")

    @asynccontextmanager
    async def _factory(*, size=(200, 50)):
        from swarmee_river.tui.app import get_swarmee_tui_class
        SwarmeeTUI = get_swarmee_tui_class()

        transport = MockTransport()

        class _TestApp(SwarmeeTUI):
            _test_transport_factory = staticmethod(lambda: transport)

        app = _TestApp()
        async with app.run_test(size=size) as pilot:
            yield app, pilot, transport

    return _factory
```

**What it does:**

1. **Filesystem isolation** &mdash; sets `SWARMEE_STATE_DIR` to a pytest `tmp_path` so tests never read/write real state files.
2. **Disables startup overhead** &mdash; `SWARMEE_PREFLIGHT=disabled` and `SWARMEE_PROJECT_MAP=disabled` skip file-system scanning.
3. **Transport injection** &mdash; creates a `_TestApp` subclass with `_test_transport_factory` pointing at the `MockTransport` instance.
4. **Headless Textual** &mdash; `app.run_test(size=...)` launches the app with a virtual terminal. The returned `Pilot` can simulate keypresses, mouse clicks, and timing pauses.

**The yielded tuple:**

| Value | Type | Use |
|-------|------|-----|
| `app` | `SwarmeeTUI` instance | Read `app.state.*`, call internal methods directly |
| `pilot` | `textual.testing.Pilot` | `await pilot.pause()`, `await pilot.press("enter")`, etc. |
| `transport` | `MockTransport` | `transport.emit_event(...)`, `transport.sent_commands` |

---

## 5) Writing your first test

### Skeleton

```python
"""tests/test_tui_e2e.py (or a new test file)"""

from __future__ import annotations
import pytest
from tests.tui_harness import tui_app_factory  # noqa: F401

pytestmark = pytest.mark.asyncio


async def _wait_for(condition, *, pilot, attempts=20, delay=0.05):
    """Poll condition() up to `attempts` times, pausing between checks."""
    for _ in range(attempts):
        if condition():
            return True
        await pilot.pause(delay=delay)
    return False


async def test_my_scenario(tui_app_factory):
    async with tui_app_factory() as (app, pilot, transport):
        # 1. Establish the daemon-ready state (required for most handlers)
        transport.emit_ready(session_id="my-test-session")
        reached = await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)
        assert reached, "daemon never became ready"

        # 2. Emit the event(s) you want to test
        transport.emit_event({
            "event": "text_delta",
            "text": "Hello from the agent",
        })

        # 3. Allow the streaming thread to process
        await pilot.pause(delay=0.15)

        # 4. Assert the expected state change
        assert "Hello from the agent" in "\n".join(app._transcript_fallback_lines)
```

### Required pattern

Every test follows the same four-step pattern:

1. **`emit_ready`** &mdash; The daemon handshake. Most event handlers check `app.state.daemon.ready` or assume a valid `session_id` exists. Without this step, events are often silently dropped.

2. **`emit_event`** (or `emit`) &mdash; Feed the event(s) that exercise the code path under test.

3. **`await pilot.pause()`** &mdash; The Textual event loop must run for state mutations to propagate. The streaming thread queues state changes via `_call_from_thread_safe()`, which must be processed by the async main loop. A `delay=0.15` is usually sufficient for synchronous event processing; use `_wait_for()` with higher `attempts` for async operations (timeline refresh, session restore).

4. **`assert app.state.*`** &mdash; Check the state dataclass tree rather than widget text. State assertions are deterministic and independent of rendering timing.

### The `_wait_for()` helper

Use `_wait_for()` when the operation involves asynchronous background work (e.g., timeline refresh is scheduled with a 0.35s delay):

```python
reached = await _wait_for(
    lambda: len(app.state.session.timeline_events) > 0,
    pilot=pilot,
    attempts=60,   # 60 * 0.05s = 3s max wait
    delay=0.05,
)
assert reached, "timeline_events never populated"
```

`_wait_for()` returns `True` if the condition was met, `False` if it timed out. The default (`attempts=20, delay=0.05`) gives a 1-second window, which is adequate for synchronous event processing.

---

## 6) Event reference

These are the events your tests can emit via `transport.emit_event(...)`. Each event must have an `"event"` key. The handlers are in `src/swarmee_river/tui/event_router.py`.

### Connection & session events

| Event | Required fields | State effect |
|-------|----------------|--------------|
| `ready` | `session_id` | `daemon.ready = True`, `daemon.session_id` set, transcript line written |
| `attached` | `session_id`, `clients?` | Same as `ready` + "shared session" transcript line |
| `session_available` | `session_id`, `turn_count?` | `daemon.available_restore_session_id` set, restore prompt in transcript |
| `session_restored` | `session_id`, `turn_count?` | Active session changed, `available_restore_session_id` cleared |
| `replay_turn` | `role` (`"user"` / `"assistant"`), `text`, `timestamp?` | Rendered in transcript (user bubble or assistant bubble) |
| `replay_complete` | `turn_count?` | "Session restored" transcript line |
| `turn_complete` | `exit_status` (`"ok"`, `"interrupted"`, `"error"`) | `_finalize_turn()` called, timeline refresh scheduled |
| `model_info` | `provider?`, `tier?`, `model?`, `model_id?` | Updates daemon model metadata and model selector widget |

### Streaming events

| Event | Required fields | State effect |
|-------|----------------|--------------|
| `text_delta` | `text` | Appends to `_streaming_buffer`, schedules flush to transcript |
| `message_delta` | `text` | Alias for `text_delta` |
| `text_complete` | `text?` | Flushes buffer, finalises assistant message in transcript |
| `thinking` | `text` | Records a thinking/reasoning block |

### Tool events

| Event | Required fields | State effect |
|-------|----------------|--------------|
| `tool_start` | `tool_use_id`, `tool` (name) | Creates `_tool_blocks[tid]` record, `daemon.run_tool_count += 1` |
| `tool_progress` | `tool_use_id`, `content?`, `stream?`, `elapsed_s?` | Appends to tool output buffer |
| `tool_input` | `tool_use_id`, `tool`, `input` | Sets tool input dict (may arrive after `tool_start`) |
| `tool_result` | `tool_use_id`, `status` (`"success"` / `"error"`), `duration_s?`, `tool?` | Renders result line; if `status != "success"` increments `session.error_count` |
| `consent_prompt` | `context`, `options?` | Shows approval UI for tool execution |

### Plan events

| Event | Required fields | State effect |
|-------|----------------|--------------|
| `plan` | `rendered?`, `plan_json?` | `plan.text` set, `plan.current_steps` populated from `plan_json`, `plan.received_structured_plan = True` |
| `plan_step_update` | `step_index` (int), `status` (`"in_progress"` / `"completed"`) | Updates `plan.current_step_statuses[step_index]`, recomputes `plan.step_counter` |
| `plan_complete` | &mdash; | Marks all steps completed, sets `plan.completion_announced = True` |

### Usage & context events

| Event | Required fields | State effect |
|-------|----------------|--------------|
| `context` | `prompt_tokens_est?`, `budget_tokens?` | Updates `daemon.last_prompt_tokens_est` and `daemon.last_budget_tokens` |
| `usage` | `usage?` (dict), `cost_usd?` | Updates `daemon.last_usage` and `daemon.last_cost_usd` |
| `compact_complete` | `compacted?`, `before_tokens_est?`, `after_tokens_est?` | Updates prompt tokens, shows notification |

### Error & warning events

| Event | Required fields | State effect |
|-------|----------------|--------------|
| `error` | `message` (or `text`), `category?`, `retry_after_s?`, `tool_use_id?`, `next_tier?` | `session.error_count += 1`, classified and added to issues, toast notification |
| `warning` | `text` | `session.warning_count += 1`, added to issues |

### Example payloads

```python
# Ready handshake
transport.emit_ready(session_id="my-session-id")

# Streaming text
transport.emit_event({"event": "text_delta", "text": "Here is the answer..."})
transport.emit_event({"event": "text_complete", "text": ""})

# Tool lifecycle
transport.emit_event({
    "event": "tool_start",
    "tool_use_id": "tu-abc",
    "tool": "bash",
})
transport.emit_event({
    "event": "tool_result",
    "tool_use_id": "tu-abc",
    "tool": "bash",
    "status": "success",
    "duration_s": 0.5,
})

# Error
transport.emit_event({
    "event": "error",
    "message": "Rate limit exceeded",
    "category": "transient",
    "retry_after_s": 5,
})

# Turn complete (triggers timeline refresh)
transport.emit_event({"event": "turn_complete", "exit_status": "ok"})

# LLM call inspector events
transport.emit_event({
    "event": "before_model_call",
    "model_call_id": "mc-1",
    "system_prompt_chars": 1000,
    "tool_count": 5,
    "tool_schema_chars": 2500,
    "message_breakdown": {"user": 2, "assistant": 1},
})
transport.emit_event({
    "event": "after_model_call",
    "model_call_id": "mc-1",
    "model_id": "claude-sonnet-4-6",
    "usage": {"inputTokens": 300, "outputTokens": 80},
    "duration_s": 1.2,
})

# Plan with steps
transport.emit_event({
    "event": "plan",
    "plan_json": {
        "summary": "Refactor auth module",
        "steps": [
            {"description": "Extract token validation"},
            {"description": "Add unit tests"},
            {"description": "Update imports"},
        ],
    },
})
transport.emit_event({"event": "plan_step_update", "step_index": 0, "status": "completed"})
transport.emit_event({"event": "plan_step_update", "step_index": 1, "status": "in_progress"})
```

---

## 7) Asserting state

Prefer state assertions over widget/rendering assertions. The `AppState` tree is the single source of truth for all UI; it updates synchronously within the event handler call.

### `app.state.daemon` (`DaemonState`)

| Field | Type | Common assertions |
|-------|------|-------------------|
| `ready` | `bool` | `assert app.state.daemon.ready` after `emit_ready()` |
| `session_id` | `str \| None` | `assert app.state.daemon.session_id == "expected-id"` |
| `query_active` | `bool` | True while a run is in progress |
| `model_id` | `str \| None` | Set by `model_info` event |
| `current_model` | `str \| None` | Display-friendly model name |
| `run_tool_count` | `int` | Incremented by each `tool_start` event |
| `last_usage` | `dict \| None` | Set by `usage` event |
| `last_prompt_tokens_est` | `int \| None` | Set by `context` event |
| `proc` | `MockTransport` | The transport instance (`app.state.daemon.proc is transport`) |

### `app.state.session` (`SessionState`)

| Field | Type | Common assertions |
|-------|------|-------------------|
| `timeline_events` | `list[dict]` | Timeline entries after refresh; check `e.get("event")` for type |
| `issues` | `list[dict]` | Error/warning issues; check `i.get("text")` for message content |
| `error_count` | `int` | Incremented by `error` events and failed tool results |
| `warning_count` | `int` | Incremented by `warning` events |
| `view_mode` | `str` | `"timeline"` or `"issues"` |

### `app.state.plan` (`PlanState`)

| Field | Type | Common assertions |
|-------|------|-------------------|
| `text` | `str` | Rendered plan text |
| `current_steps` | `list[str]` | Step descriptions extracted from `plan_json` |
| `current_step_statuses` | `list[str]` | `"pending"`, `"in_progress"`, or `"completed"` per step |
| `step_counter` | `int` | Count of completed steps |
| `received_structured_plan` | `bool` | True after a `plan` event with `plan_json` |
| `completion_announced` | `bool` | True once all steps are completed |

### Transcript and command assertions

```python
# Transcript output (list of plain-text lines captured from all writes)
transcript = "\n".join(app._transcript_fallback_lines)
assert "expected text" in transcript

# Commands the TUI sent back to the daemon
assert any(c.get("cmd") == "interrupt" for c in transport.sent_commands)
assert transport.sent_commands[-1] == {"cmd": "query", "text": "hello", "auto_approve": False}
```

---

## 8) Testing timeline refresh (disk-based)

The Session Timeline is populated from JSONL log files on disk, not from real-time events. The refresh is triggered asynchronously after `turn_complete`. Testing this requires **pre-seeding a log file**.

### How it works in production

1. The daemon's `JSONLLoggerHooks` (`src/swarmee_river/hooks/jsonl_logger.py`) writes events to `<SWARMEE_STATE_DIR>/logs/<timestamp>_<session_id>.jsonl`.
2. On `turn_complete`, the TUI schedules `_refresh_session_timeline_async()` with a 0.35-second delay.
3. That function calls `build_session_graph_index(session_id)` which discovers log files matching `*_{session_id}.jsonl` via glob, parses them, and returns an index.
4. The index events are written to `app.state.session.timeline_events`.

### Test walkthrough (from Scenario 7)

```python
async def test_timeline_updates_after_turn_complete(tui_app_factory, tmp_path):
    import json, time

    session_id = "timeline-roundtrip-session"

    # Step 1: Create the log file in the fixture's tmp_path
    state_dir = tmp_path / ".swarmee"         # matches SWARMEE_STATE_DIR
    logs_dir = state_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{ts}_{session_id}.jsonl"

    events = [
        {"event": "before_model_call", "model_call_id": "mc-1",
         "system_prompt_chars": 800, "tool_count": 3, "tool_schema_chars": 1200,
         "message_breakdown": {"user": 1}, "ts": "2026-02-28T10:00:00"},
        {"event": "after_model_call", "model_call_id": "mc-1",
         "model_id": "claude-sonnet-4-6",
         "usage": {"inputTokens": 200, "outputTokens": 50},
         "duration_s": 0.9, "ts": "2026-02-28T10:00:01"},
        {"event": "after_tool_call", "tool": "bash", "toolUseId": "tu-1",
         "duration_s": 0.3, "result": '{"ok": true}', "ts": "2026-02-28T10:00:02"},
        {"event": "after_invocation", "invocation_id": "inv-1",
         "duration_s": 2.1, "ts": "2026-02-28T10:00:03"},
    ]
    log_file.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )

    # Step 2: Start the app and trigger timeline refresh
    async with tui_app_factory() as (app, pilot, transport):
        transport.emit_ready(session_id=session_id)
        await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)

        transport.emit_event({"event": "turn_complete", "exit_status": "ok"})

        # Step 3: Wait for the async refresh (0.35s delay + build time)
        reached = await _wait_for(
            lambda: len(app.state.session.timeline_events) > 0,
            pilot=pilot,
            attempts=60,   # 3 seconds total
            delay=0.05,
        )

        # Step 4: Assert
        assert reached, "timeline_events never populated"
        event_types = [e.get("event") for e in app.state.session.timeline_events]
        assert "after_model_call" in event_types
        assert "after_tool_call" in event_types
```

### Key points

- **Filename format:** `{YYYYMMDD_HHMMSS}_{session_id}.jsonl`. The glob pattern is `*_{session_id}.jsonl`, so the timestamp prefix can be anything.
- **`SWARMEE_STATE_DIR`** is set by the fixture to `tmp_path / ".swarmee"`. Log files go in `SWARMEE_STATE_DIR/logs/`.
- **The `session_id` in `emit_ready()` must match the log filename.** The refresh function discovers logs by globbing for the active session_id.
- **Timing:** The 0.35s scheduled delay means `pilot.pause(delay=0.15)` alone is not enough. Use `_wait_for()` with `attempts=60, delay=0.05` (3s budget).

---

## 9) Testing commands (TUI &rarr; daemon)

When the TUI sends a command to the daemon (e.g., `interrupt`, `query`, `shutdown_session`), the `MockTransport.send_command()` method captures it in `_commands`. Read with `transport.sent_commands`.

### Sending a command directly

```python
from swarmee_river.tui.transport import send_daemon_command

send_daemon_command(app.state.daemon.proc, {"cmd": "interrupt"})
assert any(c.get("cmd") == "interrupt" for c in transport.sent_commands)
```

### Verifying commands triggered by app methods

```python
# _start_fresh_session() internally sends shutdown_session
app._start_fresh_session()
await pilot.pause(delay=0.15)
assert any(c.get("cmd") == "shutdown_session" for c in transport.sent_commands)
```

### Checking command payloads

```python
# Find the most recent query command
query_cmds = [c for c in transport.sent_commands if c.get("cmd") == "query"]
assert len(query_cmds) >= 1
assert query_cmds[-1]["text"] == "expected prompt text"
```

---

## 10) Common patterns and pitfalls

### Always emit `ready` first

Most event handlers assume the daemon is ready. The `ready` event sets `app.state.daemon.ready = True` and `app.state.daemon.session_id`. Without it:
- Session-dependent operations silently no-op.
- Timeline refresh has no session_id to glob for.
- Context/SOP sync never fires.

```python
# WRONG — events may be silently dropped
transport.emit_event({"event": "text_delta", "text": "hello"})

# CORRECT
transport.emit_ready(session_id="test-1")
await _wait_for(lambda: app.state.daemon.ready, pilot=pilot)
transport.emit_event({"event": "text_delta", "text": "hello"})
```

### Always `await pilot.pause()` after emitting

The `MockTransport` queue is consumed by a background thread (`_stream_daemon_output`). That thread calls `_call_from_thread_safe()` to schedule state mutations on the Textual event loop. Without `pilot.pause()`, the event loop never runs and the state update is not visible.

```python
transport.emit_event({"event": "error", "message": "boom"})
# WRONG — state hasn't updated yet
assert app.state.session.error_count == 1

# CORRECT
await pilot.pause(delay=0.15)
assert app.state.session.error_count == 1
```

### Use `_wait_for()` for async operations

Some operations involve scheduled timers or background `asyncio.to_thread()` calls:
- **Timeline refresh** &mdash; 0.35s scheduled delay + disk I/O
- **Session restore** &mdash; multi-step process with daemon round-trips

For these, a single `pilot.pause()` is insufficient. Use `_wait_for()`:

```python
reached = await _wait_for(
    lambda: len(app.state.session.timeline_events) > 0,
    pilot=pilot,
    attempts=60,
    delay=0.05,
)
assert reached, "descriptive error message"
```

### Prefer state assertions over widget text

Widget rendering is asynchronous and involves Textual's layout engine. State assertions are synchronous within the event handler and always deterministic:

```python
# FRAGILE — rendering timing varies
widget = app.query_one("#some-widget")
assert "text" in widget.renderable

# ROBUST
assert app.state.session.error_count == 1
assert any("text" in i.get("text", "") for i in app.state.session.issues)
```

### Mind the `SWARMEE_STATE_DIR` for log tests

The fixture sets `SWARMEE_STATE_DIR` to `tmp_path / ".swarmee"`. If you pre-seed log files, they must go under that path:

```python
logs_dir = tmp_path / ".swarmee" / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
```

### Use `MockTransport` import for type hints

If your test needs to reference the `MockTransport` type (e.g., for type annotations or isinstance checks):

```python
from tests.tui_harness import MockTransport
```

---

## 11) Running the tests

```bash
# Run all e2e tests
pytest tests/test_tui_e2e.py -v

# Run a specific scenario
pytest tests/test_tui_e2e.py -k "test_daemon_becomes_ready" -v

# Run via hatch (uses the project's test environment)
hatch test -- tests/test_tui_e2e.py -v
```

### pytest-asyncio configuration

The test file uses `pytestmark = pytest.mark.asyncio` at module level, which marks all `async def test_*` functions as asyncio tests. The project's `pyproject.toml` configures pytest-asyncio. If you create a new test file, include this at the top:

```python
from __future__ import annotations
import pytest
from tests.tui_harness import tui_app_factory  # noqa: F401

pytestmark = pytest.mark.asyncio
```

The `tui_app_factory` import is necessary to make the fixture available to pytest. The `# noqa: F401` suppresses the "imported but unused" lint warning (the fixture is used implicitly by pytest's injection mechanism).

---

## Quick-reference: new test checklist

1. Import `tui_app_factory` from `tests.tui_harness`
2. Mark the module with `pytestmark = pytest.mark.asyncio`
3. Define `async def test_*(tui_app_factory):`
4. Open `async with tui_app_factory() as (app, pilot, transport):`
5. Call `transport.emit_ready(session_id=...)` and wait for ready
6. Emit test events via `transport.emit_event({...})`
7. `await pilot.pause(delay=...)` or use `_wait_for()` for async operations
8. Assert against `app.state.*` and `transport.sent_commands`
