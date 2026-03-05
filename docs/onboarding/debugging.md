# Debugging and Diagnostics

This guide covers where to look when things go wrong — from a tool that didn't fire to a session that won't restore.

---

## JSONL Event Logs

Every invocation is recorded in `.swarmee/logs/` as a JSONL file:

```
.swarmee/
  logs/
    session_<id>.jsonl
    session_<id>_<invocation_id>.jsonl
```

Each line is a JSON event. Key event types:

| Event type | What it means |
|-----------|--------------|
| `message` | A conversation turn (user or assistant) |
| `tool_use` | The agent called a tool — contains tool name and input |
| `tool_result` | The tool's return value |
| `reasoning` | Agent's internal reasoning (if reasoning mode is on) |
| `error` | An exception or failure |
| `session_start` | Session initialization |
| `context_summary` | Context was trimmed via summarization |

**Quick tailing:**
```bash
tail -f .swarmee/logs/session_*.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        ev = json.loads(line)
        print(ev.get('type'), '-', str(ev)[:120])
    except: pass
"
```

Or use the built-in command:
```bash
swarmee diagnostics tail
```

---

## Built-in Diagnostics Commands

### CLI

```bash
# Stream recent log events to the terminal
swarmee diagnostics tail

# Check configuration, credentials, and connectivity
swarmee diagnostics doctor

# Collect a support bundle (logs + settings + diagnostics)
swarmee diagnostics bundle
# Produces: swarmee_bundle_<timestamp>.zip
```

### TUI

Navigate to the **Settings** tab > **Diagnostics** section. Buttons for tail, doctor, and bundle are available there.

### REPL

```
> :diagnostics tail
> :diagnostics doctor
> :diagnostics bundle
```

---

## Session Replay

To step through a specific invocation event-by-event:

```
> :replay <invocation_id>
```

Find the invocation ID in the session log filename or in `:session list`.

---

## Common Errors and Fixes

### Tool not found / not available

**Symptom:** The agent says a tool doesn't exist, or the TUI Tools tab doesn't show your tool.

**Causes and fixes:**

1. **File not in `tools/`** — confirm the file exists at `tools/my_tool.py`
2. **Missing `@tool` decorator** — the `@tool` decorator from `strands` is required
3. **Import error at module load** — a syntax error or missing dependency prevents the module from loading. Check:
   ```bash
   python3 -c "import tools.my_tool"
   ```
   Fix any errors reported.
4. **Missing `set_permissions()` call** — tool won't be registered. Add it.
5. **Tool allowlist in profile** — if the active agent profile has `tool_names` set, only listed tools are available. Add your tool to the list or clear it.

---

### Permission denied / tool blocked

**Symptom:** The agent says it cannot use a tool because of permission restrictions.

**Causes and fixes:**

1. **Plan mode** — `write` and `execute` tools are blocked during planning. The agent cannot use them until you approve the plan. This is by design.
2. **Tool policy blocklist** — check `.swarmee/settings.json` for a `tool_policy` section. Your tool may be blocked explicitly.
3. **Wrong permission declared** — if a tool is declared `read` but the model is trying to use it in a context that expects `execute`, verify your `set_permissions()` call.

---

### Context too large / token limit exceeded

**Symptom:** Error about context window or token limit; model stops responding mid-session.

**Fixes:**

1. **Trigger compaction manually:**
   - TUI: press `/compact` or use the compact button in the Run tab
   - REPL: `> :compact`

2. **Tune the context manager settings** in `.swarmee/settings.json`:
   ```json
   {
     "context": {
       "manager": "summarize",
       "max_prompt_tokens": 20000,
       "preserve_recent_messages": 10
     }
   }
   ```
   Reduce `max_prompt_tokens` to trigger summarization earlier.

3. **Start a new session** if the current one is too degraded:
   ```
   > :session new
   ```

---

### Daemon connection lost

**Symptom:** TUI shows "daemon disconnected" or REPL stops responding to queries.

**Fixes:**

1. **Check if the daemon is running:**
   ```bash
   swarmee daemon status
   ```

2. **Restart the daemon:**
   ```bash
   swarmee daemon restart
   # or in the TUI: Settings tab > Daemon section > Restart
   ```

3. **Check the broker socket:**
   ```bash
   cat .swarmee/runtime.json
   ```
   If the file is stale (process no longer running), delete it and restart.

4. **Port conflict:** If `swarmee serve` fails to bind, check if another process is using the port:
   ```bash
   lsof -i :<port>
   ```

---

### Plan approval stuck / approve button unresponsive

**Symptom:** The agent generated a plan but the Approve/Cancel buttons don't respond, or `:approve` does nothing.

**Fixes:**

1. **REPL:** Type `:approve` explicitly — make sure there's an active plan:
   ```
   > :plan show    # see the current plan
   > :approve
   ```

2. **TUI:** Click the Approve button in the **Run** tab, or press the `A` key while focused on the plan panel.

3. **No plan active:** The agent may have already executed (no plan pending). Check the transcript for results.

4. **Daemon not ready:** Check daemon status (see above). If the daemon is down, the plan cannot execute.

---

### SOP not taking effect

**Symptom:** You enabled an SOP but the agent isn't following the procedure.

**Fixes:**

1. **Confirm it's active:**
   ```
   > :sop list
   ```
   Active SOPs are marked with a checkmark.

2. **SOP name mismatch:** The name in the frontmatter (`name: my-workflow`) must match exactly what you passed to `:sop enable`. Check the file's `---` block.

3. **File not discovered:** SOPs must be in `./sops/*.sop.md` or `packs/<name>/sops/*.sop.md`. Confirm the path.

4. **Re-activate after edit:** If you edited the SOP file, deactivate and re-activate it to force a reload:
   ```
   > :sop disable my-workflow
   > :sop enable my-workflow
   ```

5. **SOP content too long:** Very long SOPs can crowd the system prompt. Keep SOPs under ~2000 words.

---

### Session won't restore

**Symptom:** `:session load <name>` reports an error or loads an empty session.

**Fixes:**

1. **List available sessions:**
   ```
   > :session list
   ```

2. **Check the log file directly:**
   ```bash
   ls .swarmee/logs/
   ```
   Saved sessions have human-readable names; auto-saved sessions use UUIDs.

3. **Session state may be from a different directory:** Sessions are stored relative to the working directory. Run `swarmee` from the same directory where you saved the session.

---

## Collecting a Support Bundle

If you need to share diagnostic information:

```bash
swarmee diagnostics bundle
```

This produces a ZIP file (`swarmee_bundle_<timestamp>.zip`) containing:
- Recent log files (last N sessions)
- Sanitized `settings.json` (secrets removed)
- Output of `diagnostics doctor`
- Environment information (Python version, package versions)

**Note:** Review the bundle before sharing. It may contain conversation content from recent sessions.

---

## Enabling Verbose Logging

For deeper debugging, increase log verbosity in `.swarmee/settings.json`:

```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

Debug logs include detailed transport messages between the TUI and daemon, which can help diagnose connection issues.

---

## Next Steps

- [getting-started.md](getting-started.md) — revisit setup if you're seeing installation-related errors
- [custom-tools.md](custom-tools.md) — check tool anatomy if a custom tool isn't loading
- [../agent_context_lifecycle.md](../agent_context_lifecycle.md) — deep dive on context assembly and trimming
