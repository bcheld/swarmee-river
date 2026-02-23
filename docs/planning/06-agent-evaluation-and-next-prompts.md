# Agent Performance Evaluation & Next Prompts (Batch 4)

## Batch 3 Agent Performance Assessment

| Prompt | Goal | Outcome | Grade |
|--------|------|---------|-------|
| C1 | Inline tool input details + coalescing timer | `_format_tool_input_oneliner()` implemented with correct tool-specific patterns (shell→`$`, file_read→`←`, file_write→`→`, editor→`✎`, http_request→method+URL). `render_tool_start_line_with_input()` added. Coalescing timer not implemented — agents went with the simpler approach of showing input on both start and result lines. | B+ |
| C2 | Streaming thinking indicator + ThinkingBar | `ThinkingBar(Static)` widget created with `show_thinking()` / `hide_thinking()`. `render_thinking_indicator()` function with char count, elapsed time, and preview. Timer-based animation cycling through dots. `/thinking` command added. Thinking events now accumulated in `_thinking_buffer` and rendered. | A |
| C3 | Scroll position preservation on mode toggle | Scroll proportion capture/restore implemented via `_get_scroll_proportion()` / `_set_scroll_proportion()` with 50ms timer delay for layout pass. Special case for bottom-following (proportion > 0.95). | A |
| C4 | Tool input on result line (fallback) | Merged into C1 — `render_tool_result_line()` now accepts `tool_input` kwarg, renders via `_format_tool_input_oneliner()`. Result lines show `"✓ shell (2.3s) — $ git status"`. | A |

### What went right
- **ThinkingBar is a strong addition**: The floating dock widget approach cleanly avoids the RichLog update limitation. The animation, char count, elapsed time, and preview line provide Claude Code-level thinking feedback.
- **Tool input visibility restored**: Both start and result lines now show what the tool is doing. The `_format_tool_input_oneliner()` function has good coverage of common tool types.
- **Scroll preservation works**: Proportional mapping is approximate but effective — users retain their reading position across mode toggles.

### What could improve
- **Coalescing timer was skipped**: The start→input event coalescing (100ms timer to combine tool_start + tool_input) wasn't attempted. This means the start line is written before input arrives, then re-rendered on tool_input. For fast tools this creates visual flicker. Acceptable for now but worth revisiting.
- **C1 was more complex than needed**: The prompt explored three different approaches before settling. Future prompts should lead with the simplest approach and note alternatives as fallbacks, not the reverse.

---

## Triage of User's 7 New Items

The user provided 7 items spanning near-term features and long-term architecture. Here's how they map to actionability:

| # | Item | Type | Batch 4 Action |
|---|------|------|----------------|
| 1 | MS Office file support (docx/xlsx/pptx) | **Feature — ready to implement** | Prompt D1 |
| 2 | Sidebar UX plan/structure | **Planning document** | Created as `06-sidebar-ux-plan.md` |
| 3 | Production environment prep (S3, Snowflake, Athena) | **Feature — ready to implement** | Prompt D2 |
| 4 | Shared agent runtime service | **Architecture assessment** | Assessment below |
| 5 | Agent studio inside TUI | **Architecture assessment** | Assessment below |
| 6 | Event-sourced session graph | **Architecture assessment** | Assessment below |
| 7 | S3 session logs + enterprise KB promotion | **Feature — ready to implement** | Prompt D3 |

### Architecture Assessments (Items 4, 5, 6)

These three items are interconnected and require design decisions before coding. They're assessed here rather than turned into prompts.

---

#### Item 4: Shared Agent Runtime Service

**Question**: Can we promote the TUI daemon into a shared runtime that TUI, CLI, and notebooks attach to?

**Current state**: The daemon (`--tui-daemon` in swarmee.py:1515) is tightly coupled to the TUI:
- Communication is JSONL over stdin/stdout pipes — requires parent process to spawn it
- Event dispatch (`_emit_tui_event`, swarmee.py:261) writes to stdout, not a socket
- Consent protocol uses `threading.Event` coordination tied to the daemon command loop
- Session state lives in the daemon's memory (Agent object accumulates messages in-process)
- CLI mode (`run_repl()` in cli/repl.py) builds its own agent runtime in-process — no daemon involved

**What would need to change**:
1. **IPC transport**: Replace stdin/stdout pipes with a socket (Unix domain socket or localhost TCP). Both TUI and CLI would connect as clients.
2. **Session multiplexing**: The daemon currently assumes one session. Multiple surfaces need session routing.
3. **Consent protocol**: Currently blocking — the daemon pauses until consent arrives. With multiple clients, consent must be routed to the originating surface.
4. **Lifecycle management**: Currently the daemon dies when TUI exits (child process). A shared daemon needs independent lifecycle (daemonize, PID file, graceful shutdown).
5. **State synchronization**: All surfaces need to see the same transcript, plan, and tool state. This requires a pub/sub event model rather than request/response.

**Readiness**: **Not ready for a coding prompt.** Requires a design document (similar to `06-sidebar-ux-plan.md`) that specifies the IPC protocol, session routing, and lifecycle model. Recommend targeting Batch 6-7 after the session graph (Item 6) is designed, since the event-sourced model naturally supports multi-surface replay.

---

#### Item 5: Agent Studio Inside TUI

**Question**: Can we add a UI-driven way to manage agents in real time?

**Current state**:
- `agent_graph` tool (tools.py:105) manages agent hierarchies programmatically
- `swarm` tool (tools.py:107) orchestrates multi-agent workflows
- The TUI has no awareness of sub-agents — it sees tool calls and results but can't introspect the agent tree
- The daemon emits events for the top-level agent only; sub-agent events are opaque

**What "Agent Studio" could mean**:
1. **Agent tree visualization**: See parent→child agent relationships, which agent is active
2. **Agent configuration**: Change system prompts, tools, model tier per agent at runtime
3. **Agent creation**: Define new agents via UI form (name, system prompt, tools, model)
4. **Workflow builder**: Visual graph of agent handoffs (swarm topology)
5. **Live monitoring**: Token usage, tool calls, errors per agent

**Dependencies**:
- Sub-agent event propagation through the daemon (currently missing)
- Agent registry / catalog (currently only ad-hoc via agent_graph tool)
- Sidebar "Tools & Config" tab from the sidebar UX plan

**Readiness**: **Not ready for a coding prompt.** Needs its own design document. The simplest useful first step would be an "Agents" sidebar tab that lists active agents from the agent_graph and shows their status. Recommend targeting Batch 7+.

---

#### Item 6: Event-Sourced Session Graph

**Question**: Can we model sessions as a timeline of events with branching, export, and cross-surface replay?

**Current state**:
- `SessionStore` (session/store.py) uses append-only JSONL for messages (`messages.jsonl`)
- Messages are Strands message dicts (role/content/tool_use), not structured events
- No concept of "branches" — each session is a linear sequence
- Session metadata tracks turn_count and message_count but not event types
- The daemon emits structured events (text_delta, tool_start, tool_result, thinking, etc.) but these are TUI display events, not stored

**What event-sourced sessions would provide**:
1. **Event log**: Every significant action (user message, tool call, plan generation, error, consent decision) as a typed event with timestamp
2. **Timeline navigation**: Jump to any point in the session, replay from there
3. **Branching**: "What if I'd said X instead?" — fork the session at a branch point
4. **Cross-surface replay**: Open a TUI session in CLI or notebook by replaying events
5. **Export**: Structured export (JSON, Markdown, HTML) of session timelines

**Implementation sketch**:
- New `EventStore` alongside `SessionStore`
- Event types: `user_message`, `assistant_text`, `tool_start`, `tool_input`, `tool_result`, `plan_generated`, `plan_approved`, `consent_requested`, `consent_given`, `error`, `thinking`, `model_change`, `context_change`
- Each event: `{event_id, session_id, parent_event_id, timestamp, type, payload}`
- `parent_event_id` enables branching (fork from any event)
- Replay: read events in order, reconstruct state

**Readiness**: **Partially ready.** The daemon already emits structured events. The gap is persisting them (currently they're display-only) and adding event IDs + parent references. This could be a Batch 5-6 prompt once the event schema is designed. For now, Prompt D3 (S3 session logs) lays groundwork by persisting session data externally.

---

## Prompt Batch 4

### Prompt D1: MS Office file support (docx, xlsx, pptx)

**Why this is high-impact:** Users working with enterprise content need to read and modify Office files. The current `file_read` tool (file_ops.py:171) is text-only and `_is_binary_file()` (file_ops.py:35) skips binary files entirely. Python has mature libraries (`python-docx`, `openpyxl`, `python-pptx`) that can read and write these formats. The agent can read file contents as structured text and write modifications back.

```
TASK: Create an `office` tool that reads and writes Microsoft Office files (docx, xlsx, pptx) by extracting/injecting text content using python-docx, openpyxl, and python-pptx.

CONTEXT:
- File: tools/file_ops.py — current file_read (line 171) is text-only; _is_binary_file (line 35) skips binary files
- File: src/swarmee_river/tools.py — _CUSTOM_TOOLS dict (line 86) is where new tools are registered
- File: tools/artifact.py — example of a well-structured tool with multiple actions (list/get/upload/store_in_kb)
- The @tool decorator from strands is used for all custom tools
- Tools return {"status": "success|error", "content": [{"text": "..."}]}
- The agent's file_read can't handle binary formats — this tool fills that gap

REQUIREMENTS:

### Part A: Create the tool file

1. Create tools/office.py with an `office` tool supporting these actions:

   action="read" (default):
   - Detect format from file extension (.docx, .xlsx, .pptx)
   - For .docx: Extract all paragraph text, preserving heading levels as markdown (# H1, ## H2, etc.)
     - Tables: render as markdown tables
     - Lists: render as markdown bullet/numbered lists
     - Ignore images/embedded objects (note their presence: "[Image: description]" if alt text exists)
   - For .xlsx: Extract sheet data as markdown tables
     - Parameter: sheet_name (optional, defaults to active sheet)
     - Parameter: max_rows (default 200, cap at 1000)
     - Include column headers from first row
     - Numeric formatting: preserve 2 decimal places for floats
     - Empty cells: render as empty string
   - For .pptx: Extract slide text in order
     - Format: "## Slide N: {title}\n{body text}\n\n" for each slide
     - Include speaker notes if present: "> Note: {notes}"
   - Parameters: path (required), sheet_name (optional), max_rows (int), max_chars (int, default 12000)
   - Return extracted text as markdown in the standard tool response format

   action="write":
   - For .docx: Accept markdown-formatted text and create/overwrite a docx file
     - Parse markdown headings → Word heading styles
     - Parse markdown paragraphs → Word paragraphs
     - Parse markdown tables → Word tables
     - Parse markdown lists → Word list items
   - For .xlsx: Accept a list of rows (as JSON array of arrays) and write to a sheet
     - Parameter: sheet_name (optional, defaults to "Sheet1")
     - First row treated as headers (bold)
     - Auto-fit column widths (approximate)
   - For .pptx: Accept a list of slides (as JSON array of {title, body, notes?}) and create/overwrite
     - Use a clean blank slide layout
     - Title → title placeholder, body → content placeholder
   - Parameters: path (required), content (required — text for docx, JSON for xlsx/pptx), sheet_name (optional)
   - Create parent directories if needed
   - Return success with byte count and path

   action="info":
   - Return file metadata without reading full content
   - .docx: page count (approximate from paragraph count), word count, heading outline
   - .xlsx: sheet names, row counts per sheet, column headers per sheet
   - .pptx: slide count, slide titles

   action="modify":
   - For .xlsx: Modify specific cells without rewriting the whole file
     - Parameters: path, sheet_name, changes (JSON array of {row, col, value})
     - Row/col are 1-indexed
   - For .docx: Find-and-replace text across all paragraphs
     - Parameters: path, find (str), replace (str), max_replacements (int, default all)
   - Return count of modifications made

2. Import structure:
   - Use try/except for each library import (python-docx, openpyxl, python-pptx)
   - Return clear error message if a library is missing: "python-docx is required. Install with: pip install python-docx"
   - Each format handler should be a separate private function (_read_docx, _read_xlsx, etc.)

3. Path safety:
   - Use the same safe_cwd() pattern from file_ops.py (import from swarmee_river.utils.path_utils)
   - Refuse to write outside the project directory
   - Resolve symlinks before checking

### Part B: Register the tool

4. In src/swarmee_river/tools.py:
   - Add import: from tools.office import office
   - Add to _CUSTOM_TOOLS: "office": office

### Part C: Update file_read to suggest office tool for binary Office files

5. In tools/file_ops.py, update file_read():
   - After the existing text read attempt, if the file has an Office extension (.docx, .xlsx, .pptx, .doc, .xls, .ppt):
     - Return a helpful error: "This is a binary Office file. Use the `office` tool instead: office(action='read', path='{path}')"
   - This teaches the agent to use the right tool without us having to update the system prompt

### Part D: Dependencies

6. Add python-docx, openpyxl, and python-pptx to the project's optional dependencies.
   - Check pyproject.toml or setup.cfg for the existing dependency format
   - Add under an "office" extras group if the project uses extras, otherwise add as optional dependencies
   - These should NOT be hard requirements — the tool gracefully degrades if they're missing

CONSTRAINTS:
- Each private handler function (_read_docx, _write_xlsx, etc.) should be self-contained — don't create shared helpers that add coupling
- Truncate all output to max_chars (default 12000) to prevent overwhelming the context window
- For .xlsx read: if a sheet has > max_rows rows, truncate and note "... (N more rows)"
- For .docx write: don't try to preserve complex formatting (styles, colors, fonts) — just structural elements (headings, paragraphs, tables, lists)
- The tool must work when only SOME of the libraries are installed (e.g., only openpyxl but not python-docx)
- Don't add the office tool to _OPTIONAL_STRANDS_TOOL_NAMES — it's a custom tool, not a strands tool
```

---

### Prompt D2: Production data source tools (S3 browser, Snowflake query, Athena query)

**Why this is high-impact:** Moving to production means agents need to find and query data where it lives — S3 buckets, Snowflake warehouses, and Athena tables. The existing `use_aws` strands tool is a generic AWS CLI wrapper, not purpose-built for data exploration. Dedicated tools with guardrails (read-only defaults, result size limits, cost awareness) let agents safely explore production data.

```
TASK: Create three production data access tools — s3_browser, snowflake_query, and athena_query — with read-only defaults and result size guardrails.

CONTEXT:
- File: src/swarmee_river/tools.py — tool registration pattern (line 86-108, _CUSTOM_TOOLS dict)
- File: tools/retrieve.py — example of a boto3-based tool with defensive error handling
- File: tools/artifact.py — example of multi-action tool with S3 upload
- File: tools/store_in_kb.py — example of async background processing pattern
- The use_aws strands tool exists (tools.py line 61) but is a generic AWS CLI wrapper
- These tools need to be SAFE for production: read-only by default, size-limited results, no DDL

REQUIREMENTS:

### Part A: S3 Browser Tool

1. Create tools/s3_browser.py with an `s3_browser` tool:

   action="list_buckets":
   - List S3 buckets accessible to the current credentials
   - Return: bucket name, creation date, region (if available)
   - Parameter: prefix (optional filter on bucket name)

   action="list_objects" (default):
   - List objects in a bucket with optional prefix
   - Parameters: bucket (required), prefix (optional), max_keys (default 100, cap 1000), delimiter (default "/")
   - Return: key, size (human-readable), last_modified, storage_class
   - Show common prefixes (folders) separately from objects
   - Truncate long listings with "... and N more objects"

   action="head":
   - Get object metadata without downloading
   - Parameters: bucket (required), key (required)
   - Return: size, content_type, last_modified, metadata, storage_class, etag

   action="read":
   - Download and return object content as text
   - Parameters: bucket (required), key (required), max_bytes (default 100KB, cap 10MB), encoding (default "utf-8")
   - For text files: return content directly
   - For JSON files: pretty-print
   - For CSV/TSV: render as markdown table (first 50 rows)
   - For binary: return "[Binary file: {size} bytes, content_type: {type}]"
   - For large files: truncate with "... (truncated at {max_bytes} bytes, total: {size})"

   action="search":
   - S3 Select or simple prefix-based search
   - Parameters: bucket, prefix, query (for S3 Select on CSV/JSON), max_results (default 20)
   - If file is CSV/JSON and query provided: use S3 Select (SQL expression)
   - Otherwise: list matching keys

2. Safety:
   - NO write/delete/put operations
   - All reads size-limited (max_bytes parameter)
   - boto3 client created per-call (no cached credentials)
   - Region from AWS_REGION env var with us-east-1 fallback

### Part B: Snowflake Query Tool

3. Create tools/snowflake_query.py with a `snowflake_query` tool:

   action="query" (default):
   - Execute a read-only SQL query against Snowflake
   - Parameters: query (required), database (optional), schema (optional), warehouse (optional), max_rows (default 200, cap 5000), max_chars (default 12000)
   - Return results as a markdown table
   - Connection parameters from environment: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD (or SNOWFLAKE_PRIVATE_KEY_PATH), SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE

   action="describe":
   - Describe a table's schema
   - Parameters: table (required), database (optional), schema (optional)
   - Return: column name, data type, nullable, default, comment
   - Uses DESCRIBE TABLE statement

   action="list_tables":
   - List tables in a schema
   - Parameters: database (optional), schema (optional), pattern (optional LIKE filter)
   - Uses SHOW TABLES

   action="list_databases":
   - List accessible databases
   - Uses SHOW DATABASES

   action="list_schemas":
   - List schemas in a database
   - Parameters: database (optional)
   - Uses SHOW SCHEMAS

4. Safety:
   - Parse SQL to reject DDL/DML: block CREATE, DROP, ALTER, INSERT, UPDATE, DELETE, MERGE, TRUNCATE, GRANT, REVOKE
   - Use a simple regex check on the query text (not a full parser) — if any blocked keyword appears as a standalone word at the start of a statement, reject
   - Set statement_timeout via session parameter (default 30s, configurable via SNOWFLAKE_QUERY_TIMEOUT)
   - Connection created per-call using snowflake-connector-python
   - Try/except import with clear install message: "snowflake-connector-python is required"

### Part C: Athena Query Tool

5. Create tools/athena_query.py with an `athena_query` tool:

   action="query" (default):
   - Execute a read-only SQL query via Athena
   - Parameters: query (required), database (optional), workgroup (optional), output_location (optional S3 path for results), max_rows (default 200, cap 5000), max_chars (default 12000)
   - Uses boto3 athena client: start_query_execution → poll get_query_execution → get_query_results
   - Poll interval: 1s, timeout: 120s (configurable via ATHENA_QUERY_TIMEOUT)
   - Return results as markdown table
   - Environment: ATHENA_DATABASE, ATHENA_WORKGROUP, ATHENA_OUTPUT_LOCATION, AWS_REGION

   action="describe":
   - Describe a table using Glue catalog
   - Parameters: table (required), database (optional)
   - Uses boto3 glue client get_table()
   - Return: column name, type, comment, partition info

   action="list_tables":
   - List tables via Glue catalog
   - Parameters: database (optional), pattern (optional filter)
   - Uses boto3 glue get_tables()

   action="list_databases":
   - List databases via Glue catalog
   - Uses boto3 glue get_databases()

   action="query_status":
   - Check status of a running query
   - Parameters: query_execution_id (required)
   - Return: state, data scanned, execution time, error if failed

6. Safety:
   - Same DDL/DML blocking regex as Snowflake tool
   - Athena is inherently read-only for most workgroups, but belt-and-suspenders
   - Include data scanned in response (cost awareness): "Data scanned: 1.2 GB"
   - Warn if data scanned > 1GB: "⚠ Large scan: {size}. Consider adding WHERE clauses or partitions."

### Part D: Register all three tools

7. In src/swarmee_river/tools.py:
   - Import all three tools
   - Add to _CUSTOM_TOOLS: "s3_browser": s3_browser, "snowflake_query": snowflake_query, "athena_query": athena_query

### Part E: Dependencies

8. Add to optional dependencies:
   - boto3 (already available for KB tools, but note it's needed for s3_browser and athena_query)
   - snowflake-connector-python under a "snowflake" extras group
   - Each tool must handle missing dependencies gracefully with install instructions

CONSTRAINTS:
- Every tool must return results within max_chars (default 12000) to avoid context window overload
- Every tool must timeout (Snowflake: 30s, Athena: 120s, S3: 15s per operation)
- NO write operations in any tool — these are read-only data exploration tools
- Each tool is a separate file (s3_browser.py, snowflake_query.py, athena_query.py) — don't bundle into one module
- Use the @tool decorator from strands consistently
- The Snowflake connection should NOT cache connections — create and close per-call for safety
- Athena polling should not block the event loop — the tool runs synchronously (Strands tools are sync) but should respect the timeout
- markdown table rendering: use | col1 | col2 | format with alignment row, truncate cell values > 100 chars
```

---

### Prompt D3: S3 session log persistence and enterprise KB promotion

**Why this is high-impact:** Session data currently lives only on the local filesystem (`.swarmee/sessions/`). For production use, sessions need to persist to S3 for durability, compliance, and cross-machine access. Additionally, promoting high-quality session content (successful plans, useful agent outputs) to enterprise Knowledge Bases creates a feedback loop where agents learn from past sessions.

```
TASK: Add S3 session log export and enterprise KB promotion capabilities to the session system.

CONTEXT:
- File: src/swarmee_river/session/store.py — SessionStore class (line 45), save_messages() (line 183), load_messages() (line 238)
- File: tools/store_in_kb.py — existing KB ingestion pattern with background thread (line 155)
- File: tools/artifact.py — existing S3 upload pattern (line 88-98)
- File: src/swarmee_river/artifacts.py — ArtifactStore with index.jsonl pattern (line 42)
- Sessions are stored in .swarmee/sessions/<session_id>/ with meta.json, messages.jsonl, state.json, last_plan.json
- The existing store_in_kb tool uses Bedrock's inline CUSTOM data source ingestion
- S3 uploads in artifact.py use simple boto3 put_object

REQUIREMENTS:

### Part A: S3 Session Export

1. Add a `session_s3` tool in tools/session_s3.py:

   action="export" (default):
   - Export a session to S3 in a structured format
   - Parameters: session_id (optional — defaults to current session), s3_bucket (optional, from SWARMEE_SESSION_S3_BUCKET env), s3_prefix (optional, default "swarmee/sessions/")
   - Uploads to: s3://{bucket}/{prefix}/{session_id}/
     - meta.json — session metadata
     - messages.jsonl — append-only message log
     - state.json — agent state snapshot
     - last_plan.json — most recent plan
     - summary.md — auto-generated human-readable session summary
   - The summary.md should contain:
     - Session ID, timestamps, model info, turn count
     - User messages (first 200 chars each) as a conversation outline
     - Plan summaries (if any plans were generated)
     - Tool usage summary (tool name → call count)
     - Error summary (if any errors occurred)
   - Return: S3 URI of exported session, file count, total bytes

   action="list":
   - List sessions available in S3
   - Parameters: s3_bucket, s3_prefix, max_results (default 20)
   - List s3://{bucket}/{prefix}/ common prefixes (session IDs)
   - For each, read meta.json to get created_at, updated_at, turn_count
   - Return sorted by updated_at descending

   action="import":
   - Import a session from S3 into local storage
   - Parameters: session_id, s3_bucket, s3_prefix
   - Download all session files to .swarmee/sessions/{session_id}/
   - Skip if session already exists locally (return message suggesting --force)
   - Parameter: force (bool, default false) — overwrite existing local session

   action="sync":
   - Bi-directional sync: export if local is newer, import if S3 is newer
   - Compare updated_at timestamps in meta.json
   - Parameters: session_id (optional — sync all if omitted), s3_bucket, s3_prefix

2. Auto-export hook:
   - Add a `SessionS3Hooks` class in src/swarmee_river/hooks/session_s3.py
   - Implements an after_invocation hook that exports the session to S3 after each agent turn
   - Only activates when SWARMEE_SESSION_S3_BUCKET is set
   - Uses background thread (like store_in_kb) to avoid blocking
   - Debounce: don't export more than once per 30 seconds

### Part B: Generate session summary

3. Add a `_generate_session_summary()` function in the session_s3 tool:
   - Input: meta dict + messages list
   - Output: markdown-formatted summary string
   - Extract conversation outline from user messages
   - Count tool usage by tool name
   - List plans (if any plan events exist)
   - List errors (if any error events exist)
   - Keep summary under 5000 chars
   - This is a STATIC analysis — no LLM call needed. Just parse the message dicts.

### Part C: KB Promotion

4. Add action="promote_to_kb" to the session_s3 tool:
   - Promote session content to a Bedrock Knowledge Base for enterprise search
   - Parameters: session_id, knowledge_base_id (optional, from SWARMEE_KNOWLEDGE_BASE_ID env), content_filter (optional: "plans", "outputs", "all")
   - content_filter="plans": promote only generated plans (from last_plan.json and plan events in messages)
   - content_filter="outputs": promote assistant responses that were marked as useful or that completed successfully
   - content_filter="all": promote the full session summary + all plans + key outputs
   - Uses the same Bedrock inline ingestion pattern as store_in_kb.py
   - Each promoted document includes metadata: session_id, timestamp, content_type, original_user_query
   - Return: count of documents promoted, KB ID

5. Add action="promote_artifact" for promoting individual artifacts:
   - Parameters: artifact_id, knowledge_base_id
   - Reads the artifact via ArtifactStore, ingests into KB
   - This is a convenience wrapper — artifact.py already has store_in_kb action, but this adds session context metadata

### Part D: Register the tool and hook

6. In src/swarmee_river/tools.py:
   - Import: from tools.session_s3 import session_s3
   - Add to _CUSTOM_TOOLS: "session_s3": session_s3

7. In src/swarmee_river/swarmee.py, in the hooks setup section (around line 742-759):
   - If SWARMEE_SESSION_S3_BUCKET is set, add SessionS3Hooks to the hooks list
   - Import conditionally to avoid breaking when boto3 is missing

### Part E: Environment variables

8. New environment variables:
   - SWARMEE_SESSION_S3_BUCKET — S3 bucket for session export (enables S3 session features)
   - SWARMEE_SESSION_S3_PREFIX — prefix within bucket (default "swarmee/sessions/")
   - SWARMEE_SESSION_S3_AUTO_EXPORT — "true" to enable auto-export after each turn (default "false")
   - SWARMEE_SESSION_KB_PROMOTE_ON_COMPLETE — "true" to auto-promote session summary to KB when session ends

CONSTRAINTS:
- S3 operations must handle missing credentials gracefully — return clear error, don't crash
- Background thread for auto-export must be daemon thread (won't prevent process exit)
- Session summary generation must be pure Python (no LLM call) — just parse message dicts
- KB promotion must reuse the existing store_in_kb pattern (Bedrock CUSTOM data source inline ingestion)
- All S3 keys should use forward slashes and avoid special characters
- Don't modify SessionStore itself — the S3 layer is an overlay, not a replacement for local storage
- Import boto3 inside functions (lazy) to avoid import-time failures when not using AWS
```

---

### Prompt D4: Interactive SOP toggle in sidebar (Phase 1 of sidebar UX plan)

**Why this is high-impact:** The sidebar UX plan (06-sidebar-ux-plan.md) identifies six improvements, but the highest-value first step is making SOPs interactive. Currently the SOPs tab is browse-only — users see SOP names but can't activate/deactivate them from the sidebar. They have to type `/sop activate <name>`. This prompt converts the SOP list into toggleable items with state synced to the Context tab.

```
TASK: Make the SOPs sidebar tab interactive with activate/deactivate toggle switches, and sync active SOP state with the Context tab.

CONTEXT:
- File: src/swarmee_river/tui/app.py — compose() SOPs tab at line 1856-1859: Static header + VerticalScroll(id="sop_list")
- File: src/swarmee_river/tui/app.py — _refresh_sop_catalog() and _render_sop_panel() handle SOP listing
- File: src/swarmee_river/tui/app.py — Context tab at line 1834-1855: context source management
- File: src/swarmee_river/tui/app.py — _set_daemon_sop_override() sends SOP activation to daemon
- File: src/swarmee_river/tui/app.py — _render_context_sources_panel() renders active context sources
- The SOPs tab currently renders SOP items as Static widgets in a VerticalScroll
- Active SOPs are tracked but only through the /sop activate command
- Context sources include type "sop" entries

THE PROBLEM:
- Users see SOPs listed but can't interact with them except via typed commands
- Active SOPs don't appear in the Context tab automatically
- No visual distinction between active and inactive SOPs
- No SOP preview without typing /sop preview <name>

REQUIREMENTS:

### Part A: SOPListItem widget

1. Create a new SOPListItem widget in widgets.py:
   class SOPListItem(Static):
       """Interactive SOP entry with toggle and preview."""
       - Properties: sop_name (str), sop_path (str), source (str — "local", "pack", "strands"), is_active (reactive bool)
       - Layout:
         ┌───────────────────────────────────┐
         │ 📋 SOP Name          [source] [⊙] │
         │    First line of SOP content...    │
         └───────────────────────────────────┘
       - The [⊙] is a Switch (Textual built-in widget) for activate/deactivate
       - "First line" is a dim 1-line preview of the SOP content (truncated to 60 chars)
       - When is_active=True: left border accent color, switch ON
       - When is_active=False: default border, switch OFF

2. SOPListItem should emit a custom message on toggle:
   class SOPToggled(Message):
       def __init__(self, sop_name: str, sop_path: str, activated: bool):
           ...

### Part B: Update _render_sop_panel()

3. Replace the current Static-based SOP items with SOPListItem widgets:
   - Mount SOPListItem for each SOP in the catalog
   - Set is_active=True for SOPs that are currently in _active_sop_names or context sources
   - Sort: active SOPs first, then alphabetical

4. Add a search/filter Input at the top of the SOPs panel:
   - Typing filters the SOP list by name (case-insensitive substring match)
   - Empty input shows all SOPs

### Part C: Handle SOPToggled message

5. In app.py, add an on_sop_toggled handler:
   - If activated:
     - Add SOP to active context sources (same as /sop activate does)
     - Call _set_daemon_sop_override() to notify the daemon
     - Add an "sop" type entry to context sources
     - Refresh the Context tab to show the newly active SOP
   - If deactivated:
     - Remove SOP from active context sources
     - Call _set_daemon_sop_override() with updated list
     - Remove the "sop" entry from context sources
     - Refresh the Context tab

### Part D: Cross-tab sync

6. When an SOP is activated via the Context tab (existing "Add SOP" flow):
   - Find the matching SOPListItem in the SOPs tab and set is_active=True
   - This keeps both tabs in sync regardless of which surface the user interacts with

7. When an SOP is activated via /sop activate command:
   - Same sync — update the SOPListItem toggle state

### Part E: SOP preview expand

8. Add a Collapsible to SOPListItem that shows the full SOP content when clicked:
   - Default: collapsed (only 1-line preview visible)
   - Click the item (not the toggle) to expand
   - Expanded view: full SOP markdown content in a read-only TextArea (max 50 lines, scrollable)
   - Only one SOP should be expanded at a time (collapse others when one is expanded)

CONSTRAINTS:
- SOPListItem must be lightweight — the SOP catalog may have 20-50 items
- Don't load full SOP content until the user expands it (lazy load)
- The Switch widget should be compact=True to save vertical space
- Keep the existing /sop commands working — this is an ADDITIONAL interaction surface, not a replacement
- The search filter should be debounced (200ms) to avoid re-rendering on every keystroke
- Active SOP state must survive tab switches (don't reset when user navigates away and back)
```

---

## Execution Order

1. **D1** (MS Office) — Independent, no dependencies. Gives agents a major new capability for enterprise workflows.
2. **D4** (SOP toggles) — Independent, no dependencies. First tangible sidebar UX improvement. Quick win that demonstrates the sidebar UX direction.
3. **D2** (Production data tools) — Independent, but benefits from testing in actual production environments. Larger scope — consider sending S3 browser first, then Snowflake/Athena as follow-ups if the agent struggles with the combined scope.
4. **D3** (S3 session logs + KB promotion) — Depends on S3 access working (D2 validates this). Also the most architectural prompt — lays groundwork for event-sourced sessions.

D1 and D4 are fully independent and can be sent in parallel.
D2 and D3 are independent of each other but both require AWS credentials in the environment.

---

## Items Deferred to Future Batches

| Item | Why Deferred | Target |
|------|-------------|--------|
| Shared agent runtime service | Needs design document for IPC protocol, session routing, lifecycle management | Batch 6-7 |
| Agent studio inside TUI | Needs sub-agent event propagation and agent registry design | Batch 7+ |
| Event-sourced session graph | Needs event schema design; D3 lays groundwork with S3 persistence | Batch 5-6 |
| Sidebar Phases 2-4 | Depends on Phase 1 (D4) landing successfully | Batch 5-7 |
