# Multi-Agent Delegation: strand, swarm, and use_agent

Swarmee River supports three delegation patterns that let the orchestrator hand work off to specialized sub-agents. Each pattern serves a different need.

---

## Why Delegate?

| Reason | Benefit |
|--------|---------|
| **Specialization** | Give a sub-agent only the tools it needs; keeps it focused |
| **Parallelism** | Run multiple sub-agents simultaneously to cut wall-clock time |
| **Token isolation** | Each sub-agent has its own context window; long investigations don't crowd out the main session |
| **Repeatability** | Wrap a complex procedure in a named tool call that can be re-used across sessions |

The orchestrator decides when to delegate — you don't invoke sub-agents directly. You influence this through SOPs (e.g., "use a swarm to parallelize investigation and implementation"), agent profiles, and prompts.

---

## `strand` — Sequential Sub-Agent

`strand` runs a single nested agent with its own tool set and system prompt. The orchestrator calls `strand(...)`, waits for the result, and continues.

**When to use:**
- You need a specialized tool set the main agent shouldn't have (e.g., database access)
- You want to isolate a complex sub-task with its own instructions
- The sub-task is long and would fill the main context window

**How the orchestrator calls it:**

The `strand` tool is available to the orchestrator. You can guide its use via a SOP or system prompt:

```markdown
## Rules
- For all data transformation tasks, delegate to a strand with access to `python_repl` and `file_write`.
  Pass the full data file path and the expected output format.
```

**Example invocation the orchestrator might make:**
```
strand(
  query="Read the CSV at /data/sales.csv, calculate monthly totals, and write a summary to /data/summary.txt",
  tools=["python_repl", "file_read", "file_write"],
  system_prompt="You are a data transformation specialist. Use pandas. Always validate the input before processing."
)
```

**Result:** The strand runs, produces output, and that output is returned to the orchestrator's context as the tool result.

---

## `swarm` — Parallel Multi-Agent

`swarm` spawns multiple agents simultaneously and waits for all to complete before returning results. It is the most powerful delegation pattern for time-sensitive workflows.

**When to use:**
- You have N independent sub-tasks that don't depend on each other
- You want to run investigation and implementation in parallel
- You want multiple specialist reviewers to evaluate the same artifact

**How to guide swarm usage:**

In a SOP or system prompt:

```markdown
## Delegation Strategy
When asked to investigate a bug and propose a fix:
1. Use swarm to run two parallel agents:
   - Agent 1 (Investigator): tools=[git, file_read, file_search] — find the root cause
   - Agent 2 (Proposer): tools=[file_read, file_list] — read surrounding code and draft a fix approach
2. Synthesize both results before writing any code.
```

The swarm SOP in `sops/swarm.sop.md` provides a detailed template for this pattern.

**Token cost note:** Each agent in the swarm has its own full context window. Swarms are powerful but more expensive than a single strand. Prefer swarm when parallelism saves significant wall-clock time on tasks where latency matters.

---

## `use_agent` — Profile-Based Delegation

`use_agent` delegates to a named **agent profile** that you've defined in `.swarmee/settings.json`. The sub-agent inherits all settings from that profile (SOPs, tools, system prompt snippet).

**When to use:**
- You have a well-defined reusable specialist (e.g., a security reviewer you always want applied consistently)
- The sub-agent's behavior is defined by a profile, not inline instructions

**Example:** If you've defined a `security-reviewer` profile (see [sops-and-profiles.md](sops-and-profiles.md)), the orchestrator can invoke it:

```
use_agent(agent_id="security-reviewer", query="Review the authentication changes in this diff: ...")
```

The `security-reviewer` profile's SOPs, tool allowlist, and system prompt snippet all apply automatically.

---

## Team Presets

Team presets are pre-configured swarm layouts defined in `.swarmee/settings.json`. They let you quickly spin up a consistent multi-agent team without repeating configuration each time.

**TUI:** Agents tab > Team section > configure preset agents

**settings.json structure:**
```json
{
  "team_presets": [
    {
      "name": "Bug Hunt Team",
      "agents": [
        {
          "id": "investigator",
          "role": "Find the root cause",
          "tool_names": ["git", "file_read", "file_search"],
          "sop_names": ["bugfix"]
        },
        {
          "id": "implementer",
          "role": "Write the fix",
          "tool_names": ["file_read", "file_write", "git", "run_checks"],
          "sop_names": ["code-change"]
        }
      ]
    }
  ]
}
```

Activate a preset via the TUI Agents tab or by naming it in your prompt: "Use the Bug Hunt Team to fix this issue."

---

## Passing Context Between Agents

Sub-agents start with a fresh context window. To give them useful information:

**Pass it in the `query`:**
```
strand(
  query=f"Here is the error log:\n\n{error_log}\n\nFind the root cause in the codebase.",
  tools=[...]
)
```

**Use artifact IDs:** If a previous tool call produced a large result stored as an artifact, pass the artifact ID:
```
strand(
  query="Analyze the data in artifact ID abc123 and produce a summary.",
  tools=["file_read"]
)
```

**Summarize before passing:** For very large results, ask the orchestrator to summarize first:
```markdown
## Rules
- Before delegating to a strand, summarize any context longer than 2000 characters.
  Pass the summary, not the raw content.
```

---

## Cost and Token Implications

| Pattern | Tokens used | Best for |
|---------|-------------|---------|
| Single agent | 1x | Most tasks |
| `strand` | 1x + sub-agent context | Isolated long sub-tasks |
| `swarm` (N agents) | 1x + N sub-agent contexts | N parallel independent tasks |
| `use_agent` | 1x + profile agent context | Reusable specialist profiles |

Each sub-agent processes the full query + its own system prompt + any results it produces. For cost-sensitive workloads, prefer passing concise, targeted queries to sub-agents rather than dumping entire conversation histories.

---

## Example: Parallel Investigator + Implementer

This is the most common real-world swarm pattern. It investigates a bug and drafts a fix simultaneously, then the orchestrator synthesizes both.

**Add to your system prompt or a SOP:**

```markdown
## Bug Resolution Strategy
When given a bug report:

1. Launch a swarm with two agents running in parallel:

   Agent A — Investigator:
   - tools: [git, file_read, file_search, file_list]
   - task: Find the root cause. Identify the exact file, line, and reason.
   - output: A brief root cause report (< 500 words)

   Agent B — Context Reader:
   - tools: [file_read, file_list]
   - task: Read the module where the bug likely lives. Understand the design intent.
   - output: A summary of the relevant code structure and design decisions

2. Wait for both to complete.
3. Synthesize: combine the root cause with the design context.
4. Write the fix yourself (do not delegate the fix to a sub-agent).
5. Run tests using the run_checks tool.
```

---

## Next Steps

- [sops-and-profiles.md](sops-and-profiles.md) — define the agent profiles that `use_agent` delegates to
- [debugging.md](debugging.md) — diagnose issues when a sub-agent doesn't behave as expected
- [../agent_context_lifecycle.md](../agent_context_lifecycle.md) — how sub-agent contexts are assembled
