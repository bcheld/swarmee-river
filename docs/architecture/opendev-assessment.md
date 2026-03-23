# OpenDev Architecture Assessment: Swarmee River

An assessment of the Swarmee River project against the principles and architectural patterns
described in the [OpenDev paper](https://arxiv.org/html/2603.05344v1) — a comprehensive
treatment of production-grade, terminal-native AI coding agents.

**Date**: 2026-03-18

## Executive Summary

Swarmee River implements many of the core patterns described in the OpenDev paper,
particularly around planning/execution separation, tool permission systems, session
persistence, and observability. The project shows maturity in safety layering, context
budget management, and human-in-the-loop workflows.

Key areas of strength:
- **Defense-in-depth safety** — multi-layer tool policy, consent, and permission system
- **Planning/execution separation** — structured WorkPlan with approval gates
- **Session persistence** — full conversation state with resume capability
- **Observability** — comprehensive JSONL event logging with redaction
- **Context management** — budget-aware summarization with shared-prefix caching

Key areas for improvement:
- **Multi-model routing** — limited to tier-based selection; no per-phase model specialization
- **Instruction fade-out counteraction** — system reminders exist but lack event-driven triggers
- **Extended ReAct loop** — no explicit self-critique or thinking phases in the execution cycle
- **Lazy tool discovery** — MCP integration exists but tool schemas loaded eagerly
- **Evaluation framework** — logging is strong but no structured benchmarking or drift detection

## Assessment by Theme

Each theme is assessed with a maturity rating:
- **Strong** — well-implemented, aligned with paper recommendations
- **Partial** — core concepts present but gaps remain
- **Weak** — minimal or no implementation of the described pattern

---

### 1. Compound AI Systems Architecture
**Rating: Partial**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Per-workflow model binding | 5 specialized model roles (normal, thinking, critique, vision, compaction) | Tier-based selection (fast/balanced/deep/long) — coarser granularity |
| Fallback chains | Seamless degradation across providers | Single provider per session; error escalation to higher tier exists |
| User-configurable routing | Model routing without code changes | Settings-based tier selection; no per-phase configuration |
| Lazy client initialization | Defer API key validation until first use | Model loaded at startup via `load_model()` |

**Strengths**: Tier system provides meaningful model differentiation. Provider abstraction (Bedrock, OpenAI, Ollama, GitHub Copilot) is solid.

**Gaps**: No separate model slots for thinking, self-critique, vision, or compaction. The paper's five-role model routing would allow using cheaper models for summarization and more capable models for reasoning.

**Recommendations**:
1. Introduce per-phase model configuration (e.g., `compaction_model`, `thinking_model`) in settings
2. Add fallback chain support so a failed Bedrock call can fall back to OpenAI
3. Implement lazy model client initialization — defer provider setup until first LLM call

---

### 2. Extended ReAct Execution Framework
**Rating: Partial**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Six-phase execution | Pre-check, thinking, self-critique, action, execution, post-processing | Standard ReAct loop via Strands SDK |
| Self-critique phase | Internal error correction before tool execution | Not implemented |
| Adaptive compaction in loop | Compaction integrated into reasoning cycle | Compaction happens via conversation manager, outside the core loop |
| Chain-of-thought traces | Optional CoT for transparency | Extended thinking available on `deep` tier only |

**Strengths**: The Strands SDK provides a solid ReAct loop. Extended thinking on deep tier enables deliberation for complex tasks.

**Gaps**: No explicit self-critique step where the model reviews its planned action before execution. No pre-check phase to evaluate context health before each iteration.

**Recommendations**:
1. Add a pre-tool-call self-critique hook that asks the model to validate its tool choice
2. Integrate context budget checking as a pre-check phase in each loop iteration
3. Consider a `think` tool (already available via Strands) as a standard part of the tool set to encourage deliberation

---

### 3. Defense-in-Depth Safety Architecture
**Rating: Strong**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Prompt-level guardrails | Security policies in system prompt | Profile-based system prompt with SOPs |
| Schema-level restrictions | Plan-mode whitelists, per-agent tool lists | `tool_permissions.py` + plan-mode allowlist |
| Runtime approval | Manual/Semi-Auto/Auto levels | `tool_consent.py` with interactive approval, remember, and session override |
| Tool-level validation | Blocklists, stale-read detection, output truncation | Tool result limiter, diff review, tool message repair |
| Lifecycle hooks | Pre-tool blocking, argument mutation | Full hook system (11 hook modules) |

**Strengths**: This is Swarmee River's strongest alignment with the paper. The five-layer safety model maps closely:
1. **Prompt-level**: Profile prompts + SOPs
2. **Schema-level**: `tool_permissions.py` with read/write/execute declarations, plan-mode allowlists
3. **Runtime approval**: `tool_consent.py` with diff previews, remember flags, session-scoped overrides
4. **Tool-level**: Result limiter (8KB), tool message repair, diff review hooks
5. **Lifecycle hooks**: 11 hook modules covering policy, consent, logging, retry, metrics

**Gaps**: No dangerous-pattern blocklist for shell commands (e.g., `rm -rf /`). Approval persistence is session-scoped but not pattern-based for recurring safe operations.

**Recommendations**:
1. Add a `DANGEROUS_PATTERNS` blocklist for shell commands (destructive filesystem ops, network exfiltration patterns)
2. Implement pattern-based auto-approval for recurring safe operations (e.g., `git status` always allowed)
3. Add approval saturation monitoring — track consent prompt frequency and warn if user is experiencing fatigue

---

### 4. Agent Scaffolding Pattern
**Rating: Partial**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Eager construction | Complete assembly before first prompt | Model and tools loaded at startup; prompt assets loaded eagerly |
| Single parameterized class | No class hierarchy for agent types | Uses Strands `Agent` class with configuration; profiles provide parameterization |
| Dependency injection | AgentDependencies dataclass | Settings-based injection; no formal DI container |
| Subagent isolation | Lightweight deps omitting session_manager | Shared-prefix forking with `SubAgentDeps`-like pattern |

**Strengths**: Profiles system provides clean parameterization without subclassing. Shared-prefix forking is a sophisticated approach to subagent creation that the paper doesn't fully address.

**Gaps**: No formal `AgentFactory` entry point — agent construction is spread across `swarmee.py`, `tools.py`, and profile resolution. Tool schema building and system prompt composition happen in multiple places.

**Recommendations**:
1. Consolidate agent construction into a dedicated factory that executes a clear three-phase build: skills/packs discovery, tool registration, prompt assembly
2. Ensure `build_system_prompt()` and `build_tool_schemas()` equivalents complete before any agent method is callable
3. Add a `refresh_tools()` method for MCP server discovery without full re-instantiation

---

### 5. Context Engineering
**Rating: Strong**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Dynamic system prompt | Priority-ordered conditional composition | `PromptCacheState` with SHA256-based change detection |
| Tool result optimization | Per-tool summarization, offloading | Artifact store for large outputs; result limiter at 8KB |
| Dual-memory architecture | Episodic + working memory | Session persistence (episodic) + recent messages (working) |
| Context-aware reminders | Event-triggered at decision points | `<system-reminder>` injection via prompt cache |
| Adaptive compaction | Progressive stages as budget depletes | `BudgetedSummarizingConversationManager` with 3 strategies |
| Context retrieval pipeline | Anchor-based tool selection, agentic search | `project_context` tool with summary/search/read modes |

**Strengths**: The `BudgetedSummarizingConversationManager` is well-designed with three strategies (balanced, cache_safe, long_running). Shared-prefix forking for compaction reuses the parent agent's cached prompt. The artifact store elegantly offloads large tool results while preserving summaries in context.

**Gaps**: No explicit context retrieval pipeline with layered selection (anchor-based → agentic search → assembly → optimization). System reminders are injected but not truly event-driven — they're updated on prompt cache changes rather than triggered by specific behavioral events.

**Recommendations**:
1. Implement event-driven reminder injection (e.g., trigger a safety reminder when the model attempts a blocked tool 3+ times)
2. Add a context health dashboard — expose token budget utilization to the user during sessions
3. Consider implementing the paper's four-layer retrieval pipeline: anchor-based tool filtering → multi-step search → context assembly → final optimization

---

### 6. Tool System Architecture
**Rating: Strong**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Registry architecture | Separate registration, schema, dispatch | `tools.py` handles registration; permissions separate; hooks handle dispatch middleware |
| File operations | Line-numbered reads, fuzzy edit matching | `file_read` with line numbers; `editor` and `patch_apply` for edits |
| Shell execution | Staged pipeline with output truncation | `shell` tool with result limiting and consent |
| Web interaction | Browser-engine fetching, search | `http_request` tool; no browser-engine or screenshot tools |
| Semantic analysis (LSP) | Four-layer LSP abstraction | Not implemented |
| Task management | Lightweight kanban | `todoread`/`todowrite` tools |
| MCP integration | Lazy discovery via search_tools | MCP support via Strands SDK |
| Subagent delegation | Isolated instances with filtered tools | `strand`, `swarm`, `agent_graph` tools |

**Strengths**: 30+ built-in tools covering file ops, shell, git, multi-agent, knowledge management, and cloud services. The tool permission system (read/write/execute) is clean and well-integrated with plan-mode enforcement. Multi-agent tools (strand, swarm, agent_graph) provide flexible delegation patterns.

**Gaps**: No LSP-based semantic analysis tools. No browser-engine web fetching or screenshot capture for vision models. No batch_tool for parallel execution of identical operations.

**Recommendations**:
1. Add LSP integration for semantic code navigation (find_symbol, find_references, rename_symbol)
2. Implement a `batch_tool` for parallel execution of identical operations across multiple inputs
3. Add browser-based web tools (screenshot capture, JS-rendered page fetching) for vision model integration
4. Implement schema filtering at build time — only include tool schemas relevant to the current agent's allowlist

---

### 7. Planning vs. Execution Separation
**Rating: Strong**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Schema-level enforcement | Write tools absent from planner | Plan mode restricts to read-permission tools only |
| Structured plan output | Seven-section plan format | `WorkPlan` with goal, context, files, steps, verification, risks |
| User approval gate | Present plan for approval | Interactive `:y/:n/:replan` flow |
| Re-planning support | Mid-execution re-planning | `:replan` command with step feedback |
| Concurrent exploration | Parallel subagent spawning | `strand` tool available in plan mode |

**Strengths**: This is well-implemented. Plan mode enforces read-only tool access at the schema level. The `WorkPlan` structure closely mirrors the paper's recommended seven-section format. The replan flow with step-level feedback is more sophisticated than the paper describes.

**Gaps**: Planning is delegated to the same agent instance with a mode switch rather than a separate planner subagent. This means the agent can potentially get stuck in plan mode (the exact anti-pattern the paper warns about).

**Recommendations**:
1. Consider delegating planning to a dedicated subagent (as the paper recommends) to eliminate the risk of getting stuck in plan mode
2. Add plan quality metrics — track revision count, approval rate, tools-expected-vs-used accuracy
3. Implement automatic plan-mode detection based on task complexity heuristics (the paper suggests this as an option)

---

### 8. Session Persistence and Continuity
**Rating: Strong**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Full conversation persistence | JSON with atomic writes | `SessionStore` with messages.json + messages.log |
| Session index | Resumption and discovery | Session list/load/save commands |
| Operation log | Shadow git snapshots | File diff review hook captures before/after |
| Cross-session knowledge | Strategy learning from feedback | Prompt assets and SOPs persist across sessions |
| Configuration hierarchy | Four-tier resolution | CLI flags → env vars → .swarmee/settings.json → defaults |

**Strengths**: Session management is comprehensive. The four-tier configuration hierarchy matches the paper's recommendation. Session state includes messages, agent state, and last plan.

**Gaps**: No atomic writes (the paper emphasizes this for corruption prevention). No formal operation log with undo capability via git snapshots. Cross-session learning is manual (user must update SOPs/prompts).

**Recommendations**:
1. Implement atomic session writes (write to temp file, then rename) to prevent corruption on crash
2. Add a git-snapshot-based undo system — capture repo state before each mutating tool call
3. Consider automated cross-session learning — extract patterns from successful sessions into SOPs

---

### 9. Instruction Fade-Out Counteraction
**Rating: Weak**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Event-driven reminders | Triggered by specific behavioral events | System reminders exist but are state-change driven, not event-driven |
| Guardrail counters | Track safety metric violations | Plan-mode loop detection (3-attempt, 4-repeat limits) |
| Template resolution | Adapt reminder text to current state | Prompt cache updates based on content hashing |
| Decision trees | Explicit tool selection guidance | Not implemented |

**Strengths**: The plan-mode loop detection (blocking after 3 attempts or 4 identical calls) is a good guardrail counter. System reminders update based on state changes.

**Gaps**: No event-driven reminder system that detects behavioral drift and injects targeted guidance. No explicit decision trees for tool selection. No tracking of attention patterns over conversation length.

**Recommendations**:
1. Implement event detectors for common failure modes: repeated tool errors, goal drift, permission fatigue
2. Add guardrail counters that track safety violations across the session and inject reminders at thresholds
3. Create explicit decision trees for tool selection (e.g., "for file search: use `file_search` for content, `glob` for paths, `project_context` for broad exploration")
4. Consider injecting behavioral reminders at fixed turn intervals in long sessions

---

### 10. Multi-Model Routing
**Rating: Partial**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Five model roles | Normal, thinking, self-critique, vision, compaction | Tier-based (fast/balanced/deep/long) |
| Per-role configuration | Independent model binding | Tier selection in settings |
| Fallback chains | Provider failover | Error escalation to higher tier; no cross-provider fallback |
| Provider-level caching | Static prompt caching | Shared-prefix forking; Bedrock `cache_tools` support |

**Strengths**: The tier system provides meaningful differentiation. Shared-prefix caching is well-implemented for Bedrock. Extended thinking on deep tier enables deliberation.

**Gaps**: No separate model for compaction (uses same model), vision (not supported), or self-critique. No cross-provider fallback chains.

**Recommendations**:
1. Add a dedicated compaction model slot — compaction can use a cheaper/faster model
2. Implement cross-provider fallback (e.g., Bedrock → OpenAI) for resilience
3. Add vision model support for screenshot/image analysis workflows
4. Allow per-phase model override in settings (e.g., `model.compaction_tier: "fast"`)

---

### 11. Progressive Degradation and Graceful Failure
**Rating: Partial**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Staged compaction | Progressive reduction as budget depletes | Three compaction strategies with budget awareness |
| Iteration caps | Maximum iterations with doom-loop detection | Plan-mode loop detection; no general iteration cap |
| Approval saturation | Pattern-based auto-approval | Session-scoped remember flags; no pattern detection |
| Network unavailability | Graceful degradation | Error classification handles transient errors with retry |

**Strengths**: Error classification into transient/escalatable/tool_error/auth_error/fatal is well-designed. The compaction strategies provide graceful context degradation. The stall monitor detects hung operations.

**Gaps**: No general iteration cap for the main agent loop. No doom-loop detection outside plan mode. No approval saturation monitoring.

**Recommendations**:
1. Add a configurable maximum iteration count for the main agent loop
2. Implement general doom-loop detection — identify when the agent is making the same tool call repeatedly without progress
3. Track approval prompt frequency and offer to switch to auto-approve for recurring safe patterns
4. Make token budget consumption visible to users during execution

---

### 12. Transparency and Observability
**Rating: Strong**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Event-driven logging | All significant state changes | JSONL logger with before/after events for invocations, model calls, tool calls |
| Plan presentation | Approval gates before execution | Interactive plan review with diff preview |
| Approval history | Pattern detection | Session-scoped consent memory |
| Session transcripts | Debugging and learning | Full message history + JSONL event logs |
| Cost tracking | Token consumption per operation | TUI metrics hook with per-provider pricing |

**Strengths**: The JSONL event logger is comprehensive, capturing the full lifecycle of each invocation. Redaction of sensitive data is built-in. Cost tracking with per-provider pricing is a practical production feature not covered in the paper.

**Gaps**: No structured audit trail beyond JSONL logs. No aggregation or analysis tools for historical sessions. Compaction events not explicitly logged with before/after token counts.

**Recommendations**:
1. Log compaction events with before/after token counts and strategy used
2. Add a session analytics command that summarizes tool usage patterns, error rates, and token efficiency across sessions
3. Consider structured audit logs (separate from diagnostic JSONL) for compliance use cases

---

### 13. Extensibility Architecture
**Rating: Strong**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Custom agents | User-defined agent specs | Profiles system + runtime agent tools |
| Skills/prompts | Three-tier hierarchy | Prompt assets + SOPs + packs |
| Lifecycle hooks | Pre-tool blocking, argument mutation | 11 hook modules |
| MCP integration | External tool servers | Supported via Strands SDK |
| Configuration-driven | Customize without code | `.swarmee/settings.json` + profiles + prompts.json |

**Strengths**: The packs system (tools + SOPs + prompts as units) is a clean abstraction not explicitly covered in the paper. The 11-module hook system provides extensive extensibility. Profile-based agent configuration enables configuration-driven customization.

**Gaps**: No unified extension path for internal and external tools — custom tools follow a different loading path than built-in tools. No formal plugin API beyond hooks and packs.

**Recommendations**:
1. Unify the tool loading path so custom tools, pack tools, and built-in tools follow the same registration flow
2. Document the hook API formally for third-party extension development
3. Add a `search_tools` equivalent for discovering available pack tools at runtime

---

### 14. Lazy Loading and Bounded Growth
**Rating: Weak**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Tool discovery on-demand | MCP tools discovered per-need | Tools loaded eagerly at startup |
| Model client deferral | HTTP client slots deferred | Model loaded at startup |
| Index self-healing | Deterministic outside agent loop | Not implemented |
| search_tools for targeted discovery | Keyword-scored tool search | Not implemented as a standalone tool |

**Strengths**: The `project_context` tool provides on-demand codebase exploration rather than upfront loading.

**Gaps**: All tools are loaded and registered at startup regardless of whether they'll be used. Model clients are initialized eagerly. No `search_tools` mechanism for lazy MCP tool discovery.

**Recommendations**:
1. Implement lazy tool schema loading — only include full schemas for tools the agent is likely to use based on the task
2. Add a `search_tools` tool for on-demand discovery of available tools (especially useful with packs and MCP)
3. Defer model client initialization until the first LLM call
4. Implement deterministic index maintenance outside the agent loop for any cached state

---

### 15. Production Readiness
**Rating: Partial**

| Aspect | Paper Recommendation | Swarmee River Status |
|--------|---------------------|---------------------|
| Atomicity | Atomic session writes | Standard file writes (no atomic rename) |
| Observability | Full event logging | JSONL logger + diagnostics + cost tracking |
| Reversibility | Git-snapshot undo | File diff review hook; no formal undo system |
| Resilience | Fallback chains, timeout enforcement | Error classification + retry; stall monitoring |
| Extensibility | Plugin architecture | Hooks + packs + profiles |

**Strengths**: Observability is production-grade. Error classification with automatic retry/escalation is robust. The stall monitor provides resilience against hung operations.

**Gaps**: No atomic file operations for session persistence. No formal undo capability. Timeout enforcement exists for stall detection but not as explicit per-tool timeouts.

**Recommendations**:
1. Implement atomic session writes (write to `.tmp`, then `os.rename()`)
2. Add per-tool execution timeouts (configurable in settings)
3. Implement a formal undo system using git snapshots before mutating operations
4. Add health check endpoints for the runtime service daemon

---

## Priority Recommendations

### High Priority (significant impact, moderate effort)

1. **Event-driven instruction reminders** — Implement behavioral event detectors that inject targeted system reminders when the agent shows signs of drift, repeated errors, or safety violations. This addresses the paper's core concern about instruction fade-out in long sessions.

2. **Dangerous pattern blocklist** — Add a `DANGEROUS_PATTERNS` list for shell commands to block destructive operations (`rm -rf /`, `DROP TABLE`, etc.) regardless of user consent settings.

3. **General doom-loop detection** — Extend the plan-mode loop detection to the main execution loop. Detect when the agent is repeating identical tool calls without progress.

4. **Atomic session writes** — Prevent session corruption by writing to temp files and using atomic rename.

### Medium Priority (meaningful improvement, moderate effort)

5. **Per-phase model routing** — Allow separate model configuration for compaction, thinking, and normal execution. Use cheaper models for summarization.

6. **Lazy tool schema loading** — Only include schemas for tools relevant to the current task/agent, reducing prompt size and improving token efficiency.

7. **Git-snapshot undo** — Capture repo state before mutating tool calls to enable operation-level undo.

8. **Context budget visibility** — Show users their current token budget utilization during long sessions.

### Lower Priority (nice-to-have, higher effort)

9. **LSP integration** — Add semantic code analysis tools for cross-file navigation and refactoring.

10. **Cross-provider fallback chains** — Enable automatic failover between model providers.

11. **Self-critique phase** — Add an optional pre-tool-call validation step where the model reviews its planned action.

12. **Session analytics** — Build aggregation and analysis over historical session data for learning and optimization.

---

## Detailed Theme Assessments

For deeper analysis of specific themes, see the companion documents:

- [Safety Architecture Deep Dive](opendev-safety-deep-dive.md)
- [Context Engineering Deep Dive](opendev-context-deep-dive.md)
