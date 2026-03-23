# OpenDev Context Engineering: Deep Dive Assessment

Companion to [opendev-assessment.md](opendev-assessment.md). Focuses on the context engineering
patterns described in the OpenDev paper and how Swarmee River manages context as a finite resource.

## The Paper's Core Thesis

> Context pressure is the "central design constraint" for long-horizon agents.

The paper argues that context window management is not a secondary concern but the primary
architectural constraint. Systems must actively manage context rather than hoping models will
handle it implicitly.

---

## 1. Dynamic System Prompt Construction

**Paper**: Priority-ordered conditional composition that loads sections only when contextually
relevant. Provider-specific conditional sections. Mode-specific variants. Provider-level prompt
caching. Two-tier fallback.

**Swarmee River Implementation**:

```
Profile → prompt_refs → PromptAsset resolution → base prompt
  + RuntimeEnv snapshot → environment context
  + Pack prompts → domain-specific sections
  + SOPs → operational procedures
  + PromptCacheState → SHA256-based change detection → <system-reminder> injection
```

**Strengths**:
- **Composable prompt assets**: The `prompt_refs` system allows profiles to select exactly which
  prompt sections to include
- **SHA256-based caching**: The `PromptCacheState` tracks content hashes to detect when updates
  are needed, enabling provider-level prompt caching (Bedrock `cache_tools`)
- **Mode-specific content**: Plan mode injects different prompt sections than execute mode
- **Dynamic reminders**: `<system-reminder>` blocks injected into user message prefix only when
  content changes, keeping the API-level system prompt stable for caching

**Gaps**:
- No priority ordering — all referenced prompt assets are included equally
- No conditional inclusion based on task context (e.g., include git guidance only when git tools
  are active)
- No explicit two-tier fallback (essential prompts vs. nice-to-have prompts that can be dropped
  under pressure)

**Recommendations**:
1. Add priority levels to prompt assets (critical, standard, optional) so compaction can drop
   optional sections first
2. Implement task-aware prompt conditioning — only include prompts relevant to the active tool set
3. Add a prompt budget with graceful degradation: include all prompts if budget allows, drop
   optional prompts if tight, warn if even critical prompts are at risk

---

## 2. Tool Result Optimization

**Paper**: Per-tool-type summarization, large output offloading, agent-aware truncation hints,
interaction with compaction to prevent double-processing.

**Swarmee River Implementation**:

- **Artifact store** (`artifacts.py`): Large tool results offloaded to `.swarmee/artifacts/`
  with indexed metadata
- **Tool result limiter** (`hooks/tool_result_limiter.py`): Truncates results at 8KB with
  metadata preservation
- **Compaction awareness**: `BudgetedSummarizingConversationManager` handles tool result
  compaction with strategy-specific behavior:
  - `balanced`: Compact all old tool results
  - `cache_safe`: Keep 2 most recent file_read/search results raw
  - `long_running`: Keep 4 most recent results raw

**Strengths**:
- The artifact store is a clean implementation of the paper's "offloading" pattern
- Strategy-specific compaction behavior is sophisticated and practical
- Result limiter prevents context explosion from verbose tools

**Gaps**:
- No per-tool-type summarization — all tools get the same truncation treatment
- No agent-aware truncation hints — the truncation doesn't understand what information the
  model most needs from each tool type
- No explicit interaction tracking between limiter and compaction (risk of double-processing)

**Recommendations**:
1. Implement per-tool summarization strategies (e.g., for `file_read`: keep first/last N lines;
   for `shell`: keep exit code + last N lines; for `file_search`: keep file list, drop content)
2. Add truncation hints to tool definitions — let tools declare what should be preserved when
   their output is truncated
3. Mark limiter-truncated results so the compaction system knows not to re-summarize them

---

## 3. Dual-Memory Architecture

**Paper**: Episodic memory (full conversation history) for long-term continuity combined with
working memory (recent observations) for immediate decision-making. Combined injection balances
context constraints with reasoning capability.

**Swarmee River Implementation**:

- **Episodic memory**: Full message history in `SessionStore` (messages.json)
- **Working memory**: Recent messages in the active conversation buffer
- **Compaction bridge**: `BudgetedSummarizingConversationManager` decides what to keep verbatim
  (working memory) vs. what to summarize (older episodic memory)
- **Knowledge base**: `store_in_kb` / `retrieve` tools for persistent knowledge storage (Bedrock KB)
- **Prompt assets**: Reusable prompt sections that persist across sessions

**Strengths**:
- Clean separation between full history (session store) and active context (conversation manager)
- The compaction strategies explicitly manage the episodic → working memory transition
- Knowledge base integration provides external episodic storage beyond session scope

**Gaps**:
- No explicit working memory buffer — the "working memory" is implicitly the most recent N
  messages that haven't been compacted
- No structured extraction of key facts/decisions from episodic memory into a summary buffer
- Knowledge base is optional and provider-specific (Bedrock only)

**Recommendations**:
1. Implement an explicit working memory buffer that extracts key facts, decisions, and constraints
   from the conversation into a structured summary
2. Update the working memory buffer on each compaction cycle — don't just truncate old messages,
   extract their key insights first
3. Make knowledge base integration provider-agnostic for cross-provider deployments

---

## 4. Context-Aware System Reminders

**Paper**: Event detectors trigger reminders at decision points. Template resolution adapts
reminder text to current state. Guardrail counters track safety metric violations. Graceful
degradation when reminder tokens exceed budget.

**Swarmee River Implementation**:

- **Prompt cache reminders**: `PromptCacheState` tracks changes to SOPs, project context, plan
  progress, and injects `<system-reminder>` blocks when content changes
- **Plan-mode loop detection**: Counts blocked tool attempts (3 max) and identical call repeats
  (4 max) — acts as a guardrail counter
- **Mode-specific reminders**: Different content injected in plan vs. execute mode

**Strengths**:
- The `<system-reminder>` injection pattern keeps the system prompt stable for caching while
  still providing dynamic guidance
- Loop detection is an effective guardrail counter for plan mode

**Gaps**:
- Reminders are triggered by state changes, not behavioral events — the system doesn't detect
  "the model just attempted the same failed tool call 3 times" outside plan mode
- No reminder budget with graceful degradation — if too many reminders accumulate, they all get
  injected
- No event detector framework — each detection is ad-hoc rather than pluggable

**Recommendations**:
1. Build an event detector framework:
   ```python
   class EventDetector(Protocol):
       def check(self, event: AgentEvent) -> Optional[Reminder]: ...

   # Example detectors:
   RepeatedToolFailureDetector   # 3+ failures of same tool
   GoalDriftDetector             # model discussing unrelated topics
   ApprovalFatigueDetector       # >5 consent prompts in 2 minutes
   TokenBudgetWarningDetector    # <20% budget remaining
   ```
2. Add a reminder token budget — cap total reminder tokens and prioritize by severity
3. Implement turn-interval reminders for long sessions (e.g., re-inject core safety guidance
   every N turns)

---

## 5. Adaptive Context Compaction

**Paper**: Progressive compaction stages as token budget depletes. Self-healing indexes for
consistency. Deterministic operations outside agent loop to prevent divergence.

**Swarmee River Implementation**:

- **Three strategies** in `BudgetedSummarizingConversationManager`:
  - `balanced`: Default compaction with mixed verbatim/summary retention
  - `cache_safe`: Prioritizes prompt cache stability, keeps recent tool results raw
  - `long_running`: Aggressive compaction for extended sessions
- **Shared-prefix compaction**: Uses a text fork of the parent agent to generate summaries,
  reusing the cached prompt prefix
- **Token budget tracking**: Estimates token usage and triggers compaction when budget exceeded
- **Tool result deduplication**: Tracks result frequencies to compact repetitive tool calls

**Strengths**:
- Three strategies provide task-appropriate compaction behavior
- Shared-prefix forking for compaction is efficient (reuses cached prompt)
- Tool result frequency tracking for deduplication is a sophisticated optimization

**Gaps**:
- No progressive stages within a strategy — compaction is binary (compact or don't)
- No self-healing indexes — if compaction produces inconsistent state, no recovery mechanism
- Compaction runs within the agent's conversation manager, not as a deterministic external process

**Recommendations**:
1. Implement progressive compaction stages within each strategy:
   - Stage 1 (75% budget): Compact tool results older than N turns
   - Stage 2 (85% budget): Summarize all non-recent messages
   - Stage 3 (95% budget): Aggressive summarization, keep only working memory
2. Add consistency validation after compaction — verify that referenced artifacts/files still
   have valid pointers
3. Consider running compaction as a deterministic operation outside the agent loop to prevent
   divergence from repeated compaction within the reasoning cycle

---

## 6. Context Retrieval and Assembly Pipeline

**Paper**: Four-layer pipeline:
1. Anchor-based tool selection reduces schema size
2. Multi-step agentic search via Code Explorer
3. Context assembly combines retrieved segments
4. Optimization applies final truncation and summarization

**Swarmee River Implementation**:

- **Layer 1 (tool selection)**: All tool schemas included; no anchor-based filtering
- **Layer 2 (agentic search)**: `project_context` tool provides summary/search/read modes;
  `strand` tool can delegate exploration to child agents
- **Layer 3 (context assembly)**: Prompt cache composes system prompt + reminders + profile
  prompts into a single context
- **Layer 4 (optimization)**: Tool result limiter + compaction provide output-level optimization

**Strengths**:
- `project_context` is an effective single tool for multi-mode codebase exploration
- The `strand` delegation for exploration matches the paper's "agentic search" concept

**Gaps**:
- No anchor-based tool filtering — all tools are always included in the schema
- No explicit context assembly pipeline — composition happens across multiple disconnected systems
- No final optimization pass that considers total context size and makes trade-offs

**Recommendations**:
1. Implement anchor-based tool filtering: analyze the user's prompt to identify relevant tool
   categories and only include those schemas
2. Create a unified context assembly pipeline that receives all context sources (system prompt,
   reminders, tool schemas, conversation history) and makes global optimization decisions
3. Add a final context optimization pass before each LLM call that trims low-priority content
   if total tokens exceed a target threshold

---

## Summary: Context Engineering Maturity

| Component | Rating | Key Strength | Key Gap |
|-----------|--------|-------------|---------|
| Dynamic System Prompt | Strong | SHA256 caching + composable assets | No priority ordering |
| Tool Result Optimization | Strong | Artifact offloading + strategy-aware compaction | No per-tool summarization |
| Dual-Memory Architecture | Partial | Clean episodic/working separation | No structured working memory extraction |
| Context-Aware Reminders | Partial | `<system-reminder>` injection pattern | Not event-driven |
| Adaptive Compaction | Strong | Three strategies + shared-prefix forking | No progressive stages |
| Retrieval Pipeline | Partial | `project_context` multi-mode tool | No anchor-based schema filtering |

**Overall**: Context engineering is one of Swarmee River's stronger areas, with sophisticated
compaction strategies and efficient caching. The main gaps are around event-driven behavioral
guidance and context-aware schema filtering — areas where the system could be more dynamic
and responsive to the specific task at hand.
