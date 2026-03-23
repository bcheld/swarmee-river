# OpenDev Safety Architecture: Deep Dive Assessment

Companion to [opendev-assessment.md](opendev-assessment.md). Focuses on the defense-in-depth
safety model described in the OpenDev paper and how Swarmee River implements each layer.

## The Five-Layer Model

The paper describes five independent safety layers, each addressing a different abstraction level.
A failure in any single layer should not compromise overall safety.

### Layer 1: Prompt-Level Guardrails

**Paper**: Security policies, action safety rules, read-before-edit patterns, git workflow guidance,
error recovery instructions baked into the system prompt.

**Swarmee River Implementation**:
- **Profile-based system prompts** (`profiles/models.py`): Named configurations with `prompt_refs` pointing to prompt assets
- **SOPs** (Standard Operating Procedures): Injected as `<system-reminder>` blocks via prompt cache
- **Pack prompts**: Packs bundle domain-specific prompt sections with tools

**Assessment**: Well-implemented. The prompt composition system is flexible and supports
conditional inclusion. However, safety-specific prompt content is mixed with general
instructions — there's no dedicated "safety prompt" layer that's always present regardless
of profile.

**Recommendation**: Create a base safety prompt asset that is always included, independent of
profile selection. This ensures safety guidance is never accidentally omitted when switching
profiles.

---

### Layer 2: Schema-Level Tool Restrictions

**Paper**: Plan-mode whitelists, per-subagent `allowed_tools` lists, MCP discovery gating.

**Swarmee River Implementation**:
- **Tool permissions** (`tool_permissions.py`): Each tool declares `read`, `write`, or `execute` permissions
- **Plan-mode allowlist** (`hooks/tool_policy.py`): Only `read`-permission tools in plan mode, with specific shell command whitelisting
- **Per-agent tool lists**: Profiles define `tool_names` allowlists
- **Tier profiles**: `harness.tier_profiles` provide tier-specific tool allowlists/blocklists
- **Environment overrides**: `SWARMEE_ENABLE_TOOLS` / `SWARMEE_DISABLE_TOOLS`

**Assessment**: Strong implementation with multiple levels of schema restriction. The plan-mode
shell command whitelist (`ls`, `grep`, `git show`, etc.) is a practical addition not in the paper.

**Gap**: Tool schemas are not filtered at build time — all tools are included in the prompt and
then restricted at runtime via hooks. This means the model sees tools it cannot use, wasting
context tokens and potentially causing confusion.

**Recommendation**: Filter tool schemas at agent build time so the model only sees tools it's
actually allowed to use in the current mode. This reduces prompt size and eliminates invalid
tool call attempts.

---

### Layer 3: Runtime Approval System

**Paper**: Manual/Semi-Auto/Auto approval levels, pattern/command/prefix/danger rules, persistent
permission caching.

**Swarmee River Implementation**:
- **Tool consent hook** (`hooks/tool_consent.py`):
  - High-risk tool set: `shell`, `editor`, `patch_apply`, `http_request`
  - Interactive approval with diff preview for mutating tools
  - Session-scoped "remember" flags
  - Permission rules with `when` conditions (command_glob, path_regex, host_glob)
  - Actions: `allow`, `ask`, `deny`
- **Safety settings**: `safety.tool_consent` configurable per-project
- **Non-interactive fallback**: Blocks high-risk tools when no interactive session

**Assessment**: The diff preview before editor/patch_apply approval is a strong feature not
explicitly covered in the paper. Permission rules with glob/regex conditions are flexible.

**Gaps**:
1. No approval fatigue monitoring — no tracking of how often the user is prompted
2. No pattern-based auto-approval learning (e.g., "always allow `git status`")
3. Session-scoped memory only — approvals don't persist across sessions
4. No Semi-Auto mode that auto-approves safe patterns while prompting for novel ones

**Recommendations**:
1. Track approval prompt frequency per session and warn if > N prompts/minute
2. Implement pattern-based auto-approval: after N consecutive approvals of the same pattern, offer to auto-approve
3. Consider cross-session approval persistence for explicitly whitelisted patterns
4. Add a Semi-Auto mode that auto-approves tools matching known-safe patterns

---

### Layer 4: Tool-Level Validation

**Paper**: `DANGEROUS_PATTERNS` blocklists, stale-read detection, output truncation, execution timeouts.

**Swarmee River Implementation**:
- **Tool result limiter** (`hooks/tool_result_limiter.py`): Truncates results to 8KB default
- **Tool message repair** (`hooks/tool_message_repair.py`): Fixes orphaned tool-use/result pairs
- **File diff review** (`hooks/file_diff_review.py`): Pre/post snapshot comparison
- **Max tokens retry** (`hooks/max_tokens_retry.py`): Auto-retry with increased budget

**Assessment**: Output truncation and message repair are well-implemented. The diff review
hook provides good visibility into mutations.

**Gaps**:
1. **No dangerous pattern blocklist** — no static list of blocked shell commands
2. **No stale-read detection** — the model can edit a file it hasn't recently read
3. **No per-tool execution timeouts** — stall detection exists but isn't per-tool
4. **No input validation** — tool inputs aren't validated beyond Strands schema checking

**Recommendations**:
1. Add a `DANGEROUS_PATTERNS` blocklist:
   ```python
   DANGEROUS_PATTERNS = [
       r"rm\s+(-rf?|--recursive)\s+/",  # recursive delete from root
       r"mkfs\.",                         # format filesystem
       r"dd\s+.*of=/dev/",              # raw disk write
       r":\(\)\{.*\|.*&\s*\};:",        # fork bomb
       r"DROP\s+(DATABASE|TABLE|SCHEMA)", # destructive SQL
       r">\s*/dev/sd",                   # overwrite disk
   ]
   ```
2. Implement stale-read detection: track which files the model has read and warn if editing an unread file
3. Add configurable per-tool timeouts in settings
4. Validate shell command inputs against the blocklist before execution

---

### Layer 5: Lifecycle Hooks

**Paper**: User-defined pre-tool blocking (exit code 2), argument mutation, JSON stdin protocol.

**Swarmee River Implementation**:
- **11 hook modules** in `hooks/`:
  - `tool_policy.py` — allowlist/blocklist enforcement
  - `tool_consent.py` — interactive approval
  - `tool_result_limiter.py` — output truncation
  - `tool_message_repair.py` — conversation integrity
  - `file_diff_review.py` — mutation tracking
  - `max_tokens_retry.py` — automatic retry
  - `jsonl_logger.py` — event logging
  - `tui_metrics.py` — cost/usage tracking
  - Plus hooks for notebook truncation, stall detection, etc.

**Assessment**: The hook system is extensive and well-architected. Each hook module is focused
on a single concern. The `BeforeToolCallEvent` / `AfterToolCallEvent` pattern provides clean
interception points.

**Gap**: No user-extensible hook system — hooks are all internal. Users can't add custom pre-tool
blocking or argument mutation without modifying source code.

**Recommendation**: Expose a user-extensible hook API where users can register custom hooks via
configuration (e.g., `.swarmee/hooks/` directory with Python files or shell scripts that receive
JSON stdin and return modified arguments or exit code 2 to block).

---

## Cross-Layer Analysis

### Independence of Failure Domains

The paper emphasizes that safety layers should have independent failure domains. In Swarmee River:

| Layer Failure | Other Layers Still Protect? |
|--------------|---------------------------|
| System prompt omitted | Schema restrictions + consent + validation still enforce |
| Tool permissions misconfigured | Consent hook still prompts for high-risk tools |
| Consent skipped (non-interactive) | Tool policy hook blocks non-allowed tools |
| Result limiter disabled | Message repair + diff review still function |
| Hook system bypassed | Strands SDK level validation remains |

**Assessment**: Layers are reasonably independent. However, the consent hook and tool policy hook
share configuration state (`safety` settings), meaning a misconfiguration could affect both layers
simultaneously.

**Recommendation**: Ensure that the high-risk tool set in `tool_consent.py` is hardcoded (not
configurable) as a last-resort safety net, even if all other policy rules are misconfigured.

---

## Summary

| Layer | Rating | Key Gap |
|-------|--------|---------|
| 1. Prompt-Level | Strong | No mandatory safety prompt |
| 2. Schema-Level | Strong | Schemas not filtered at build time |
| 3. Runtime Approval | Partial | No fatigue monitoring or pattern learning |
| 4. Tool Validation | Partial | No dangerous pattern blocklist |
| 5. Lifecycle Hooks | Strong | Not user-extensible |
| **Cross-Layer** | Strong | Shared configuration between layers 2-3 |
