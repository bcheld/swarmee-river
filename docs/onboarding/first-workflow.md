# Your First Agentic Workflow: A Code Review Agent

This tutorial walks you through building a complete agentic workflow from scratch. By the end you will have:

- A **Standard Operating Procedure** (SOP) that guides the agent through a structured code review
- An **agent profile** that bundles the SOP with the right tools
- A working **session** you can run and iterate on

No code changes required — just config and a Markdown file.

---

## What We're Building

A **Code Review Agent** that:
1. Reads the git diff of recent changes
2. Checks each changed file against a structured review checklist (the SOP)
3. Posts findings in a structured format in the transcript

The agent will use the built-in `git`, `file_read`, and `file_list` tools — no custom tools needed for this walkthrough.

---

## Step 1: Design the Workflow

Before writing anything, think through what the agent needs:

| Need | Solution |
|------|----------|
| Read changed files | `git` tool (reads diff) + `file_read` tool |
| Structured review process | SOP with numbered checklist |
| Consistent output format | SOP specifies the output template |
| Scope the agent's tool access | Agent profile with explicit `tool_names` |

The **SOP** defines *how* to do the review. The **agent profile** binds the SOP to the right tools and system prompt.

---

## Step 2: Write the SOP

Create a file at `sops/code-review.sop.md`:

```markdown
---
name: code-review
version: 0.1.0
description: Structured code review — read diff, check each file, report findings.
---

# SOP: Code Review

## Inputs
- The git diff of recent changes (use the `git` tool with `git diff HEAD~1`)
- Or a specific commit or branch range provided by the user

## Review Checklist
For each changed file, evaluate:

1. **Correctness** — Does the logic match the stated intent? Any obvious bugs?
2. **Edge cases** — Are inputs validated? What happens on empty, null, or large inputs?
3. **Tests** — Are relevant tests added or updated?
4. **Security** — Any injection risks, hardcoded credentials, or unsafe operations?
5. **Readability** — Are names clear? Is complex logic commented?

## Output Format
Produce a structured review with this template for each file:

```
### <filename>

**Summary:** One-sentence description of the change.

**Findings:**
- [CRITICAL] Description (if any)
- [WARNING] Description (if any)
- [INFO] Description (if any)

**Verdict:** APPROVE / REQUEST_CHANGES / NEEDS_INFO
```

## Rules
- Use CRITICAL for bugs or security issues that must be fixed before merge.
- Use WARNING for issues that should be addressed but are not blocking.
- Use INFO for style suggestions or non-blocking observations.
- If a file has no findings, still write the header and a one-line summary.
- Always end with an overall verdict: APPROVE, REQUEST_CHANGES, or NEEDS_INFO.
```

---

## Step 3: Create the Agent Profile

You can create the profile either via the TUI or directly in `settings.json`.

### Via TUI

1. Run `swarmee tui`
2. Navigate to the **Agents** tab
3. Click **New Agent** (or press `N`)
4. Fill in:
   - **ID:** `code-reviewer`
   - **Name:** `Code Reviewer`
   - **System prompt snippet:** `You are a careful code reviewer. Follow the code-review SOP precisely. Be specific and actionable in your findings.`
5. Under **SOPs**, check `code-review`
6. Under **Tools**, enter: `git, file_read, file_list, file_search`
7. Click **Save**

### Via settings.json

Add this entry to the `agents` array in `.swarmee/settings.json`:

```json
{
  "id": "code-reviewer",
  "name": "Code Reviewer",
  "prompt": "You are a careful code reviewer. Follow the code-review SOP precisely. Be specific and actionable in your findings.",
  "sop_names": ["code-review"],
  "tool_names": ["git", "file_read", "file_list", "file_search"]
}
```

---

## Step 4: Activate the SOP

Before running a session, the SOP needs to be active. There are three ways:

**TUI:** Tooling tab > SOPs > check the `code-review` row

**REPL:**
```
> :sop enable code-review
```

**Profile** (already done above via `sop_names`): the SOP activates automatically when the profile is loaded.

---

## Step 5: Run the Session

```bash
swarmee tui
# or
swarmee
```

Send a prompt:
```
Review the changes in the last commit.
```

### What happens next

1. The agent reads the system prompt (which now includes your SOP content)
2. It generates a **plan** — a list of tool calls it intends to make
3. The plan appears in the **Run tab** (TUI) or inline in the REPL
4. Review the plan and **approve** it:
   - TUI: click the **Approve** button, or press `A`
   - REPL: type `:approve`
5. The agent executes: calls `git` to read the diff, `file_read` for any files it wants to inspect, then produces the structured review
6. The review appears in the **Transcript**

---

## Step 6: Iterate

Try these variations to see how profile + SOP changes affect output:

**Tighten the tool set** — edit the profile to only allow `git`:
```json
"tool_names": ["git"]
```
The agent will still read the diff but won't be able to read individual files. Notice how it adapts.

**Add a focus** — prepend to your prompt:
```
Focus only on security findings. Review the last 3 commits.
```

**Edit the SOP** — add a step to the checklist in `sops/code-review.sop.md`, then run the session again. Changes take effect immediately (SOP content is re-read on each session start).

---

## Step 7: Save the Session

To return to this conversation later:

```
> :session save code-review-demo
```

To restore it:

```bash
swarmee --session code-review-demo
# or in the REPL
> :session load code-review-demo
```

To list saved sessions:

```
> :session list
```

---

## What You Learned

- **SOPs** define structured workflows in plain Markdown — no code required
- **Agent profiles** bind SOPs + tools + a system prompt snippet into a reusable configuration
- The **plan/approve/execute** cycle gives you visibility and control before anything runs
- **SOP edits** take effect immediately without restarting

---

## Next Steps

- [custom-tools.md](custom-tools.md) — extend the agent with a custom tool (e.g., post the review to a GitHub PR)
- [sops-and-profiles.md](sops-and-profiles.md) — full reference for SOP format and profile configuration
- [delegation.md](delegation.md) — run multiple agents in parallel (e.g., separate investigator and fix-proposer agents)
