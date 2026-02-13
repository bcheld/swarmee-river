---
name: swarm
version: 0.1.0
description: Safe defaults for multi-agent swarms (roles, handoffs, and tool use).
---

# SOP: Swarm / multi-agent mode

## Goal
Use multiple agents to parallelize thinking while keeping execution safe and auditable.

## Roles (recommended)
- **Planner**: proposes a plan, risks, and open questions (no tools).
- **Researcher**: reads code and gathers evidence (read-only tools).
- **Executor**: performs edits and runs commands (high-risk tools with consent).
- **Reviewer**: checks changes against acceptance criteria.

## Handoff rules
1) Start with a plan and explicit acceptance criteria.
2) Only one agent (Executor) should run high-risk tools (`shell`, `editor`, `file_write`, `http_request`).
3) All agents should write outputs as short, structured notes.
4) If execution deviates from the approved plan, stop and re-plan.

## Safety defaults
- Keep swarm limits conservative (`node_timeout`, `execution_timeout`, `max_handoffs`).
- Prefer local artifacts + logs under `.swarmee/`.

