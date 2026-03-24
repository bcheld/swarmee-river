from __future__ import annotations

import json
import re
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PlanPrecondition(str, Enum):
    CONVERSATION_CONTEXT = "conversation_context"


class PlanStep(BaseModel):
    description: str = Field(..., description="What will be done in this step.")
    files_to_read: list[str] = Field(default_factory=list, description="Files expected to be read (paths or globs).")
    files_to_edit: list[str] = Field(default_factory=list, description="Files expected to be edited (paths or globs).")
    tools_expected: list[str] = Field(default_factory=list, description="Tools expected to be used.")
    commands_expected: list[str] = Field(default_factory=list, description="Shell commands expected to be run.")
    risks: list[str] = Field(default_factory=list, description="Risks and mitigations.")

    @field_validator(
        "files_to_read",
        "files_to_edit",
        "tools_expected",
        "commands_expected",
        "risks",
        mode="before",
    )
    @classmethod
    def _coerce_none_to_list(cls, v: Any) -> Any:
        return v if v is not None else []


class WorkPlan(BaseModel):
    kind: Literal["work_plan"] = "work_plan"
    summary: str = Field(..., description="Short summary of the intended changes.")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions being made.")
    questions: list[str] = Field(default_factory=list, description="Questions to ask the user before executing.")
    preconditions: list[PlanPrecondition] = Field(
        default_factory=list,
        description="Structured prerequisites that must still hold at execution time.",
    )
    steps: list[PlanStep] = Field(default_factory=list, description="Ordered execution steps.")
    confirmation_prompt: str = Field(
        default="Approve this plan? Type :y to execute, :n to cancel, or :replan to regenerate.",
        description="Prompt shown to the user to confirm execution.",
    )


    @field_validator("assumptions", "questions", "preconditions", "steps", mode="before")
    @classmethod
    def _coerce_none_lists(cls, v: Any) -> Any:
        return v if v is not None else []


class PlanningContextSnapshot(BaseModel):
    parent_prefix_hash: str = Field(default="", description="Planning-time shared-prefix hash.")
    parent_message_count: int = Field(default=0, ge=0, description="Planning-time parent message count.")


class PlanningToolSummary(BaseModel):
    tool: str = Field(..., description="Tool name used during planning.")
    count: int = Field(default=0, ge=0, description="How many times the tool was used during planning.")
    last_input_descriptor: str = Field(default="", description="Short descriptor of the last tool input.")
    result_summary: str = Field(default="", description="Short digest of the most recent result.")


class PlanningContext(BaseModel):
    version: Literal[1] = 1
    findings_digest: str = Field(default="", description="Concise digest of planning-time findings.")
    reasoning_digest: str = Field(
        default="",
        description="Evidence-backed digest of planning conclusions without raw chain-of-thought.",
    )
    used_tools: list[str] = Field(default_factory=list, description="Unique tools used during planning.")
    fallback_tools: list[str] = Field(
        default_factory=list,
        description="Safe planner-used tools that execution may reuse without replanning.",
    )
    tool_summaries: list[PlanningToolSummary] = Field(
        default_factory=list,
        description="Per-tool planning activity summaries.",
    )
    context_snapshot: PlanningContextSnapshot | None = Field(
        default=None,
        description="Planning-time parent conversation snapshot used for execute-time validation.",
    )
    artifact_id: str | None = Field(default=None, description="Artifact id for the persisted planning-context record.")
    artifact_path: str | None = Field(
        default=None,
        description="Artifact path for the persisted planning-context record.",
    )

    @field_validator("used_tools", "fallback_tools", "tool_summaries", mode="before")
    @classmethod
    def _coerce_context_lists(cls, v: Any) -> Any:
        return v if v is not None else []


class PendingWorkPlan(BaseModel):
    kind: Literal["pending_work_plan"] = "pending_work_plan"
    plan_run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    original_request: str = Field(..., description="Original request that the plan is answering.")
    current_plan: WorkPlan = Field(..., description="Current canonical WorkPlan under review.")
    revision_count: int = Field(default=0, ge=0, description="How many plan revisions have been generated.")
    feedback_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured step/question feedback applied while revising the plan.",
    )
    planning_context: PlanningContext | None = Field(
        default=None,
        description="Non-user-facing planning-time findings carried into execute mode.",
    )

    @property
    def summary(self) -> str:
        return self.current_plan.summary

    @property
    def assumptions(self) -> list[str]:
        return list(self.current_plan.assumptions)

    @property
    def questions(self) -> list[str]:
        return list(self.current_plan.questions)

    @property
    def steps(self) -> list[PlanStep]:
        return list(self.current_plan.steps)

    @property
    def confirmation_prompt(self) -> str:
        return self.current_plan.confirmation_prompt


def ensure_work_plan(plan: Any) -> WorkPlan:
    if isinstance(plan, WorkPlan):
        return plan
    if isinstance(plan, PendingWorkPlan):
        return plan.current_plan
    if isinstance(plan, dict):
        if "current_plan" in plan:
            pending = pending_work_plan_from_payload(plan)
            if pending is not None:
                return pending.current_plan
        return WorkPlan.model_validate(plan)
    raise TypeError("Expected WorkPlan-compatible payload")


def pending_work_plan_from_payload(payload: Any) -> PendingWorkPlan | None:
    if isinstance(payload, PendingWorkPlan):
        return payload
    if not isinstance(payload, dict):
        return None
    try:
        return PendingWorkPlan.model_validate(payload)
    except Exception:
        return None


def new_pending_work_plan(
    *,
    original_request: str,
    plan: WorkPlan,
    revision_count: int = 0,
    feedback_history: list[dict[str, Any]] | None = None,
    plan_run_id: str | None = None,
    planning_context: PlanningContext | dict[str, Any] | None = None,
) -> PendingWorkPlan:
    return PendingWorkPlan(
        plan_run_id=str(plan_run_id or "").strip() or uuid.uuid4().hex,
        original_request=str(original_request or "").strip(),
        current_plan=ensure_work_plan(plan),
        revision_count=max(0, int(revision_count or 0)),
        feedback_history=[dict(item) for item in (feedback_history or []) if isinstance(item, dict)],
        planning_context=(
            planning_context
            if isinstance(planning_context, PlanningContext)
            else (PlanningContext.model_validate(planning_context) if isinstance(planning_context, dict) else None)
        ),
    )


def build_plan_revision_prompt(
    pending: PendingWorkPlan,
    *,
    step_feedback: list[str],
    question_feedback: list[str],
) -> str:
    lines = [
        "Revise the WorkPlan for the original request below.",
        "",
        "Original request:",
        pending.original_request.strip(),
        "",
        "Current WorkPlan JSON:",
        json.dumps(pending.current_plan.model_dump(), indent=2, ensure_ascii=False),
    ]
    if step_feedback:
        lines.extend(["", "Step feedback:"])
        lines.extend(step_feedback)
    if question_feedback:
        lines.extend(["", "Question responses:"])
        lines.extend(question_feedback)
    lines.extend(
        [
            "",
            "Return a complete replacement WorkPlan.",
            "Stop exploring once you have enough context to produce the revised plan.",
            "Do not restate the old plan outside the WorkPlan response.",
        ]
    )
    return "\n".join(lines).strip()


_WORK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(implement|fix|refactor|rewrite|add|remove|delete|rename|update|upgrade)\b", re.IGNORECASE),
    re.compile(r"\b(create|generate|scaffold|bootstrap)\b", re.IGNORECASE),
    re.compile(r"\b(run|rerun|execute)\b.*\b(tests?|pytest|mypy|ruff|lint|format|build)\b", re.IGNORECASE),
    re.compile(r"\b(debug|reproduce|investigate)\b", re.IGNORECASE),
    re.compile(r"\b(open|create)\b.*\b(pr|pull request)\b", re.IGNORECASE),
    re.compile(r"\b(edit|change)\b.*\b(file|code|repo|repository)\b", re.IGNORECASE),
]


def classify_intent(prompt: str) -> str:
    """
    Best-effort heuristic classifier for interactive UX decisions.

    Returns:
        "work" if the prompt likely requests code changes or execution,
        otherwise "info".
    """
    text = (prompt or "").strip()
    if not text:
        return "info"

    for pattern in _WORK_PATTERNS:
        if pattern.search(text):
            return "work"

    return "info"


def structured_plan_prompt() -> str:
    return (
        "You are Swarmee River in PLAN mode.\n"
        "- Your job is to produce a WorkPlan, not to solve or draft the final answer.\n"
        "- You may use safe read-only tools to gather context before planning: "
        "file_read, notebook_read, file_search, file_list, glob, grep, retrieve, project_context, "
        "athena_query, conservative read-only shell commands, and single-expression "
        "read-only python_repl queries.\n"
        "- Stop exploring as soon as you have enough context to emit a concrete WorkPlan.\n"
        "- Do NOT produce any text output. Your final response MUST be a single "
        "WorkPlan tool call with no preceding text.\n"
        "- Do NOT draft a solution, implementation, or answer in assistant text before the WorkPlan.\n"
        "- Put all analysis, reasoning, and recommendations into the WorkPlan fields "
        "(summary, assumptions, steps).\n"
        "- Use WorkPlan.preconditions for execution-time requirements. Add "
        "`conversation_context` when the plan depends on planning-time conversation state or "
        "tool findings that must still be available during execution.\n"
        "- If you cannot produce a valid WorkPlan, ask focused questions in WorkPlan.questions.\n"
        "- Produce a concrete, minimally sufficient plan for the user's request.\n"
        "- If important details are missing, include them as questions (keep them specific).\n"
        "- Steps should reference likely files/tools/commands.\n"
        "- Keep the plan short: 3\u20138 steps unless strictly necessary.\n"
    )


class PlanExecutionReplanRequired(RuntimeError):
    def __init__(self, *, pending_plan: PendingWorkPlan, warning: str) -> None:
        super().__init__(warning)
        self.pending_plan = pending_plan
        self.warning = warning
