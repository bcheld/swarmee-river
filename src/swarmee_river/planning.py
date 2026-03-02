from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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
    steps: list[PlanStep] = Field(default_factory=list, description="Ordered execution steps.")
    confirmation_prompt: str = Field(
        default="Approve this plan? Type :y to execute, :n to cancel, or :replan to regenerate.",
        description="Prompt shown to the user to confirm execution.",
    )


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
        "- You may use read-only tools (file_read, file_search, file_list, glob, grep, "
        "project_context, retrieve) to gather context before planning.\n"
        "- Do NOT produce any text output. Your final response MUST be a single "
        "WorkPlan tool call with no preceding text.\n"
        "- Put all analysis, reasoning, and recommendations into the WorkPlan fields "
        "(summary, assumptions, steps).\n"
        "- If you cannot produce a valid WorkPlan, ask focused questions in WorkPlan.questions.\n"
        "- Produce a concrete, minimally sufficient plan for the user's request.\n"
        "- If important details are missing, include them as questions (keep them specific).\n"
        "- Steps should reference likely files/tools/commands.\n"
        "- Keep the plan short: 3\u20138 steps unless strictly necessary.\n"
    )
