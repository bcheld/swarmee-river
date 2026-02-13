from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    description: str = Field(..., description="What will be done in this step.")
    files_to_read: list[str] = Field(default_factory=list, description="Files expected to be read (paths or globs).")
    files_to_edit: list[str] = Field(default_factory=list, description="Files expected to be edited (paths or globs).")
    tools_expected: list[str] = Field(default_factory=list, description="Tools expected to be used.")
    commands_expected: list[str] = Field(default_factory=list, description="Shell commands expected to be run.")
    risks: list[str] = Field(default_factory=list, description="Risks and mitigations.")


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


def structured_plan_prompt() -> str:
    return (
        "You are Swarmee River in PLAN mode.\n"
        "- Do NOT execute tools.\n"
        "- Produce a concrete, minimally sufficient plan for the user's request.\n"
        "- If important details are missing, include them as questions (keep them specific).\n"
        "- Steps should reference likely files/tools/commands.\n"
        "- Keep the plan short: 3–8 steps unless strictly necessary.\n"
    )

