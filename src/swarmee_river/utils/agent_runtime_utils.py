from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def plan_json_for_execution(plan: Any) -> str:
    payload = plan.model_dump(exclude={"confirmation_prompt"})
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_plan_text(plan: Any) -> str:
    lines: list[str] = ["\nProposed plan:", f"- Summary: {plan.summary}"]
    if plan.assumptions:
        lines.append("- Assumptions:")
        lines.extend([f"  - {a}" for a in plan.assumptions])
    if plan.questions:
        lines.append("- Questions:")
        lines.extend([f"  - {q}" for q in plan.questions])
    if plan.steps:
        lines.append("- Steps:")
        for i, step in enumerate(plan.steps, start=1):
            lines.append(f"  {i}. {step.description}")
            if step.files_to_read:
                lines.append(f"     - read: {', '.join(step.files_to_read)}")
            if step.files_to_edit:
                lines.append(f"     - edit: {', '.join(step.files_to_edit)}")
            if step.tools_expected:
                lines.append(f"     - tools: {', '.join(step.tools_expected)}")
            if step.commands_expected:
                lines.append(f"     - cmds: {', '.join(step.commands_expected)}")
            if step.risks:
                lines.append(f"     - risks: {', '.join(step.risks)}")
    return "\n".join(lines).strip()


def build_base_system_prompt(
    *,
    raw_system_prompt: str,
    runtime_environment_prompt_section: str | None,
    pack_prompt_sections: list[str],
    tool_usage_rules: str,
    system_reminder_rules: str,
) -> str:
    base_prompt_parts: list[str] = [raw_system_prompt, tool_usage_rules, system_reminder_rules]
    if runtime_environment_prompt_section:
        base_prompt_parts.append(runtime_environment_prompt_section)
    if pack_prompt_sections:
        base_prompt_parts.extend(pack_prompt_sections)
    return "\n\n".join([p for p in base_prompt_parts if p]).strip()


def resolve_effective_sop_paths(*, cli_sop_paths: str | None, pack_sop_paths: list[Path]) -> str | None:
    effective_sop_paths: str | None = cli_sop_paths
    if pack_sop_paths:
        pack_paths_str = os.pathsep.join(str(p) for p in pack_sop_paths)
        effective_sop_paths = (
            pack_paths_str if not effective_sop_paths else os.pathsep.join([effective_sop_paths, pack_paths_str])
        )
    return effective_sop_paths
