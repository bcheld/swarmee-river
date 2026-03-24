from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from swarmee_river.artifacts import ArtifactStore, tools_expected_from_plan
from swarmee_river.context.budgeted_summarizing_conversation_manager import (
    _iter_tool_result_records,
    _tool_activity_descriptor,
)
from swarmee_river.planning import (
    PendingWorkPlan,
    PlanningContext,
    PlanningContextSnapshot,
    PlanningToolSummary,
    PlanPrecondition,
    WorkPlan,
    build_plan_revision_prompt,
)
from swarmee_river.tool_permissions import STRANDS_TOOL_PERMISSIONS, get_permissions
from swarmee_river.utils.fork_utils import SharedPrefixForkSnapshot

_REASONING_KEYS = {"reasoningContent", "reasoningText", "reasoning_content", "reasoning_text"}
_AUTO_FALLBACK_TOOL_ALLOWLIST = {"athena_query"}
_AUTO_FALLBACK_TOOL_BLOCKLIST = {"WorkPlan", "plan_progress", "python_repl", "shell"}


def _sanitize_reasoning_payload(value: Any) -> Any | None:
    if isinstance(value, list):
        cleaned_items: list[Any] = []
        for item in value:
            cleaned = _sanitize_reasoning_payload(item)
            if cleaned is None or cleaned == {} or cleaned == []:
                continue
            cleaned_items.append(cleaned)
        return cleaned_items
    if isinstance(value, dict):
        cleaned_dict: dict[str, Any] = {}
        for key, nested in value.items():
            if key in _REASONING_KEYS:
                continue
            cleaned = _sanitize_reasoning_payload(nested)
            if cleaned is None:
                continue
            if cleaned == {}:
                continue
            if cleaned == [] and key != "content":
                continue
            cleaned_dict[key] = cleaned
        return cleaned_dict
    return value


def sanitize_planning_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_messages: list[dict[str, Any]] = []
    for item in messages:
        cleaned = _sanitize_reasoning_payload(item)
        if not isinstance(cleaned, dict):
            continue
        cleaned_messages.append(cleaned)
    return cleaned_messages


def _truncate(text: str, *, limit: int = 240) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(1, limit - 3)].rstrip() + "..."


def _text_items(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _iter_tool_uses(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
    tool_uses: list[tuple[str, dict[str, Any], str]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            tool_use = item.get("toolUse")
            if not isinstance(tool_use, dict):
                continue
            tool_name = str(tool_use.get("name") or "").strip()
            if not tool_name:
                continue
            tool_input = tool_use.get("input")
            tool_use_id = str(tool_use.get("toolUseId") or tool_use.get("id") or "").strip()
            tool_uses.append((tool_name, tool_input if isinstance(tool_input, dict) else {}, tool_use_id))
    return tool_uses


def _tool_result_excerpt(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return _truncate(" ".join(lines[:3]), limit=220)


def _build_reasoning_digest(plan: WorkPlan) -> str:
    parts: list[str] = []
    summary = str(plan.summary or "").strip()
    if summary:
        parts.append(summary)
    assumptions = [str(item).strip() for item in (plan.assumptions or []) if str(item).strip()]
    if assumptions:
        parts.append("Assumptions: " + "; ".join(assumptions[:3]))
    risk_items: list[str] = []
    for step in plan.steps or []:
        for risk in step.risks or []:
            text = str(risk).strip()
            if text:
                risk_items.append(text)
    if risk_items:
        parts.append("Execution cautions: " + "; ".join(risk_items[:2]))
    return _truncate(" ".join(parts), limit=420)


def _build_findings_digest(
    *,
    planning_messages: list[dict[str, Any]],
    tool_result_records: list[Any],
    fallback_text: str,
) -> str:
    findings: list[str] = []
    for record in tool_result_records[-6:]:
        excerpt = _tool_result_excerpt(getattr(record, "text", ""))
        descriptor = _tool_activity_descriptor(getattr(record, "tool_name", ""), getattr(record, "tool_input", {}))
        if excerpt:
            findings.append(f"{record.tool_name} ({descriptor}): {excerpt}")
        else:
            findings.append(f"{record.tool_name} ({descriptor})")
    if findings:
        return "\n".join(f"- {item}" for item in findings[:6])

    assistant_texts: list[str] = []
    for message in planning_messages:
        role = str(message.get("role", "")).strip().lower()
        if role != "assistant":
            continue
        assistant_texts.extend(_text_items(message))
    if assistant_texts:
        return "\n".join(f"- {_truncate(text, limit=220)}" for text in assistant_texts[:4])

    return f"- {_truncate(fallback_text, limit=220)}" if fallback_text.strip() else ""


def _permissions_for_tool(tool_name: str, tools_dict: dict[str, Any] | None) -> frozenset[str]:
    if isinstance(tools_dict, dict):
        tool_obj = tools_dict.get(tool_name)
        if tool_obj is not None:
            declared = get_permissions(tool_obj)
            if declared is not None:
                return declared
    sdk = STRANDS_TOOL_PERMISSIONS.get(tool_name)
    return sdk if sdk is not None else frozenset()


def planner_fallback_tools(*, used_tools: list[str], tools_dict: dict[str, Any] | None = None) -> list[str]:
    fallback: set[str] = set()
    for tool_name in used_tools:
        if tool_name in _AUTO_FALLBACK_TOOL_BLOCKLIST:
            continue
        perms = _permissions_for_tool(tool_name, tools_dict)
        if "write" in perms:
            continue
        if tool_name in _AUTO_FALLBACK_TOOL_ALLOWLIST or ("read" in perms and "write" not in perms):
            fallback.add(tool_name)
    return sorted(fallback)


def build_planning_context(
    *,
    plan: WorkPlan,
    child_messages: list[dict[str, Any]],
    snapshot: SharedPrefixForkSnapshot,
    artifact_store: ArtifactStore,
    plan_run_id: str,
    request: str,
    tools_dict: dict[str, Any] | None = None,
) -> PlanningContext:
    planning_messages = (
        child_messages[snapshot.parent_message_count :]
        if snapshot.parent_message_count > 0
        else child_messages
    )
    sanitized_messages = sanitize_planning_messages(
        planning_messages if isinstance(planning_messages, list) else []
    )
    tool_uses = _iter_tool_uses(sanitized_messages)
    tool_result_records = _iter_tool_result_records(sanitized_messages)

    summaries_by_tool: dict[str, dict[str, Any]] = {}
    for tool_name, tool_input, _tool_use_id in tool_uses:
        entry = summaries_by_tool.setdefault(
            tool_name,
            {"tool": tool_name, "count": 0, "last_input_descriptor": "", "result_summary": ""},
        )
        entry["count"] = int(entry["count"]) + 1
        entry["last_input_descriptor"] = _tool_activity_descriptor(tool_name, tool_input)
    for record in tool_result_records:
        entry = summaries_by_tool.setdefault(
            record.tool_name,
            {"tool": record.tool_name, "count": 0, "last_input_descriptor": "", "result_summary": ""},
        )
        if not entry["last_input_descriptor"]:
            entry["last_input_descriptor"] = _tool_activity_descriptor(record.tool_name, record.tool_input)
        excerpt = _tool_result_excerpt(record.text)
        if excerpt:
            entry["result_summary"] = excerpt

    used_tools = sorted(summaries_by_tool.keys())
    fallback_tools = planner_fallback_tools(used_tools=used_tools, tools_dict=tools_dict)
    tool_summaries = [
        PlanningToolSummary.model_validate(summary)
        for summary in sorted(summaries_by_tool.values(), key=lambda item: str(item["tool"]))
    ]
    findings_digest = _build_findings_digest(
        planning_messages=sanitized_messages,
        tool_result_records=tool_result_records,
        fallback_text=plan.summary,
    )
    reasoning_digest = _build_reasoning_digest(plan)
    context_snapshot = PlanningContextSnapshot(
        parent_prefix_hash=snapshot.prefix_hash,
        parent_message_count=snapshot.parent_message_count,
    )

    artifact_payload = {
        "version": 1,
        "plan_run_id": plan_run_id,
        "request": request,
        "current_plan": plan.model_dump(mode="json"),
        "used_tools": used_tools,
        "fallback_tools": fallback_tools,
        "tool_summaries": [item.model_dump(mode="json") for item in tool_summaries],
        "findings_digest": findings_digest,
        "reasoning_digest": reasoning_digest,
        "context_snapshot": context_snapshot.model_dump(mode="json"),
        "planning_messages": sanitized_messages,
    }
    artifact_ref = artifact_store.write_text(
        kind="planning_context",
        text=json.dumps(artifact_payload, indent=2, ensure_ascii=False, default=str),
        suffix="json",
        metadata={"request": request, "plan_run_id": plan_run_id},
    )

    return PlanningContext(
        findings_digest=findings_digest,
        reasoning_digest=reasoning_digest,
        used_tools=used_tools,
        fallback_tools=fallback_tools,
        tool_summaries=tool_summaries,
        context_snapshot=context_snapshot,
        artifact_id=artifact_ref.artifact_id,
        artifact_path=str(artifact_ref.path),
    )


def maybe_add_conversation_context_precondition(
    plan: WorkPlan,
    *,
    planning_context: PlanningContext | None,
) -> WorkPlan:
    if planning_context is None:
        return plan
    existing = list(plan.preconditions or [])
    if PlanPrecondition.CONVERSATION_CONTEXT in existing:
        return plan
    snapshot = planning_context.context_snapshot
    needs_context = bool(planning_context.used_tools) or bool(snapshot and snapshot.parent_message_count > 0)
    if not needs_context:
        return plan
    return plan.model_copy(update={"preconditions": [*existing, PlanPrecondition.CONVERSATION_CONTEXT]})


def execution_allowed_tools(
    plan: WorkPlan,
    *,
    pending_plan: PendingWorkPlan | None = None,
    tools_dict: dict[str, Any] | None = None,
) -> list[str]:
    allowed = {tool_name for tool_name in tools_expected_from_plan(plan) if tool_name != "WorkPlan"}
    if pending_plan is not None and pending_plan.planning_context is not None:
        allowed.update(
            planner_fallback_tools(
                used_tools=pending_plan.planning_context.used_tools,
                tools_dict=tools_dict,
            )
        )
        allowed.update(str(name).strip() for name in pending_plan.planning_context.fallback_tools if str(name).strip())
    allowed.add("plan_progress")
    return sorted(allowed)


def render_planning_context_preamble(pending_plan: PendingWorkPlan | None) -> str:
    planning_context = pending_plan.planning_context if pending_plan is not None else None
    if planning_context is None:
        return ""
    lines = [
        "Planning Context (reference only):",
        "This summarizes evidence gathered during planning.",
        "Use it as reference only; do not skip plan steps, assume work is already done, or claim completion early.",
    ]
    if planning_context.findings_digest:
        lines.extend(["", "Findings:", planning_context.findings_digest])
    if planning_context.reasoning_digest:
        lines.extend(["", "Reasoning digest:", planning_context.reasoning_digest])
    if planning_context.tool_summaries:
        lines.append("")
        lines.append("Planner tool activity:")
        for item in planning_context.tool_summaries[:6]:
            line = f"- {item.tool} x{item.count}"
            if item.last_input_descriptor:
                line += f"; last={item.last_input_descriptor}"
            if item.result_summary:
                line += f"; result={item.result_summary}"
            lines.append(line)
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class PlanPreconditionValidation:
    valid: bool
    failed_precondition: PlanPrecondition | None = None
    message: str = ""


def validate_plan_preconditions(
    pending_plan: PendingWorkPlan,
    *,
    current_snapshot: SharedPrefixForkSnapshot | None,
) -> PlanPreconditionValidation:
    preconditions = list(pending_plan.current_plan.preconditions or [])
    if not preconditions:
        return PlanPreconditionValidation(valid=True)

    for precondition in preconditions:
        if precondition != PlanPrecondition.CONVERSATION_CONTEXT:
            continue
        planning_context = pending_plan.planning_context
        planning_snapshot = planning_context.context_snapshot if planning_context is not None else None
        if planning_snapshot is None or not planning_snapshot.parent_prefix_hash:
            return PlanPreconditionValidation(
                valid=False,
                failed_precondition=precondition,
                message=(
                    "Approved plan requires planning-time conversation context, but the planning snapshot is missing."
                ),
            )
        if current_snapshot is None:
            return PlanPreconditionValidation(
                valid=False,
                failed_precondition=precondition,
                message=(
                    "Approved plan requires a live parent conversation, but execution has no compatible parent "
                    "conversation snapshot."
                ),
            )
        if current_snapshot.parent_message_count != planning_snapshot.parent_message_count:
            return PlanPreconditionValidation(
                valid=False,
                failed_precondition=precondition,
                message=(
                    "Approved plan depends on planning-time conversation/tool findings, but the current conversation "
                    f"message count changed (planned={planning_snapshot.parent_message_count}, "
                    f"current={current_snapshot.parent_message_count})."
                ),
            )
        if current_snapshot.prefix_hash != planning_snapshot.parent_prefix_hash:
            return PlanPreconditionValidation(
                valid=False,
                failed_precondition=precondition,
                message=(
                    "Approved plan depends on planning-time conversation/tool findings, but the current "
                    "conversation snapshot no longer matches the planning snapshot."
                ),
            )
    return PlanPreconditionValidation(valid=True)


def build_precondition_failure_replan_prompt(
    pending_plan: PendingWorkPlan,
    *,
    validation: PlanPreconditionValidation,
) -> tuple[str, dict[str, Any], str]:
    failed_name = (
        validation.failed_precondition.value
        if isinstance(validation.failed_precondition, PlanPrecondition)
        else "unknown"
    )
    reason = validation.message.strip() or "Approved plan no longer matches the planning context."
    feedback_lines = [
        f"- Execution precondition failed: {failed_name}.",
        f"- {reason}",
        "- Regenerate the WorkPlan against the current conversation and current repository state.",
        "- Preserve still-valid intent, but refresh assumptions, fallback tools, and execution steps.",
    ]
    feedback_entry = {
        "kind": "precondition_failure",
        "failed_precondition": failed_name,
        "message": reason,
    }
    warning = "Approved plan no longer matches planning context; regenerated the plan before execution."
    return (
        build_plan_revision_prompt(pending_plan, step_feedback=feedback_lines, question_feedback=[]),
        feedback_entry,
        warning,
    )
