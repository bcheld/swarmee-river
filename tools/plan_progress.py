from __future__ import annotations

from typing import Any

from strands import tool


@tool
def plan_progress(
    *,
    step: int | None = None,
    step_index: int | None = None,
    status: str,
    note: str | None = None,
) -> dict[str, Any]:
    """
    Report plan execution progress from execute mode.

    Agent-facing usage:
    - Use 1-based `step` when following the written plan.
    - Optionally pass 0-based `step_index` directly.
    - `status` must be `in_progress` or `completed`.
    """
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in {"in_progress", "completed"}:
        return {
            "status": "error",
            "content": [{"text": "status must be one of: in_progress, completed"}],
        }

    resolved_index: int | None = None
    if isinstance(step_index, int):
        resolved_index = step_index
    elif isinstance(step, int):
        if step == 0:
            resolved_index = 0
        elif step > 0:
            resolved_index = step - 1
    if resolved_index is None or resolved_index < 0:
        return {
            "status": "error",
            "content": [{"text": "step (1-based) or step_index (0-based) is required"}],
        }

    summary = f"plan progress accepted: step_index={resolved_index}, status={normalized_status}"
    if isinstance(note, str) and note.strip():
        summary = f"{summary}, note={note.strip()}"
    return {"status": "success", "content": [{"text": summary}]}
