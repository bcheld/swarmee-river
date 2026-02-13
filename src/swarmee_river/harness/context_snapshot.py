from __future__ import annotations

import os
from dataclasses import dataclass

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.project_map import build_project_map, render_project_map_summary, save_project_map
from tools.project_context import run_project_context


def _truthy(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


@dataclass(frozen=True)
class ContextSnapshot:
    preflight_prompt_section: str | None
    project_map_prompt_section: str | None


def build_context_snapshot(
    *,
    artifact_store: ArtifactStore,
    interactive: bool,
    default_preflight_level: str | None = None,
) -> ContextSnapshot:
    """
    Build a lightweight repo context snapshot (preflight + project map) and return prompt sections.

    Controlled by existing env vars:
    - SWARMEE_PREFLIGHT=enabled|disabled
    - SWARMEE_PREFLIGHT_LEVEL=summary|summary+tree|summary+files
    - SWARMEE_PREFLIGHT_MAX_CHARS
    - SWARMEE_PROJECT_MAP=enabled|disabled
    """
    preflight_prompt_section: str | None = None
    project_map_prompt_section: str | None = None

    if _truthy(os.getenv("SWARMEE_PREFLIGHT", "enabled"), True):
        level = (
            os.getenv("SWARMEE_PREFLIGHT_LEVEL")
            or default_preflight_level
            or "summary"
        ).strip().lower()
        max_chars = int(os.getenv("SWARMEE_PREFLIGHT_MAX_CHARS", "8000"))
        actions = ["summary"]
        if level == "summary+tree":
            actions.append("tree")
        elif level == "summary+files":
            actions.append("files")

        preflight_parts: list[str] = []
        for action in actions:
            try:
                result = run_project_context(action=action, max_chars=max_chars)
                if result.get("status") == "success":
                    preflight_parts.append(result.get("content", [{"text": ""}])[0].get("text", ""))
            except Exception:
                continue
        preflight_text = "\n\n".join([p for p in preflight_parts if p]).strip()
        if preflight_text:
            artifact_store.write_text(
                kind="context_snapshot",
                text=preflight_text,
                suffix="txt",
                metadata={"source": "project_context", "level": level},
            )
            preflight_prompt_section = f"Project context snapshot:\n{preflight_text}"
            should_print_preflight = _truthy(os.getenv("SWARMEE_PREFLIGHT_PRINT", "disabled"), False)
            if interactive and should_print_preflight:
                print("\n[preflight]\n" + preflight_text + "\n")

    if _truthy(os.getenv("SWARMEE_PROJECT_MAP", "enabled"), True):
        try:
            pm = build_project_map()
            pm_path = save_project_map(pm)
            project_map_prompt_section = render_project_map_summary(pm) + f"\n(project_map: {pm_path})"
        except Exception:
            project_map_prompt_section = None
    else:
        project_map_prompt_section = None

    return ContextSnapshot(
        preflight_prompt_section=preflight_prompt_section,
        project_map_prompt_section=project_map_prompt_section,
    )
