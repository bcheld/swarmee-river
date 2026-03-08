from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.diff_review import (
    build_unified_diff,
    resolve_mutating_tool_paths,
    snapshot_file,
    summarize_diff_stats,
)
from swarmee_river.hooks._compat import register_hook_callback
from swarmee_river.state_paths import artifacts_dir as _default_artifacts_dir
from swarmee_river.utils.env_utils import truthy
from swarmee_river.utils.stdio_utils import write_stdout_jsonl


class FileDiffReviewHooks(HookProvider):
    def __init__(self, *, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = _default_artifacts_dir() if artifacts_dir is None else artifacts_dir

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        register_hook_callback(registry, BeforeToolCallEvent, self.before_tool_call)
        register_hook_callback(registry, AfterToolCallEvent, self.after_tool_call)

    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if event.cancel_tool:
            return

        tool_use = event.tool_use if isinstance(event.tool_use, dict) else {}
        tool_name = str(tool_use.get("name", "")).strip()
        tool_input = tool_use.get("input")
        try:
            base, paths = resolve_mutating_tool_paths(tool_name, tool_input if isinstance(tool_input, dict) else None)
        except Exception:
            return
        if not paths:
            return

        snapshots = {}
        for display_path, target in paths:
            try:
                snapshots[display_path] = snapshot_file(target, display_path=display_path)
            except Exception:
                continue
        if not snapshots:
            return

        sw = event.invocation_state.setdefault("swarmee", {})
        review_state = sw.setdefault("file_diff_review", {})
        tool_use_id = str(tool_use.get("toolUseId", "")).strip() or "unknown_tool_use"
        review_state[tool_use_id] = {
            "base": str(base),
            "tool": tool_name,
            "paths": list(snapshots.keys()),
            "snapshots": snapshots,
            "dry_run": bool((tool_input or {}).get("dry_run")) if isinstance(tool_input, dict) else False,
        }

    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        tool_use = event.tool_use if isinstance(event.tool_use, dict) else {}
        tool_use_id = str(tool_use.get("toolUseId", "")).strip() or "unknown_tool_use"
        sw = event.invocation_state.get("swarmee", {}) if isinstance(event.invocation_state, dict) else {}
        review_state = sw.get("file_diff_review", {}) if isinstance(sw, dict) else {}
        pending = review_state.pop(tool_use_id, None) if isinstance(review_state, dict) else None
        if not isinstance(pending, dict):
            return
        if pending.get("dry_run"):
            return

        result = event.result if isinstance(event.result, dict) else {}
        if str(result.get("status", "")).strip().lower() != "success":
            return

        base = Path(str(pending.get("base", "")).strip() or ".").expanduser().resolve()
        before = pending.get("snapshots", {})
        path_order = pending.get("paths", [])
        if not isinstance(before, dict) or not isinstance(path_order, list):
            return

        textual_diffs: list[str] = []
        changed_paths: list[str] = []
        non_text_paths: list[dict[str, Any]] = []
        for display_path in path_order:
            snapshot_before = before.get(display_path)
            if snapshot_before is None:
                continue
            target = (base / str(display_path)).expanduser().resolve()
            try:
                snapshot_after = snapshot_file(target, display_path=str(display_path))
            except Exception:
                continue

            before_kind = getattr(snapshot_before, "kind", "")
            after_kind = snapshot_after.kind
            if before_kind in {"text", "missing"} and after_kind in {"text", "missing"}:
                diff_text = build_unified_diff(snapshot_before, snapshot_after)
                if not diff_text:
                    continue
                changed_paths.append(str(display_path))
                textual_diffs.append(diff_text)
                continue

            if (
                getattr(snapshot_before, "digest", None) == snapshot_after.digest
                and getattr(snapshot_before, "kind", None) == snapshot_after.kind
                and getattr(snapshot_before, "exists", None) == snapshot_after.exists
            ):
                continue
            changed_paths.append(str(display_path))
            non_text_paths.append(
                {
                    "path": str(display_path),
                    "before": str(before_kind or "unknown"),
                    "after": str(after_kind or "unknown"),
                }
            )

        if not changed_paths:
            return

        body = "\n\n".join(chunk for chunk in textual_diffs if chunk.strip()).strip()
        if not body:
            body = "Non-text file changes captured for this tool call. See artifact metadata for details."

        stats = summarize_diff_stats(body)
        stats["files_changed"] = len(changed_paths)
        stats["non_text_changes"] = len(non_text_paths)

        session_id = (os.getenv("SWARMEE_SESSION_ID") or "").strip() or None
        invocation_id = str(sw.get("invocation_id", "")).strip() if isinstance(sw, dict) else ""
        tool_name = str(pending.get("tool", "")).strip() or str(tool_use.get("name", "")).strip() or "unknown_tool"
        store = ArtifactStore(self.artifacts_dir)
        artifact = store.write_text(
            kind="file_diff",
            text=body,
            suffix="diff",
            metadata={
                "session_id": session_id,
                "invocation_id": invocation_id or None,
                "toolUseId": tool_use_id,
                "tool": tool_name,
                "cwd": str(base),
                "touched_paths": [str(item) for item in path_order],
                "changed_paths": changed_paths,
                "non_text_paths": non_text_paths,
                "stats": stats,
            },
        )

        if truthy(os.getenv("SWARMEE_TUI_EVENTS")):
            write_stdout_jsonl(
                {
                    "event": "file_diff",
                    "artifact_id": artifact.artifact_id,
                    "artifact_path": str(artifact.path),
                    "tool_use_id": tool_use_id,
                    "tool": tool_name,
                    "paths": changed_paths,
                    "touched_paths": [str(item) for item in path_order],
                    "stats": stats,
                }
            )
