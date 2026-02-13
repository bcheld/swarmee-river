from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from strands.hooks import HookRegistry, HookProvider
from strands.hooks.events import AfterToolCallEvent

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.hooks._compat import register_hook_callback


def _truthy_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


class ToolResultLimiterHooks(HookProvider):
    """
    Reduce prompt bloat by truncating large tool results before they are added to the agent conversation.

    The full original tool result text is persisted to a local artifact file so it can be recovered later.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        max_text_chars: int | None = None,
        artifacts_dir: Path | None = None,
    ) -> None:
        self.enabled = _truthy_env("SWARMEE_LIMIT_TOOL_RESULTS", True) if enabled is None else enabled
        self.max_text_chars = (
            int(os.getenv("SWARMEE_TOOL_RESULT_MAX_CHARS", "8000")) if max_text_chars is None else max_text_chars
        )
        default_artifacts_dir = Path.cwd() / ".swarmee" / "artifacts"
        self.artifacts_dir = default_artifacts_dir if artifacts_dir is None else artifacts_dir

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        register_hook_callback(registry, AfterToolCallEvent, self.after_tool_call)

    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        if not self.enabled:
            return

        result = event.result
        if not isinstance(result, dict):
            return

        content = result.get("content")
        if not isinstance(content, list):
            return

        tool_use = event.tool_use or {}
        tool_use_id = tool_use.get("toolUseId") or result.get("toolUseId") or "unknown_tool_use"
        tool_name = tool_use.get("name") or "unknown_tool"

        changed = False
        new_content: list[dict[str, Any]] = []

        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                new_content.append(item)
                continue

            if len(text) <= self.max_text_chars:
                new_content.append(item)
                continue

            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            iso_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            artifact_path = self.artifacts_dir / f"{ts}_{tool_name}_{tool_use_id}.txt"
            artifact_meta_path = self.artifacts_dir / f"{ts}_{tool_name}_{tool_use_id}.meta.json"

            artifact_path.write_text(text, encoding="utf-8", errors="replace")
            artifact_meta_path.write_text(
                json.dumps(
                    {
                        "toolUseId": tool_use_id,
                        "tool": tool_name,
                        "original_chars": len(text),
                        "saved_to": str(artifact_path),
                        "created_at": ts,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            try:
                ArtifactStore(self.artifacts_dir).append_index(
                    {
                        "id": f"{ts}_{tool_name}_{tool_use_id}",
                        "kind": "tool_result",
                        "path": str(artifact_path),
                        "meta_path": str(artifact_meta_path),
                        "created_at": iso_ts,
                        "toolUseId": tool_use_id,
                        "tool": tool_name,
                        "original_chars": len(text),
                        "kept_chars": self.max_text_chars,
                    }
                )
            except Exception:
                pass

            truncated = text[: self.max_text_chars]
            suffix = (
                "\n\n"
                f"[tool result truncated: kept {self.max_text_chars} chars of {len(text)}; "
                f"full output saved to {artifact_path}]"
            )
            new_item = dict(item)
            new_item["text"] = truncated + suffix
            new_content.append(new_item)
            changed = True

        if changed:
            result["content"] = new_content
            # Note: mutating the dict in-place is the most compatible approach across Strands versions.
