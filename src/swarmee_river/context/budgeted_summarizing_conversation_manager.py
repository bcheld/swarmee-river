from __future__ import annotations

import json
import math
from typing import Any

from strands.agent.conversation_manager import SummarizingConversationManager


def _extract_message_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    role = message.get("role")
    if isinstance(role, str):
        parts.append(f"role={role}")

    content = message.get("content", [])
    if not isinstance(content, list):
        return "\n".join(parts)

    for item in content:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("text"), str):
            parts.append(item["text"])
        if isinstance(item.get("toolUse"), dict):
            tool_use = item["toolUse"]
            parts.append(f"toolUse={tool_use.get('name')}")
            if "input" in tool_use:
                try:
                    parts.append(json.dumps(tool_use["input"], ensure_ascii=False))
                except Exception:
                    parts.append(str(tool_use["input"]))
        if isinstance(item.get("toolResult"), dict):
            tool_result = item["toolResult"]
            parts.append(f"toolResult={tool_result.get('status')}")
            if "content" in tool_result:
                try:
                    parts.append(json.dumps(tool_result["content"], ensure_ascii=False))
                except Exception:
                    parts.append(str(tool_result["content"]))

    return "\n".join(parts)


def estimate_tokens(*, system_prompt: str | None, messages: list[dict[str, Any]], chars_per_token: int) -> int:
    total_chars = 0
    if system_prompt:
        total_chars += len(system_prompt)
    for message in messages:
        if isinstance(message, dict):
            total_chars += len(_extract_message_text(message))

    divisor = max(1, chars_per_token)
    return int(math.ceil(total_chars / divisor))


class BudgetedSummarizingConversationManager(SummarizingConversationManager):
    """
    A token-budgeted conversation manager that proactively summarizes older context.

    Strands' built-in `SummarizingConversationManager` only summarizes on overflow.
    This implementation estimates the prompt token size and triggers summarization
    before overflow to prevent runaway context growth.
    """

    def __init__(
        self,
        *,
        max_prompt_tokens: int | None = None,
        chars_per_token: int | None = None,
        max_reduce_passes: int | None = None,
        summary_ratio: float = 0.3,
        preserve_recent_messages: int = 10,
        summarization_system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            summary_ratio=summary_ratio,
            preserve_recent_messages=preserve_recent_messages,
            summarization_system_prompt=summarization_system_prompt,
        )

        # Settings are provided by the caller (CLI/TUI/notebook) via structured
        # `.swarmee/settings.json` fields. Avoid environment-variable fallbacks.
        self.max_prompt_tokens = int(max_prompt_tokens) if max_prompt_tokens is not None else 20000
        self.chars_per_token = int(chars_per_token) if chars_per_token is not None else 4
        self.max_reduce_passes = int(max_reduce_passes) if max_reduce_passes is not None else 4
        self.enabled = True

    def reduce_context(self, agent: "Any", e: Exception | None = None, **kwargs: Any) -> None:
        messages = getattr(agent, "messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            return
        try:
            super().reduce_context(agent, e=e, **kwargs)
        except Exception as exc:
            text = str(exc or "").strip().lower()
            if "cannot summarize" in text and "insufficient messages" in text:
                return
            raise

    def apply_management(self, agent: "Any", **kwargs: Any) -> None:
        if not self.enabled:
            return

        for _ in range(max(1, self.max_reduce_passes)):
            tokens = estimate_tokens(
                system_prompt=getattr(agent, "system_prompt", None),
                messages=getattr(agent, "messages", []),
                chars_per_token=self.chars_per_token,
            )
            if tokens <= self.max_prompt_tokens:
                return
            # Summarize once; re-check and repeat if still over budget.
            self.reduce_context(agent, e=None, **kwargs)
