from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from strands.agent.conversation_manager import SummarizingConversationManager

_CACHE_COMPACTED_PREFIX = "[cache-compacted]"
_COMPACTABLE_TOOLS = {"file_read", "file_search"}
_STRATEGY_RAW_RESULT_KEEP = {
    "cache_safe": 2,
    "long_running": 4,
}
SWARMEE_SUMMARIZATION_SYSTEM_PROMPT = """You are compressing prior conversation state for a future model turn.
Return only a concise plain-text summary of durable facts, decisions, open questions,
and exact file or path references that still matter.
Do not emit chain-of-thought, reasoning narration, tool calls, tool transcripts, XML, JSON, or role labels.
If tools were used, summarize only the durable outcome in plain language."""


@dataclass
class _ToolResultRecord:
    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]
    tool_result: dict[str, Any]
    text: str


def _tool_result_text(tool_result: dict[str, Any]) -> str:
    parts: list[str] = []
    content = tool_result.get("content")
    if not isinstance(content, list):
        return ""
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts).strip()


def estimate_tool_schema_chars(tools: Any) -> int:
    if tools is None:
        return 0
    if isinstance(tools, dict):
        values = list(tools.values())
    elif isinstance(tools, (list, tuple, set)):
        values = list(tools)
    else:
        values = [tools]

    total = 0
    for tool in values:
        try:
            total += len(json.dumps(tool, ensure_ascii=False, sort_keys=True, default=str))
        except Exception:
            total += len(str(tool))
    return total


def _tool_result_signature(record: _ToolResultRecord) -> tuple[str, str]:
    input_payload = record.tool_input if isinstance(record.tool_input, dict) else {}
    encoded = json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(record.text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{record.tool_name}|{encoded}", digest


def _tool_result_descriptor(record: _ToolResultRecord) -> str:
    tool_name = record.tool_name
    payload = record.tool_input if isinstance(record.tool_input, dict) else {}
    if tool_name == "file_read":
        path = str(payload.get("path") or "").strip() or "(unknown path)"
        start_line = payload.get("start_line")
        max_lines = payload.get("max_lines")
        if isinstance(start_line, int) and isinstance(max_lines, int) and max_lines > 0:
            end_line = start_line + max_lines - 1
            return f"path={path} lines={start_line}-{end_line}"
        return f"path={path}"
    if tool_name == "file_search":
        query = str(payload.get("query") or "").strip() or "(unknown query)"
        return f"query={query}"
    return tool_name


def _summarize_tool_result(record: _ToolResultRecord, *, duplicate: bool) -> str:
    descriptor = _tool_result_descriptor(record)
    original_chars = len(record.text)
    if duplicate:
        detail = "duplicate excerpt elided; same input/content appeared earlier"
    else:
        detail = "older excerpt elided for cache stability; rerun the tool if exact text is needed"
    return (
        f"{_CACHE_COMPACTED_PREFIX} {record.tool_name} {descriptor}; "
        f"original_chars={original_chars}; {detail}."
    )


def _iter_tool_result_records(messages: list[dict[str, Any]]) -> list[_ToolResultRecord]:
    tool_uses: dict[str, tuple[str, dict[str, Any]]] = {}
    records: list[_ToolResultRecord] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            tool_use = item.get("toolUse")
            if isinstance(tool_use, dict):
                tool_use_id = str(tool_use.get("toolUseId") or tool_use.get("id") or "").strip()
                if not tool_use_id:
                    continue
                tool_name = str(tool_use.get("name") or "unknown").strip() or "unknown"
                tool_input = tool_use.get("input")
                tool_uses[tool_use_id] = (tool_name, tool_input if isinstance(tool_input, dict) else {})
            tool_result = item.get("toolResult")
            if not isinstance(tool_result, dict):
                continue
            tool_use_id = str(tool_result.get("toolUseId") or "").strip()
            if not tool_use_id:
                continue
            tool_name, tool_input = tool_uses.get(tool_use_id, ("unknown", {}))
            text = _tool_result_text(tool_result)
            if not text:
                continue
            records.append(
                _ToolResultRecord(
                    tool_name=tool_name,
                    tool_use_id=tool_use_id,
                    tool_input=tool_input,
                    tool_result=tool_result,
                    text=text,
                )
            )
    return records


def _extract_reasoning_fragments(value: Any, *, depth: int = 4, in_reasoning: bool = False) -> list[str]:
    if depth <= 0:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if in_reasoning and text else []
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            chunks.extend(_extract_reasoning_fragments(item, depth=depth - 1, in_reasoning=in_reasoning))
        return chunks
    if not isinstance(value, dict):
        return []

    chunks: list[str] = []
    for key, nested in value.items():
        normalized = str(key or "").strip().lower().replace("-", "").replace("_", "")
        next_reasoning = in_reasoning or normalized in {
            "reasoning",
            "reasoningtext",
            "reasoningcontent",
            "thinking",
        }
        if next_reasoning and normalized in {"text", "content", "summary", "value"} and isinstance(nested, str):
            text = nested.strip()
            if text:
                chunks.append(text)
            continue
        chunks.extend(_extract_reasoning_fragments(nested, depth=depth - 1, in_reasoning=next_reasoning))
    return chunks


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
        for reasoning in _extract_reasoning_fragments(item):
            parts.append(f"reasoning={reasoning}")
        if isinstance(item.get("toolUse"), dict):
            tool_use = item["toolUse"]
            tool_name = str(tool_use.get("name") or "unknown").strip() or "unknown"
            tool_use_id = str(tool_use.get("toolUseId") or tool_use.get("id") or "").strip()
            if tool_use_id:
                parts.append(f"toolUseId={tool_use_id}")
            parts.append(f"toolUse={tool_name}")
            if "input" in tool_use:
                try:
                    parts.append(json.dumps(tool_use["input"], ensure_ascii=False, sort_keys=True))
                except Exception:
                    parts.append(str(tool_use["input"]))
        if isinstance(item.get("toolResult"), dict):
            tool_result = item["toolResult"]
            tool_use_id = str(tool_result.get("toolUseId") or "").strip()
            status = str(tool_result.get("status") or "unknown").strip() or "unknown"
            if tool_use_id:
                parts.append(f"toolResultId={tool_use_id}")
            parts.append(f"toolResult={status}")
            if "content" in tool_result:
                try:
                    parts.append(json.dumps(tool_result["content"], ensure_ascii=False, sort_keys=True))
                except Exception:
                    parts.append(str(tool_result["content"]))

    return "\n".join(parts)

def estimate_tokens(
    *,
    system_prompt: str | None,
    messages: list[dict[str, Any]],
    chars_per_token: int,
    tool_schema_chars: int = 0,
) -> int:
    total_chars = 0
    if system_prompt:
        total_chars += len(system_prompt)
    total_chars += max(0, int(tool_schema_chars or 0))
    for message in messages:
        if isinstance(message, dict):
            total_chars += len(_extract_message_text(message))

    divisor = max(1, chars_per_token)
    return int(math.ceil(total_chars / divisor))


def estimate_tokens_for_agent(agent: Any, *, chars_per_token: int = 4) -> int:
    return estimate_tokens(
        system_prompt=getattr(agent, "system_prompt", None),
        messages=getattr(agent, "messages", []),
        chars_per_token=chars_per_token,
        tool_schema_chars=estimate_tool_schema_chars(getattr(agent, "tools", None)),
    )


def _tool_activity_descriptor(tool_name: str, tool_input: dict[str, Any] | None) -> str:
    return _tool_result_descriptor(
        _ToolResultRecord(
            tool_name=tool_name,
            tool_use_id="summary",
            tool_input=tool_input if isinstance(tool_input, dict) else {},
            tool_result={},
            text="",
        )
    )


def _summarize_tool_use(tool_name: str, tool_input: dict[str, Any] | None) -> str:
    descriptor = _tool_activity_descriptor(tool_name, tool_input)
    return f"[tool request] {tool_name}; {descriptor}"


def _summarize_tool_result_for_summary(
    *,
    tool_name: str,
    tool_input: dict[str, Any] | None,
    tool_result: dict[str, Any],
) -> str:
    descriptor = _tool_activity_descriptor(tool_name, tool_input)
    status = str(tool_result.get("status") or "unknown").strip() or "unknown"
    text = _tool_result_text(tool_result)
    original_chars = len(text)
    return f"[tool result] {tool_name}; {descriptor}; status={status}; chars={original_chars}"


def _sanitize_messages_for_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_uses: dict[str, tuple[str, dict[str, Any]]] = {}
    sanitized: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip() or "user"
        content = message.get("content")
        if not isinstance(content, list):
            continue

        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

            tool_use = item.get("toolUse")
            if isinstance(tool_use, dict):
                tool_use_id = str(tool_use.get("toolUseId") or tool_use.get("id") or "").strip()
                tool_name = str(tool_use.get("name") or "unknown").strip() or "unknown"
                tool_input = tool_use.get("input")
                normalized_input = tool_input if isinstance(tool_input, dict) else {}
                if tool_use_id:
                    tool_uses[tool_use_id] = (tool_name, normalized_input)
                parts.append(_summarize_tool_use(tool_name, normalized_input))

            tool_result = item.get("toolResult")
            if isinstance(tool_result, dict):
                tool_use_id = str(tool_result.get("toolUseId") or "").strip()
                tool_name, tool_input = tool_uses.get(tool_use_id, ("unknown", {}))
                parts.append(
                    _summarize_tool_result_for_summary(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_result=tool_result,
                    )
                )

        merged = "\n".join(part for part in parts if part).strip()
        if not merged:
            continue
        sanitized.append({"role": role, "content": [{"text": merged}]})

    return sanitized


def _sanitize_summary_message(message: Any) -> dict[str, Any]:
    parts: list[str] = []
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        elif isinstance(message.get("text"), str) and str(message.get("text")).strip():
            parts.append(str(message.get("text")).strip())

    summary_text = "\n\n".join(parts).strip() or "Conversation summary unavailable."
    return {"role": "user", "content": [{"text": summary_text}]}


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
        strategy: str = "balanced",
        compaction_mode: str = "auto",
        stable_tool_names: list[str] | None = None,
        compactable_tool_names: list[str] | None = None,
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
        self.strategy = strategy.strip().lower() if isinstance(strategy, str) else "balanced"
        self.compaction_mode = compaction_mode.strip().lower() if isinstance(compaction_mode, str) else "auto"
        self.stable_tool_names = [str(name).strip() for name in (stable_tool_names or []) if str(name).strip()]
        configured_compactable = (
            compactable_tool_names if compactable_tool_names is not None else list(_COMPACTABLE_TOOLS)
        )
        self.compactable_tool_names = {
            str(name).strip() for name in configured_compactable if str(name).strip()
        } or set(_COMPACTABLE_TOOLS)
        self._last_compacted_read_results = 0

    def estimate_tokens_for_agent(self, agent: "Any") -> int:
        return estimate_tokens_for_agent(agent, chars_per_token=self.chars_per_token)

    def _generate_summary(self, messages: list[dict[str, Any]], agent: "Any") -> dict[str, Any]:
        sanitized_messages = _sanitize_messages_for_summary(messages)
        summary_message = super()._generate_summary(sanitized_messages, agent)
        return _sanitize_summary_message(summary_message)

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

    def _trim_history_to_budget(self, agent: "Any") -> int:
        messages = getattr(agent, "messages", [])
        if not isinstance(messages, list) or not messages:
            return 0
        minimum_keep = max(2, int(self.preserve_recent_messages))
        trimmed = 0
        while len(messages) > minimum_keep:
            tokens = self.estimate_tokens_for_agent(agent)
            if tokens <= self.max_prompt_tokens:
                break
            remove_count = min(max(1, len(messages) // 8), len(messages) - minimum_keep)
            if remove_count <= 0:
                break
            del messages[:remove_count]
            trimmed += remove_count
        return trimmed

    def _compact_tool_results_for_strategy(self, agent: "Any") -> int:
        keep_count = _STRATEGY_RAW_RESULT_KEEP.get(self.strategy)
        messages = getattr(agent, "messages", [])
        if keep_count is None or not isinstance(messages, list) or not messages:
            return 0

        records = [
            record
            for record in _iter_tool_result_records(messages)
            if record.tool_name in self.compactable_tool_names and not record.text.startswith(_CACHE_COMPACTED_PREFIX)
        ]
        if not records:
            return 0

        latest_by_signature: dict[tuple[str, str], int] = {}
        for idx, record in enumerate(records):
            latest_by_signature[_tool_result_signature(record)] = idx

        keep_start = max(0, len(records) - keep_count)
        raw_keep_indices = set(range(keep_start, len(records)))
        compacted = 0

        for idx, record in enumerate(records):
            signature = _tool_result_signature(record)
            latest_idx = latest_by_signature.get(signature, idx)
            should_compact = latest_idx != idx or idx not in raw_keep_indices
            if not should_compact:
                continue
            summary = _summarize_tool_result(record, duplicate=latest_idx != idx)
            record.tool_result["content"] = [{"text": summary}]
            compacted += 1

        return compacted

    def estimate_uncached_tail_tokens(self, agent: "Any") -> int:
        messages = getattr(agent, "messages", [])
        if not isinstance(messages, list) or not messages:
            return 0
        last_assistant_idx = -1
        for idx, message in enumerate(messages):
            if isinstance(message, dict) and str(message.get("role", "")).strip().lower() == "assistant":
                last_assistant_idx = idx
        tail_messages = messages[last_assistant_idx + 1 :] if last_assistant_idx >= 0 else messages
        return estimate_tokens(system_prompt=None, messages=tail_messages, chars_per_token=self.chars_per_token)

    def cache_diagnostics_for_agent(self, agent: "Any") -> dict[str, int]:
        records = _iter_tool_result_records(getattr(agent, "messages", []) or [])
        raw_recent = sum(
            1
            for record in records
            if record.tool_name in self.compactable_tool_names and not record.text.startswith(_CACHE_COMPACTED_PREFIX)
        )
        return {
            "uncached_tail_tokens_est": self.estimate_uncached_tail_tokens(agent),
            "recent_read_tool_results": raw_recent,
            "compacted_read_results": int(self._last_compacted_read_results),
        }

    def compact_to_budget(self, agent: "Any", **kwargs: Any) -> dict[str, Any]:
        before_tokens = self.estimate_tokens_for_agent(agent)
        self._last_compacted_read_results = 0
        if not self.enabled or self.compaction_mode == "manual":
            return {
                "before_tokens_est": before_tokens,
                "after_tokens_est": before_tokens,
                "within_budget": before_tokens <= self.max_prompt_tokens,
                "summary_passes": 0,
                "trimmed_messages": 0,
                "compacted_read_results": 0,
            }

        compacted_read_results = self._compact_tool_results_for_strategy(agent)
        self._last_compacted_read_results = compacted_read_results
        summary_passes = 0
        for _ in range(max(1, self.max_reduce_passes)):
            tokens = self.estimate_tokens_for_agent(agent)
            if tokens <= self.max_prompt_tokens:
                return {
                    "before_tokens_est": before_tokens,
                    "after_tokens_est": tokens,
                    "within_budget": True,
                    "summary_passes": summary_passes,
                    "trimmed_messages": 0,
                    "compacted_read_results": compacted_read_results,
                }
            self.reduce_context(agent, e=None, **kwargs)
            summary_passes += 1
        trimmed_messages = self._trim_history_to_budget(agent)
        after_tokens = self.estimate_tokens_for_agent(agent)
        return {
            "before_tokens_est": before_tokens,
            "after_tokens_est": after_tokens,
            "within_budget": after_tokens <= self.max_prompt_tokens,
            "summary_passes": summary_passes,
            "trimmed_messages": trimmed_messages,
            "compacted_read_results": compacted_read_results,
        }

    def apply_management(self, agent: "Any", **kwargs: Any) -> None:
        if not self.enabled:
            return
        if self.compaction_mode == "manual":
            return
        self.compact_to_budget(agent, **kwargs)
