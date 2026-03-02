from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def format_system_reminder(chunks: Iterable[str]) -> str:
    parts = [str(c).strip() for c in chunks if isinstance(c, str) and c.strip()]
    if not parts:
        return ""
    body = "\n\n".join(parts)
    return f"<system-reminder>\n{body}\n</system-reminder>"


def inject_system_reminder(*, user_query: str, reminder: str | None) -> str:
    query = (user_query or "").strip()
    prefix = (reminder or "").strip()
    if not prefix:
        return query
    if not query:
        return prefix
    return f"{prefix}\n\n{query}"


@dataclass
class PromptCacheState:
    """
    Tracks dynamic prompt sections and queues reminders only when content changes.

    The intent is to keep the API-level system prompt and tool schema stable so
    provider-side prompt caching can reuse the prefix, while still letting the agent
    receive updates via a tagged user message prefix.
    """

    sent_hashes: dict[str, str] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)

    def queue_if_changed(self, key: str, text: str | None) -> None:
        normalized = (text or "").strip()
        digest = _hash_text(normalized)
        if self.sent_hashes.get(key) == digest:
            return
        self.sent_hashes[key] = digest
        if normalized:
            self.pending.append(normalized)

    def queue_one_off(self, text: str | None) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return
        self.pending.append(normalized)

    def pop_reminder(self) -> str:
        reminder = format_system_reminder(self.pending)
        self.pending.clear()
        return reminder
