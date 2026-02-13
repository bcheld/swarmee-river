from __future__ import annotations

import re


_WORK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(implement|fix|refactor|rewrite|add|remove|delete|rename|update|upgrade)\b", re.IGNORECASE),
    re.compile(r"\b(create|generate|scaffold|bootstrap)\b", re.IGNORECASE),
    re.compile(r"\b(run|rerun|execute)\b.*\b(tests?|pytest|mypy|ruff|lint|format|build)\b", re.IGNORECASE),
    re.compile(r"\b(debug|reproduce|investigate)\b", re.IGNORECASE),
    re.compile(r"\b(open|create)\b.*\b(pr|pull request)\b", re.IGNORECASE),
    re.compile(r"\b(edit|change)\b.*\b(file|code|repo|repository)\b", re.IGNORECASE),
]


def classify_intent(prompt: str) -> str:
    """
    Best-effort heuristic classifier for interactive UX decisions.

    Returns:
        "work" if the prompt likely requests code changes or execution,
        otherwise "info".
    """
    text = (prompt or "").strip()
    if not text:
        return "info"

    for pattern in _WORK_PATTERNS:
        if pattern.search(text):
            return "work"

    return "info"

