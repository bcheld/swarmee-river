from __future__ import annotations


def truncate(text: str, max_chars: int) -> str:
    """
    Truncate text to a maximum number of characters, appending a truncation notice if needed.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated to {max_chars} chars) ..."
