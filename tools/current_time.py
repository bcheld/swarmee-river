from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from strands import tool


@tool
def current_time(*, utc: bool = True) -> dict[str, Any]:
    """
    Cross-platform fallback for `strands_tools.current_time`.

    Returns an ISO-8601 timestamp.
    """
    now = datetime.now(timezone.utc) if utc else datetime.now().astimezone()
    return {"status": "success", "content": [{"text": now.isoformat()}]}
