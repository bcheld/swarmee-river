from __future__ import annotations

import os
from typing import Any

from strands import tool


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… (truncated to {max_chars} chars) …"


@tool
def retrieve(
    *,
    text: str,
    knowledgeBaseId: str | None = None,
    knowledge_base_id: str | None = None,
    max_results: int = 5,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Cross-platform fallback for `strands_tools.retrieve` (Amazon Bedrock Knowledge Bases).

    Notes:
    - Uses boto3 `bedrock-agent-runtime.retrieve`.
    - Returns a text summary of retrieved chunks (best-effort).
    """
    query = (text or "").strip()
    if not query:
        return {"status": "error", "content": [{"text": "text is required"}]}

    kb_id = (
        knowledgeBaseId
        or knowledge_base_id
        or os.getenv("SWARMEE_KNOWLEDGE_BASE_ID")
        or os.getenv("STRANDS_KNOWLEDGE_BASE_ID")
        or ""
    ).strip()
    if not kb_id:
        return {"status": "error", "content": [{"text": "knowledge_base_id (knowledgeBaseId) is required"}]}

    try:
        import boto3
    except Exception as exc:
        return {"status": "error", "content": [{"text": f"boto3 is required for retrieve: {exc}"}]}

    region = (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2").strip()
    n = max(1, int(max_results))

    try:
        client = boto3.client("bedrock-agent-runtime", region_name=region)
        response = client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": n}},
        )
    except Exception as exc:
        return {"status": "error", "content": [{"text": f"retrieve failed: {exc}"}]}

    results = response.get("retrievalResults") if isinstance(response, dict) else None
    if not isinstance(results, list) or not results:
        return {"status": "success", "content": [{"text": "(no retrieval results)"}]}

    lines: list[str] = ["# Retrieved context", f"- results: {len(results)}", ""]
    for idx, item in enumerate(results[:n], start=1):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        chunk = ""
        if isinstance(content, dict):
            chunk = str(content.get("text") or "")
        if not chunk:
            chunk = str(item.get("text") or "")
        chunk = chunk.strip()

        score = item.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else ""
        header = f"## {idx}" + (f" (score={score_str})" if score_str else "")
        lines.append(header)
        if chunk:
            lines.append(chunk)
        else:
            lines.append("(empty)")
        lines.append("")

    return {"status": "success", "content": [{"text": _truncate("\n".join(lines).strip(), max_chars)}]}
