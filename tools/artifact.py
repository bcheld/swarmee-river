from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from strands import tool

from swarmee_river.artifacts import ArtifactStore

from tools.store_in_kb import store_in_kb as _store_in_kb


def _truthy_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


@tool
def artifact(
    action: str = "list",
    artifact_id: str | None = None,
    path: str | None = None,
    kind: str | None = None,
    limit: int = 25,
    max_chars: int = 12000,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    knowledge_base_id: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """
    Manage Swarmee artifacts stored under `.swarmee/artifacts/`.

    Actions:
    - list: show recent artifacts from `.swarmee/artifacts/index.jsonl`
    - get: read an artifact by id (preferred) or by path
    - upload: upload an artifact to S3 (requires bucket)
    - store_in_kb: store artifact contents in a Bedrock Knowledge Base
    """
    store = ArtifactStore()
    action = (action or "").strip().lower()

    if action == "list":
        entries = store.list(limit=limit, kind=kind)
        if not entries:
            return {"status": "success", "content": [{"text": "No artifacts found."}]}

        lines: list[str] = ["# Artifacts", ""]
        for e in entries:
            lines.append(f"- `{e.get('id')}` ({e.get('kind')}) -> {e.get('path')}")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    if action == "get":
        resolved_path: Path | None = None
        if artifact_id:
            meta = store.get_by_id(artifact_id)
            if meta and meta.get("path"):
                resolved_path = Path(str(meta["path"]))
        if resolved_path is None and path:
            resolved_path = Path(path).expanduser()

        if resolved_path is None:
            return {"status": "error", "content": [{"text": "artifact_id or path is required for action=get"}]}
        if not resolved_path.exists() or not resolved_path.is_file():
            return {"status": "error", "content": [{"text": f"File not found: {resolved_path}"}]}

        try:
            text = store.read_text(resolved_path, max_chars=max_chars)
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Failed to read artifact: {e}"}]}
        return {"status": "success", "content": [{"text": text}]}

    if action == "upload":
        bucket = s3_bucket or os.getenv("SWARMEE_ARTIFACT_S3_BUCKET")
        prefix = (s3_prefix or os.getenv("SWARMEE_ARTIFACT_S3_PREFIX") or "swarmee/artifacts").strip("/")
        if not bucket:
            return {"status": "error", "content": [{"text": "Missing s3_bucket (or SWARMEE_ARTIFACT_S3_BUCKET)."}]}
        if not artifact_id and not path:
            return {"status": "error", "content": [{"text": "artifact_id or path is required for action=upload"}]}

        meta = store.get_by_id(artifact_id) if artifact_id else None
        resolved_path = Path(str(meta["path"])) if meta and meta.get("path") else Path(path or "").expanduser()
        if not resolved_path.exists() or not resolved_path.is_file():
            return {"status": "error", "content": [{"text": f"File not found: {resolved_path}"}]}

        try:
            import boto3
        except Exception:
            return {"status": "error", "content": [{"text": "boto3 is required for S3 upload."}]}

        key = f"{prefix}/{resolved_path.name}"
        try:
            s3 = boto3.client("s3")
            s3.put_object(Bucket=bucket, Key=key, Body=resolved_path.read_bytes())
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Upload failed: {e}"}]}
        return {"status": "success", "content": [{"text": f"Uploaded to s3://{bucket}/{key}"}]}

    if action == "store_in_kb":
        if not artifact_id and not path:
            return {"status": "error", "content": [{"text": "artifact_id or path is required for action=store_in_kb"}]}

        meta = store.get_by_id(artifact_id) if artifact_id else None
        resolved_path = Path(str(meta["path"])) if meta and meta.get("path") else Path(path or "").expanduser()
        if not resolved_path.exists() or not resolved_path.is_file():
            return {"status": "error", "content": [{"text": f"File not found: {resolved_path}"}]}

        try:
            text = store.read_text(resolved_path, max_chars=None)
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Failed to read artifact: {e}"}]}

        kb_id = knowledge_base_id or os.getenv("SWARMEE_KNOWLEDGE_BASE_ID") or os.getenv("STRANDS_KNOWLEDGE_BASE_ID")
        if not kb_id:
            return {"status": "error", "content": [{"text": "Missing knowledge_base_id (or SWARMEE_KNOWLEDGE_BASE_ID)."}]}

        doc_title = title or f"Artifact: {resolved_path.name}"
        return _store_in_kb(content=text, title=doc_title, knowledge_base_id=kb_id)

    return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}

