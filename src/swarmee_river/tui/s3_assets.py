"""S3 import helpers for tooling assets (prompts, SOPs, tools config, KBs)."""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any


def _s3_client() -> Any:
    """Return a boto3 S3 client with standard timeout/retry config."""
    try:
        import boto3
        from botocore.config import Config
    except Exception as exc:
        raise RuntimeError("boto3 is required for S3 import. Install with: pip install boto3") from exc

    region = (os.getenv("AWS_REGION") or "us-east-1").strip() or "us-east-1"
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(connect_timeout=15, read_timeout=15, retries={"max_attempts": 2}),
    )


def _resolve_bucket() -> str:
    """Resolve S3 bucket from environment."""
    raw = os.getenv("SWARMEE_SESSION_S3_BUCKET", "").strip()
    if not raw:
        raise RuntimeError(
            "SWARMEE_SESSION_S3_BUCKET environment variable is required for S3 import."
        )
    return raw


def _tooling_prefix() -> str:
    """Base S3 prefix for tooling assets."""
    raw = os.getenv("SWARMEE_TOOLING_S3_PREFIX", "").strip()
    return raw or "swarmee/tooling"


def _list_objects(bucket: str, prefix: str) -> list[dict[str, Any]]:
    """List S3 objects under a prefix, returning key + size metadata."""
    client = _s3_client()
    results: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            results.append({
                "key": obj["Key"],
                "size": obj.get("Size", 0),
                "last_modified": str(obj.get("LastModified", "")),
            })
    return results


def _get_object_text(bucket: str, key: str) -> str:
    """Download an S3 object as UTF-8 text."""
    client = _s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8", errors="replace")


def _get_object_json(bucket: str, key: str) -> Any:
    """Download an S3 object and parse as JSON."""
    text = _get_object_text(bucket, key)
    return json.loads(text)


# ── Asset-type specific imports ────────────────────────────────────────────

def import_prompts_from_s3(
    bucket: str | None = None,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Import prompt template definitions from S3.

    Looks for ``*.prompt.md`` files and/or a ``prompts.json`` index.
    """
    bucket = bucket or _resolve_bucket()
    base = prefix or f"{_tooling_prefix()}/prompts"
    if not base.endswith("/"):
        base += "/"

    prompts: list[dict[str, Any]] = []

    # Try a prompts.json index first.
    with contextlib.suppress(Exception):
        index = _get_object_json(bucket, f"{base}prompts.json")
        if isinstance(index, list):
            prompts.extend(index)
            return prompts

    # Fall back to listing *.prompt.md files.
    for obj in _list_objects(bucket, base):
        key = obj["key"]
        if key.endswith(".prompt.md"):
            with contextlib.suppress(Exception):
                content = _get_object_text(bucket, key)
                name = key.rsplit("/", 1)[-1].removesuffix(".prompt.md")
                prompts.append({"id": name, "name": name, "content": content, "source": "s3"})
    return prompts


def import_sops_from_s3(
    bucket: str | None = None,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Import SOP definitions from S3 (``*.sop.md`` files)."""
    bucket = bucket or _resolve_bucket()
    base = prefix or f"{_tooling_prefix()}/sops"
    if not base.endswith("/"):
        base += "/"

    sops: list[dict[str, Any]] = []
    for obj in _list_objects(bucket, base):
        key = obj["key"]
        if key.endswith(".sop.md"):
            with contextlib.suppress(Exception):
                content = _get_object_text(bucket, key)
                name = key.rsplit("/", 1)[-1].removesuffix(".sop.md")
                sops.append({"name": name, "content": content, "source": "s3", "path": f"s3://{bucket}/{key}"})
    return sops


def import_tools_config_from_s3(
    bucket: str | None = None,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Import tool metadata overrides from S3 (``tools.json``)."""
    bucket = bucket or _resolve_bucket()
    base = prefix or f"{_tooling_prefix()}/tools"
    key = f"{base.rstrip('/')}/tools.json"

    with contextlib.suppress(Exception):
        data = _get_object_json(bucket, key)
        if isinstance(data, dict):
            return data
    return {}


def import_kbs_from_s3(
    bucket: str | None = None,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Import knowledge base definitions from S3 (``kbs.json``)."""
    bucket = bucket or _resolve_bucket()
    base = prefix or f"{_tooling_prefix()}/kbs"
    key = f"{base.rstrip('/')}/kbs.json"

    with contextlib.suppress(Exception):
        data = _get_object_json(bucket, key)
        if isinstance(data, list):
            return data
    return []


__all__ = [
    "import_kbs_from_s3",
    "import_prompts_from_s3",
    "import_sops_from_s3",
    "import_tools_config_from_s3",
]
