from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from typing import Any

from strands import tool

from swarmee_river.tool_permissions import set_permissions
from swarmee_river.utils.aws_config import resolve_runtime_aws_region
from swarmee_river.utils.text_utils import truncate


def _success(text: str, *, max_chars: int) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": truncate(text, max_chars)}]}


def _error(text: str, *, max_chars: int) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": truncate(text, max_chars)}]}


def _aws_region() -> str:
    return resolve_runtime_aws_region()


def _s3_client() -> Any:
    try:
        import boto3
        from botocore.config import Config
    except Exception as exc:  # pragma: no cover - boto3 is a project dependency today.
        raise RuntimeError(f"boto3 is required. Install with: pip install boto3 ({exc})") from exc

    return boto3.client(
        "s3",
        region_name=_aws_region(),
        config=Config(connect_timeout=15, read_timeout=15, retries={"max_attempts": 2}),
    )


def _fmt_dt(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _fmt_size(num_bytes: Any) -> str:
    try:
        n = float(num_bytes)
    except Exception:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while n >= 1024 and idx < len(units) - 1:
        n /= 1024
        idx += 1
    return f"{n:.1f} {units[idx]}" if idx > 0 else f"{int(n)} {units[idx]}"


def _cell(value: Any, *, limit: int = 100) -> str:
    text = str(value or "")
    text = text.replace("\n", " ").replace("|", "\\|")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(_cell(h) for h in headers) + " |"
    align_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join([header_line, align_line, *body])


def _looks_textual(*, key: str, content_type: str) -> bool:
    ext = key.lower().rsplit(".", 1)[-1] if "." in key else ""
    if ext in {"txt", "md", "json", "jsonl", "csv", "tsv", "log", "xml", "yaml", "yml"}:
        return True

    ctype = (content_type or "").lower()
    if ctype.startswith("text/"):
        return True
    markers = ["json", "csv", "xml", "yaml", "x-ndjson", "javascript", "sql"]
    return any(marker in ctype for marker in markers)


def _render_delimited_table(text: str, *, delimiter: str, max_data_rows: int = 50) -> str:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader]
    if not rows:
        return "(no rows)"

    headers = rows[0]
    data = rows[1 : 1 + max_data_rows]
    rendered = _markdown_table(headers, data)
    if len(rows) - 1 > max_data_rows:
        rendered += f"\n\n... ({len(rows) - 1 - max_data_rows} more rows)"
    return rendered


def _list_buckets(*, prefix: str | None, max_chars: int) -> dict[str, Any]:
    try:
        s3 = _s3_client()
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    try:
        response = s3.list_buckets()
    except Exception as exc:
        return _error(f"Failed to list buckets: {exc}", max_chars=max_chars)

    buckets = response.get("Buckets") if isinstance(response, dict) else None
    if not isinstance(buckets, list) or not buckets:
        return _success("(no accessible buckets)", max_chars=max_chars)

    bucket_rows: list[list[Any]] = []
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        name = str(bucket.get("Name") or "").strip()
        if not name:
            continue
        if prefix and not name.startswith(prefix):
            continue

        region = "unknown"
        try:
            loc = s3.get_bucket_location(Bucket=name)
            location_constraint = loc.get("LocationConstraint") if isinstance(loc, dict) else None
            region = location_constraint or "us-east-1"
        except Exception:
            region = "unknown"

        bucket_rows.append([name, _fmt_dt(bucket.get("CreationDate")), region])

    if not bucket_rows:
        return _success("(no buckets matched the prefix)", max_chars=max_chars)

    table = _markdown_table(["Bucket", "Created", "Region"], bucket_rows)
    return _success(f"# S3 Buckets\n\n{table}", max_chars=max_chars)


def _list_objects(
    *,
    bucket: str | None,
    prefix: str | None,
    delimiter: str,
    max_keys: int,
    max_chars: int,
) -> dict[str, Any]:
    name = (bucket or "").strip()
    if not name:
        return _error("bucket is required for action=list_objects", max_chars=max_chars)

    try:
        s3 = _s3_client()
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    requested = max(1, min(1000, int(max_keys)))

    kwargs: dict[str, Any] = {"Bucket": name, "MaxKeys": min(1000, requested + 1)}
    if prefix:
        kwargs["Prefix"] = prefix
    if delimiter:
        kwargs["Delimiter"] = delimiter

    try:
        response = s3.list_objects_v2(**kwargs)
    except Exception as exc:
        return _error(f"Failed to list objects in s3://{name}: {exc}", max_chars=max_chars)

    contents = response.get("Contents") if isinstance(response, dict) else None
    objects = contents if isinstance(contents, list) else []

    shown = objects[:requested]
    hidden = max(0, len(objects) - len(shown))
    if hidden == 0 and bool(response.get("IsTruncated")):
        hidden = 1

    prefixes = response.get("CommonPrefixes") if isinstance(response, dict) else None
    common_prefixes = []
    if isinstance(prefixes, list):
        for item in prefixes:
            if isinstance(item, dict) and item.get("Prefix"):
                common_prefixes.append(str(item["Prefix"]))

    lines: list[str] = [f"# s3://{name}/{prefix or ''}", ""]

    if common_prefixes:
        lines.append("## Common Prefixes")
        for cp in common_prefixes:
            lines.append(f"- `{cp}`")
        lines.append("")

    if shown:
        rows = [
            [
                str(obj.get("Key") or ""),
                _fmt_size(obj.get("Size") or 0),
                _fmt_dt(obj.get("LastModified")),
                str(obj.get("StorageClass") or "STANDARD"),
            ]
            for obj in shown
            if isinstance(obj, dict)
        ]
        lines.append("## Objects")
        lines.append(_markdown_table(["Key", "Size", "Last Modified", "Storage Class"], rows))
    else:
        lines.append("(no objects)")

    if hidden > 0:
        lines.append("")
        lines.append(f"... and {hidden} more objects")

    return _success("\n".join(lines).strip(), max_chars=max_chars)


def _head_object(*, bucket: str | None, key: str | None, max_chars: int) -> dict[str, Any]:
    name = (bucket or "").strip()
    obj_key = (key or "").strip()
    if not name:
        return _error("bucket is required for action=head", max_chars=max_chars)
    if not obj_key:
        return _error("key is required for action=head", max_chars=max_chars)

    try:
        s3 = _s3_client()
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    try:
        head = s3.head_object(Bucket=name, Key=obj_key)
    except Exception as exc:
        return _error(f"Failed to head s3://{name}/{obj_key}: {exc}", max_chars=max_chars)

    lines = [
        f"# Metadata: s3://{name}/{obj_key}",
        "",
        f"- size: {_fmt_size(head.get('ContentLength') or 0)}",
        f"- content_type: {head.get('ContentType') or ''}",
        f"- last_modified: {_fmt_dt(head.get('LastModified'))}",
        f"- storage_class: {head.get('StorageClass') or 'STANDARD'}",
        f"- etag: {head.get('ETag') or ''}",
    ]

    metadata = head.get("Metadata") if isinstance(head, dict) else None
    if isinstance(metadata, dict) and metadata:
        lines.append("- metadata:")
        for mk, mv in sorted(metadata.items()):
            lines.append(f"  - {mk}: {mv}")

    return _success("\n".join(lines), max_chars=max_chars)


def _read_object(
    *,
    bucket: str | None,
    key: str | None,
    max_bytes: int,
    encoding: str,
    max_chars: int,
) -> dict[str, Any]:
    name = (bucket or "").strip()
    obj_key = (key or "").strip()
    if not name:
        return _error("bucket is required for action=read", max_chars=max_chars)
    if not obj_key:
        return _error("key is required for action=read", max_chars=max_chars)

    cap = max(1, min(10 * 1024 * 1024, int(max_bytes)))

    try:
        s3 = _s3_client()
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    try:
        head = s3.head_object(Bucket=name, Key=obj_key)
    except Exception as exc:
        return _error(f"Failed to read metadata for s3://{name}/{obj_key}: {exc}", max_chars=max_chars)

    total_size = int(head.get("ContentLength") or 0)
    content_type = str(head.get("ContentType") or "")

    get_kwargs: dict[str, Any] = {"Bucket": name, "Key": obj_key}
    if total_size > 0:
        get_kwargs["Range"] = f"bytes=0-{min(cap, total_size) - 1}"

    try:
        response = s3.get_object(**get_kwargs)
        body = response["Body"].read(cap)
    except Exception as exc:
        return _error(f"Failed to read s3://{name}/{obj_key}: {exc}", max_chars=max_chars)

    is_text = _looks_textual(key=obj_key, content_type=content_type)
    if (not is_text) and (b"\x00" in body):
        return _success(
            f"[Binary file: {total_size} bytes, content_type: {content_type or 'application/octet-stream'}]",
            max_chars=max_chars,
        )
    if not is_text:
        sample = body[:2048]
        if sample:
            non_text = sum(1 for b in sample if b < 9 or (13 < b < 32))
            if (non_text / len(sample)) > 0.30:
                return _success(
                    f"[Binary file: {total_size} bytes, content_type: {content_type or 'application/octet-stream'}]",
                    max_chars=max_chars,
                )

    try:
        text = body.decode(encoding, errors="replace")
    except Exception:
        return _success(
            f"[Binary file: {total_size} bytes, content_type: {content_type or 'application/octet-stream'}]",
            max_chars=max_chars,
        )

    key_lower = obj_key.lower()
    rendered = text

    if key_lower.endswith(".json") or key_lower.endswith(".jsonl") or "json" in content_type.lower():
        try:
            parsed = json.loads(text)
            rendered = json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            rendered = text
    elif key_lower.endswith(".csv") or "csv" in content_type.lower():
        rendered = _render_delimited_table(text, delimiter=",")
    elif key_lower.endswith(".tsv") or "tab-separated" in content_type.lower():
        rendered = _render_delimited_table(text, delimiter="\t")

    if total_size > cap:
        rendered = rendered.rstrip() + f"\n\n... (truncated at {cap} bytes, total: {total_size})"

    return _success(rendered, max_chars=max_chars)


def _search(
    *,
    bucket: str | None,
    prefix: str | None,
    query: str | None,
    max_results: int,
    max_chars: int,
) -> dict[str, Any]:
    name = (bucket or "").strip()
    if not name:
        return _error("bucket is required for action=search", max_chars=max_chars)

    result_cap = max(1, min(1000, int(max_results)))

    try:
        s3 = _s3_client()
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    use_select = bool(query and prefix and prefix.lower().endswith((".csv", ".json", ".jsonl")))
    if use_select:
        expr = (query or "").strip().rstrip(";")
        if not expr:
            return _error("query is required for S3 Select", max_chars=max_chars)

        if not re.search(r"\blimit\s+\d+\b", expr, flags=re.IGNORECASE):
            expr = f"{expr} LIMIT {result_cap}"

        input_serialization: dict[str, Any]
        if prefix and prefix.lower().endswith(".csv"):
            input_serialization = {"CSV": {"FileHeaderInfo": "USE"}, "CompressionType": "NONE"}
        else:
            input_serialization = {"JSON": {"Type": "DOCUMENT"}, "CompressionType": "NONE"}

        try:
            select_resp = s3.select_object_content(
                Bucket=name,
                Key=prefix,
                ExpressionType="SQL",
                Expression=expr,
                InputSerialization=input_serialization,
                OutputSerialization={"JSON": {}},
            )
        except Exception as exc:
            return _error(f"S3 Select failed: {exc}", max_chars=max_chars)

        chunks: list[bytes] = []
        for event in select_resp.get("Payload", []):
            if "Records" in event and isinstance(event["Records"], dict):
                payload = event["Records"].get("Payload")
                if isinstance(payload, (bytes, bytearray)):
                    chunks.append(bytes(payload))

        if not chunks:
            return _success("(no matching rows)", max_chars=max_chars)

        combined = b"".join(chunks).decode("utf-8", errors="replace").strip()
        return _success(combined or "(no matching rows)", max_chars=max_chars)

    list_kwargs: dict[str, Any] = {
        "Bucket": name,
        "MaxKeys": min(1000, result_cap + 1),
    }
    if prefix:
        list_kwargs["Prefix"] = prefix

    try:
        response = s3.list_objects_v2(**list_kwargs)
    except Exception as exc:
        return _error(f"Failed to list matching keys: {exc}", max_chars=max_chars)

    contents = response.get("Contents") if isinstance(response, dict) else None
    objects = contents if isinstance(contents, list) else []

    shown = objects[:result_cap]
    hidden = max(0, len(objects) - len(shown))
    if hidden == 0 and bool(response.get("IsTruncated")):
        hidden = 1

    lines: list[str] = [f"# Matching keys in s3://{name}/{prefix or ''}", ""]
    for obj in shown:
        if not isinstance(obj, dict):
            continue
        lines.append(f"- `{obj.get('Key')}` ({_fmt_size(obj.get('Size') or 0)})")

    if len(lines) == 2:
        lines.append("(no matches)")

    if hidden > 0:
        lines.append("")
        lines.append(f"... and {hidden} more objects")

    return _success("\n".join(lines).strip(), max_chars=max_chars)


@tool
def s3_browser(
    action: str = "list_objects",
    bucket: str | None = None,
    key: str | None = None,
    prefix: str | None = None,
    max_keys: int = 100,
    delimiter: str = "/",
    max_bytes: int = 100 * 1024,
    encoding: str = "utf-8",
    query: str | None = None,
    max_results: int = 20,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Read-only S3 exploration tool with size guardrails.

    Actions:
    - list_buckets: list accessible buckets
    - list_objects (default): list objects in a bucket/prefix
    - head: read object metadata
    - read: download bounded bytes and render text/JSON/CSV/TSV
    - search: S3 Select for CSV/JSON object, or prefix key search fallback
    """
    mode = (action or "list_objects").strip().lower()

    if mode == "list_buckets":
        return _list_buckets(prefix=prefix, max_chars=max_chars)
    if mode == "list_objects":
        return _list_objects(
            bucket=bucket,
            prefix=prefix,
            delimiter=delimiter,
            max_keys=max_keys,
            max_chars=max_chars,
        )
    if mode == "head":
        return _head_object(bucket=bucket, key=key, max_chars=max_chars)
    if mode == "read":
        return _read_object(
            bucket=bucket,
            key=key,
            max_bytes=max_bytes,
            encoding=encoding,
            max_chars=max_chars,
        )
    if mode == "search":
        return _search(
            bucket=bucket,
            prefix=prefix,
            query=query,
            max_results=max_results,
            max_chars=max_chars,
        )

    return _error(f"Unknown action: {mode}", max_chars=max_chars)


set_permissions(s3_browser, "read")
