from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from strands import tool

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.session.store import SessionStore
from swarmee_river.utils.text_utils import truncate

_DEFAULT_S3_PREFIX = "swarmee/sessions"
_SESSION_FILE_NAMES = ("meta.json", "messages.jsonl", "state.json", "last_plan.json", "summary.md")


def _iso_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _success(text: str, *, max_chars: int) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": truncate(text, max_chars)}]}


def _error(text: str, *, max_chars: int) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": truncate(text, max_chars)}]}


def _sanitize_s3_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip())
    cleaned = cleaned.strip(".-")
    return cleaned or "session"


def _normalize_prefix(prefix: str | None) -> str:
    raw = (prefix or os.getenv("SWARMEE_SESSION_S3_PREFIX") or _DEFAULT_S3_PREFIX).strip()
    raw = raw.replace("\\", "/")
    raw = re.sub(r"/+", "/", raw).strip("/")
    return raw or _DEFAULT_S3_PREFIX


def _resolve_bucket(bucket: str | None) -> str:
    value = (bucket or os.getenv("SWARMEE_SESSION_S3_BUCKET") or "").strip()
    if not value:
        raise ValueError("Missing s3_bucket (or SWARMEE_SESSION_S3_BUCKET).")
    return value


def _resolve_session_id(session_id: str | None) -> str:
    sid = (session_id or os.getenv("SWARMEE_SESSION_ID") or "").strip()
    if not sid:
        raise ValueError("session_id is required (or set SWARMEE_SESSION_ID)")
    return sid


def _s3_client() -> Any:
    try:
        import boto3
        from botocore.config import Config
    except Exception as exc:
        raise RuntimeError("boto3 is required. Install with: pip install boto3") from exc

    region = (os.getenv("AWS_REGION") or "us-east-1").strip() or "us-east-1"
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(connect_timeout=15, read_timeout=15, retries={"max_attempts": 2}),
    )


def _kb_client() -> Any:
    try:
        import boto3
        from botocore.config import Config
    except Exception as exc:
        raise RuntimeError("boto3 is required. Install with: pip install boto3") from exc

    region = (os.getenv("AWS_REGION") or "us-east-1").strip() or "us-east-1"
    return boto3.client(
        "bedrock-agent",
        region_name=region,
        config=Config(connect_timeout=15, read_timeout=15, retries={"max_attempts": 2}),
    )


def _session_dir(session_id: str) -> Path:
    return SessionStore().root_dir / session_id


def _load_json_file(path: Path, default: Any) -> Any:
    if not path.exists() or not path.is_file():
        return default
    try:
        loaded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default
    return loaded if loaded is not None else default


def _extract_text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_extract_text_from_content(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("toolUse") or value.get("toolResult"):
            nested = value.get("content")
            return _extract_text_from_content(nested) if nested is not None else ""
        for key in ("text", "data", "delta", "output_text", "outputText", "textDelta"):
            raw = value.get(key)
            if isinstance(raw, str) and raw:
                return raw
        nested = value.get("content")
        if nested is not None:
            return _extract_text_from_content(nested)
    return ""


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    extracted = _extract_text_from_content(message.get("content"))
    if extracted:
        return extracted
    for key in ("text", "data", "output_text"):
        raw = message.get(key)
        if isinstance(raw, str) and raw:
            return raw
    return ""


def _iter_dict_nodes(value: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            nodes.append(current)
            for nested in current.values():
                if isinstance(nested, (dict, list)):
                    stack.append(nested)
        elif isinstance(current, list):
            for nested in current:
                if isinstance(nested, (dict, list)):
                    stack.append(nested)
    return nodes


def _load_messages_for_summary(session_path: Path) -> list[Any]:
    messages_log = session_path / "messages.jsonl"
    latest: list[Any] | None = None
    if messages_log.exists() and messages_log.is_file():
        lines = messages_log.read_text(encoding="utf-8", errors="replace").splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            with contextlib.suppress(Exception):
                parsed = json.loads(line)
                if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
                    latest = list(parsed["messages"])
                elif isinstance(parsed, list):
                    latest = list(parsed)

    if latest is not None:
        return latest

    messages_json = session_path / "messages.json"
    loaded = _load_json_file(messages_json, default=[])
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("messages"), list):
        return list(loaded["messages"])
    return []


def _first_user_query(messages: list[Any]) -> str:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        text = _message_text(message).strip()
        if text:
            return text
    return ""


def _plan_summary_from_payload(value: Any) -> str:
    if isinstance(value, dict):
        summary = str(value.get("summary") or "").strip()
        if summary:
            return summary
        steps = value.get("steps")
        if isinstance(steps, list) and steps:
            first_step = steps[0]
            if isinstance(first_step, dict):
                desc = str(first_step.get("description") or "").strip()
                if desc:
                    return desc
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped.splitlines()[0][:200]
    return ""


def _parse_timestamp(ts: Any) -> float:
    if not isinstance(ts, str) or not ts.strip():
        return 0.0
    raw = ts.strip()
    with contextlib.suppress(ValueError):
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    return 0.0


def _safe_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def _safe_jsonl_snapshot(messages: list[Any]) -> bytes:
    payload = {
        "version": 1,
        "saved_at": _iso_ts(),
        "message_count": len(messages),
        "turn_count": sum(
            1
            for item in messages
            if isinstance(item, dict) and str(item.get("role", "")).strip().lower() == "user"
        ),
        "messages": messages,
    }
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode("utf-8")


def _build_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def _cell(value: Any) -> str:
        text = str(value or "")
        text = text.replace("\n", " ").replace("|", "\\|")
        return text[:100] + "..." if len(text) > 100 else text

    top = "| " + " | ".join(_cell(h) for h in headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_cell(col) for col in row) + " |" for row in rows]
    return "\n".join([top, sep, *body])


def _generate_session_summary(meta: dict[str, Any], messages: list[Any]) -> str:
    session_id = str(meta.get("id") or os.getenv("SWARMEE_SESSION_ID") or "unknown").strip() or "unknown"
    created_at = str(meta.get("created_at") or "")
    updated_at = str(meta.get("updated_at") or "")
    provider = str(meta.get("provider") or "")
    tier = str(meta.get("tier") or "")
    model_id = str(meta.get("model_id") or "")

    user_outline: list[str] = []
    tool_counts: dict[str, int] = {}
    plan_summaries: list[str] = []
    errors: list[str] = []

    meta_plan = _plan_summary_from_payload(meta.get("last_plan"))
    if meta_plan:
        plan_summaries.append(meta_plan)

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        msg_text = _message_text(message).strip()

        if role == "user" and msg_text:
            compact = " ".join(msg_text.split())
            user_outline.append(compact[:200])

        if str(message.get("event", "")).strip().lower() == "plan":
            summary = _plan_summary_from_payload(
                message.get("plan_json") or message.get("plan") or message.get("rendered")
            )
            if summary:
                plan_summaries.append(summary)

        if str(message.get("status", "")).strip().lower() == "error":
            if msg_text:
                errors.append(msg_text[:300])

        for node in _iter_dict_nodes(message.get("content")):
            tool_use = node.get("toolUse")
            if isinstance(tool_use, dict):
                tool_name = str(tool_use.get("name") or "").strip()
                if tool_name:
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

            tool_result = node.get("toolResult")
            if isinstance(tool_result, dict):
                status = str(tool_result.get("status") or "").strip().lower()
                if status == "error":
                    err_text = _extract_text_from_content(tool_result.get("content")).strip()
                    if err_text:
                        errors.append(err_text[:300])

            if str(node.get("event", "")).strip().lower() == "plan":
                node_summary = _plan_summary_from_payload(
                    node.get("plan_json") or node.get("plan") or node.get("rendered")
                )
                if node_summary:
                    plan_summaries.append(node_summary)

            node_error = node.get("error")
            if isinstance(node_error, str) and node_error.strip():
                errors.append(node_error.strip()[:300])

    if "turn_count" in meta and isinstance(meta.get("turn_count"), int):
        turn_count = int(meta["turn_count"])
    else:
        turn_count = sum(
            1
            for item in messages
            if isinstance(item, dict) and str(item.get("role", "")).strip().lower() == "user"
        )

    unique_plans: list[str] = []
    seen_plans: set[str] = set()
    for plan in plan_summaries:
        compact = " ".join(plan.split()).strip()
        if not compact:
            continue
        token = compact.lower()
        if token in seen_plans:
            continue
        seen_plans.add(token)
        unique_plans.append(compact[:240])

    unique_errors: list[str] = []
    seen_errors: set[str] = set()
    for err in errors:
        compact = " ".join(err.split()).strip()
        if not compact:
            continue
        token = compact.lower()
        if token in seen_errors:
            continue
        seen_errors.add(token)
        unique_errors.append(compact[:300])

    lines: list[str] = ["# Session Summary", ""]
    lines.extend(
        [
            f"- Session ID: `{session_id}`",
            f"- Created: {created_at or '(unknown)'}",
            f"- Updated: {updated_at or '(unknown)'}",
            f"- Provider: {provider or '(unknown)'}",
            f"- Tier: {tier or '(unknown)'}",
            f"- Model: {model_id or '(unknown)'}",
            f"- Turn count: {turn_count}",
            "",
        ]
    )

    lines.append("## Conversation Outline")
    if user_outline:
        for idx, item in enumerate(user_outline[:50], start=1):
            lines.append(f"{idx}. {item}")
    else:
        lines.append("(no user messages found)")
    lines.append("")

    lines.append("## Plan Summaries")
    if unique_plans:
        for item in unique_plans[:20]:
            lines.append(f"- {item}")
    else:
        lines.append("(no plans detected)")
    lines.append("")

    lines.append("## Tool Usage")
    if tool_counts:
        rows = [[name, count] for name, count in sorted(tool_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        lines.append(_build_markdown_table(["Tool", "Call Count"], rows))
    else:
        lines.append("(no tool usage detected)")
    lines.append("")

    lines.append("## Errors")
    if unique_errors:
        for item in unique_errors[:20]:
            lines.append(f"- {item}")
    else:
        lines.append("(no explicit errors detected)")

    return truncate("\n".join(lines).strip(), 5000)


def _load_session_snapshot(session_id: str) -> tuple[Path, dict[str, Any], list[Any], dict[str, Any], dict[str, Any]]:
    session_path = _session_dir(session_id)
    if not session_path.exists() or not session_path.is_dir():
        raise FileNotFoundError(f"Session not found: {session_id}")

    meta = _load_json_file(session_path / "meta.json", default={})
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("id", session_id)
    meta.setdefault("updated_at", _iso_ts())
    meta.setdefault("created_at", meta.get("updated_at") or _iso_ts())

    messages = _load_messages_for_summary(session_path)
    state = _load_json_file(session_path / "state.json", default={})
    last_plan = _load_json_file(session_path / "last_plan.json", default={})

    if not isinstance(state, dict):
        state = {"value": state}
    if not isinstance(last_plan, dict):
        last_plan = {"value": last_plan}

    return session_path, meta, messages, state, last_plan


def export_session_to_s3(
    *,
    session_id: str,
    s3_bucket: str,
    s3_prefix: str,
) -> dict[str, Any]:
    session_path, meta, messages, state, last_plan = _load_session_snapshot(session_id)

    summary_meta = dict(meta)
    if last_plan:
        summary_meta["last_plan"] = last_plan
    summary_md = _generate_session_summary(summary_meta, messages)

    files: dict[str, bytes] = {}
    files["meta.json"] = _safe_json_bytes(meta)

    messages_log_path = session_path / "messages.jsonl"
    if messages_log_path.exists() and messages_log_path.is_file():
        files["messages.jsonl"] = messages_log_path.read_bytes()
    else:
        files["messages.jsonl"] = _safe_jsonl_snapshot(messages)

    files["state.json"] = _safe_json_bytes(state or {})
    files["last_plan.json"] = _safe_json_bytes(last_plan or {})
    files["summary.md"] = (summary_md + "\n").encode("utf-8")

    s3 = _s3_client()
    safe_id = _sanitize_s3_segment(session_id)
    key_prefix = f"{_normalize_prefix(s3_prefix)}/{safe_id}"

    total_bytes = 0
    for filename in _SESSION_FILE_NAMES:
        body = files[filename]
        key = f"{key_prefix}/{filename}"
        s3.put_object(Bucket=s3_bucket, Key=key, Body=body)
        total_bytes += len(body)

    return {
        "s3_uri": f"s3://{s3_bucket}/{key_prefix}/",
        "file_count": len(_SESSION_FILE_NAMES),
        "total_bytes": total_bytes,
        "summary": summary_md,
    }


def _list_s3_session_ids(*, bucket: str, prefix: str, max_results: int) -> list[str]:
    s3 = _s3_client()
    normalized_prefix = _normalize_prefix(prefix)
    root_prefix = f"{normalized_prefix}/"

    session_ids: list[str] = []
    token: str | None = None
    cap = max(1, min(1000, int(max_results)))

    while len(session_ids) < cap:
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": root_prefix,
            "Delimiter": "/",
            "MaxKeys": 1000,
        }
        if token:
            kwargs["ContinuationToken"] = token

        response = s3.list_objects_v2(**kwargs)
        prefixes = response.get("CommonPrefixes") if isinstance(response, dict) else None
        if isinstance(prefixes, list):
            for item in prefixes:
                if not isinstance(item, dict):
                    continue
                raw_prefix = str(item.get("Prefix") or "")
                sid = raw_prefix[len(root_prefix) :].strip("/") if raw_prefix.startswith(root_prefix) else ""
                if sid:
                    session_ids.append(sid)
                    if len(session_ids) >= cap:
                        break

        token = response.get("NextContinuationToken") if isinstance(response, dict) else None
        if not token:
            break

    return session_ids[:cap]


def _list_sessions_from_s3(*, bucket: str, prefix: str, max_results: int) -> list[dict[str, Any]]:
    s3 = _s3_client()
    normalized_prefix = _normalize_prefix(prefix)
    session_ids = _list_s3_session_ids(bucket=bucket, prefix=normalized_prefix, max_results=max_results)

    sessions: list[dict[str, Any]] = []
    for sid in session_ids:
        meta_key = f"{normalized_prefix}/{sid}/meta.json"
        meta: dict[str, Any] = {"id": sid}
        with contextlib.suppress(Exception):
            response = s3.get_object(Bucket=bucket, Key=meta_key)
            payload = response.get("Body").read().decode("utf-8", errors="replace")
            loaded = json.loads(payload)
            if isinstance(loaded, dict):
                meta = loaded
                meta.setdefault("id", sid)
        sessions.append(meta)

    sessions.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return sessions[: max(1, int(max_results))]


def import_session_from_s3(
    *,
    session_id: str,
    s3_bucket: str,
    s3_prefix: str,
    force: bool,
) -> dict[str, Any]:
    s3 = _s3_client()
    store = SessionStore()

    safe_sid = _sanitize_s3_segment(session_id)
    key_prefix = f"{_normalize_prefix(s3_prefix)}/{safe_sid}/"

    objects: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": s3_bucket, "Prefix": key_prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3.list_objects_v2(**kwargs)
        contents = response.get("Contents") if isinstance(response, dict) else []
        if isinstance(contents, list):
            objects.extend(item for item in contents if isinstance(item, dict))
        token = response.get("NextContinuationToken") if isinstance(response, dict) else None
        if not token:
            break

    if not objects:
        raise FileNotFoundError(f"No S3 session found at s3://{s3_bucket}/{key_prefix}")

    local_dir = store.root_dir / session_id
    if local_dir.exists() and not force:
        raise FileExistsError("Session already exists locally. Re-run with force=True (or --force) to overwrite.")

    if force and local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    total_bytes = 0
    for item in objects:
        key = str(item.get("Key") or "")
        if not key.startswith(key_prefix):
            continue
        rel = key[len(key_prefix) :]
        if not rel or rel.endswith("/"):
            continue

        rel_posix = PurePosixPath(rel)
        if any(part in {"", ".", ".."} for part in rel_posix.parts):
            continue

        response = s3.get_object(Bucket=s3_bucket, Key=key)
        body = response.get("Body").read()

        target = local_dir.joinpath(*rel_posix.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)

        file_count += 1
        total_bytes += len(body)

    return {
        "session_id": session_id,
        "local_dir": str(local_dir),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _resolve_kb_id(kb_id: str | None) -> str:
    resolved = (
        kb_id
        or os.getenv("SWARMEE_KNOWLEDGE_BASE_ID")
        or os.getenv("STRANDS_KNOWLEDGE_BASE_ID")
        or ""
    ).strip()
    if not resolved:
        raise ValueError("knowledge_base_id is required (or set SWARMEE_KNOWLEDGE_BASE_ID)")
    return resolved


def _resolve_custom_data_source_id(kb_client: Any, knowledge_base_id: str) -> str:
    data_sources = kb_client.list_data_sources(knowledgeBaseId=knowledge_base_id)
    summaries = data_sources.get("dataSourceSummaries") if isinstance(data_sources, dict) else None
    if not isinstance(summaries, list) or not summaries:
        raise RuntimeError(f"No data sources found for knowledge base {knowledge_base_id}.")

    first_source_id: str | None = None
    first_source_type: str | None = None

    for source in summaries:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("dataSourceId") or "").strip()
        if not source_id:
            continue

        source_detail = kb_client.get_data_source(knowledgeBaseId=knowledge_base_id, dataSourceId=source_id)
        source_type = (
            source_detail.get("dataSource", {})
            .get("dataSourceConfiguration", {})
            .get("type")
            if isinstance(source_detail, dict)
            else None
        )
        source_type_str = str(source_type or "").strip().upper()

        if first_source_id is None:
            first_source_id = source_id
            first_source_type = source_type_str
        if source_type_str == "CUSTOM":
            return source_id

    if first_source_id is None:
        raise RuntimeError(f"No suitable data source found for knowledge base {knowledge_base_id}.")

    raise RuntimeError(
        "Knowledge base has no CUSTOM data source for inline ingestion "
        f"(first source type: {first_source_type or 'unknown'})."
    )


def _ingest_documents_to_kb(*, knowledge_base_id: str, docs: list[dict[str, Any]]) -> int:
    if not docs:
        return 0

    kb_client = _kb_client()
    data_source_id = _resolve_custom_data_source_id(kb_client, knowledge_base_id)

    payload_documents: list[dict[str, Any]] = []
    for item in docs:
        doc_id = f"session_{_sanitize_s3_segment(str(item.get('session_id') or 'doc'))}_{uuid.uuid4().hex[:10]}"
        serialized = json.dumps(item, ensure_ascii=False)
        payload_documents.append(
            {
                "content": {
                    "dataSourceType": "CUSTOM",
                    "custom": {
                        "customDocumentIdentifier": {"id": doc_id},
                        "inlineContent": {
                            "textContent": {"data": serialized},
                            "type": "TEXT",
                        },
                        "sourceType": "IN_LINE",
                    },
                }
            }
        )

    kb_client.ingest_knowledge_base_documents(
        knowledgeBaseId=knowledge_base_id,
        dataSourceId=data_source_id,
        documents=payload_documents,
    )

    return len(payload_documents)


def _extract_plan_docs(
    *,
    meta: dict[str, Any],
    messages: list[Any],
    last_plan: dict[str, Any],
    session_id: str,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    last_plan_summary = _plan_summary_from_payload(last_plan)
    if last_plan_summary:
        docs.append(
            {
                "session_id": session_id,
                "timestamp": _iso_ts(),
                "content_type": "plan",
                "original_user_query": _first_user_query(messages),
                "content": json.dumps(last_plan, ensure_ascii=False, indent=2, default=str),
            }
        )

    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("event", "")).strip().lower() != "plan":
            continue
        plan_payload = message.get("plan_json") or message.get("plan") or message.get("rendered")
        summary = _plan_summary_from_payload(plan_payload)
        if not summary:
            continue
        docs.append(
            {
                "session_id": session_id,
                "timestamp": _iso_ts(),
                "content_type": "plan",
                "original_user_query": _first_user_query(messages),
                "content": summary,
            }
        )

    if not docs and isinstance(meta.get("last_plan"), dict):
        docs.append(
            {
                "session_id": session_id,
                "timestamp": _iso_ts(),
                "content_type": "plan",
                "original_user_query": _first_user_query(messages),
                "content": json.dumps(meta.get("last_plan"), ensure_ascii=False, indent=2, default=str),
            }
        )

    return docs


def _extract_output_docs(*, messages: list[Any], session_id: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    original_query = _first_user_query(messages)

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        if role != "assistant":
            continue
        text = _message_text(message).strip()
        if not text:
            continue
        if "error" in text.lower():
            continue
        docs.append(
            {
                "session_id": session_id,
                "timestamp": _iso_ts(),
                "content_type": "output",
                "original_user_query": original_query,
                "content": text,
            }
        )

    return docs[:20]


def promote_session_to_kb(
    *,
    session_id: str,
    knowledge_base_id: str,
    content_filter: str,
) -> dict[str, Any]:
    _session_path, meta, messages, _state, last_plan = _load_session_snapshot(session_id)

    filter_name = (content_filter or "all").strip().lower()
    if filter_name not in {"plans", "outputs", "all"}:
        raise ValueError("content_filter must be one of: plans, outputs, all")

    docs: list[dict[str, Any]] = []

    if filter_name in {"plans", "all"}:
        docs.extend(_extract_plan_docs(meta=meta, messages=messages, last_plan=last_plan, session_id=session_id))

    if filter_name in {"outputs", "all"}:
        docs.extend(_extract_output_docs(messages=messages, session_id=session_id))

    if filter_name == "all":
        summary_meta = dict(meta)
        if last_plan:
            summary_meta["last_plan"] = last_plan
        summary = _generate_session_summary(summary_meta, messages)
        docs.insert(
            0,
            {
                "session_id": session_id,
                "timestamp": _iso_ts(),
                "content_type": "summary",
                "original_user_query": _first_user_query(messages),
                "content": summary,
            },
        )

    if not docs:
        raise ValueError(f"No promotable content found for session {session_id}")

    promoted_count = _ingest_documents_to_kb(knowledge_base_id=knowledge_base_id, docs=docs)
    return {
        "session_id": session_id,
        "knowledge_base_id": knowledge_base_id,
        "promoted_count": promoted_count,
    }


def promote_artifact_to_kb(*, artifact_id: str, knowledge_base_id: str) -> dict[str, Any]:
    store = ArtifactStore()
    meta = store.get_by_id(artifact_id)
    if not isinstance(meta, dict):
        raise FileNotFoundError(f"Artifact not found: {artifact_id}")

    path = Path(str(meta.get("path") or "")).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Artifact file not found: {path}")

    text = store.read_text(path, max_chars=None)
    doc = {
        "session_id": (os.getenv("SWARMEE_SESSION_ID") or "").strip() or "",
        "timestamp": _iso_ts(),
        "content_type": "artifact",
        "original_user_query": "",
        "artifact_id": artifact_id,
        "artifact_kind": str(meta.get("kind") or ""),
        "artifact_path": str(path),
        "content": text,
    }
    promoted_count = _ingest_documents_to_kb(knowledge_base_id=knowledge_base_id, docs=[doc])
    return {
        "artifact_id": artifact_id,
        "knowledge_base_id": knowledge_base_id,
        "promoted_count": promoted_count,
    }


@tool
def session_s3(
    action: str = "export",
    session_id: str | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    max_results: int = 20,
    force: bool = False,
    knowledge_base_id: str | None = None,
    content_filter: str = "all",
    artifact_id: str | None = None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    S3 overlay for session persistence and enterprise KB promotion.

    Actions:
    - export (default): upload local session files and generated summary to S3
    - list: list session prefixes in S3 and summarize metadata from meta.json
    - import: download a session from S3 into local session storage
    - sync: compare updated_at and import/export whichever is newer
    - promote_to_kb: ingest session-derived content into Bedrock Knowledge Base
    - promote_artifact: ingest a local artifact into Bedrock Knowledge Base with session metadata
    """
    mode = (action or "export").strip().lower()

    try:
        if mode == "export":
            sid = _resolve_session_id(session_id)
            bucket = _resolve_bucket(s3_bucket)
            prefix = _normalize_prefix(s3_prefix)
            exported = export_session_to_s3(session_id=sid, s3_bucket=bucket, s3_prefix=prefix)
            return _success(
                (
                    f"Exported session `{sid}` to {exported['s3_uri']}\n"
                    f"- files: {exported['file_count']}\n"
                    f"- bytes: {exported['total_bytes']}"
                ),
                max_chars=max_chars,
            )

        if mode == "list":
            bucket = _resolve_bucket(s3_bucket)
            prefix = _normalize_prefix(s3_prefix)
            sessions = _list_sessions_from_s3(bucket=bucket, prefix=prefix, max_results=max_results)
            if not sessions:
                return _success("(no sessions found in S3)", max_chars=max_chars)

            rows = [
                [
                    str(item.get("id") or ""),
                    str(item.get("created_at") or ""),
                    str(item.get("updated_at") or ""),
                    str(item.get("turn_count") or ""),
                ]
                for item in sessions
            ]
            table = _build_markdown_table(["Session", "Created", "Updated", "Turns"], rows)
            return _success(f"# S3 Sessions\n\n{table}", max_chars=max_chars)

        if mode == "import":
            sid = _resolve_session_id(session_id)
            bucket = _resolve_bucket(s3_bucket)
            prefix = _normalize_prefix(s3_prefix)
            imported = import_session_from_s3(session_id=sid, s3_bucket=bucket, s3_prefix=prefix, force=bool(force))
            return _success(
                (
                    f"Imported session `{sid}` to `{imported['local_dir']}`\n"
                    f"- files: {imported['file_count']}\n"
                    f"- bytes: {imported['total_bytes']}"
                ),
                max_chars=max_chars,
            )

        if mode == "sync":
            bucket = _resolve_bucket(s3_bucket)
            prefix = _normalize_prefix(s3_prefix)
            store = SessionStore()

            if session_id:
                session_ids = [_resolve_session_id(session_id)]
            else:
                local_ids = {str(item.get("id") or "").strip() for item in store.list(limit=500)}
                local_ids = {sid for sid in local_ids if sid}
                s3_ids = set(_list_s3_session_ids(bucket=bucket, prefix=prefix, max_results=500))
                session_ids = sorted(local_ids | s3_ids)

            if not session_ids:
                return _success("(no sessions to sync)", max_chars=max_chars)

            exported = 0
            imported = 0
            unchanged = 0
            notes: list[str] = []

            s3 = _s3_client()
            for sid in session_ids:
                local_meta = _load_json_file(_session_dir(sid) / "meta.json", default={})
                if not isinstance(local_meta, dict):
                    local_meta = {}
                local_updated = _parse_timestamp(local_meta.get("updated_at") or local_meta.get("created_at"))

                safe_sid = _sanitize_s3_segment(sid)
                meta_key = f"{prefix}/{safe_sid}/meta.json"
                s3_meta: dict[str, Any] = {}
                with contextlib.suppress(Exception):
                    response = s3.get_object(Bucket=bucket, Key=meta_key)
                    parsed = json.loads(response.get("Body").read().decode("utf-8", errors="replace"))
                    if isinstance(parsed, dict):
                        s3_meta = parsed
                s3_updated = _parse_timestamp(s3_meta.get("updated_at") or s3_meta.get("created_at"))

                if local_updated <= 0 and s3_updated <= 0:
                    unchanged += 1
                    notes.append(f"- `{sid}` unchanged (no metadata timestamps)")
                    continue

                if local_updated > s3_updated:
                    export_session_to_s3(session_id=sid, s3_bucket=bucket, s3_prefix=prefix)
                    exported += 1
                    notes.append(f"- `{sid}` exported (local newer)")
                elif s3_updated > local_updated:
                    import_session_from_s3(session_id=sid, s3_bucket=bucket, s3_prefix=prefix, force=True)
                    imported += 1
                    notes.append(f"- `{sid}` imported (S3 newer)")
                else:
                    unchanged += 1
                    notes.append(f"- `{sid}` unchanged")

            summary = [
                "# Session Sync",
                "",
                f"- exported: {exported}",
                f"- imported: {imported}",
                f"- unchanged: {unchanged}",
                "",
                *notes[:50],
            ]
            return _success("\n".join(summary).strip(), max_chars=max_chars)

        if mode == "promote_to_kb":
            sid = _resolve_session_id(session_id)
            kb_id = _resolve_kb_id(knowledge_base_id)
            promoted = promote_session_to_kb(session_id=sid, knowledge_base_id=kb_id, content_filter=content_filter)
            return _success(
                (
                    f"Promoted session `{sid}` to KB `{kb_id}`\n"
                    f"- documents: {promoted['promoted_count']}\n"
                    f"- content_filter: {content_filter}"
                ),
                max_chars=max_chars,
            )

        if mode == "promote_artifact":
            raw_artifact_id = (artifact_id or "").strip()
            if not raw_artifact_id:
                return _error("artifact_id is required for action=promote_artifact", max_chars=max_chars)
            kb_id = _resolve_kb_id(knowledge_base_id)
            promoted = promote_artifact_to_kb(artifact_id=raw_artifact_id, knowledge_base_id=kb_id)
            return _success(
                (
                    f"Promoted artifact `{raw_artifact_id}` to KB `{kb_id}`\n"
                    f"- documents: {promoted['promoted_count']}"
                ),
                max_chars=max_chars,
            )

        return _error(f"Unknown action: {mode}", max_chars=max_chars)

    except FileExistsError as exc:
        return _error(str(exc), max_chars=max_chars)
    except FileNotFoundError as exc:
        return _error(str(exc), max_chars=max_chars)
    except ValueError as exc:
        return _error(str(exc), max_chars=max_chars)
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)
    except Exception as exc:
        return _error(f"session_s3 failed: {exc}", max_chars=max_chars)
