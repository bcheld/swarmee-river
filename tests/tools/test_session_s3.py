from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.session.store import SessionStore
from swarmee_river.state_paths import set_state_dir_override
from tools.session_s3 import _generate_session_summary, session_s3


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803
        payload = Body if isinstance(Body, bytes) else bytes(Body)
        self.objects[(Bucket, Key)] = payload

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        payload = self.objects[(Bucket, Key)]
        return {"Body": _Body(payload)}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs.get("Bucket") or "")
        prefix = str(kwargs.get("Prefix") or "")
        delimiter = kwargs.get("Delimiter")

        keys = sorted([key for (b, key) in self.objects.keys() if b == bucket and key.startswith(prefix)])

        if delimiter == "/":
            common: set[str] = set()
            for key in keys:
                suffix = key[len(prefix) :]
                first = suffix.split("/", 1)[0]
                if first:
                    common.add(f"{prefix}{first}/")
            return {
                "CommonPrefixes": [{"Prefix": item} for item in sorted(common)],
                "IsTruncated": False,
            }

        return {
            "Contents": [{"Key": key, "Size": len(self.objects[(bucket, key)])} for key in keys],
            "IsTruncated": False,
        }


@pytest.fixture
def session_workspace(tmp_path: Path) -> SessionStore:
    set_state_dir_override(tmp_path / ".swarmee", cwd=tmp_path)
    try:
        yield SessionStore()
    finally:
        set_state_dir_override(None)


def _text(result: dict[str, object]) -> str:
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            value = first.get("text")
            if isinstance(value, str):
                return value
    return ""


def test_generate_session_summary_includes_tools_plans_and_errors() -> None:
    meta = {
        "id": "sess-1",
        "created_at": "2026-02-20T10:00:00",
        "updated_at": "2026-02-20T10:30:00",
        "provider": "openai",
        "tier": "deep",
        "model_id": "gpt-5.2",
        "last_plan": {"summary": "Ship S3 session export"},
    }
    messages = [
        {"role": "user", "content": [{"text": "Export my session and summarize it"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "t1", "name": "file_read", "input": {"path": "a.py"}}},
                {"text": "Working on it."},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "status": "error",
                        "content": [{"text": "File not found"}],
                    }
                }
            ],
        },
    ]

    summary = _generate_session_summary(meta, messages)

    assert "Session ID: `sess-1`" in summary
    assert "## Conversation Outline" in summary
    assert "## Plan Summaries" in summary
    assert "## Tool Usage" in summary
    assert "file_read" in summary
    assert "## Errors" in summary


def test_session_s3_export_uploads_expected_files(
    session_workspace: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = session_workspace.create(session_id="sess-1", meta={"provider": "openai", "tier": "deep"})
    meta = session_workspace.read_meta(sid)
    meta["model_id"] = "gpt-5.2"
    session_workspace.save(sid, meta=meta, state={"ok": True}, last_plan={"summary": "Plan summary"})
    session_workspace.save_messages(
        sid,
        [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"text": "world"}]},
        ],
    )

    fake_s3 = _FakeS3()
    monkeypatch.setattr("tools.session_s3._s3_client", lambda: fake_s3)

    result = session_s3(action="export", session_id=sid, s3_bucket="bucket", s3_prefix="swarmee/sessions")

    assert result.get("status") == "success"
    text = _text(result)
    assert "files: 5" in text
    keys = [key for (bucket, key) in fake_s3.objects.keys() if bucket == "bucket"]
    assert f"swarmee/sessions/{sid}/meta.json" in keys
    assert f"swarmee/sessions/{sid}/messages.jsonl" in keys
    assert f"swarmee/sessions/{sid}/state.json" in keys
    assert f"swarmee/sessions/{sid}/last_plan.json" in keys
    assert f"swarmee/sessions/{sid}/summary.md" in keys


def test_session_s3_list_sorts_by_updated_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_s3 = _FakeS3()
    fake_s3.put_object(
        Bucket="bucket",
        Key="sessions/s1/meta.json",
        Body=json.dumps({"id": "s1", "updated_at": "2026-01-01T00:00:00"}).encode("utf-8"),
    )
    fake_s3.put_object(
        Bucket="bucket",
        Key="sessions/s2/meta.json",
        Body=json.dumps({"id": "s2", "updated_at": "2026-02-01T00:00:00"}).encode("utf-8"),
    )
    monkeypatch.setattr("tools.session_s3._s3_client", lambda: fake_s3)

    result = session_s3(action="list", s3_bucket="bucket", s3_prefix="sessions", max_results=10)

    assert result.get("status") == "success"
    text = _text(result)
    assert text.index("s2") < text.index("s1")


def test_session_s3_import_respects_force(
    session_workspace: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "sess-2"
    existing_dir = session_workspace.root_dir / sid
    existing_dir.mkdir(parents=True, exist_ok=True)

    fake_s3 = _FakeS3()
    fake_s3.put_object(Bucket="bucket", Key=f"sessions/{sid}/meta.json", Body=b"{}\n")
    fake_s3.put_object(Bucket="bucket", Key=f"sessions/{sid}/messages.jsonl", Body=b"[]\n")
    monkeypatch.setattr("tools.session_s3._s3_client", lambda: fake_s3)

    result_no_force = session_s3(action="import", session_id=sid, s3_bucket="bucket", s3_prefix="sessions")
    assert result_no_force.get("status") == "error"
    assert "force=True" in _text(result_no_force)

    result_force = session_s3(
        action="import",
        session_id=sid,
        s3_bucket="bucket",
        s3_prefix="sessions",
        force=True,
    )
    assert result_force.get("status") == "success"
    assert (existing_dir / "meta.json").exists()


def test_session_s3_promote_to_kb_and_promote_artifact(
    session_workspace: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = session_workspace.create(session_id="sess-3", meta={"provider": "openai"})
    session_workspace.save(sid, meta=session_workspace.read_meta(sid), last_plan={"summary": "Ship plan"})
    session_workspace.save_messages(
        sid,
        [
            {"role": "user", "content": [{"text": "do the thing"}]},
            {"role": "assistant", "content": [{"text": "completed successfully"}]},
        ],
    )

    captured_docs: list[dict[str, Any]] = []

    def _fake_ingest(*, knowledge_base_id: str, docs: list[dict[str, Any]]) -> int:
        assert knowledge_base_id == "kb-123"
        captured_docs.extend(docs)
        return len(docs)

    monkeypatch.setattr("tools.session_s3._ingest_documents_to_kb", _fake_ingest)

    promote_result = session_s3(
        action="promote_to_kb",
        session_id=sid,
        knowledge_base_id="kb-123",
        content_filter="plans",
    )
    assert promote_result.get("status") == "success"
    assert any(doc.get("content_type") == "plan" for doc in captured_docs)

    artifact_store = ArtifactStore()
    ref = artifact_store.write_text(kind="note", text="artifact content", suffix="md")

    promote_artifact_result = session_s3(
        action="promote_artifact",
        artifact_id=ref.artifact_id,
        knowledge_base_id="kb-123",
    )
    assert promote_artifact_result.get("status") == "success"
    assert any(doc.get("content_type") == "artifact" for doc in captured_docs)
