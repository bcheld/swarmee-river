from __future__ import annotations

import json
from pathlib import Path

from swarmee_river.session.store import SessionStore


def test_session_store_create_list_delete(tmp_path: Path) -> None:
    store = SessionStore(root_dir=tmp_path / "sessions")
    sid = store.create(meta={"cwd": "/tmp/demo"})

    entries = store.list()
    assert entries
    assert any(e.get("id") == sid for e in entries)

    store.delete(sid)
    assert not (tmp_path / "sessions" / sid).exists()


def test_session_store_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(root_dir=tmp_path / "sessions")
    sid = store.create(meta={"cwd": "/tmp/demo", "provider": "openai", "tier": "deep"})

    messages = [{"role": "user", "content": [{"text": "hi"}]}]
    state = {"foo": "bar"}
    last_plan = {"kind": "work_plan", "summary": "do thing", "steps": []}

    store.save(sid, meta={"cwd": "/tmp/demo2"}, messages=messages, state=state, last_plan=last_plan)
    meta, loaded_messages, loaded_state, loaded_last_plan = store.load(sid)

    assert meta["id"] == sid
    assert meta["cwd"] == "/tmp/demo2"
    assert loaded_messages == messages
    assert loaded_state == state
    assert loaded_last_plan == last_plan


def test_session_store_save_messages_and_load_messages_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(root_dir=tmp_path / "sessions")
    sid = store.create(meta={"cwd": str(tmp_path)})
    messages = [{"role": "user", "content": [{"text": f"msg {i}"}]} for i in range(220)]

    summary = store.save_messages(sid, messages)
    loaded = store.load_messages(sid)

    assert summary["version"] == 1
    assert summary["message_count"] == 200
    assert len(loaded) == 200
    assert loaded[0]["content"][0]["text"] == "msg 20"
    assert loaded[-1]["content"][0]["text"] == "msg 219"


def test_session_store_load_messages_ignores_corrupt_lines(tmp_path: Path) -> None:
    store = SessionStore(root_dir=tmp_path / "sessions")
    sid = store.create(meta={"cwd": str(tmp_path)})
    session_dir = tmp_path / "sessions" / sid
    log_path = session_dir / "messages.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("not-json\n" + json.dumps({"version": 1, "messages": [{"role": "user", "content": [{"text": "ok"}]}]}) + "\n")

    loaded = store.load_messages(sid)
    assert loaded == [{"role": "user", "content": [{"text": "ok"}]}]


def test_session_store_load_messages_handles_version_mismatch(tmp_path: Path) -> None:
    store = SessionStore(root_dir=tmp_path / "sessions")
    sid = store.create(meta={"cwd": str(tmp_path)})
    session_dir = tmp_path / "sessions" / sid
    log_path = session_dir / "messages.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"version": 99, "messages": [{"role": "user", "content": [{"text": "nope"}]}]}) + "\n")

    loaded = store.load_messages(sid)
    assert loaded == []
