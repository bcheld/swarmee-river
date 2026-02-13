from __future__ import annotations

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
