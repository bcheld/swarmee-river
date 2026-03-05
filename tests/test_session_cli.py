from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from swarmee_river import swarmee
from swarmee_river.session.store import SessionStore
from swarmee_river.state_paths import set_state_dir_override


def _run_main(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "argv", argv)
    with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as exc:
        swarmee.main()
    code = int(exc.value.code) if isinstance(exc.value.code, int) else 1
    return code, stdout.getvalue()


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, str):
                fh.write(row.rstrip("\n") + "\n")
            else:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_main_dispatches_session_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_run(args: list[str]) -> int:
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(swarmee, "_run_session_command", _fake_run)
    monkeypatch.setattr(sys, "argv", ["swarmee", "session", "list", "--limit", "7"])

    with pytest.raises(SystemExit) as exc:
        swarmee.main()

    assert exc.value.code == 0
    assert captured["args"] == ["list", "--limit", "7"]


def test_session_list_and_index_build_graph_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_state_dir_override(tmp_path / ".swarmee", cwd=tmp_path)
    try:
        store = SessionStore()
        session_id = "sid-index"
        store.create(meta={"cwd": str(tmp_path), "turn_count": 2}, session_id=session_id)
        store.save_messages(
            session_id,
            [
                {"role": "user", "content": [{"text": "hello"}]},
                {"role": "assistant", "content": [{"text": "world"}]},
                {"role": "user", "content": [{"text": "next"}]},
                {"role": "assistant", "content": [{"text": "done"}]},
            ],
            max_messages=10,
        )
        _write_jsonl(
            tmp_path / ".swarmee" / "logs" / f"20260223_120001_{session_id}.jsonl",
            [
                {"event": "after_tool_call", "tool": "file_read", "duration_s": 0.1, "result": '{"ok": true}'},
                {"event": "after_model_call", "duration_s": 0.3},
            ],
        )

        code, output = _run_main(["swarmee", "session", "list", "--limit", "5"], monkeypatch)
        assert code == 0
        assert "session_id\tupdated_at\tturn_count\tcwd" in output
        assert f"{session_id}\t" in output

        code, output = _run_main(["swarmee", "session", "index", "--session", session_id], monkeypatch)
        assert code == 0
        assert f"Indexed session: {session_id}" in output
        assert "Turns: 2" in output
        assert "Tool calls: 1" in output

        graph_path = tmp_path / ".swarmee" / "sessions" / session_id / "graph_index.json"
        assert graph_path.exists()
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        assert graph["session_id"] == session_id
        assert graph["stats"]["turns"] == 2
        assert graph["tools"]["counts"] == {"file_read": 1}
    finally:
        set_state_dir_override(None)


def test_session_export_markdown_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_state_dir_override(tmp_path / ".swarmee", cwd=tmp_path)
    try:
        store = SessionStore()
        session_id = "sid-export"
        store.create(meta={"cwd": str(tmp_path)}, session_id=session_id)
        store.save_messages(
            session_id,
            [
                {"role": "user", "content": [{"text": "Find bug"}]},
                {"role": "assistant", "content": [{"text": "I will inspect logs."}]},
                {"role": "user", "content": [{"text": "Ship fix"}]},
                {"role": "assistant", "content": [{"text": "Done and verified."}]},
            ],
            max_messages=20,
        )
        base_meta = store.read_meta(session_id)
        store.save(session_id, meta=base_meta, last_plan={"summary": "Fix issue and verify tests", "steps": []})
        _write_jsonl(
            tmp_path / ".swarmee" / "logs" / f"20260223_121500_{session_id}.jsonl",
            [
                {"event": "after_tool_call", "tool": "file_search", "duration_s": 0.2, "result": '{"ok": true}'},
                {"event": "after_tool_call", "tool": "shell", "duration_s": 0.4, "result": '{"error":"boom"}'},
                {"event": "after_invocation", "duration_s": 1.0},
            ],
        )

        code, output = _run_main(
            ["swarmee", "session", "export", "--session", session_id, "--format", "md"],
            monkeypatch,
        )
        assert code == 0
        assert "# Session Export" in output
        assert "## Tool Summary" in output
        assert "## Errors" in output
        assert "## Last Plan" in output
        assert "Fix issue and verify tests" in output
        assert "### Turn 1" in output
        assert "Find bug" in output

        out_path = tmp_path / "out" / "export.json"
        code, output = _run_main(
            [
                "swarmee",
                "session",
                "export",
                "--session",
                session_id,
                "--format",
                "json",
                "--out",
                str(out_path),
            ],
            monkeypatch,
        )
        assert code == 0
        assert f"Wrote export: {out_path}" in output
        exported = json.loads(out_path.read_text(encoding="utf-8"))
        assert exported["session_id"] == session_id
        assert exported["tools"]["counts"] == {"file_search": 1, "shell": 1}
    finally:
        set_state_dir_override(None)


def test_session_branch_uses_user_turns_and_wont_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_state_dir_override(tmp_path / ".swarmee", cwd=tmp_path)
    try:
        store = SessionStore()
        source_sid = "sid-source"
        store.create(meta={"cwd": str(tmp_path)}, session_id=source_sid)
        source_messages = [
            {"role": "system", "content": [{"text": "system"}]},
            {"role": "user", "content": [{"toolUse": {"name": "file_read", "input": {"path": "x"}}}]},
            {"role": "user", "content": [{"text": "Turn 1"}]},
            {"role": "assistant", "content": [{"text": "A1"}]},
            {"role": "user", "content": [{"text": "Turn 2"}]},
            {"role": "assistant", "content": [{"text": "A2"}]},
            {"role": "user", "content": [{"text": "Turn 3"}]},
            {"role": "assistant", "content": [{"text": "A3"}]},
        ]
        store.save_messages(source_sid, source_messages, max_messages=50)

        code, output = _run_main(
            [
                "swarmee",
                "session",
                "branch",
                "--from-session",
                source_sid,
                "--turn",
                "2",
                "--new-session",
                "sid-branch",
            ],
            monkeypatch,
        )
        assert code == 0
        assert "Branched session: sid-branch" in output

        branched_messages = store.load_messages("sid-branch", max_messages=1000)
        user_texts = [
            swarmee._extract_text_from_message_for_replay(msg).strip()
            for msg in branched_messages
            if isinstance(msg, dict) and str(msg.get("role", "")).strip().lower() == "user"
        ]
        assert "Turn 1" in user_texts
        assert "Turn 2" in user_texts
        assert "Turn 3" not in user_texts

        code, output = _run_main(
            [
                "swarmee",
                "session",
                "branch",
                "--from-session",
                source_sid,
                "--turn",
                "1",
                "--new-session",
                "sid-branch",
            ],
            monkeypatch,
        )
        assert code == 1
        assert "Error:" in output
    finally:
        set_state_dir_override(None)
