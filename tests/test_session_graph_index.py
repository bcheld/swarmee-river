from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from swarmee_river.session.graph_index import (
    build_session_graph_index,
    load_session_graph_index,
    write_session_graph_index,
)


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, str):
                fh.write(row.rstrip("\n") + "\n")
            else:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_session_graph_index_builds_and_persists_with_corruption_tolerance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    session_id = "session-graph-mvp"
    state_root = tmp_path / ".swarmee"
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(state_root))
    messages_log = state_root / "sessions" / session_id / "messages.jsonl"

    final_messages = [
        {"role": "system", "content": [{"text": "system prompt"}]},
        {"role": "user", "content": [{"toolUse": {"name": "list_dir", "input": {"path": "."}}}]},
        {"role": "user", "content": [{"text": "Find all tests"}]},
        {"role": "assistant", "content": [{"text": "Scanning the repository now."}]},
        {"role": "assistant", "content": [{"toolUse": {"name": "file_search", "input": {"pattern": "test_"}}}]},
        {"role": "user", "content": [{"text": "Thanks"}]},
        {"role": "assistant", "content": [{"text": "Done."}]},
    ]
    _write_jsonl(
        messages_log,
        [
            "this is not json",
            {"version": 1, "messages": [{"role": "user", "content": [{"text": "older snapshot"}]}]},
            '{"version": 1, "messages": [',
            {"version": 1, "messages": final_messages},
        ],
    )

    old_log = state_root / "logs" / f"20260223_100000_{session_id}.jsonl"
    new_log = state_root / "logs" / f"20260223_120000_{session_id}.jsonl"
    _write_jsonl(
        old_log,
        [
            {"event": "after_tool_call", "tool": "ignored_tool", "duration_s": 0.01, "result": '{"ok": true}'},
        ],
    )
    _write_jsonl(
        new_log,
        [
            "{broken-json",
            {
                "event": "after_tool_call",
                "ts": "2026-02-23T12:00:00",
                "tool": "list_dir",
                "toolUseId": "tool-1",
                "duration_s": 0.2,
                "result": '{"ok": true}',
            },
            {
                "event": "after_tool_call",
                "ts": "2026-02-23T12:00:01",
                "tool": "calc",
                "toolUseId": "tool-2",
                "duration_s": 0.1,
                "result": "{truncated-json",
            },
            {
                "event": "before_model_call",
                "ts": "2026-02-23T12:00:01.5",
                "model_call_id": "model-1",
                "messages": 8,
                "system_prompt_chars": 4200,
                "tool_count": 15,
                "tool_schema_chars": 18000,
                "model_id": "gpt-4",
                "message_breakdown": {"user": 3, "assistant": 2, "tool": 3},
            },
            {
                "event": "after_model_call",
                "ts": "2026-02-23T12:00:02",
                "model_call_id": "model-1",
                "duration_s": 1.25,
                "model_id": "gpt-4",
                "usage": {"input_tokens": 2500, "output_tokens": 600, "cache_read_input_tokens": 1800},
            },
            {
                "event": "after_invocation",
                "ts": "2026-02-23T12:00:03",
                "invocation_id": "inv-1",
                "duration_s": 2.0,
            },
            {
                "event": "after_tool_call",
                "ts": "2026-02-23T12:00:04",
                "tool": "failing_tool",
                "toolUseId": "tool-3",
                "duration_s": 0.5,
                "result": '{"error": "boom"}',
            },
        ],
    )
    os.utime(old_log, (1000, 1000))
    os.utime(new_log, (2000, 2000))

    index = build_session_graph_index(session_id, cwd=tmp_path)

    assert sorted(index.keys()) == [
        "events",
        "generated_at",
        "schema",
        "schema_version",
        "session_id",
        "sources",
        "stats",
        "tools",
        "turns",
    ]
    assert index["schema"] == "session_graph_index"
    assert index["schema_version"] == 1
    assert index["session_id"] == session_id
    assert isinstance(index["generated_at"], str) and index["generated_at"]
    assert index["sources"] == {
        "messages_log": str(messages_log),
        "logs_files": [str(old_log), str(new_log)],
    }

    # Both log files are now merged: old_log contributes ignored_tool (success=True,
    # no error), new_log contributes list_dir, calc, model_call, invocation, failing_tool.
    assert index["stats"] == {"turns": 2, "tools": 4, "errors": 1}
    assert index["tools"]["counts"] == {"calc": 1, "failing_tool": 1, "ignored_tool": 1, "list_dir": 1}

    turns = index["turns"]
    assert [turn["user_text"] for turn in turns] == ["Find all tests", "Thanks"]
    assert [turn["assistant_text"] for turn in turns] == ["Scanning the repository now.", "Done."]

    assert [event["event"] for event in index["events"]] == [
        "after_tool_call",  # ignored_tool from old_log
        "after_tool_call",  # list_dir from new_log
        "after_tool_call",  # calc from new_log
        "after_model_call",
        "after_invocation",
        "after_tool_call",  # failing_tool
    ]
    calc_event = next(event for event in index["events"] if event.get("tool") == "calc")
    assert "success" not in calc_event
    assert "error" not in calc_event
    failing_event = next(event for event in index["events"] if event.get("tool") == "failing_tool")
    assert failing_event["success"] is False
    assert failing_event["error"] == "boom"

    # Verify enriched model call event (before_model_call merged into after_model_call).
    model_event = next(event for event in index["events"] if event["event"] == "after_model_call")
    assert model_event["model_call_id"] == "model-1"
    assert model_event["model_id"] == "gpt-4"
    assert model_event["usage"] == {"input_tokens": 2500, "output_tokens": 600, "cache_read_input_tokens": 1800}
    # Context composition fields merged from before_model_call.
    assert model_event["system_prompt_chars"] == 4200
    assert model_event["tool_count"] == 15
    assert model_event["tool_schema_chars"] == 18000
    assert model_event["messages"] == 8
    assert model_event["message_breakdown"] == {"user": 3, "assistant": 2, "tool": 3}

    monkeypatch.chdir(tmp_path)
    written = write_session_graph_index(session_id, index)
    assert written == state_root / "sessions" / session_id / "graph_index.json"

    loaded = load_session_graph_index(session_id)
    assert loaded == index
