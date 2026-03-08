from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.hooks.file_diff_review import FileDiffReviewHooks


def _before_event(tool_name: str, tool_use_id: str, tool_input: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        tool_use={"name": tool_name, "toolUseId": tool_use_id, "input": tool_input},
        invocation_state={},
        cancel_tool=None,
    )


def _after_event(before: SimpleNamespace, result: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        tool_use=before.tool_use,
        invocation_state=before.invocation_state,
        result=result,
    )


def test_editor_replace_creates_file_diff_artifact_and_emits_event(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_SESSION_ID", "sid-editor")
    monkeypatch.setenv("SWARMEE_TUI_EVENTS", "1")
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr("swarmee_river.hooks.file_diff_review.write_stdout_jsonl", emitted.append)
    artifacts_dir = tmp_path / ".swarmee" / "artifacts"
    target = tmp_path / "notes.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    hook = FileDiffReviewHooks(artifacts_dir=artifacts_dir)
    before = _before_event(
        "editor",
        "tool-1",
        {"command": "replace", "path": "notes.txt", "old_str": "hello", "new_str": "goodbye", "cwd": str(tmp_path)},
    )
    hook.before_tool_call(before)

    target.write_text("goodbye\nworld\n", encoding="utf-8")
    hook.after_tool_call(_after_event(before, {"status": "success", "content": [{"text": "ok"}]}))

    entries = ArtifactStore(artifacts_dir=artifacts_dir).list(kind="file_diff")
    assert len(entries) == 1
    entry = entries[0]
    meta = entry.get("meta")
    assert isinstance(meta, dict)
    assert meta["session_id"] == "sid-editor"
    assert meta["tool"] == "editor"
    assert meta["changed_paths"] == ["notes.txt"]
    body = Path(str(entry["path"])).read_text(encoding="utf-8")
    assert "--- a/notes.txt" in body
    assert "+++ b/notes.txt" in body
    assert "-hello" in body
    assert "+goodbye" in body
    assert len(emitted) == 1
    assert emitted[0]["event"] == "file_diff"
    assert emitted[0]["paths"] == ["notes.txt"]


def test_patch_apply_multifile_creates_single_diff_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_SESSION_ID", "sid-patch")
    artifacts_dir = tmp_path / ".swarmee" / "artifacts"
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")

    patch_text = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-one
+ONE
diff --git a/b.txt b/b.txt
--- a/b.txt
+++ b/b.txt
@@ -1 +1 @@
-two
+TWO
"""

    hook = FileDiffReviewHooks(artifacts_dir=artifacts_dir)
    before = _before_event(
        "patch_apply",
        "tool-2",
        {"patch": patch_text, "cwd": str(tmp_path)},
    )
    hook.before_tool_call(before)

    (tmp_path / "a.txt").write_text("ONE\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("TWO\n", encoding="utf-8")
    hook.after_tool_call(_after_event(before, {"status": "success", "content": [{"text": "ok"}]}))

    entries = ArtifactStore(artifacts_dir=artifacts_dir).list(kind="file_diff")
    assert len(entries) == 1
    meta = entries[0].get("meta")
    assert isinstance(meta, dict)
    assert meta["changed_paths"] == ["a.txt", "b.txt"]
    assert meta["stats"]["files_changed"] == 2


def test_failed_or_dry_run_tool_call_does_not_create_diff_artifact(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / ".swarmee" / "artifacts"
    target = tmp_path / "notes.txt"
    target.write_text("hello\n", encoding="utf-8")

    hook = FileDiffReviewHooks(artifacts_dir=artifacts_dir)

    failed_before = _before_event(
        "editor",
        "tool-failed",
        {"command": "replace", "path": "notes.txt", "old_str": "hello", "new_str": "bye", "cwd": str(tmp_path)},
    )
    hook.before_tool_call(failed_before)
    target.write_text("bye\n", encoding="utf-8")
    hook.after_tool_call(_after_event(failed_before, {"status": "error", "content": [{"text": "boom"}]}))

    dry_run_before = _before_event(
        "patch_apply",
        "tool-dry-run",
        {
            "patch": "--- a/notes.txt\n+++ b/notes.txt\n@@ -1 +1 @@\n-hello\n+bye\n",
            "cwd": str(tmp_path),
            "dry_run": True,
        },
    )
    hook.before_tool_call(dry_run_before)
    hook.after_tool_call(_after_event(dry_run_before, {"status": "success", "content": [{"text": "ok"}]}))

    assert ArtifactStore(artifacts_dir=artifacts_dir).list(kind="file_diff") == []


def test_noop_write_does_not_create_diff_artifact(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / ".swarmee" / "artifacts"
    target = tmp_path / "same.txt"
    target.write_text("same\n", encoding="utf-8")

    hook = FileDiffReviewHooks(artifacts_dir=artifacts_dir)
    before = _before_event(
        "editor",
        "tool-3",
        {"command": "write", "path": "same.txt", "file_text": "same\n", "cwd": str(tmp_path)},
    )
    hook.before_tool_call(before)
    hook.after_tool_call(_after_event(before, {"status": "success", "content": [{"text": "ok"}]}))

    assert ArtifactStore(artifacts_dir=artifacts_dir).list(kind="file_diff") == []


def test_new_and_deleted_text_files_render_as_unified_diffs(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / ".swarmee" / "artifacts"

    create_hook = FileDiffReviewHooks(artifacts_dir=artifacts_dir)
    create_before = _before_event(
        "editor",
        "tool-new",
        {"command": "write", "path": "new.txt", "file_text": "created\n", "cwd": str(tmp_path)},
    )
    create_hook.before_tool_call(create_before)
    (tmp_path / "new.txt").write_text("created\n", encoding="utf-8")
    create_hook.after_tool_call(_after_event(create_before, {"status": "success", "content": [{"text": "ok"}]}))

    delete_target = tmp_path / "gone.txt"
    delete_target.write_text("remove me\n", encoding="utf-8")
    delete_hook = FileDiffReviewHooks(artifacts_dir=artifacts_dir)
    delete_before = _before_event(
        "patch_apply",
        "tool-delete",
        {"patch": "--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-remove me\n", "cwd": str(tmp_path)},
    )
    delete_hook.before_tool_call(delete_before)
    delete_target.unlink()
    delete_hook.after_tool_call(_after_event(delete_before, {"status": "success", "content": [{"text": "ok"}]}))

    entries = ArtifactStore(artifacts_dir=artifacts_dir).list(kind="file_diff")
    bodies = [Path(str(entry["path"])).read_text(encoding="utf-8") for entry in entries]

    assert any("+++ b/new.txt" in body and "+created" in body for body in bodies)
    assert any("--- a/gone.txt" in body and "-remove me" in body for body in bodies)


def test_binary_change_creates_summary_only_diff_artifact(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / ".swarmee" / "artifacts"
    target = tmp_path / "image.bin"
    target.write_bytes(b"\x00\x01before")

    hook = FileDiffReviewHooks(artifacts_dir=artifacts_dir)
    before = _before_event(
        "editor",
        "tool-4",
        {"command": "write", "path": "image.bin", "file_text": "after\n", "cwd": str(tmp_path)},
    )
    hook.before_tool_call(before)

    target.write_bytes(b"\x00\x01after")
    hook.after_tool_call(_after_event(before, {"status": "success", "content": [{"text": "ok"}]}))

    entries = ArtifactStore(artifacts_dir=artifacts_dir).list(kind="file_diff")
    assert len(entries) == 1
    meta = entries[0].get("meta")
    assert isinstance(meta, dict)
    assert meta["non_text_paths"] == [{"path": "image.bin", "before": "binary", "after": "binary"}]
    body = Path(str(entries[0]["path"])).read_text(encoding="utf-8")
    assert "Non-text file changes captured" in body
