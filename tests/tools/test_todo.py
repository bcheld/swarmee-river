from __future__ import annotations

from pathlib import Path

from tools.todo import todoread, todowrite


def _content_text(result: dict) -> str:
    return (result.get("content") or [{"text": ""}])[0].get("text", "")


def test_todowrite_uses_swarmee_state_dir_env(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "custom-state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(state_root))

    result = todowrite("- [ ] Implement todo tools", cwd=str(project_root))

    assert result.get("status") == "success"
    todo_file = state_root / "todo.md"
    assert todo_file.exists()
    assert todo_file.read_text(encoding="utf-8") == "- [ ] Implement todo tools"


def test_todowrite_respects_relative_state_dir_from_cwd(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("SWARMEE_STATE_DIR", ".local-state")

    result = todowrite("- [ ] relative path", cwd=str(project_root))

    assert result.get("status") == "success"
    todo_file = project_root / ".local-state" / "todo.md"
    assert todo_file.exists()
    assert todo_file.read_text(encoding="utf-8") == "- [ ] relative path"


def test_todowrite_then_todoread_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / "state"))
    todos = "- [x] done\n- [ ] next"

    write_result = todowrite(todos, cwd=str(tmp_path))
    read_result = todoread(cwd=str(tmp_path))

    assert write_result.get("status") == "success"
    assert read_result.get("status") == "success"
    assert _content_text(read_result) == todos


def test_todoread_returns_no_todos_when_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / "empty-state"))

    result = todoread(cwd=str(tmp_path))

    assert result.get("status") == "success"
    assert _content_text(result) == "(no todos)"
