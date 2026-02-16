from __future__ import annotations

from pathlib import Path

from tools.file_ops import file_list, file_read, file_search


def test_file_list_includes_created_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")

    result = file_list(cwd=str(tmp_path))

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "a.txt" in text


def test_file_search_finds_match(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")

    result = file_search("needle", cwd=str(tmp_path))

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "needle" in text


def test_file_read_reads_with_line_numbers(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("line1\nline2\n", encoding="utf-8")

    result = file_read("a.txt", cwd=str(tmp_path), start_line=2, max_lines=1)

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "2 | line2" in text


def test_file_search_falls_back_without_rg(tmp_path: Path, monkeypatch) -> None:
    import tools.file_ops as file_ops

    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")

    monkeypatch.setattr(file_ops, "_run_rg", lambda *_args, **_kwargs: (None, "", "rg not found"))
    result = file_ops.file_search("needle", cwd=str(tmp_path))

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "a.txt" in text
    assert "needle" in text


def test_file_search_invalid_regex_returns_error(tmp_path: Path, monkeypatch) -> None:
    import tools.file_ops as file_ops

    (tmp_path / "a.txt").write_text("anything\n", encoding="utf-8")

    monkeypatch.setattr(file_ops, "_run_rg", lambda *_args, **_kwargs: (None, "", "rg not found"))
    result = file_ops.file_search("[", cwd=str(tmp_path))

    assert result.get("status") == "error"
