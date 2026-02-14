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
