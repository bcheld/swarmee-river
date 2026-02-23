from __future__ import annotations

from pathlib import Path

from tools.office import office


def test_office_requires_path(tmp_path: Path) -> None:
    result = office(action="read", path="", cwd=str(tmp_path))
    assert result.get("status") == "error"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "path is required" in text.lower()


def test_office_write_refuses_outside_cwd(tmp_path: Path) -> None:
    result = office(
        action="write",
        path="../escape.docx",
        content="# Heading",
        cwd=str(tmp_path),
    )
    assert result.get("status") == "error"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "outside cwd" in text.lower()


def test_office_rejects_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = office(action="read", path="notes.txt", cwd=str(tmp_path))
    assert result.get("status") == "error"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "unsupported office format" in text.lower()
