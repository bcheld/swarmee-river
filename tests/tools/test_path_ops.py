from __future__ import annotations

from pathlib import Path

from tools.path_ops import glob as glob_tool
from tools.path_ops import list as list_tool


def _content_text(result: dict) -> str:
    return (result.get("content") or [{"text": ""}])[0].get("text", "")


def test_list_excludes_hidden_by_default(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / ".hidden").write_text("secret\n", encoding="utf-8")
    (tmp_path / "dir").mkdir()

    result = list_tool(cwd=str(tmp_path))

    assert result.get("status") == "success"
    text = _content_text(result)
    assert "a.txt" in text
    assert "dir/" in text
    assert ".hidden" not in text


def test_list_can_include_hidden(tmp_path: Path) -> None:
    (tmp_path / ".hidden").write_text("secret\n", encoding="utf-8")

    result = list_tool(cwd=str(tmp_path), include_hidden=True)

    assert result.get("status") == "success"
    text = _content_text(result)
    assert ".hidden" in text


def test_glob_requires_double_star_for_recursive_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("top\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("nested\n", encoding="utf-8")

    result = glob_tool("*.txt", cwd=str(tmp_path))

    assert result.get("status") == "success"
    text = _content_text(result)
    assert "a.txt" in text
    assert "sub/b.txt" not in text

    result2 = glob_tool("**/*.txt", cwd=str(tmp_path))
    assert result2.get("status") == "success"
    text2 = _content_text(result2)
    assert "a.txt" in text2
    assert "sub/b.txt" in text2


def test_glob_skips_git_dir_by_default(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "secret.txt").write_text("nope\n", encoding="utf-8")

    result = glob_tool("**/secret.txt", cwd=str(tmp_path))

    assert result.get("status") == "success"
    text = _content_text(result)
    assert "secret.txt" not in text


def test_glob_can_include_dirs(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.txt").write_text("x\n", encoding="utf-8")

    result = glob_tool("data", cwd=str(tmp_path), include_dirs=True)

    assert result.get("status") == "success"
    text = _content_text(result)
    assert "data/" in text


def test_glob_rejects_parent_traversal(tmp_path: Path) -> None:
    result = glob_tool("../*.txt", cwd=str(tmp_path))
    assert result.get("status") == "error"
