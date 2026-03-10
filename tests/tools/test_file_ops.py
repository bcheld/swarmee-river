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


def test_file_read_uses_cache_friendly_defaults(tmp_path: Path) -> None:
    body = "\n".join(f"line {idx} {'x' * 40}" for idx in range(1, 301))
    (tmp_path / "big.txt").write_text(body, encoding="utf-8")

    result = file_read("big.txt", cwd=str(tmp_path))

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "truncated to 4000 chars" in text
    assert "121 |" not in text


def test_file_read_suggests_office_tool_for_office_extensions(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    path.write_bytes(b"PK\x03\x04not-a-real-docx")

    result = file_read("sample.docx", cwd=str(tmp_path))

    assert result.get("status") == "error"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "This is a binary Office file" in text
    assert "office(action='read', path='sample.docx')" in text


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


def test_file_search_passes_explicit_search_root_to_rg(tmp_path: Path, monkeypatch) -> None:
    import tools.file_ops as file_ops

    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_rg(args, *, cwd, timeout_s=15):
        captured["args"] = list(args)
        captured["cwd"] = cwd
        captured["timeout_s"] = timeout_s
        return 0, "a.txt:1:needle\n", ""

    monkeypatch.setattr(file_ops, "_run_rg", _fake_run_rg)
    result = file_ops.file_search("needle", cwd=str(tmp_path))

    assert result.get("status") == "success"
    assert captured["args"][-1] == "."


def test_run_rg_uses_devnull_stdin(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    import tools.file_ops as file_ops

    class _Completed:
        returncode = 1
        stdout = ""
        stderr = ""

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return _Completed()

    monkeypatch.setattr(file_ops.subprocess, "run", _fake_run)
    code, out, err = file_ops._run_rg(["needle", "."], cwd=tmp_path)

    assert code == 1
    assert out == ""
    assert err == ""
    assert captured.get("stdin") is subprocess.DEVNULL
