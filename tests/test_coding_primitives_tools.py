from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


def test_git_tool_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.git import git

    monkeypatch.chdir(tmp_path)
    _run(["git", "init", "-q"], cwd=tmp_path)

    result = git(action="status")

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert isinstance(text, str)


def test_run_checks_runs_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.run_checks import run_checks

    monkeypatch.chdir(tmp_path)

    result = run_checks(action="run", commands=["echo ok"], timeout_s=60)

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "ok" in text.lower()


def test_patch_apply_dry_run_then_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.patch_apply import patch_apply

    monkeypatch.chdir(tmp_path)
    _run(["git", "init", "-q"], cwd=tmp_path)

    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello\n", encoding="utf-8")
    _run(["git", "add", "hello.txt"], cwd=tmp_path)
    _run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "-m",
            "init",
            "-q",
        ],
        cwd=tmp_path,
    )

    file_path.write_text("hello world\n", encoding="utf-8")
    patch = _run(["git", "diff"], cwd=tmp_path).stdout
    assert patch.strip()

    # Reset file back to the pre-patch state so `git apply` can apply cleanly.
    file_path.write_text("hello\n", encoding="utf-8")

    dry = patch_apply(patch=patch, dry_run=True, timeout_s=60)
    assert dry.get("status") == "success"

    applied = patch_apply(patch=patch, dry_run=False, timeout_s=60)
    assert applied.get("status") == "success"
    assert file_path.read_text(encoding="utf-8") == "hello world\n"
