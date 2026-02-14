from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

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


def test_git_tool_commit_uses_streaming_and_extended_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.git as git_tool

    captured: dict[str, object] = {}

    def _fake_run_streaming(
        cmd: list[str],
        *,
        cwd: Path,
        timeout_s: int,
        heartbeat_s: int = 15,
    ) -> tuple[int, str, str]:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout_s"] = timeout_s
        captured["heartbeat_s"] = heartbeat_s
        return 0, "[main abc123] test\n 1 file changed", ""

    monkeypatch.setattr(git_tool, "_run_streaming", _fake_run_streaming)
    with mock.patch("builtins.print") as mock_print:
        result = git_tool.git(action="commit", message="test", timeout_s=30)

    assert captured["cmd"] == ["git", "commit", "-m", "test"]
    assert captured["timeout_s"] == 1800
    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "abc123" in text
    assert any("running commit" in str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert any("commit finished" in str(call.args[0]) for call in mock_print.call_args_list if call.args)


def test_git_tool_commit_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.git as git_tool

    def _fake_run_streaming(
        _cmd: list[str],
        *,
        cwd: Path,
        timeout_s: int,
        heartbeat_s: int = 15,
    ) -> tuple[int, str, str]:
        del cwd
        del timeout_s
        del heartbeat_s
        return 1, "", "Command timed out after 1800s."

    monkeypatch.setattr(git_tool, "_run_streaming", _fake_run_streaming)
    result = git_tool.git(action="commit", message="test", timeout_s=5)

    assert result.get("status") == "error"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "timed out" in text.lower()
