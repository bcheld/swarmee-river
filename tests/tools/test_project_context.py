from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


@pytest.mark.parametrize("action", ["summary", "git_status", "files", "tree", "search", "read"])
def test_project_context_actions_work_without_rg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    import tools.file_ops as file_ops
    import tools.project_context as project_context

    monkeypatch.chdir(tmp_path)
    _run(["git", "init", "-q"], cwd=tmp_path)

    (tmp_path / "a.txt").write_text("needle\nline2\n", encoding="utf-8")

    # Simulate a minimal environment without ripgrep.
    monkeypatch.setattr(file_ops, "_run_rg", lambda *_args, **_kwargs: (None, "", "rg not found"))

    real_subprocess_run = subprocess.run

    def git_only_run(cmd: list[str], *, cwd: Path, timeout_s: int = 15) -> tuple[int, str, str]:
        assert cmd and cmd[0] == "git"
        completed = real_subprocess_run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        return int(completed.returncode), completed.stdout or "", completed.stderr or ""

    monkeypatch.setattr(project_context, "_run", git_only_run)

    kwargs: dict[str, object] = {}
    if action == "search":
        kwargs["query"] = "needle"
    if action == "read":
        kwargs["path"] = "a.txt"

    result = project_context.run_project_context(action=action, **kwargs)

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert isinstance(text, str)
    assert text
