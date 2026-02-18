from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from swarmee_river import project_map
from swarmee_river.cli import diagnostics
from tools import git as git_tool
from tools.patch_apply import patch_apply
from tools.project_context import _run as project_context_run
from tools.run_checks import run_checks
from tools.shell import shell


def _ok_result(stdout: str = "", stderr: str = "", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_git_run_uses_replace_errors() -> None:
    with mock.patch("tools.git.subprocess.run", return_value=_ok_result()) as run:
        git_tool._run(["git", "status"], cwd=Path("."), timeout_s=1)  # noqa: SLF001
    assert run.call_args.kwargs["encoding"] == "utf-8"
    assert run.call_args.kwargs["errors"] == "replace"


def test_git_streaming_uses_replace_errors() -> None:
    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = iter(())
            self.returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            return None

    with mock.patch("tools.git.subprocess.Popen", return_value=_FakeProc()) as popen:
        git_tool._run_streaming(["git", "status"], cwd=Path("."), timeout_s=1)  # noqa: SLF001
    assert popen.call_args.kwargs["encoding"] == "utf-8"
    assert popen.call_args.kwargs["errors"] == "replace"


def test_shell_tool_uses_replace_errors() -> None:
    with mock.patch("tools.shell.subprocess.run", return_value=_ok_result()) as run:
        shell(command="echo ok")
    assert run.call_args.kwargs["encoding"] == "utf-8"
    assert run.call_args.kwargs["errors"] == "replace"


def test_run_checks_uses_replace_errors(tmp_path: Path) -> None:
    with mock.patch("tools.run_checks.subprocess.run", return_value=_ok_result(stdout="ok\n")) as run:
        run_checks(action="run", commands=["echo ok"], cwd=str(tmp_path), timeout_s=1)
    assert run.call_args.kwargs["encoding"] == "utf-8"
    assert run.call_args.kwargs["errors"] == "replace"


def test_project_context_uses_replace_errors() -> None:
    with mock.patch("tools.project_context.subprocess.run", return_value=_ok_result()) as run:
        project_context_run(["git", "status"], cwd=Path("."), timeout_s=1)
    assert run.call_args.kwargs["encoding"] == "utf-8"
    assert run.call_args.kwargs["errors"] == "replace"


def test_patch_apply_uses_replace_errors(tmp_path: Path) -> None:
    with mock.patch("tools.patch_apply.subprocess.run", return_value=_ok_result()) as run:
        patch_apply(patch="diff --git a/a.txt b/a.txt\n", cwd=str(tmp_path), dry_run=True, timeout_s=1)
    assert run.call_args.kwargs["encoding"] == "utf-8"
    assert run.call_args.kwargs["errors"] == "replace"


def test_project_map_uses_replace_errors() -> None:
    with mock.patch("swarmee_river.project_map.subprocess.run", return_value=_ok_result()) as run:
        project_map._run(["git", "status"], cwd=Path("."), timeout_s=1)  # noqa: SLF001
    assert run.call_args.kwargs["encoding"] == "utf-8"
    assert run.call_args.kwargs["errors"] == "replace"


def test_diagnostics_uses_replace_errors() -> None:
    with mock.patch("swarmee_river.cli.diagnostics.subprocess.run", return_value=_ok_result()) as run:
        diagnostics._run_git(["status"], cwd=Path("."), timeout_s=1)  # noqa: SLF001
    assert run.call_args.kwargs["encoding"] == "utf-8"
    assert run.call_args.kwargs["errors"] == "replace"
