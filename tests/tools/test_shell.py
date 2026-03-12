from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from tools.shell import shell


def test_shell_uses_devnull_stdin_in_non_interactive_mode() -> None:
    with mock.patch("tools.shell.run_subprocess_capture_interruptible") as run:
        run.return_value = SimpleNamespace(stdout="", stderr="", returncode=0, interrupted=False, timed_out=False)

        result = shell(command="echo ok", non_interactive_mode=True)

        assert result["status"] == "success"
        assert run.call_args.kwargs["stdin"] is not None


def test_shell_allows_inherited_stdin_in_interactive_mode() -> None:
    with mock.patch("tools.shell.run_subprocess_capture_interruptible") as run:
        run.return_value = SimpleNamespace(stdout="", stderr="", returncode=0, interrupted=False, timed_out=False)

        result = shell(command="echo ok", non_interactive_mode=False)

        assert result["status"] == "success"
        assert run.call_args.kwargs["stdin"] is None


def test_shell_reports_interrupt() -> None:
    with mock.patch("tools.shell.run_subprocess_capture_interruptible") as run:
        run.return_value = SimpleNamespace(
            stdout="",
            stderr="Command interrupted.",
            returncode=1,
            interrupted=True,
            timed_out=False,
        )

        result = shell(command="sleep 5", non_interactive_mode=True)

        assert result["status"] == "error"
        combined = result["content"][1]["text"]
        assert "Command interrupted." in combined
