from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest import mock

from tools.shell import shell


def test_shell_uses_devnull_stdin_in_non_interactive_mode() -> None:
    with mock.patch("tools.shell.subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout="", stderr="", returncode=0)

        result = shell(command="echo ok", non_interactive_mode=True)

        assert result["status"] == "success"
        assert run.call_args.kwargs["stdin"] is subprocess.DEVNULL


def test_shell_allows_inherited_stdin_in_interactive_mode() -> None:
    with mock.patch("tools.shell.subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout="", stderr="", returncode=0)

        result = shell(command="echo ok", non_interactive_mode=False)

        assert result["status"] == "success"
        assert run.call_args.kwargs["stdin"] is None
