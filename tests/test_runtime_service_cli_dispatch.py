from __future__ import annotations

import sys

import pytest

from swarmee_river import swarmee


def test_main_dispatches_serve_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_run(args: list[str]) -> int:
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(swarmee, "_run_serve_command", _fake_run)
    monkeypatch.setattr(sys, "argv", ["swarmee", "serve", "--port", "7342"])

    with pytest.raises(SystemExit) as exc:
        swarmee.main()

    assert exc.value.code == 0
    assert captured["args"] == ["--port", "7342"]


def test_main_dispatches_attach_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_run(args: list[str]) -> int:
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(swarmee, "_run_attach_command", _fake_run)
    monkeypatch.setattr(sys, "argv", ["swarmee", "attach", "--session", "abc", "--tail"])

    with pytest.raises(SystemExit) as exc:
        swarmee.main()

    assert exc.value.code == 0
    assert captured["args"] == ["--session", "abc", "--tail"]


def test_main_dispatches_daemon_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_run(args: list[str]) -> int:
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(swarmee, "_run_daemon_command", _fake_run)
    monkeypatch.setattr(sys, "argv", ["swarmee", "daemon", "stop"])

    with pytest.raises(SystemExit) as exc:
        swarmee.main()

    assert exc.value.code == 0
    assert captured["args"] == ["stop"]


def test_main_dispatches_daemon_stop_all(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_run(args: list[str]) -> int:
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(swarmee, "_run_daemon_command", _fake_run)
    monkeypatch.setattr(sys, "argv", ["swarmee", "daemon", "stop", "all"])

    with pytest.raises(SystemExit) as exc:
        swarmee.main()

    assert exc.value.code == 0
    assert captured["args"] == ["stop", "all"]


def test_main_dispatches_diagnostics_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_render(*, args: list[str], cwd, surface: str) -> str:  # noqa: ANN001
        captured["args"] = list(args)
        captured["surface"] = surface
        return "ok"

    monkeypatch.setattr(swarmee, "render_diagnostics_command_for_surface", _fake_render)
    monkeypatch.setattr(sys, "argv", ["swarmee", "diagnostics", "doctor"])

    swarmee.main()

    assert captured["args"] == ["doctor"]
    assert captured["surface"] == "cli"
