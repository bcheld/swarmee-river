#!/usr/bin/env python3
"""
Smoke tests for optional TUI scaffolding.
"""

from __future__ import annotations

from swarmee_river.tui import app as tui_app


def test_run_tui_missing_textual(monkeypatch, capsys):
    real_import_module = tui_app.importlib.import_module

    def _mock_import_module(name: str, package: str | None = None):
        if name.startswith("textual"):
            raise ImportError("No module named 'textual'")
        return real_import_module(name, package)

    monkeypatch.setattr(tui_app.importlib, "import_module", _mock_import_module)

    exit_code = tui_app.run_tui()

    captured = capsys.readouterr()
    assert exit_code != 0
    assert 'pip install "swarmee-river[tui]"' in captured.err
    assert 'pip install -e ".[tui]"' in captured.err
