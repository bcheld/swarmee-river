from __future__ import annotations

import json
from pathlib import Path

from tools.notebook_read import notebook_read


def _write_notebook(path: Path) -> None:
    payload = {
        "cells": [
            {"cell_type": "markdown", "id": "markdown-0", "metadata": {}, "source": ["# Title\n", "Intro text\n"]},
            {
                "cell_type": "code",
                "id": "code-1",
                "execution_count": 1,
                "metadata": {},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": ["hello\n"]}],
                "source": ["print('hello')\n"],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_notebook_read_renders_markdown_and_code(tmp_path: Path) -> None:
    notebook_path = tmp_path / "demo.ipynb"
    _write_notebook(notebook_path)

    result = notebook_read("demo.ipynb", cwd=str(tmp_path))

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "[markdown:0]" in text
    assert "[code:1]" in text
    assert "print('hello')" in text


def test_notebook_read_can_include_outputs(tmp_path: Path) -> None:
    notebook_path = tmp_path / "demo.ipynb"
    _write_notebook(notebook_path)

    result = notebook_read("demo.ipynb", cwd=str(tmp_path), include_outputs=True)

    assert result.get("status") == "success"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "Outputs:" in text
    assert "hello" in text


def test_notebook_read_outside_scope_is_blocked(tmp_path: Path) -> None:
    outside_dir = tmp_path.parent / "outside-nb"
    outside_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = outside_dir / "secret.ipynb"
    _write_notebook(notebook_path)

    result = notebook_read(str(notebook_path), cwd=str(tmp_path))

    assert result.get("status") == "error"
    text = (result.get("content") or [{"text": ""}])[0].get("text", "")
    assert "outside the current scope" in text
    assert "change cwd/scope" in text
