from __future__ import annotations

from pathlib import Path

import pytest

from swarmee_river.state_paths import scope_root, state_dir


def test_scope_root_prefers_nearest_git_ancestor(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b" / "c"
    nested.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)

    assert scope_root(cwd=nested) == repo_root.resolve()
    assert state_dir(cwd=nested) == repo_root.resolve() / ".swarmee"


def test_scope_root_falls_back_to_home_when_no_git(tmp_path: Path) -> None:
    assert scope_root(cwd=tmp_path) == Path.home().expanduser().resolve()


def test_state_dir_relative_override_resolves_from_scope_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "src"
    nested.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SWARMEE_STATE_DIR", ".custom-state")

    assert state_dir(cwd=nested) == nested.resolve() / ".custom-state"
