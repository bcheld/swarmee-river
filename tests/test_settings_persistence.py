"""Settings persistence safety tests (doc 11 F1/F3).

A corrupt settings.json must never be silently replaced with defaults, and
save failures must surface to the user instead of pretending to succeed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import swarmee_river.settings as settings_module
from swarmee_river.settings import SwarmeeSettings, load_settings, save_settings
from swarmee_river.tui.mixins.settings import SettingsMixin


class _Host(SettingsMixin):
    def __init__(self) -> None:
        self.notifications: list[tuple[str, str]] = []
        self.transcript: list[str] = []

    def notify(self, message, *, title=None, severity="information", timeout=None):  # noqa: D401
        self.notifications.append((severity, str(message)))

    def _write_transcript_line(self, text: str) -> None:
        self.transcript.append(text)


@pytest.fixture
def project_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".swarmee").mkdir()
    return tmp_path


def _settings_path(project_dir: Path) -> Path:
    return project_dir / ".swarmee" / "settings.json"


def test_save_settings_is_atomic_and_leaves_no_temp_files(project_dir) -> None:
    path = _settings_path(project_dir)
    save_settings(SwarmeeSettings.from_dict({}), path=path)
    assert json.loads(path.read_text(encoding="utf-8"))
    leftovers = [p for p in path.parent.iterdir() if p.name != path.name]
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_corrupt_settings_file_notifies_once_and_falls_back_to_defaults(project_dir) -> None:
    path = _settings_path(project_dir)
    path.write_text('{"models": invalid-json', encoding="utf-8")

    host = _Host()
    payload, returned_path = host._load_project_settings_payload()
    assert returned_path == path
    assert isinstance(payload, dict) and payload  # defaults
    assert len(host.notifications) == 1
    assert host.notifications[0][0] == "error"

    # A second load does not re-notify while still corrupt.
    host._load_project_settings_payload()
    assert len(host.notifications) == 1


def test_save_backs_up_corrupt_file_before_overwriting(project_dir) -> None:
    path = _settings_path(project_dir)
    original = '{"models": invalid-json'
    path.write_text(original, encoding="utf-8")

    host = _Host()
    payload, _ = host._load_project_settings_payload()
    assert host._save_project_settings_payload(payload, path) is True

    backups = list(path.parent.glob("settings.json.broken-*"))
    assert len(backups) == 1, "the unparseable file must be preserved"
    assert backups[0].read_text(encoding="utf-8") == original
    assert json.loads(path.read_text(encoding="utf-8"))  # new file is valid


def test_save_failure_is_reported_not_silent(project_dir, monkeypatch) -> None:
    path = _settings_path(project_dir)

    def failing_save(settings, path=None):
        raise OSError("disk full")

    monkeypatch.setattr(settings_module, "save_settings", failing_save)

    host = _Host()
    payload, _ = host._load_project_settings_payload()
    assert host._save_project_settings_payload(payload, path) is False
    assert any(sev == "error" and "failed to save" in msg for sev, msg in host.notifications)


def test_legacy_default_tier_is_honored_on_load(project_dir) -> None:
    """A hand-edited `models.default_tier` must not be clobbered by the
    defaults template's materialized `default_selection`."""
    path = _settings_path(project_dir)
    path.write_text(json.dumps({"models": {"default_tier": "fast"}}), encoding="utf-8")
    assert load_settings(path).models.default_tier == "fast"


def test_settings_round_trip_preserves_custom_values(project_dir) -> None:
    path = _settings_path(project_dir)
    path.write_text(json.dumps({"models": {"default_tier": "fast"}}), encoding="utf-8")

    host = _Host()
    payload, _ = host._load_project_settings_payload()
    assert payload["models"]["default_tier"] == "fast"
    assert host._save_project_settings_payload(payload, path) is True
    assert load_settings(path).models.default_tier == "fast"
