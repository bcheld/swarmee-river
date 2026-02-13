from __future__ import annotations

from unittest import mock

from swarmee_river.harness.context_snapshot import build_context_snapshot


def test_interactive_preflight_is_silent_by_default(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_PREFLIGHT", "enabled")
    monkeypatch.setenv("SWARMEE_PROJECT_MAP", "disabled")
    monkeypatch.delenv("SWARMEE_PREFLIGHT_PRINT", raising=False)

    artifact_store = mock.MagicMock()

    with (
        mock.patch("swarmee_river.harness.context_snapshot.run_project_context") as run_project_context,
        mock.patch("builtins.print") as print_mock,
    ):
        run_project_context.return_value = {"status": "success", "content": [{"text": "snapshot"}]}

        snapshot = build_context_snapshot(
            artifact_store=artifact_store,
            interactive=True,
            default_preflight_level="summary",
        )

        assert snapshot.preflight_prompt_section == "Project context snapshot:\nsnapshot"
        print_mock.assert_not_called()


def test_interactive_preflight_prints_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_PREFLIGHT", "enabled")
    monkeypatch.setenv("SWARMEE_PROJECT_MAP", "disabled")
    monkeypatch.setenv("SWARMEE_PREFLIGHT_PRINT", "enabled")

    artifact_store = mock.MagicMock()

    with (
        mock.patch("swarmee_river.harness.context_snapshot.run_project_context") as run_project_context,
        mock.patch("builtins.print") as print_mock,
    ):
        run_project_context.return_value = {"status": "success", "content": [{"text": "snapshot"}]}

        build_context_snapshot(
            artifact_store=artifact_store,
            interactive=True,
            default_preflight_level="summary",
        )

        print_mock.assert_called_once()
