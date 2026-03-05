from __future__ import annotations

from unittest import mock

from swarmee_river.project_map import build_context_snapshot
from swarmee_river.settings import RuntimeConfig


def test_interactive_preflight_is_silent_by_default() -> None:
    runtime = RuntimeConfig(preflight_enabled=True, project_map_enabled=False, preflight_print=False)

    artifact_store = mock.MagicMock()

    with (
        mock.patch("swarmee_river.project_map.run_project_context") as run_project_context,
        mock.patch("builtins.print") as print_mock,
    ):
        run_project_context.return_value = {"status": "success", "content": [{"text": "snapshot"}]}

        snapshot = build_context_snapshot(
            artifact_store=artifact_store,
            interactive=True,
            runtime=runtime,
            default_preflight_level="summary",
        )

        assert snapshot.preflight_prompt_section == "Project context snapshot:\nsnapshot"
        print_mock.assert_not_called()


def test_interactive_preflight_prints_when_enabled() -> None:
    runtime = RuntimeConfig(preflight_enabled=True, project_map_enabled=False, preflight_print=True)

    artifact_store = mock.MagicMock()

    with (
        mock.patch("swarmee_river.project_map.run_project_context") as run_project_context,
        mock.patch("builtins.print") as print_mock,
    ):
        run_project_context.return_value = {"status": "success", "content": [{"text": "snapshot"}]}

        build_context_snapshot(
            artifact_store=artifact_store,
            interactive=True,
            runtime=runtime,
            default_preflight_level="summary",
        )

        print_mock.assert_called_once()
