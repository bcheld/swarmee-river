from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.tui.agent_studio import (
    build_activated_agent_sidebar_items,
    build_activated_agent_table_rows,
    build_builder_agent_table_rows,
)
from swarmee_river.tui.sidebar_artifacts import build_artifact_table_rows
from swarmee_river.tui.sidebar_session import build_session_timeline_table_rows
from swarmee_river.tui.views.settings import build_env_table_rows, build_models_table_rows


def test_build_session_timeline_table_rows_maps_expected_columns() -> None:
    rows = build_session_timeline_table_rows(
        [
            {
                "id": "ev-1",
                "event": "after_tool_call",
                "tool": "shell",
                "duration_s": 0.5,
                "success": True,
                "ts": "",
            }
        ]
    )

    assert len(rows) == 1
    event_id, summary, kind, when = rows[0]
    assert event_id == "ev-1"
    assert "shell" in summary
    assert kind == "tool"
    assert when == ""


def test_build_artifact_table_rows_truncates_long_paths() -> None:
    long_path = "/tmp/" + ("deep/" * 30) + "artifact.txt"
    rows = build_artifact_table_rows(
        [
            {
                "id": "artifact-1",
                "kind": "tool_result",
                "path": long_path,
                "created_at": "2026-02-23T10:01:00",
                "meta": {"name": "Tool output"},
            }
        ]
    )

    assert len(rows) == 1
    item_id, name, kind, created_at, shown_path = rows[0]
    assert item_id == long_path
    assert name == "Tool output"
    assert kind == "tool_result"
    assert created_at == "2026-02-23T10:01:00"
    assert shown_path.startswith("...")
    assert len(shown_path) == 96


def test_agent_table_row_builders_map_overview_and_builder() -> None:
    items = build_activated_agent_sidebar_items(
        [
            {
                "id": "triage",
                "name": "Triage",
                "summary": "Handles triage",
                "provider": "openai",
                "tier": "balanced",
                "activated": True,
            }
        ]
    )

    overview_rows = build_activated_agent_table_rows(items)
    assert overview_rows == [("triage", "Triage", "Handles triage", "openai/balanced", "yes")]

    builder_rows = build_builder_agent_table_rows(
        [
            {
                "id": "triage",
                "agent": {
                    "id": "triage",
                    "name": "Triage",
                    "summary": "Handles triage",
                    "provider": "openai",
                    "tier": "balanced",
                    "activated": True,
                },
            }
        ]
    )
    assert builder_rows == [("triage", "Triage", "Handles triage", "openai/balanced", "active")]


def test_settings_table_row_builders_for_models_and_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abcdefgh12345678")

    pricing = SimpleNamespace(input_per_1m=1.25, output_per_1m=5.0, cached_input_per_1m=0.5)
    monkeypatch.setattr("swarmee_river.pricing.resolve_pricing", lambda provider, model_id: pricing)

    settings = SimpleNamespace(
        models=SimpleNamespace(
            providers={
                "openai": SimpleNamespace(
                    tiers={
                        "balanced": SimpleNamespace(model_id="gpt-4.1"),
                    }
                )
            }
        )
    )

    model_rows = build_models_table_rows(settings)
    assert model_rows == [
        ("openai|balanced", "openai/balanced", "gpt-4.1", " | $1.25/1M in, $5.0/1M out, $0.5/1M cached")
    ]

    env_rows = build_env_table_rows()
    api_key_row = next((row for row in env_rows if row[0] == "OPENAI_API_KEY"), None)
    assert api_key_row is not None
    assert api_key_row[1].startswith("sk-t...")
    assert api_key_row[3] == "set"
