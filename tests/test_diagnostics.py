from __future__ import annotations

import json
from pathlib import Path

from swarmee_river.cli.diagnostics import render_diagnostic_command
from swarmee_river.diagnostics import render_diagnostics_doctor
from swarmee_river.state_paths import artifacts_dir
from swarmee_river.utils.model_utils import OpenAIResponsesTransportStatus


def test_render_diagnostics_doctor_reports_openai_transport_status(tmp_path: Path, monkeypatch) -> None:
    settings_dir = tmp_path / ".swarmee"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(json.dumps({"models": {"provider": "openai"}}), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "swarmee_river.utils.model_utils.probe_openai_responses_transport",
        lambda: OpenAIResponsesTransportStatus(
            available=False,
            strands_version="1.26.0",
            openai_version="1.109.1",
            reason="Installed openai==1.109.1 is incompatible with Swarmee's OpenAI Responses runtime.",
        ),
    )

    output = render_diagnostics_doctor(cwd=tmp_path)

    assert "- selected_provider: openai" in output
    assert "- strands_agents_version: 1.26.0" in output
    assert "- openai_sdk_version: 1.109.1" in output
    assert "- openai_responses_transport: unavailable" in output
    assert "- selected_tier_available: no" in output


def test_render_diagnostic_command_diff_session_reads_captured_file_diffs(tmp_path: Path) -> None:
    (tmp_path / ".swarmee").mkdir(parents=True, exist_ok=True)
    store_dir = artifacts_dir(cwd=tmp_path)
    store_dir.mkdir(parents=True, exist_ok=True)
    diff_path = store_dir / "example.diff"
    diff_path.write_text("--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")
    (store_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "id": "artifact-1",
                "kind": "file_diff",
                "path": str(diff_path),
                "created_at": "2026-03-08T12:00:00",
                "meta": {
                    "session_id": "sid-1",
                    "tool": "editor",
                    "changed_paths": ["app.py"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rendered = render_diagnostic_command(cmd="diff", args=["session", "sid-1"], cwd=tmp_path)

    assert "# Session Diffs" in rendered
    assert "## editor - app.py" in rendered
    assert "--- a/app.py" in rendered
    assert "+new" in rendered
