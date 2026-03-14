from __future__ import annotations

import json
from pathlib import Path

from swarmee_river.utils.aws_config import resolve_runtime_athena_config, resolve_runtime_aws_region


def test_resolve_runtime_aws_region_prefers_settings_over_env(monkeypatch, tmp_path: Path) -> None:
    swarmee_dir = tmp_path / ".swarmee"
    swarmee_dir.mkdir()
    (swarmee_dir / "settings.json").write_text(
        json.dumps({"runtime": {"aws": {"region": "eu-west-1"}}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AWS_REGION", "us-west-2")

    assert resolve_runtime_aws_region() == "eu-west-1"


def test_resolve_runtime_athena_config_uses_env_when_settings_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    monkeypatch.setenv("ATHENA_DATABASE", "env_db")
    monkeypatch.setenv("ATHENA_WORKGROUP", "env_wg")
    monkeypatch.setenv("ATHENA_OUTPUT_LOCATION", "s3://env/results/")
    monkeypatch.setenv("ATHENA_QUERY_TIMEOUT", "90")

    resolved = resolve_runtime_athena_config()

    assert resolved.region == "ap-southeast-2"
    assert resolved.database == "env_db"
    assert resolved.workgroup == "env_wg"
    assert resolved.output_location == "s3://env/results/"
    assert resolved.query_timeout_seconds == 90

