from __future__ import annotations

import json
from pathlib import Path

from tools.athena_query import athena_query


def _text(result: dict[str, object]) -> str:
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            value = first.get("text")
            if isinstance(value, str):
                return value
    return ""


def test_athena_query_requires_query() -> None:
    result = athena_query(action="query", query="")

    assert result.get("status") == "error"
    assert "query is required" in _text(result).lower()


def test_athena_query_blocks_ddl() -> None:
    result = athena_query(action="query", query="CREATE TABLE x (id INT)")

    assert result.get("status") == "error"
    assert "blocked sql statement: create" in _text(result).lower()


def test_athena_query_status_requires_id() -> None:
    result = athena_query(action="query_status", query_execution_id="")

    assert result.get("status") == "error"
    assert "query_execution_id is required" in _text(result).lower()


def test_athena_query_reports_scan_warning(monkeypatch) -> None:
    class FakeAthena:
        def start_query_execution(self, **kwargs):
            return {"QueryExecutionId": "q-123"}

        def get_query_execution(self, **kwargs):
            return {
                "QueryExecution": {
                    "Status": {"State": "SUCCEEDED"},
                    "Statistics": {
                        "DataScannedInBytes": 2 * 1024 * 1024 * 1024,
                        "EngineExecutionTimeInMillis": 1234,
                    },
                }
            }

        def get_query_results(self, **kwargs):
            return {
                "ResultSet": {
                    "ResultSetMetadata": {
                        "ColumnInfo": [{"Name": "id"}, {"Name": "name"}],
                    },
                    "Rows": [
                        {"Data": [{"VarCharValue": "id"}, {"VarCharValue": "name"}]},
                        {"Data": [{"VarCharValue": "1"}, {"VarCharValue": "alpha"}]},
                    ],
                }
            }

    monkeypatch.setattr("tools.athena_query._aws_client", lambda service_name, **kwargs: FakeAthena())

    result = athena_query(action="query", query="SELECT * FROM demo")
    text = _text(result)

    assert result.get("status") == "success"
    assert "Data scanned:" in text
    assert "Large scan:" in text
    assert "| id | name |" in text


def test_athena_query_interrupts_running_query(monkeypatch) -> None:
    class FakeAthena:
        def __init__(self) -> None:
            self.stopped: list[str] = []

        def start_query_execution(self, **_kwargs):
            return {"QueryExecutionId": "q-123"}

        def get_query_execution(self, **_kwargs):
            return {"QueryExecution": {"Status": {"State": "RUNNING"}}}

        def stop_query_execution(self, *, QueryExecutionId: str):
            self.stopped.append(QueryExecutionId)
            return {}

    fake = FakeAthena()
    monkeypatch.setattr("tools.athena_query._aws_client", lambda service_name, **kwargs: fake)
    monkeypatch.setattr("tools.athena_query.interrupt_requested", lambda: True)

    result = athena_query(action="query", query="SELECT * FROM demo")

    assert result.get("status") == "error"
    assert "interrupted" in _text(result).lower()
    assert fake.stopped == ["q-123"]


def test_athena_query_uses_settings_defaults_over_env(monkeypatch, tmp_path: Path) -> None:
    swarmee_dir = tmp_path / ".swarmee"
    swarmee_dir.mkdir()
    (swarmee_dir / "settings.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "aws": {"region": "us-east-2"},
                    "athena": {
                        "database": "settings_db",
                        "workgroup": "settings_wg",
                        "output_location": "s3://settings/results/",
                        "query_timeout_seconds": 180,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("ATHENA_DATABASE", "env_db")
    monkeypatch.setenv("ATHENA_WORKGROUP", "env_wg")
    monkeypatch.setenv("ATHENA_OUTPUT_LOCATION", "s3://env/results/")
    monkeypatch.setenv("ATHENA_QUERY_TIMEOUT", "90")

    class FakeAthena:
        def __init__(self) -> None:
            self.start_kwargs: dict[str, object] | None = None

        def start_query_execution(self, **kwargs):
            self.start_kwargs = kwargs
            return {"QueryExecutionId": "q-123"}

        def get_query_execution(self, **kwargs):
            return {
                "QueryExecution": {
                    "Status": {"State": "SUCCEEDED"},
                    "Statistics": {"DataScannedInBytes": 1, "EngineExecutionTimeInMillis": 2},
                }
            }

        def get_query_results(self, **kwargs):
            return {
                "ResultSet": {
                    "ResultSetMetadata": {"ColumnInfo": [{"Name": "col"}]},
                    "Rows": [
                        {"Data": [{"VarCharValue": "col"}]},
                        {"Data": [{"VarCharValue": "1"}]},
                    ],
                }
            }

    fake = FakeAthena()
    client_calls: list[dict[str, object]] = []

    def _fake_client(service_name: str, **kwargs):
        client_calls.append({"service_name": service_name, **kwargs})
        return fake

    monkeypatch.setattr("tools.athena_query._aws_client", _fake_client)

    result = athena_query(action="query", query="SELECT 1")

    assert result.get("status") == "success"
    assert client_calls[0]["region_name"] == "us-east-2"
    assert fake.start_kwargs is not None
    assert fake.start_kwargs["QueryExecutionContext"] == {"Database": "settings_db"}
    assert fake.start_kwargs["WorkGroup"] == "settings_wg"
    assert fake.start_kwargs["ResultConfiguration"] == {"OutputLocation": "s3://settings/results/"}


def test_athena_query_explicit_args_override_settings(monkeypatch, tmp_path: Path) -> None:
    swarmee_dir = tmp_path / ".swarmee"
    swarmee_dir.mkdir()
    (swarmee_dir / "settings.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "aws": {"region": "us-east-2"},
                    "athena": {
                        "database": "settings_db",
                        "workgroup": "settings_wg",
                        "output_location": "s3://settings/results/",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    class FakeAthena:
        def __init__(self) -> None:
            self.start_kwargs: dict[str, object] | None = None

        def start_query_execution(self, **kwargs):
            self.start_kwargs = kwargs
            return {"QueryExecutionId": "q-123"}

        def get_query_execution(self, **kwargs):
            return {
                "QueryExecution": {
                    "Status": {"State": "SUCCEEDED"},
                    "Statistics": {"DataScannedInBytes": 1, "EngineExecutionTimeInMillis": 2},
                }
            }

        def get_query_results(self, **kwargs):
            return {
                "ResultSet": {
                    "ResultSetMetadata": {"ColumnInfo": [{"Name": "col"}]},
                    "Rows": [
                        {"Data": [{"VarCharValue": "col"}]},
                        {"Data": [{"VarCharValue": "1"}]},
                    ],
                }
            }

    fake = FakeAthena()
    monkeypatch.setattr("tools.athena_query._aws_client", lambda service_name, **kwargs: fake)

    result = athena_query(
        action="query",
        query="SELECT 1",
        database="arg_db",
        workgroup="arg_wg",
        output_location="s3://arg/results/",
    )

    assert result.get("status") == "success"
    assert fake.start_kwargs is not None
    assert fake.start_kwargs["QueryExecutionContext"] == {"Database": "arg_db"}
    assert fake.start_kwargs["WorkGroup"] == "arg_wg"
    assert fake.start_kwargs["ResultConfiguration"] == {"OutputLocation": "s3://arg/results/"}
