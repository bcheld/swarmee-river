from __future__ import annotations

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

    monkeypatch.setattr("tools.athena_query._aws_client", lambda service_name: FakeAthena())

    result = athena_query(action="query", query="SELECT * FROM demo")
    text = _text(result)

    assert result.get("status") == "success"
    assert "Data scanned:" in text
    assert "Large scan:" in text
    assert "| id | name |" in text
