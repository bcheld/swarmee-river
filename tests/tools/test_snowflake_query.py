from __future__ import annotations

from tools.snowflake_query import snowflake_query


def _text(result: dict[str, object]) -> str:
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            value = first.get("text")
            if isinstance(value, str):
                return value
    return ""


def test_snowflake_query_requires_query() -> None:
    result = snowflake_query(action="query", query="")

    assert result.get("status") == "error"
    assert "query is required" in _text(result).lower()


def test_snowflake_query_blocks_dml_statement() -> None:
    result = snowflake_query(action="query", query="INSERT INTO demo VALUES (1)")

    assert result.get("status") == "error"
    assert "blocked sql statement: insert" in _text(result).lower()


def test_snowflake_query_blocks_second_statement() -> None:
    result = snowflake_query(action="query", query="SELECT 1; DROP TABLE x")

    assert result.get("status") == "error"
    assert "blocked sql statement: drop" in _text(result).lower()


def test_snowflake_list_tables_builds_show_statement(monkeypatch) -> None:
    def _fake_execute_sql(**kwargs):
        return {"status": "success", "content": [{"text": kwargs["sql"]}]}

    monkeypatch.setattr("tools.snowflake_query._execute_sql", _fake_execute_sql)

    result = snowflake_query(
        action="list_tables",
        database="ANALYTICS",
        schema_name="PUBLIC",
        pattern="EVENT%",
    )

    assert result.get("status") == "success"
    text = _text(result)
    assert "SHOW TABLES IN SCHEMA ANALYTICS.PUBLIC LIKE 'EVENT%'" in text


def test_snowflake_query_surfaces_connector_error(monkeypatch) -> None:
    def _fail_connect(**kwargs):
        raise RuntimeError(
            "snowflake-connector-python is required. Install with: pip install snowflake-connector-python"
        )

    monkeypatch.setattr("tools.snowflake_query._connect", _fail_connect)

    result = snowflake_query(action="query", query="SELECT 1")

    assert result.get("status") == "error"
    assert "snowflake-connector-python is required" in _text(result)
