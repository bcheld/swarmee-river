from __future__ import annotations

import os
import re
from typing import Any

from strands import tool

from swarmee_river.tool_permissions import set_permissions
from swarmee_river.utils.text_utils import truncate

_BLOCKED_SQL_KEYWORDS = {
    "CREATE",
    "DROP",
    "ALTER",
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
}


def _success(text: str, *, max_chars: int) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": truncate(text, max_chars)}]}


def _error(text: str, *, max_chars: int) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": truncate(text, max_chars)}]}


def _cell(value: Any, *, limit: int = 100) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(_cell(h) for h in headers) + " |"
    align = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join([header, align, *body])


def _strip_leading_comments(statement: str) -> str:
    s = statement
    while True:
        stripped = s.lstrip()
        if stripped.startswith("--"):
            nl = stripped.find("\n")
            if nl == -1:
                return ""
            s = stripped[nl + 1 :]
            continue
        if stripped.startswith("/*"):
            end = stripped.find("*/")
            if end == -1:
                return ""
            s = stripped[end + 2 :]
            continue
        return stripped


def _blocked_statement_keyword(sql: str) -> str | None:
    statements = [part for part in sql.split(";") if part.strip()]
    for raw_statement in statements:
        statement = _strip_leading_comments(raw_statement)
        if not statement:
            continue
        match = re.match(r"([A-Za-z]+)\b", statement)
        if not match:
            continue
        keyword = match.group(1).upper()
        if keyword in _BLOCKED_SQL_KEYWORDS:
            return keyword
    return None


def _safe_identifier(name: str, *, field: str) -> str:
    value = (name or "").strip()
    if not value:
        raise ValueError(f"{field} is required")
    if not re.fullmatch(r"[A-Za-z0-9_$.]+", value):
        raise ValueError(f"Invalid {field}: only letters, digits, underscore, dot, and dollar are allowed")
    return value


def _timeout_seconds() -> int:
    raw = (os.getenv("SNOWFLAKE_QUERY_TIMEOUT") or "30").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 30
    return max(5, min(300, value))


def _connect(
    *,
    database: str | None,
    schema_name: str | None,
    warehouse: str | None,
):
    try:
        import snowflake.connector
    except Exception as exc:
        raise RuntimeError(
            "snowflake-connector-python is required. Install with: pip install snowflake-connector-python"
        ) from exc

    account = (os.getenv("SNOWFLAKE_ACCOUNT") or "").strip()
    user = (os.getenv("SNOWFLAKE_USER") or "").strip()
    password = os.getenv("SNOWFLAKE_PASSWORD")
    private_key_path = (os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH") or "").strip()

    if not account:
        raise RuntimeError("SNOWFLAKE_ACCOUNT is required")
    if not user:
        raise RuntimeError("SNOWFLAKE_USER is required")
    if not password and not private_key_path:
        raise RuntimeError("Set SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY_PATH")

    timeout_s = _timeout_seconds()
    kwargs: dict[str, Any] = {
        "account": account,
        "user": user,
        "database": (database or os.getenv("SNOWFLAKE_DATABASE") or "").strip() or None,
        "schema": (schema_name or os.getenv("SNOWFLAKE_SCHEMA") or "").strip() or None,
        "warehouse": (warehouse or os.getenv("SNOWFLAKE_WAREHOUSE") or "").strip() or None,
        "role": (os.getenv("SNOWFLAKE_ROLE") or "").strip() or None,
        "session_parameters": {"STATEMENT_TIMEOUT_IN_SECONDS": timeout_s},
        "login_timeout": timeout_s,
        "network_timeout": timeout_s,
    }

    if password:
        kwargs["password"] = password
    elif private_key_path:
        try:
            from cryptography.hazmat.primitives import serialization
        except Exception as exc:
            raise RuntimeError(
                "cryptography is required for SNOWFLAKE_PRIVATE_KEY_PATH auth. Install with: pip install cryptography"
            ) from exc

        try:
            with open(private_key_path, "rb") as key_file:
                key_data = key_file.read()
            passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
            passphrase_bytes = passphrase.encode("utf-8") if passphrase else None
            private_key = serialization.load_pem_private_key(key_data, password=passphrase_bytes)
            private_key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load SNOWFLAKE_PRIVATE_KEY_PATH: {exc}") from exc

        kwargs["private_key"] = private_key_bytes

    return snowflake.connector.connect(**{k: v for k, v in kwargs.items() if v is not None})


def _execute_sql(
    *,
    sql: str,
    database: str | None,
    schema_name: str | None,
    warehouse: str | None,
    max_rows: int,
    max_chars: int,
) -> dict[str, Any]:
    row_cap = max(1, min(5000, int(max_rows)))

    try:
        conn = _connect(database=database, schema_name=schema_name, warehouse=warehouse)
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)
    except Exception as exc:
        return _error(f"Snowflake connection failed: {exc}", max_chars=max_chars)

    try:
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            description = cursor.description or []
            headers = [str(col[0]) for col in description] if description else []

            if not headers:
                return _success("(statement executed with no tabular result)", max_chars=max_chars)

            rows: list[list[Any]] = []
            while len(rows) <= row_cap:
                batch = cursor.fetchmany(min(1000, row_cap + 1 - len(rows)))
                if not batch:
                    break
                rows.extend([list(item) for item in batch])

            shown_rows = rows[:row_cap]
            hidden = max(0, len(rows) - len(shown_rows))
            table = _markdown_table(headers, shown_rows)

            parts = [table]
            if hidden > 0:
                parts.append(f"\n... ({hidden} more rows)")

            return _success("\n".join(parts), max_chars=max_chars)
        finally:
            cursor.close()
    except Exception as exc:
        return _error(f"Snowflake query failed: {exc}", max_chars=max_chars)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _query(
    *,
    query: str | None,
    database: str | None,
    schema_name: str | None,
    warehouse: str | None,
    max_rows: int,
    max_chars: int,
) -> dict[str, Any]:
    sql = (query or "").strip()
    if not sql:
        return _error("query is required for action=query", max_chars=max_chars)

    blocked = _blocked_statement_keyword(sql)
    if blocked:
        return _error(f"Blocked SQL statement: {blocked}. Only read-only queries are allowed.", max_chars=max_chars)

    return _execute_sql(
        sql=sql,
        database=database,
        schema_name=schema_name,
        warehouse=warehouse,
        max_rows=max_rows,
        max_chars=max_chars,
    )


def _describe(
    *,
    table: str | None,
    database: str | None,
    schema_name: str | None,
    warehouse: str | None,
    max_chars: int,
) -> dict[str, Any]:
    raw_table = (table or "").strip()
    if not raw_table:
        return _error("table is required for action=describe", max_chars=max_chars)

    try:
        safe_table = _safe_identifier(raw_table, field="table")
    except ValueError as exc:
        return _error(str(exc), max_chars=max_chars)

    sql = f"DESCRIBE TABLE {safe_table}"
    return _execute_sql(
        sql=sql,
        database=database,
        schema_name=schema_name,
        warehouse=warehouse,
        max_rows=5000,
        max_chars=max_chars,
    )


def _list_tables(
    *,
    database: str | None,
    schema_name: str | None,
    pattern: str | None,
    warehouse: str | None,
    max_chars: int,
) -> dict[str, Any]:
    sql = "SHOW TABLES"
    try:
        db = _safe_identifier(database, field="database") if database else ""
        sch = _safe_identifier(schema_name, field="schema_name") if schema_name else ""
    except ValueError as exc:
        return _error(str(exc), max_chars=max_chars)

    if db and sch:
        sql += f" IN SCHEMA {db}.{sch}"
    elif sch:
        sql += f" IN SCHEMA {sch}"
    elif db:
        sql += f" IN DATABASE {db}"

    if pattern:
        like = pattern.replace("'", "''")
        sql += f" LIKE '{like}'"

    return _execute_sql(
        sql=sql,
        database=database,
        schema_name=schema_name,
        warehouse=warehouse,
        max_rows=5000,
        max_chars=max_chars,
    )


def _list_databases(*, warehouse: str | None, max_chars: int) -> dict[str, Any]:
    return _execute_sql(
        sql="SHOW DATABASES",
        database=None,
        schema_name=None,
        warehouse=warehouse,
        max_rows=5000,
        max_chars=max_chars,
    )


def _list_schemas(*, database: str | None, warehouse: str | None, max_chars: int) -> dict[str, Any]:
    sql = "SHOW SCHEMAS"
    if database:
        try:
            db = _safe_identifier(database, field="database")
        except ValueError as exc:
            return _error(str(exc), max_chars=max_chars)
        sql += f" IN DATABASE {db}"

    return _execute_sql(
        sql=sql,
        database=database,
        schema_name=None,
        warehouse=warehouse,
        max_rows=5000,
        max_chars=max_chars,
    )


@tool
def snowflake_query(
    action: str = "query",
    query: str | None = None,
    table: str | None = None,
    database: str | None = None,
    schema_name: str | None = None,
    warehouse: str | None = None,
    pattern: str | None = None,
    max_rows: int = 200,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Read-only Snowflake data access tool with row and statement guardrails.

    Actions:
    - query (default): execute a read-only SQL query
    - describe: describe table columns
    - list_tables: list tables in schema/database
    - list_databases: list databases
    - list_schemas: list schemas in database
    """
    mode = (action or "query").strip().lower()

    if mode == "query":
        return _query(
            query=query,
            database=database,
            schema_name=schema_name,
            warehouse=warehouse,
            max_rows=max_rows,
            max_chars=max_chars,
        )
    if mode == "describe":
        return _describe(
            table=table,
            database=database,
            schema_name=schema_name,
            warehouse=warehouse,
            max_chars=max_chars,
        )
    if mode == "list_tables":
        return _list_tables(
            database=database,
            schema_name=schema_name,
            pattern=pattern,
            warehouse=warehouse,
            max_chars=max_chars,
        )
    if mode == "list_databases":
        return _list_databases(warehouse=warehouse, max_chars=max_chars)
    if mode == "list_schemas":
        return _list_schemas(database=database, warehouse=warehouse, max_chars=max_chars)

    return _error(f"Unknown action: {mode}", max_chars=max_chars)


set_permissions(snowflake_query, "read", "execute")
