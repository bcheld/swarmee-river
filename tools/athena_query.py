from __future__ import annotations

import contextlib
import re
import time
from typing import Any

from strands import tool

from swarmee_river.tool_permissions import set_permissions
from swarmee_river.utils.aws_config import resolve_runtime_athena_config, resolve_runtime_aws_region
from swarmee_river.utils.text_utils import truncate
from swarmee_river.utils.tool_interrupts import interrupt_requested, sleep_with_interrupt

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


def _region() -> str:
    return resolve_runtime_aws_region()


def _timeout_seconds() -> int:
    value = resolve_runtime_athena_config().query_timeout_seconds
    return max(10, min(600, value))


def _aws_client(service_name: str, *, region_name: str | None = None) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except Exception as exc:
        raise RuntimeError("boto3 is required. Install with: pip install boto3") from exc

    return boto3.client(
        service_name,
        region_name=resolve_runtime_aws_region(explicit_region=region_name),
        config=Config(connect_timeout=15, read_timeout=15, retries={"max_attempts": 2}),
    )


def _fmt_size(num_bytes: Any) -> str:
    try:
        n = float(num_bytes)
    except Exception:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while n >= 1024 and idx < len(units) - 1:
        n /= 1024
        idx += 1
    return f"{n:.1f} {units[idx]}" if idx > 0 else f"{int(n)} {units[idx]}"


def _query_status_text(execution: dict[str, Any]) -> str:
    status = execution.get("Status") if isinstance(execution, dict) else {}
    state = status.get("State") if isinstance(status, dict) else "UNKNOWN"
    stats = execution.get("Statistics") if isinstance(execution, dict) else {}
    scanned = stats.get("DataScannedInBytes") if isinstance(stats, dict) else 0
    exec_ms = stats.get("EngineExecutionTimeInMillis") if isinstance(stats, dict) else 0

    lines = [
        f"- state: {state}",
        f"- data scanned: {_fmt_size(scanned)}",
        f"- execution time: {exec_ms} ms",
    ]

    if isinstance(status, dict) and status.get("StateChangeReason"):
        lines.append(f"- reason: {status.get('StateChangeReason')}")

    return "\n".join(lines)


def _query(
    *,
    query: str | None,
    database: str | None,
    workgroup: str | None,
    output_location: str | None,
    max_rows: int,
    max_chars: int,
) -> dict[str, Any]:
    sql = (query or "").strip()
    if not sql:
        return _error("query is required for action=query", max_chars=max_chars)

    blocked = _blocked_statement_keyword(sql)
    if blocked:
        return _error(f"Blocked SQL statement: {blocked}. Only read-only queries are allowed.", max_chars=max_chars)

    athena_cfg = resolve_runtime_athena_config(
        explicit_database=database,
        explicit_workgroup=workgroup,
        explicit_output_location=output_location,
    )

    try:
        athena = _aws_client("athena", region_name=athena_cfg.region)
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    db = athena_cfg.database
    wg = athena_cfg.workgroup
    out = athena_cfg.output_location
    row_cap = max(1, min(5000, int(max_rows)))

    start_args: dict[str, Any] = {"QueryString": sql}
    if db:
        start_args["QueryExecutionContext"] = {"Database": db}
    if wg:
        start_args["WorkGroup"] = wg
    if out:
        start_args["ResultConfiguration"] = {"OutputLocation": out}

    try:
        start_resp = athena.start_query_execution(**start_args)
        query_id = str(start_resp.get("QueryExecutionId") or "")
    except Exception as exc:
        return _error(f"Athena start_query_execution failed: {exc}", max_chars=max_chars)

    if not query_id:
        return _error("Athena did not return a query_execution_id", max_chars=max_chars)

    timeout_s = _timeout_seconds()
    deadline = time.monotonic() + timeout_s
    execution: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        if interrupt_requested():
            with contextlib.suppress(Exception):
                athena.stop_query_execution(QueryExecutionId=query_id)
            return _error(f"Athena query {query_id} interrupted.", max_chars=max_chars)
        try:
            exec_resp = athena.get_query_execution(QueryExecutionId=query_id)
        except Exception as exc:
            return _error(f"Failed to fetch Athena query status ({query_id}): {exc}", max_chars=max_chars)

        execution = exec_resp.get("QueryExecution") if isinstance(exec_resp, dict) else None
        if not isinstance(execution, dict):
            return _error(f"Unexpected Athena query status payload for {query_id}", max_chars=max_chars)

        state = str((execution.get("Status") or {}).get("State") or "UNKNOWN").upper()
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break
        if sleep_with_interrupt(1.0):
            with contextlib.suppress(Exception):
                athena.stop_query_execution(QueryExecutionId=query_id)
            return _error(f"Athena query {query_id} interrupted.", max_chars=max_chars)

    if not isinstance(execution, dict):
        return _error(f"Timed out waiting for Athena query {query_id}", max_chars=max_chars)

    state = str((execution.get("Status") or {}).get("State") or "UNKNOWN").upper()
    if state not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        return _error(f"Timed out waiting for Athena query {query_id} (state={state})", max_chars=max_chars)

    if state != "SUCCEEDED":
        details = _query_status_text(execution)
        return _error(f"Athena query {query_id} finished in state {state}\n\n{details}", max_chars=max_chars)

    headers: list[str] = []
    rows: list[list[str]] = []
    next_token: str | None = None
    first_page = True
    hidden = 0

    try:
        while True:
            kwargs = {"QueryExecutionId": query_id, "MaxResults": min(1000, row_cap + 1)}
            if next_token:
                kwargs["NextToken"] = next_token
            result_resp = athena.get_query_results(**kwargs)

            result_set = result_resp.get("ResultSet") if isinstance(result_resp, dict) else {}
            metadata = result_set.get("ResultSetMetadata") if isinstance(result_set, dict) else {}
            columns = metadata.get("ColumnInfo") if isinstance(metadata, dict) else []
            if not headers and isinstance(columns, list) and columns:
                headers = [str(col.get("Name") or "") for col in columns if isinstance(col, dict)]

            result_rows = result_set.get("Rows") if isinstance(result_set, dict) else []
            if not isinstance(result_rows, list):
                result_rows = []

            start_idx = 1 if first_page else 0
            for row_item in result_rows[start_idx:]:
                if not isinstance(row_item, dict):
                    continue
                cells = row_item.get("Data")
                values: list[str] = []
                if isinstance(cells, list):
                    for cell in cells:
                        if isinstance(cell, dict):
                            values.append(str(cell.get("VarCharValue") or ""))
                        else:
                            values.append("")
                rows.append(values)

                if len(rows) > row_cap:
                    hidden += 1
                    break

            next_token = result_resp.get("NextToken") if isinstance(result_resp, dict) else None
            first_page = False

            if len(rows) > row_cap:
                break
            if not next_token:
                break
    except Exception as exc:
        return _error(f"Athena get_query_results failed for {query_id}: {exc}", max_chars=max_chars)

    shown_rows = rows[:row_cap]

    if not headers and shown_rows:
        headers = [f"col_{idx + 1}" for idx in range(len(shown_rows[0]))]
    if not headers:
        headers = ["result"]

    rendered_rows = [row + [""] * (len(headers) - len(row)) for row in shown_rows]
    table = _markdown_table(headers, rendered_rows)

    stats = execution.get("Statistics") if isinstance(execution, dict) else {}
    scanned_bytes = int(stats.get("DataScannedInBytes") or 0) if isinstance(stats, dict) else 0
    scanned_text = _fmt_size(scanned_bytes)

    lines = [f"# Athena query {query_id}", "", table, "", f"Data scanned: {scanned_text}"]
    if scanned_bytes > 1024 * 1024 * 1024:
        lines.append(f"⚠ Large scan: {scanned_text}. Consider adding WHERE clauses or partitions.")

    if hidden > 0 or next_token:
        extra = hidden if hidden > 0 else 1
        lines.append(f"\n... ({extra} more rows)")

    return _success("\n".join(lines).strip(), max_chars=max_chars)


def _describe(*, table: str | None, database: str | None, max_chars: int) -> dict[str, Any]:
    table_name = (table or "").strip()
    if not table_name:
        return _error("table is required for action=describe", max_chars=max_chars)

    db = (resolve_runtime_athena_config(explicit_database=database).database or "").strip()
    if not db:
        return _error(
            "database is required for action=describe (or configure runtime.athena.database / ATHENA_DATABASE)",
            max_chars=max_chars,
        )

    try:
        glue = _aws_client("glue")
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    try:
        resp = glue.get_table(DatabaseName=db, Name=table_name)
    except Exception as exc:
        return _error(f"Glue get_table failed: {exc}", max_chars=max_chars)

    table_data = resp.get("Table") if isinstance(resp, dict) else {}
    storage = table_data.get("StorageDescriptor") if isinstance(table_data, dict) else {}

    columns = storage.get("Columns") if isinstance(storage, dict) else []
    partitions = table_data.get("PartitionKeys") if isinstance(table_data, dict) else []

    rows: list[list[Any]] = []
    if isinstance(columns, list):
        for col in columns:
            if not isinstance(col, dict):
                continue
            rows.append([col.get("Name") or "", col.get("Type") or "", col.get("Comment") or "", "no"])

    if isinstance(partitions, list):
        for col in partitions:
            if not isinstance(col, dict):
                continue
            rows.append([col.get("Name") or "", col.get("Type") or "", col.get("Comment") or "", "yes"])

    if not rows:
        return _success("(no columns found)", max_chars=max_chars)

    table_md = _markdown_table(["Column", "Type", "Comment", "Partition"], rows)
    return _success(f"# {db}.{table_name}\n\n{table_md}", max_chars=max_chars)


def _list_tables(*, database: str | None, pattern: str | None, max_chars: int) -> dict[str, Any]:
    db = (resolve_runtime_athena_config(explicit_database=database).database or "").strip()
    if not db:
        return _error(
            "database is required for action=list_tables (or configure runtime.athena.database / ATHENA_DATABASE)",
            max_chars=max_chars,
        )

    try:
        glue = _aws_client("glue")
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    rows: list[list[str]] = []
    next_token: str | None = None

    try:
        while True:
            kwargs: dict[str, Any] = {"DatabaseName": db, "MaxResults": 100}
            if next_token:
                kwargs["NextToken"] = next_token
            resp = glue.get_tables(**kwargs)

            tables = resp.get("TableList") if isinstance(resp, dict) else []
            if isinstance(tables, list):
                for tbl in tables:
                    if not isinstance(tbl, dict):
                        continue
                    name = str(tbl.get("Name") or "")
                    if pattern and not re.search(pattern, name, flags=re.IGNORECASE):
                        continue
                    tbl_type = str(tbl.get("TableType") or "")
                    created = str(tbl.get("CreateTime") or "")
                    rows.append([name, tbl_type, created])

            next_token = resp.get("NextToken") if isinstance(resp, dict) else None
            if not next_token:
                break
    except re.error as exc:
        return _error(f"Invalid pattern regex: {exc}", max_chars=max_chars)
    except Exception as exc:
        return _error(f"Glue get_tables failed: {exc}", max_chars=max_chars)

    if not rows:
        return _success("(no tables found)", max_chars=max_chars)

    return _success(_markdown_table(["Table", "Type", "Created"], rows), max_chars=max_chars)


def _list_databases(*, max_chars: int) -> dict[str, Any]:
    try:
        glue = _aws_client("glue")
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    rows: list[list[str]] = []
    next_token: str | None = None

    try:
        while True:
            kwargs: dict[str, Any] = {"MaxResults": 100}
            if next_token:
                kwargs["NextToken"] = next_token
            resp = glue.get_databases(**kwargs)

            dbs = resp.get("DatabaseList") if isinstance(resp, dict) else []
            if isinstance(dbs, list):
                for db in dbs:
                    if not isinstance(db, dict):
                        continue
                    rows.append([str(db.get("Name") or ""), str(db.get("Description") or "")])

            next_token = resp.get("NextToken") if isinstance(resp, dict) else None
            if not next_token:
                break
    except Exception as exc:
        return _error(f"Glue get_databases failed: {exc}", max_chars=max_chars)

    if not rows:
        return _success("(no databases found)", max_chars=max_chars)

    return _success(_markdown_table(["Database", "Description"], rows), max_chars=max_chars)


def _query_status(*, query_execution_id: str | None, max_chars: int) -> dict[str, Any]:
    query_id = (query_execution_id or "").strip()
    if not query_id:
        return _error("query_execution_id is required for action=query_status", max_chars=max_chars)

    try:
        athena = _aws_client("athena")
    except RuntimeError as exc:
        return _error(str(exc), max_chars=max_chars)

    try:
        response = athena.get_query_execution(QueryExecutionId=query_id)
    except Exception as exc:
        return _error(f"Athena get_query_execution failed: {exc}", max_chars=max_chars)

    execution = response.get("QueryExecution") if isinstance(response, dict) else {}
    if not isinstance(execution, dict):
        return _error("Unexpected response from Athena", max_chars=max_chars)

    lines = [f"# Athena query {query_id}", "", _query_status_text(execution)]
    return _success("\n".join(lines), max_chars=max_chars)


@tool
def athena_query(
    action: str = "query",
    query: str | None = None,
    table: str | None = None,
    database: str | None = None,
    workgroup: str | None = None,
    output_location: str | None = None,
    pattern: str | None = None,
    query_execution_id: str | None = None,
    max_rows: int = 200,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Read-only Athena query and Glue catalog inspection tool.

    Actions:
    - query (default): execute read-only SQL in Athena
    - describe: describe table schema from Glue catalog
    - list_tables: list tables in a database
    - list_databases: list Glue databases
    - query_status: inspect status/scan details for a query execution id
    """
    mode = (action or "query").strip().lower()

    if mode == "query":
        return _query(
            query=query,
            database=database,
            workgroup=workgroup,
            output_location=output_location,
            max_rows=max_rows,
            max_chars=max_chars,
        )
    if mode == "describe":
        return _describe(table=table, database=database, max_chars=max_chars)
    if mode == "list_tables":
        return _list_tables(database=database, pattern=pattern, max_chars=max_chars)
    if mode == "list_databases":
        return _list_databases(max_chars=max_chars)
    if mode == "query_status":
        return _query_status(query_execution_id=query_execution_id, max_chars=max_chars)

    return _error(f"Unknown action: {mode}", max_chars=max_chars)


set_permissions(athena_query, "read", "execute")
