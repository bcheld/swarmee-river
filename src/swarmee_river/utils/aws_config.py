from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from swarmee_river.settings import default_settings_template, load_settings_payload, migrate_legacy_env_overrides


@dataclass(frozen=True)
class ResolvedAthenaConfig:
    region: str
    database: str | None
    workgroup: str | None
    output_location: str | None
    query_timeout_seconds: int


def _configured_payload(path: Path | None = None) -> dict[str, Any]:
    payload = load_settings_payload(path)
    if payload and isinstance(payload, dict) and payload.get("env"):
        payload, _migrated, _dropped = migrate_legacy_env_overrides(payload)
    return payload if isinstance(payload, dict) else {}


def _runtime_payload(path: Path | None = None) -> dict[str, Any]:
    payload = _configured_payload(path)
    runtime = payload.get("runtime")
    return runtime if isinstance(runtime, dict) else {}


def _nested_payload_value(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def resolve_runtime_aws_region(
    *,
    explicit_region: str | None = None,
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    explicit = _normalize_optional_str(explicit_region)
    if explicit:
        return explicit

    runtime = _runtime_payload(path)
    configured = _normalize_optional_str(_nested_payload_value(runtime, "aws", "region"))
    if configured:
        return configured

    env_map = env or os.environ
    for key in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        candidate = _normalize_optional_str(env_map.get(key))
        if candidate:
            return candidate

    return default_settings_template().runtime.aws.region


def resolve_runtime_athena_config(
    *,
    explicit_database: str | None = None,
    explicit_workgroup: str | None = None,
    explicit_output_location: str | None = None,
    explicit_query_timeout_seconds: int | None = None,
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedAthenaConfig:
    runtime = _runtime_payload(path)
    env_map = env or os.environ
    defaults = default_settings_template().runtime.athena

    def _resolve_text(explicit: str | None, settings_key: str, env_key: str) -> str | None:
        direct = _normalize_optional_str(explicit)
        if direct:
            return direct
        configured = _normalize_optional_str(_nested_payload_value(runtime, "athena", settings_key))
        if configured:
            return configured
        candidate = _normalize_optional_str(env_map.get(env_key))
        if candidate:
            return candidate
        return _normalize_optional_str(getattr(defaults, settings_key))

    timeout: int | None = None
    if isinstance(explicit_query_timeout_seconds, int) and explicit_query_timeout_seconds > 0:
        timeout = explicit_query_timeout_seconds
    else:
        configured_timeout = _nested_payload_value(runtime, "athena", "query_timeout_seconds")
        if isinstance(configured_timeout, int) and configured_timeout > 0:
            timeout = configured_timeout
        else:
            env_timeout = _normalize_optional_str(env_map.get("ATHENA_QUERY_TIMEOUT"))
            if env_timeout and env_timeout.isdigit() and int(env_timeout) > 0:
                timeout = int(env_timeout)
    if timeout is None:
        timeout = defaults.query_timeout_seconds

    return ResolvedAthenaConfig(
        region=resolve_runtime_aws_region(path=path, env=env_map),
        database=_resolve_text(explicit_database, "database", "ATHENA_DATABASE"),
        workgroup=_resolve_text(explicit_workgroup, "workgroup", "ATHENA_WORKGROUP"),
        output_location=_resolve_text(explicit_output_location, "output_location", "ATHENA_OUTPUT_LOCATION"),
        query_timeout_seconds=timeout,
    )

