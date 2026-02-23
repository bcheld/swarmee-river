from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

_PROVIDER_ALIASES: dict[str, str] = {
    "ghcp": "github_copilot",
    "copilot": "github_copilot",
    "githubcopilot": "github_copilot",
    "github-copilot": "github_copilot",
    "github_copilot": "github_copilot",
}


def normalize_provider_name(value: object | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    normalized = token.replace("-", "_")
    return _PROVIDER_ALIASES.get(normalized, normalized)


def _data_home(app_name: str) -> Path:
    if os.name == "nt":
        appdata = (os.getenv("APPDATA") or "").strip()
        if appdata:
            return Path(appdata).expanduser() / app_name

    xdg_data_home = (os.getenv("XDG_DATA_HOME") or "").strip()
    base = Path(xdg_data_home).expanduser() if xdg_data_home else (Path.home() / ".local" / "share")
    return base / app_name


def auth_store_path() -> Path:
    raw = (os.getenv("SWARMEE_AUTH_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_data_home("swarmee") / "auth.json").resolve()


def opencode_auth_store_path() -> Path:
    raw = (os.getenv("SWARMEE_OPENCODE_AUTH_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_data_home("opencode") / "auth.json").resolve()


def _load_auth_payload(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_auth_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with contextlib.suppress(Exception):
        os.chmod(path, 0o600)
    return path


def _extract_record(payload: dict[str, Any], provider: str) -> dict[str, Any] | None:
    wanted = normalize_provider_name(provider)
    if not wanted:
        return None
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        key_norm = normalize_provider_name(key)
        if key_norm == wanted:
            return dict(value)
        value_provider = normalize_provider_name(value.get("provider"))
        if value_provider and value_provider == wanted:
            return dict(value)
    return None


def get_provider_record(provider: str, *, include_opencode: bool = True) -> tuple[dict[str, Any] | None, str | None]:
    wanted = normalize_provider_name(provider)
    if not wanted:
        return None, None

    swarmee_payload = _load_auth_payload(auth_store_path())
    local = _extract_record(swarmee_payload, wanted)
    if local is not None:
        return local, "swarmee"

    if include_opencode:
        opencode_payload = _load_auth_payload(opencode_auth_store_path())
        opencode = _extract_record(opencode_payload, wanted)
        if opencode is not None:
            return opencode, "opencode"

    return None, None


def set_provider_record(provider: str, record: dict[str, Any]) -> Path:
    normalized = normalize_provider_name(provider)
    if not normalized:
        raise ValueError("provider is required")
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")

    payload = _load_auth_payload(auth_store_path())
    payload[normalized] = dict(record)
    return _write_auth_payload(auth_store_path(), payload)


def delete_provider_record(provider: str) -> bool:
    normalized = normalize_provider_name(provider)
    if not normalized:
        return False
    path = auth_store_path()
    payload = _load_auth_payload(path)
    target_key = None
    for key in payload.keys():
        if normalize_provider_name(key) == normalized:
            target_key = key
            break
    if target_key is None:
        return False
    del payload[target_key]
    _write_auth_payload(path, payload)
    return True


def list_auth_records(*, include_opencode: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _append_from_payload(payload: dict[str, Any], source: str) -> None:
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            provider = normalize_provider_name(key) or normalize_provider_name(value.get("provider"))
            if not provider:
                continue
            marker = (provider, source)
            if marker in seen:
                continue
            seen.add(marker)
            records.append(
                {
                    "provider": provider,
                    "source": source,
                    "type": str(value.get("type", "unknown")).strip() or "unknown",
                    "has_key": bool(str(value.get("key", "")).strip()),
                    "has_refresh": bool(str(value.get("refresh", "")).strip()),
                    "has_access": bool(str(value.get("access", "")).strip()),
                    "expires": value.get("expires"),
                }
            )

    _append_from_payload(_load_auth_payload(auth_store_path()), "swarmee")
    if include_opencode:
        _append_from_payload(_load_auth_payload(opencode_auth_store_path()), "opencode")

    return sorted(records, key=lambda item: (str(item.get("provider")), str(item.get("source"))))


def has_provider_record(provider: str) -> bool:
    record, _source = get_provider_record(provider, include_opencode=True)
    return record is not None
