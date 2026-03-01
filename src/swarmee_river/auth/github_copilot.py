from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from swarmee_river.auth.store import get_provider_record, set_provider_record

PROVIDER_NAME = "github_copilot"
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
DEVICE_ENDPOINT = "https://github.com/login/device/code"
TOKEN_ENDPOINT = "https://github.com/login/oauth/access_token"
COPILOT_TOKEN_ENDPOINT = "https://api.github.com/copilot_internal/v2/token"
DEFAULT_BASE_URL = "https://api.githubcopilot.com"

_COPILOT_HEADERS = {
    "Accept": "application/json",
    "Editor-Version": "vscode/1.99.3",
    "Editor-Plugin-Version": "copilot-chat/0.26.7",
    "User-Agent": "GithubCopilot/0.26.7",
}


@dataclass(frozen=True)
class CopilotRuntimeCredentials:
    access_token: str
    base_url: str
    headers: dict[str, str]


def _post_form_json(
    url: str,
    data: dict[str, str],
    *,
    headers: dict[str, str] | None = None,
    timeout_s: int = 15,
) -> dict[str, Any]:
    body = urllib.parse.urlencode(data).encode("utf-8")
    request_headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    if isinstance(headers, dict):
        request_headers.update(headers)
    request = urllib.request.Request(url, method="POST", data=body, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8", errors="replace")
    data_obj = json.loads(payload or "{}")
    return data_obj if isinstance(data_obj, dict) else {}


def _get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: int = 15,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json"}
    if isinstance(headers, dict):
        request_headers.update(headers)
    request = urllib.request.Request(url, method="GET", headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8", errors="replace")
    data_obj = json.loads(payload or "{}")
    return data_obj if isinstance(data_obj, dict) else {}


def _expires_to_ms(value: object) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            raw = int(text)
            return raw if raw > 100_000_000_000 else raw * 1000
        with contextlib.suppress(Exception):
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return int(parsed.timestamp() * 1000)
    return 0


def _is_access_valid(record: dict[str, Any]) -> bool:
    access = str(record.get("access", "")).strip()
    if not access:
        return False
    raw_expires = record.get("expires")
    if raw_expires is None or (isinstance(raw_expires, str) and not raw_expires.strip()):
        return True
    expires_ms = _expires_to_ms(raw_expires)
    if expires_ms <= 0:
        return False
    now_ms = int(time.time() * 1000)
    return expires_ms > (now_ms + 60_000)


def exchange_refresh_token(refresh_token: str) -> dict[str, Any]:
    token = (refresh_token or "").strip()
    if not token:
        raise RuntimeError("Missing GitHub OAuth token for Copilot access exchange.")

    headers = {
        **_COPILOT_HEADERS,
        "Authorization": f"Bearer {token}",
    }
    payload = _get_json(COPILOT_TOKEN_ENDPOINT, headers=headers)

    access = str(payload.get("token", "")).strip()
    if not access:
        message = str(payload.get("message", "")).strip() or "Copilot token endpoint did not return a token."
        raise RuntimeError(message)

    endpoints = payload.get("endpoints")
    endpoint_api = ""
    if isinstance(endpoints, dict):
        endpoint_api = str(endpoints.get("api", "")).strip()

    return {
        "access": access,
        "expires": _expires_to_ms(payload.get("expires_at")),
        "endpoint": endpoint_api or DEFAULT_BASE_URL,
        "account_id": str(payload.get("user", "")).strip() or None,
    }


def resolve_runtime_credentials(*, refresh: bool = True) -> CopilotRuntimeCredentials:
    record, source = get_provider_record(PROVIDER_NAME, include_opencode=True)
    if not isinstance(record, dict):
        raise RuntimeError("No saved GitHub Copilot auth record found. Run `swarmee connect`.")

    record_type = str(record.get("type", "")).strip().lower()
    if record_type == "api":
        key = str(record.get("key", "")).strip()
        if not key:
            raise RuntimeError("Saved GitHub Copilot API key record is missing `key`.")
        base_url = str(record.get("base_url", "")).strip() or DEFAULT_BASE_URL
        return CopilotRuntimeCredentials(access_token=key, base_url=base_url, headers={})

    if record_type != "oauth":
        raise RuntimeError(f"Unsupported GitHub Copilot auth record type: {record_type or 'unknown'}")

    refresh_token = str(record.get("refresh", "")).strip()
    if _is_access_valid(record):
        base_url = str(record.get("endpoint", "")).strip() or DEFAULT_BASE_URL
        return CopilotRuntimeCredentials(
            access_token=str(record.get("access", "")).strip(),
            base_url=base_url,
            headers=dict(_COPILOT_HEADERS),
        )

    if not refresh:
        raise RuntimeError("GitHub Copilot access token expired. Run `swarmee connect` to refresh.")
    if not refresh_token:
        raise RuntimeError("GitHub Copilot OAuth record is missing `refresh` token.")

    refreshed = exchange_refresh_token(refresh_token)
    updated = dict(record)
    updated.update(refreshed)
    set_provider_record(PROVIDER_NAME, updated)

    return CopilotRuntimeCredentials(
        access_token=str(updated.get("access", "")).strip(),
        base_url=str(updated.get("endpoint", "")).strip() or DEFAULT_BASE_URL,
        headers=dict(_COPILOT_HEADERS),
    )


def login_device_flow(
    *,
    client_id: str | None = None,
    open_browser: bool = True,
    status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = status or (lambda _text: None)
    resolved_client_id = (
        (client_id or "").strip() or (os.getenv("SWARMEE_GITHUB_COPILOT_CLIENT_ID") or "").strip() or GITHUB_CLIENT_ID
    )

    device = _post_form_json(
        DEVICE_ENDPOINT,
        {"client_id": resolved_client_id, "scope": "read:user"},
    )
    device_code = str(device.get("device_code", "")).strip()
    user_code = str(device.get("user_code", "")).strip()
    verification_uri = str(device.get("verification_uri", "")).strip()
    interval_s = int(device.get("interval", 5)) if str(device.get("interval", "")).strip() else 5
    expires_in_s = int(device.get("expires_in", 900)) if str(device.get("expires_in", "")).strip() else 900

    if not device_code or not user_code or not verification_uri:
        raise RuntimeError("GitHub device authorization did not return the expected fields.")

    emit("GitHub Copilot sign-in required.")
    emit(f"Open: {verification_uri}")
    emit(f"Enter code: {user_code}")
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(verification_uri)

    deadline = time.time() + expires_in_s
    while True:
        if time.time() >= deadline:
            raise RuntimeError("GitHub device sign-in timed out before authorization completed.")

        time.sleep(max(1, interval_s))
        token_payload = _post_form_json(
            TOKEN_ENDPOINT,
            {
                "client_id": resolved_client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )

        refresh_token = str(token_payload.get("access_token", "")).strip()
        if refresh_token:
            refreshed = exchange_refresh_token(refresh_token)
            record = {
                "type": "oauth",
                "refresh": refresh_token,
                "access": refreshed.get("access"),
                "expires": refreshed.get("expires"),
                "endpoint": refreshed.get("endpoint"),
                "account_id": refreshed.get("account_id"),
            }
            path = set_provider_record(PROVIDER_NAME, record)
            return {
                "provider": PROVIDER_NAME,
                "path": str(path),
                "type": "oauth",
                "expires": record.get("expires"),
            }

        error_code = str(token_payload.get("error", "")).strip().lower()
        if error_code in {"authorization_pending", "slow_down"}:
            if error_code == "slow_down":
                interval_s += 5
            continue
        if error_code == "expired_token":
            raise RuntimeError("GitHub device code expired. Run connect again to restart sign-in.")
        if error_code == "access_denied":
            raise RuntimeError("GitHub sign-in was denied.")
        if error_code:
            desc = str(token_payload.get("error_description", "")).strip()
            raise RuntimeError(desc or f"GitHub OAuth error: {error_code}")
        raise RuntimeError("GitHub OAuth token endpoint returned an unexpected response.")


def save_api_key(
    api_key: str,
    *,
    base_url: str | None = None,
) -> Path:
    key = (api_key or "").strip()
    if not key:
        raise ValueError("api_key is required")
    record = {
        "type": "api",
        "key": key,
        "base_url": (base_url or "").strip() or DEFAULT_BASE_URL,
    }
    path = set_provider_record(PROVIDER_NAME, record)
    return Path(path)
