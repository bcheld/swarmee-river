from __future__ import annotations

import os
from typing import Final

# End-user supported environment variables should be secrets only.
SECRET_ENV_ALLOWLIST: Final[set[str]] = {
    # OpenAI
    "OPENAI_API_KEY",
    # GitHub Copilot (preferred) + legacy aliases
    "SWARMEE_GITHUB_COPILOT_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    # AWS SDK standard credentials (Swarmee may not read these directly, but they are
    # part of the documented "external provider env" surface).
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
}

# Internal process wiring environment variables (not user-configurable).
INTERNAL_ENV_ALLOWLIST: Final[set[str]] = {
    "PYTHONUNBUFFERED",
    "PYTHONWARNINGS",
    "SWARMEE_BROKER_IDLE_TIMEOUT",
    "SWARMEE_ENABLE_PROJECT_CONTEXT_TOOL",
    "SWARMEE_NOTEBOOK_USE_RUNTIME",
    "SWARMEE_REPAIR_TOOL_MESSAGES",
    "SWARMEE_SESSION_IDLE_TIMEOUT",
    "SWARMEE_SESSION_ID",
    "SWARMEE_SPINNERS",
    "SWARMEE_TOOLING_S3_PREFIX",
    "SWARMEE_TUI_EVENTS",
    "SWARMEE_USER_CONTEXT_SOURCE_MAX_CHARS",
    "SWARMEE_USER_CONTEXT_TOTAL_MAX_CHARS",
}

# Internal env overrides that are allowed to be persisted in `.swarmee/settings.json` under
# the legacy `env` section (migration-only). Keep this very small.
INTERNAL_SETTINGS_ENV_OVERRIDE_ALLOWLIST: Final[set[str]] = {
    "PYTHONUNBUFFERED",
    "PYTHONWARNINGS",
}


def is_secret_env_key(name: str) -> bool:
    return str(name or "").strip() in SECRET_ENV_ALLOWLIST


def is_internal_env_key(name: str) -> bool:
    return str(name or "").strip() in INTERNAL_ENV_ALLOWLIST


def getenv_secret(name: str) -> str | None:
    """Read a supported secret env var, or None if not set/allowed."""
    key = str(name or "").strip()
    if not key or key not in SECRET_ENV_ALLOWLIST:
        return None
    value = os.getenv(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def getenv_internal(name: str) -> str | None:
    """Read an internal wiring env var, or None if not set/allowed."""
    key = str(name or "").strip()
    if not key or key not in INTERNAL_ENV_ALLOWLIST:
        return None
    value = os.getenv(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def filter_project_env_overrides(raw_env: object) -> dict[str, str]:
    """
    Project settings.json used to support arbitrary env overrides.

    New policy: only allow internal wiring env vars here (migration-only). End-user
    configuration must be expressed as structured settings, not env var injection.
    """
    if not isinstance(raw_env, dict):
        return {}
    resolved: dict[str, str] = {}
    for raw_key, raw_value in raw_env.items():
        key = str(raw_key or "").strip()
        if not key or key not in INTERNAL_SETTINGS_ENV_OVERRIDE_ALLOWLIST:
            continue
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        resolved[key] = value
    return resolved
