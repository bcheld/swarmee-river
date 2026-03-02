from __future__ import annotations

import os
from typing import Optional, Tuple

_PROVIDER_ALIASES: dict[str, str] = {
    "ghcp": "github_copilot",
    "copilot": "github_copilot",
    "githubcopilot": "github_copilot",
    "github-copilot": "github_copilot",
    "github_copilot": "github_copilot",
    "aws": "bedrock",
    "amazon_bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
}


def normalize_provider_name(value: object | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    normalized = token.replace("-", "_")
    return _PROVIDER_ALIASES.get(normalized, normalized)


def has_openai_api_key() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def has_github_copilot_token() -> bool:
    for key in ("SWARMEE_GITHUB_COPILOT_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"):
        if (os.getenv(key) or "").strip():
            return True
    try:
        from swarmee_river.auth.store import has_provider_record

        if has_provider_record("github_copilot"):
            return True
    except Exception:
        return False
    return False


def has_aws_credentials() -> bool:
    """
    Check whether boto/botocore can resolve AWS credentials locally.

    This does not make a network request; it only inspects configured
    credential sources (env, profiles, process providers, etc.).
    """
    try:
        import botocore.session

        session = botocore.session.get_session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:
        return False


def resolve_model_provider(
    *,
    cli_provider: str | None,
    env_provider: str | None,
    settings_provider: str | None,
) -> Tuple[str, Optional[str]]:
    """
    Resolve provider with safe fallback:
    - Respect explicit CLI/env provider choice.
    - If provider resolves to Bedrock but AWS credentials are missing:
      - fall back to OpenAI when OPENAI_API_KEY is available
      - else fall back to GitHub Copilot when a Copilot token is available
    """
    cli = normalize_provider_name(cli_provider)
    env = normalize_provider_name(env_provider)
    settings = normalize_provider_name(settings_provider)

    if cli:
        return cli, None
    if env:
        return env, None

    selected = settings or (
        "openai" if has_openai_api_key() else "github_copilot" if has_github_copilot_token() else "bedrock"
    )
    if selected == "bedrock" and not has_aws_credentials():
        if has_openai_api_key():
            return (
                "openai",
                "No AWS credentials detected for Bedrock; falling back to OpenAI because OPENAI_API_KEY is set.",
            )
        if has_github_copilot_token():
            return (
                "github_copilot",
                "No AWS credentials detected for Bedrock; "
                "falling back to GitHub Copilot because a Copilot token is set.",
            )

    return selected, None
