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
    has_credentials, _source = resolve_aws_auth_source()
    return has_credentials


def resolve_aws_auth_source() -> tuple[bool, str]:
    """
    Return whether AWS credentials resolve and the normalized source label.

    Source labels are normalized for diagnostics:
    - env
    - profile
    - process
    - imds
    - unknown
    """
    try:
        import botocore.session

        session = botocore.session.get_session()
        credentials = session.get_credentials()
        if credentials is None:
            return False, "unknown"
        raw_method = str(getattr(credentials, "method", "") or "").strip().lower()
    except Exception:
        return False, "unknown"

    if raw_method in {"env"}:
        return True, "env"
    if raw_method in {"shared-credentials-file", "config-file", "assume-role"}:
        return True, "profile"
    if raw_method in {"custom-process"}:
        return True, "process"
    if raw_method in {"iam-role", "container-role", "ec2-credentials-file"}:
        return True, "imds"
    if raw_method:
        return True, raw_method
    return True, "unknown"


def resolve_aws_region_source() -> tuple[str | None, str]:
    """
    Resolve AWS region and label the source.

    Source labels:
    - env
    - profile_or_config
    - runtime
    - unknown
    """
    env_region = str(os.getenv("AWS_REGION") or "").strip()
    if env_region:
        return env_region, "env"

    env_default_region = str(os.getenv("AWS_DEFAULT_REGION") or "").strip()
    if env_default_region:
        return env_default_region, "env"

    try:
        import botocore.session

        session = botocore.session.get_session()
        inferred = str(session.get_config_variable("region") or "").strip()
        if inferred:
            source = "runtime"
            with_config = {}
            with_scoped = getattr(session, "get_scoped_config", None)
            if callable(with_scoped):
                try:
                    scoped = with_scoped()
                    with_config = scoped if isinstance(scoped, dict) else {}
                except Exception:
                    with_config = {}
            if str(with_config.get("region") or "").strip():
                source = "profile_or_config"
            return inferred, source
    except Exception:
        pass

    return None, "unknown"


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
