from __future__ import annotations

import os
from typing import Optional, Tuple


def has_openai_api_key() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


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
    - If provider resolves to Bedrock but AWS credentials are missing and
      OpenAI credentials are available, fall back to OpenAI.
    """
    cli = (cli_provider or "").strip().lower()
    env = (env_provider or "").strip().lower()
    settings = (settings_provider or "").strip().lower()

    if cli:
        return cli, None
    if env:
        return env, None

    selected = settings or ("openai" if has_openai_api_key() else "bedrock")
    if selected == "bedrock" and not has_aws_credentials() and has_openai_api_key():
        return (
            "openai",
            "No AWS credentials detected for Bedrock; falling back to OpenAI because OPENAI_API_KEY is set.",
        )

    return selected, None
