from __future__ import annotations

from swarmee_river.utils import provider_utils


def test_resolve_model_provider_prefers_cli() -> None:
    provider, notice = provider_utils.resolve_model_provider(
        cli_provider="openai",
        env_provider="bedrock",
        settings_provider="bedrock",
    )
    assert provider == "openai"
    assert notice is None


def test_resolve_model_provider_falls_back_to_openai_when_bedrock_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(provider_utils, "has_aws_credentials", lambda: False)
    monkeypatch.setattr(provider_utils, "has_openai_api_key", lambda: True)

    provider, notice = provider_utils.resolve_model_provider(
        cli_provider=None,
        env_provider=None,
        settings_provider="bedrock",
    )
    assert provider == "openai"
    assert notice is not None


def test_resolve_model_provider_keeps_explicit_env_provider(monkeypatch) -> None:
    monkeypatch.setattr(provider_utils, "has_aws_credentials", lambda: False)
    monkeypatch.setattr(provider_utils, "has_openai_api_key", lambda: True)

    provider, notice = provider_utils.resolve_model_provider(
        cli_provider=None,
        env_provider="bedrock",
        settings_provider="openai",
    )
    assert provider == "bedrock"
    assert notice is None
