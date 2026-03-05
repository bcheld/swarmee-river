from __future__ import annotations

from swarmee_river.auth.store import set_provider_record
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


def test_resolve_model_provider_normalizes_copilot_aliases() -> None:
    provider, notice = provider_utils.resolve_model_provider(
        cli_provider="github-copilot",
        env_provider=None,
        settings_provider=None,
    )
    assert provider == "github_copilot"
    assert notice is None


def test_normalize_provider_name_maps_aws_alias_to_bedrock() -> None:
    assert provider_utils.normalize_provider_name("aws") == "bedrock"
    assert provider_utils.normalize_provider_name("amazon-bedrock") == "bedrock"


def test_resolve_model_provider_falls_back_to_github_copilot_when_bedrock_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(provider_utils, "has_aws_credentials", lambda: False)
    monkeypatch.setattr(provider_utils, "has_openai_api_key", lambda: False)
    monkeypatch.setattr(provider_utils, "has_github_copilot_token", lambda: True)

    provider, notice = provider_utils.resolve_model_provider(
        cli_provider=None,
        env_provider=None,
        settings_provider="bedrock",
    )
    assert provider == "github_copilot"
    assert notice is not None


def test_has_github_copilot_token_reads_auth_store(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_GITHUB_COPILOT_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    # Keep auth store writes inside the test sandbox.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    set_provider_record("github_copilot", {"type": "api", "key": "abc"})

    assert provider_utils.has_github_copilot_token() is True


def test_resolve_aws_auth_source_maps_profile(monkeypatch) -> None:
    import botocore.session

    class _Creds:
        method = "shared-credentials-file"

    class _Session:
        def get_credentials(self):
            return _Creds()

    monkeypatch.setattr(botocore.session, "get_session", lambda: _Session())
    has_creds, source = provider_utils.resolve_aws_auth_source()
    assert has_creds is True
    assert source == "profile"
