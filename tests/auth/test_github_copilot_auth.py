from __future__ import annotations

from swarmee_river.auth.github_copilot import (
    DEVICE_ENDPOINT,
    TOKEN_ENDPOINT,
    login_device_flow,
    resolve_runtime_credentials,
)
from swarmee_river.auth.store import get_provider_record, set_provider_record


def test_login_device_flow_saves_oauth_record(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AUTH_PATH", str(tmp_path / "auth.json"))
    monkeypatch.setenv("SWARMEE_OPENCODE_AUTH_PATH", str(tmp_path / "opencode-auth.json"))

    def _fake_post(url: str, data: dict[str, str], **_kwargs):
        if url == DEVICE_ENDPOINT:
            return {
                "device_code": "device-1",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "interval": 0,
                "expires_in": 600,
            }
        if url == TOKEN_ENDPOINT:
            return {"access_token": "refresh-1"}
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("swarmee_river.auth.github_copilot._post_form_json", _fake_post)
    monkeypatch.setattr("swarmee_river.auth.github_copilot.time.sleep", lambda _n: None)
    monkeypatch.setattr(
        "swarmee_river.auth.github_copilot.exchange_refresh_token",
        lambda _refresh: {"access": "access-1", "expires": 9_999_999_999_999, "endpoint": "https://api.example"},
    )

    result = login_device_flow(open_browser=False)

    assert result["provider"] == "github_copilot"
    record, source = get_provider_record("github_copilot", include_opencode=False)
    assert source == "swarmee"
    assert record is not None
    assert record["type"] == "oauth"
    assert record["refresh"] == "refresh-1"
    assert record["access"] == "access-1"


def test_resolve_runtime_credentials_api_record(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AUTH_PATH", str(tmp_path / "auth.json"))
    monkeypatch.setenv("SWARMEE_OPENCODE_AUTH_PATH", str(tmp_path / "opencode-auth.json"))

    set_provider_record("github_copilot", {"type": "api", "key": "api-key-1", "base_url": "https://api.example"})
    creds = resolve_runtime_credentials(refresh=False)

    assert creds.access_token == "api-key-1"
    assert creds.base_url == "https://api.example"


def test_resolve_runtime_credentials_refreshes_expired_oauth(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AUTH_PATH", str(tmp_path / "auth.json"))
    monkeypatch.setenv("SWARMEE_OPENCODE_AUTH_PATH", str(tmp_path / "opencode-auth.json"))

    set_provider_record(
        "github_copilot",
        {
            "type": "oauth",
            "refresh": "refresh-old",
            "access": "access-old",
            "expires": 0,
            "endpoint": "https://api.githubcopilot.com",
        },
    )
    monkeypatch.setattr(
        "swarmee_river.auth.github_copilot.exchange_refresh_token",
        lambda _refresh: {"access": "access-new", "expires": 9_999_999_999_999, "endpoint": "https://api.new"},
    )

    creds = resolve_runtime_credentials(refresh=True)

    assert creds.access_token == "access-new"
    assert creds.base_url == "https://api.new"
    updated, _source = get_provider_record("github_copilot", include_opencode=False)
    assert updated is not None
    assert updated["access"] == "access-new"
