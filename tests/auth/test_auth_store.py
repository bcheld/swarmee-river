from __future__ import annotations

from swarmee_river.auth.store import (
    get_provider_record,
    list_auth_records,
    normalize_provider_name,
    set_provider_record,
)


def test_set_and_get_provider_record(tmp_path, monkeypatch) -> None:
    auth_path = tmp_path / "auth.json"
    monkeypatch.setenv("SWARMEE_AUTH_PATH", str(auth_path))
    monkeypatch.setenv("SWARMEE_OPENCODE_AUTH_PATH", str(tmp_path / "opencode-auth.json"))

    set_provider_record("github-copilot", {"type": "api", "key": "abc"})
    record, source = get_provider_record("github_copilot", include_opencode=False)

    assert source == "swarmee"
    assert record is not None
    assert record["type"] == "api"
    assert record["key"] == "abc"


def test_get_provider_record_falls_back_to_opencode_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AUTH_PATH", str(tmp_path / "empty-auth.json"))
    opencode_auth_path = tmp_path / "opencode-auth.json"
    monkeypatch.setenv("SWARMEE_OPENCODE_AUTH_PATH", str(opencode_auth_path))
    opencode_auth_path.write_text('{"github-copilot":{"type":"oauth","refresh":"refresh-1"}}', encoding="utf-8")

    record, source = get_provider_record("github_copilot", include_opencode=True)

    assert source == "opencode"
    assert record is not None
    assert record["type"] == "oauth"
    assert record["refresh"] == "refresh-1"


def test_list_auth_records_includes_both_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_AUTH_PATH", str(tmp_path / "swarmee-auth.json"))
    opencode_auth_path = tmp_path / "opencode-auth.json"
    monkeypatch.setenv("SWARMEE_OPENCODE_AUTH_PATH", str(opencode_auth_path))
    opencode_auth_path.write_text('{"github-copilot":{"type":"oauth","refresh":"refresh-1"}}', encoding="utf-8")

    set_provider_record("github_copilot", {"type": "api", "key": "xyz"})
    records = list_auth_records(include_opencode=True)

    providers = {(str(item.get("provider")), str(item.get("source"))) for item in records}
    assert ("github_copilot", "swarmee") in providers
    assert ("github_copilot", "opencode") in providers


def test_normalize_provider_name_maps_aws_to_bedrock() -> None:
    assert normalize_provider_name("aws") == "bedrock"
