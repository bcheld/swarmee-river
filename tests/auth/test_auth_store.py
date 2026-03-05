from __future__ import annotations

from swarmee_river.auth.store import (
    auth_store_path,
    get_provider_record,
    list_auth_records,
    normalize_provider_name,
    opencode_auth_store_path,
    set_provider_record,
)


def test_set_and_get_provider_record(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    set_provider_record("github-copilot", {"type": "api", "key": "abc"})
    record, source = get_provider_record("github_copilot", include_opencode=False)

    assert source == "swarmee"
    assert record is not None
    assert record["type"] == "api"
    assert record["key"] == "abc"


def test_get_provider_record_falls_back_to_opencode_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    opencode_path = opencode_auth_store_path()
    opencode_path.parent.mkdir(parents=True, exist_ok=True)
    opencode_path.write_text('{"github-copilot":{"type":"oauth","refresh":"refresh-1"}}', encoding="utf-8")

    record, source = get_provider_record("github_copilot", include_opencode=True)

    assert source == "opencode"
    assert record is not None
    assert record["type"] == "oauth"
    assert record["refresh"] == "refresh-1"


def test_list_auth_records_includes_both_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    # Seed opencode auth store.
    opencode_path = opencode_auth_store_path()
    opencode_path.parent.mkdir(parents=True, exist_ok=True)
    opencode_path.write_text('{"github-copilot":{"type":"oauth","refresh":"refresh-1"}}', encoding="utf-8")
    # Ensure swarmee auth store starts clean for this test.
    swarmee_auth = auth_store_path()
    swarmee_auth.parent.mkdir(parents=True, exist_ok=True)

    set_provider_record("github_copilot", {"type": "api", "key": "xyz"})
    records = list_auth_records(include_opencode=True)

    providers = {(str(item.get("provider")), str(item.get("source"))) for item in records}
    assert ("github_copilot", "swarmee") in providers
    assert ("github_copilot", "opencode") in providers


def test_normalize_provider_name_maps_aws_to_bedrock() -> None:
    assert normalize_provider_name("aws") == "bedrock"
