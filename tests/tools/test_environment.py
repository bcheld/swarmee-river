from __future__ import annotations

from tools.environment import environment


def _text(result: dict) -> str:
    return (result.get("content") or [{"text": ""}])[0].get("text", "")


def test_environment_set_get_unset(monkeypatch) -> None:
    monkeypatch.delenv("SWARMEE_ENV_TEST_KEY", raising=False)

    result = environment(action="set", key="SWARMEE_ENV_TEST_KEY", value="hello")
    assert result.get("status") == "success"

    get_result = environment(action="get", key="SWARMEE_ENV_TEST_KEY", redact=False)
    assert get_result.get("status") == "success"
    assert _text(get_result) == "hello"

    unset_result = environment(action="unset", key="SWARMEE_ENV_TEST_KEY")
    assert unset_result.get("status") == "success"

    missing = environment(action="get", key="SWARMEE_ENV_TEST_KEY")
    assert missing.get("status") == "success"
    assert _text(missing) == "(not set)"


def test_environment_list_hides_values_by_default(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_ENV_TEST_LIST", "alpha")

    result = environment(action="list", keys=["SWARMEE_ENV_TEST_LIST"])
    assert result.get("status") == "success"
    text = _text(result)
    assert "SWARMEE_ENV_TEST_LIST" in text
    assert "alpha" not in text


def test_environment_redacts_sensitive_values(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_SECRET_TOKEN", "shh")

    get_result = environment(action="get", key="SWARMEE_SECRET_TOKEN")
    assert get_result.get("status") == "success"
    assert _text(get_result) == "<redacted>"

    list_result = environment(action="list", keys=["SWARMEE_SECRET_TOKEN"], reveal=True)
    assert list_result.get("status") == "success"
    assert "SWARMEE_SECRET_TOKEN=<redacted>" in _text(list_result)


def test_environment_export_formats_shell_lines(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_ENV_TEST_EXPORT", "beta")

    result = environment(action="export", keys=["SWARMEE_ENV_TEST_EXPORT"], redact=False)
    assert result.get("status") == "success"
    assert _text(result).strip() == "export SWARMEE_ENV_TEST_EXPORT=beta"
