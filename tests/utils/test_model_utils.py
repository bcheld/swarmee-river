import pathlib
import unittest.mock

import pytest

import swarmee_river
import swarmee_river.utils.model_utils
from swarmee_river.settings import SwarmeeSettings, deep_merge_dict, default_settings_template


def _settings_with(override: dict) -> SwarmeeSettings:
    base = default_settings_template().to_dict()
    merged = deep_merge_dict(base, override)
    return SwarmeeSettings.from_dict(merged)


@pytest.fixture
def cwd(tmp_path):
    with unittest.mock.patch.object(swarmee_river.utils.model_utils.pathlib.Path, "cwd") as mock_cwd:
        mock_cwd.return_value = tmp_path
        yield tmp_path


@pytest.fixture
def custom_model_dir(cwd):
    dir = cwd / ".models"
    dir.mkdir()

    return dir


@pytest.fixture
def packaged_model_dir():
    return pathlib.Path(swarmee_river.models.__file__).parent


@pytest.fixture
def config_str():
    return '{"model_id": "test"}'


@pytest.fixture
def config_path(config_str, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(config_str)

    return path


def test_load_path_custom(custom_model_dir):
    model_path = custom_model_dir / "test.py"
    model_path.touch()

    tru_path = swarmee_river.utils.model_utils.load_path("test")
    exp_path = model_path
    assert tru_path == exp_path


@pytest.mark.parametrize("name", ["bedrock", "ollama", "openai", "github_copilot"])
def test_load_path_packaged(name, packaged_model_dir):
    tru_path = swarmee_river.utils.model_utils.load_path(name).resolve()
    exp_path = packaged_model_dir / f"{name}.py"
    assert tru_path == exp_path


def test_load_path_missing():
    with pytest.raises(ImportError):
        swarmee_river.utils.model_utils.load_path("invalid")


def test_load_config_str(config_str):
    tru_config = swarmee_river.utils.model_utils.load_config(config_str)
    exp_config = {"model_id": "test"}
    assert tru_config == exp_config


def test_load_config_path(config_path):
    tru_config = swarmee_river.utils.model_utils.load_config(str(config_path))
    exp_config = {"model_id": "test"}
    assert tru_config == exp_config


def test_load_config_default_none():
    assert swarmee_river.utils.model_utils.load_config("{}") is None


def test_default_model_config_openai():
    config = swarmee_river.utils.model_utils.default_model_config("openai", default_settings_template())
    assert config["model_id"]


def test_default_model_config_bedrock_uses_responsive_defaults(monkeypatch):
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", default_settings_template())
    boto_config = config["boto_client_config"]

    assert boto_config.read_timeout == 45.0
    assert boto_config.connect_timeout == 5.0
    assert boto_config.retries["max_attempts"] == 2


def test_default_model_config_bedrock_honors_timeout_and_retry_settings():
    settings = _settings_with(
        {
            "models": {
                "providers": {
                    "bedrock": {
                        "read_timeout_sec": 75.0,
                        "connect_timeout_sec": 12.0,
                        "max_retries": 2,
                    }
                }
            }
        }
    )
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    boto_config = config["boto_client_config"]

    assert boto_config.read_timeout == 75.0
    assert boto_config.connect_timeout == 12.0
    assert boto_config.retries["max_attempts"] == 2


def test_default_model_config_bedrock_invalid_settings_falls_back():
    settings = _settings_with(
        {
            "models": {
                "providers": {
                    "bedrock": {
                        "read_timeout_sec": "abc",
                        "connect_timeout_sec": 0,
                        "max_retries": -2,
                    }
                }
            }
        }
    )
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    boto_config = config["boto_client_config"]

    assert boto_config.read_timeout == 45.0
    assert boto_config.connect_timeout == 5.0
    assert boto_config.retries["max_attempts"] == 2


@pytest.mark.parametrize("token", ["disable", "disabled", "off", "false", "0", ""])
def test_default_model_config_bedrock_omits_thinking_when_disabled_in_settings(token):
    settings = _settings_with({"models": {"providers": {"bedrock": {"thinking_type": token}}}})
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    additional = config.get("additional_request_fields")
    assert isinstance(additional, dict)
    assert "thinking" not in additional


@pytest.mark.parametrize("token", ["enable", "enabled", "on", "true", "1"])
def test_default_model_config_bedrock_emits_enabled_thinking_payload_from_settings(token):
    settings = _settings_with(
        {
            "models": {
                "providers": {
                    "bedrock": {
                        "thinking_type": token,
                        "thinking_budget_tokens": 3072,
                    }
                }
            }
        }
    )
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    thinking = config["additional_request_fields"]["thinking"]
    assert thinking["type"] == "enabled"
    assert thinking["budget_tokens"] == 3072


def test_default_model_config_bedrock_invalid_budget_falls_back_to_default():
    settings = _settings_with(
        {"models": {"providers": {"bedrock": {"thinking_type": "enabled", "thinking_budget_tokens": "x"}}}}
    )
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    thinking = config["additional_request_fields"]["thinking"]
    assert thinking["type"] == "enabled"
    assert thinking["budget_tokens"] == 2048


def test_default_model_config_openai_includes_max_retries_default_zero():
    config = swarmee_river.utils.model_utils.default_model_config("openai", default_settings_template())
    assert config["client_args"]["max_retries"] == 0


def test_default_model_config_openai_honors_max_retries_settings():
    settings = _settings_with({"models": {"providers": {"openai": {"max_retries": 2}}}})
    config = swarmee_river.utils.model_utils.default_model_config("openai", settings)
    assert config["client_args"]["max_retries"] == 2


def test_default_model_config_openai_invalid_max_retries_falls_back_to_default():
    settings = _settings_with({"models": {"providers": {"openai": {"max_retries": "abc"}}}})
    config = swarmee_river.utils.model_utils.default_model_config("openai", settings)
    assert config["client_args"]["max_retries"] == 0


def test_default_model_config_github_copilot_defaults(monkeypatch):
    monkeypatch.delenv("SWARMEE_GITHUB_COPILOT_API_KEY", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "token-1")

    config = swarmee_river.utils.model_utils.default_model_config("github_copilot", default_settings_template())

    assert config["model_id"] == "gpt-4o"
    assert config["client_args"]["api_key"] == "token-1"
    assert config["client_args"]["base_url"] == "https://api.githubcopilot.com"
    assert config["client_args"]["max_retries"] == 0


def test_default_model_config_github_copilot_honors_swarmee_token_and_retries(monkeypatch):
    monkeypatch.setenv("SWARMEE_GITHUB_COPILOT_API_KEY", "preferred")
    monkeypatch.setenv("GITHUB_TOKEN", "ignored")
    settings = _settings_with(
        {
            "models": {
                "max_output_tokens": 256,
                "providers": {
                    "github_copilot": {
                        "max_retries": 2,
                        "base_url": "https://internal.example/copilot",
                    }
                },
            }
        }
    )
    config = swarmee_river.utils.model_utils.default_model_config("github_copilot", settings)

    assert config["client_args"]["api_key"] == "preferred"
    assert config["client_args"]["max_retries"] == 2
    assert config["client_args"]["base_url"] == "https://internal.example/copilot"
    assert config["params"]["max_completion_tokens"] == 256


def test_load_model(custom_model_dir):
    model_path = custom_model_dir / "test_model.py"
    model_path.write_text("def instance(**config): return config")

    tru_result = swarmee_river.utils.model_utils.load_model(model_path, {"k1": "v1"})
    exp_result = {"k1": "v1"}
    assert tru_result == exp_result
