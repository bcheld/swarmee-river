import pathlib
import unittest.mock

import pytest

import swarmee_river
import swarmee_river.utils.model_utils


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
    config = swarmee_river.utils.model_utils.default_model_config("openai")
    assert config["model_id"]


def test_default_model_config_bedrock_uses_responsive_defaults(monkeypatch):
    monkeypatch.delenv("SWARMEE_BEDROCK_READ_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("SWARMEE_BEDROCK_MAX_RETRIES", raising=False)

    config = swarmee_river.utils.model_utils.default_model_config("bedrock")
    boto_config = config["boto_client_config"]

    assert boto_config.read_timeout == 60.0
    assert boto_config.connect_timeout == 10.0
    assert boto_config.retries["max_attempts"] == 1


def test_default_model_config_bedrock_honors_timeout_and_retry_env(monkeypatch):
    monkeypatch.setenv("SWARMEE_BEDROCK_READ_TIMEOUT_SEC", "75")
    monkeypatch.setenv("SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC", "12")
    monkeypatch.setenv("SWARMEE_BEDROCK_MAX_RETRIES", "2")

    config = swarmee_river.utils.model_utils.default_model_config("bedrock")
    boto_config = config["boto_client_config"]

    assert boto_config.read_timeout == 75.0
    assert boto_config.connect_timeout == 12.0
    assert boto_config.retries["max_attempts"] == 2


def test_default_model_config_bedrock_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("SWARMEE_BEDROCK_READ_TIMEOUT_SEC", "abc")
    monkeypatch.setenv("SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC", "0")
    monkeypatch.setenv("SWARMEE_BEDROCK_MAX_RETRIES", "-2")

    config = swarmee_river.utils.model_utils.default_model_config("bedrock")
    boto_config = config["boto_client_config"]

    assert boto_config.read_timeout == 60.0
    assert boto_config.connect_timeout == 10.0
    assert boto_config.retries["max_attempts"] == 1


def test_default_model_config_openai_includes_max_retries_default_zero(monkeypatch):
    monkeypatch.delenv("SWARMEE_OPENAI_MAX_RETRIES", raising=False)
    config = swarmee_river.utils.model_utils.default_model_config("openai")
    assert config["client_args"]["max_retries"] == 0


def test_default_model_config_openai_honors_max_retries_env(monkeypatch):
    monkeypatch.setenv("SWARMEE_OPENAI_MAX_RETRIES", "2")
    config = swarmee_river.utils.model_utils.default_model_config("openai")
    assert config["client_args"]["max_retries"] == 2


def test_default_model_config_openai_ignores_invalid_max_retries(monkeypatch):
    monkeypatch.setenv("SWARMEE_OPENAI_MAX_RETRIES", "abc")
    config = swarmee_river.utils.model_utils.default_model_config("openai")
    assert "max_retries" not in config.get("client_args", {})


def test_default_model_config_github_copilot_defaults(monkeypatch):
    monkeypatch.delenv("SWARMEE_GITHUB_COPILOT_MODEL_ID", raising=False)
    monkeypatch.delenv("SWARMEE_GITHUB_COPILOT_API_KEY", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "token-1")

    config = swarmee_river.utils.model_utils.default_model_config("github_copilot")

    assert config["model_id"] == "gpt-4o"
    assert config["client_args"]["api_key"] == "token-1"
    assert config["client_args"]["base_url"] == "https://api.githubcopilot.com"
    assert config["client_args"]["max_retries"] == 0


def test_default_model_config_github_copilot_honors_swarmee_token_and_retries(monkeypatch):
    monkeypatch.setenv("SWARMEE_GITHUB_COPILOT_API_KEY", "preferred")
    monkeypatch.setenv("GITHUB_TOKEN", "ignored")
    monkeypatch.setenv("SWARMEE_GITHUB_COPILOT_MAX_RETRIES", "2")
    monkeypatch.setenv("SWARMEE_GITHUB_COPILOT_BASE_URL", "https://internal.example/copilot")
    monkeypatch.setenv("SWARMEE_MAX_TOKENS", "256")

    config = swarmee_river.utils.model_utils.default_model_config("github_copilot")

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
