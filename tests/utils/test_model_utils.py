import importlib
import importlib.metadata
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
    assert config["transport"] == "responses"


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


def test_default_model_config_bedrock_no_longer_emits_reasoning_fields_by_default():
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", default_settings_template())
    assert "additional_request_fields" not in config
    assert "cache_tools" not in config


def test_bedrock_model_capabilities_detects_current_claude_families():
    adaptive = swarmee_river.utils.model_utils.bedrock_model_capabilities("us.anthropic.claude-opus-4-6-v1:0")
    extended = swarmee_river.utils.model_utils.bedrock_model_capabilities(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    generic = swarmee_river.utils.model_utils.bedrock_model_capabilities("amazon.nova-lite-v1:0")

    assert adaptive.reasoning_mode == "adaptive"
    assert adaptive.supports_cache_tools is True
    assert extended.reasoning_mode == "extended"
    assert generic.reasoning_mode == "none"
    assert generic.supports_cache_tools is False


@pytest.mark.parametrize("token", ["disable", "disabled", "off", "false", "0", ""])
def test_sanitize_bedrock_converse_config_omits_thinking_when_disabled_in_settings(token):
    settings = _settings_with({"models": {"providers": {"bedrock": {"thinking_type": token}}}})
    tier = settings.models.providers["bedrock"].tiers["deep"]
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    config["model_id"] = tier.model_id

    swarmee_river.utils.model_utils.sanitize_bedrock_converse_config(config, tier=tier, settings=settings)

    additional = config.get("additional_request_fields")
    assert additional is None or "thinking" not in additional


@pytest.mark.parametrize("token", ["enable", "enabled", "on", "true", "1"])
def test_sanitize_bedrock_converse_config_emits_extended_thinking_payload_for_claude_45(token):
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
    tier = settings.models.providers["bedrock"].tiers["balanced"]
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    config["model_id"] = tier.model_id
    swarmee_river.utils.model_utils.sanitize_bedrock_converse_config(config, tier=tier, settings=settings)

    thinking = config["additional_request_fields"]["thinking"]
    assert thinking["type"] == "enabled"
    assert thinking["budget_tokens"] == 3072
    assert config["additional_request_fields"]["anthropic_beta"] == ["interleaved-thinking-2025-05-14"]


def test_sanitize_bedrock_converse_config_invalid_budget_falls_back_to_effort_default():
    settings = _settings_with(
        {"models": {"providers": {"bedrock": {"thinking_type": "enabled", "thinking_budget_tokens": "x"}}}}
    )
    tier = settings.models.providers["bedrock"].tiers["balanced"]
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    config["model_id"] = tier.model_id
    swarmee_river.utils.model_utils.sanitize_bedrock_converse_config(config, tier=tier, settings=settings)

    thinking = config["additional_request_fields"]["thinking"]
    assert thinking["type"] == "enabled"
    assert thinking["budget_tokens"] == 4096


def test_sanitize_bedrock_converse_config_emits_adaptive_thinking_for_opus_46():
    settings = _settings_with({})
    tier = settings.models.providers["bedrock"].tiers["deep"]
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    config["model_id"] = tier.model_id

    swarmee_river.utils.model_utils.sanitize_bedrock_converse_config(config, tier=tier, settings=settings)

    assert config["cache_tools"] == "default"
    assert config["additional_request_fields"]["thinking"] == {"type": "adaptive"}
    assert config["additional_request_fields"]["output_config"] == {"effort": "high"}
    assert "anthropic_beta" not in config["additional_request_fields"]


def test_sanitize_bedrock_converse_config_strips_reasoning_when_tool_choice_forces_tool_use():
    settings = _settings_with({})
    tier = settings.models.providers["bedrock"].tiers["deep"]
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    config["model_id"] = tier.model_id

    swarmee_river.utils.model_utils.sanitize_bedrock_converse_config(
        config,
        tier=tier,
        settings=settings,
        tool_choice={"any": {}},
    )

    additional = config.get("additional_request_fields")
    assert additional is None or "thinking" not in additional


def test_sanitize_bedrock_converse_config_skips_reasoning_for_non_claude_models():
    settings = _settings_with(
        {
            "models": {
                "providers": {
                    "bedrock": {
                        "tiers": {
                            "balanced": {
                                "model_id": "amazon.nova-lite-v1:0",
                                "reasoning": {"effort": "medium"},
                                "tooling": {"mode": "standard", "discovery": "off"},
                            }
                        }
                    }
                }
            }
        }
    )
    tier = settings.models.providers["bedrock"].tiers["balanced"]
    config = swarmee_river.utils.model_utils.default_model_config("bedrock", settings)
    config["model_id"] = tier.model_id

    capabilities = swarmee_river.utils.model_utils.sanitize_bedrock_converse_config(
        config,
        tier=tier,
        settings=settings,
    )

    assert capabilities.reasoning_mode == "none"
    assert "additional_request_fields" not in config
    assert "cache_tools" not in config


def test_default_model_config_openai_includes_max_retries_default_zero():
    config = swarmee_river.utils.model_utils.default_model_config("openai", default_settings_template())
    assert config["client_args"]["max_retries"] == 0
    assert "max_completion_tokens" not in config.get("params", {})


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


def test_sanitize_openai_responses_config_applies_guided_fields():
    settings = _settings_with(
        {
            "models": {
                "providers": {
                    "openai": {
                        "tiers": {
                            "deep": {
                                "transport": "responses",
                                "reasoning": {"effort": "high"},
                                "tooling": {"mode": "tool-heavy", "discovery": "search"},
                            }
                        }
                    }
                }
            }
        }
    )
    tier = settings.models.providers["openai"].tiers["deep"]
    config = swarmee_river.utils.model_utils.default_model_config("openai", settings)

    swarmee_river.utils.model_utils.sanitize_openai_responses_config(config, tier=tier, settings=settings)

    assert config["transport"] == "responses"
    assert config["params"]["reasoning"] == {"effort": "high"}
    assert config["params"]["parallel_tool_calls"] is True
    assert config["params"]["tool_choice"] == "auto"


@pytest.mark.parametrize("model_id", ["gpt-5-mini", "gpt-5-micro", "gpt-5-mini-preview"])
def test_sanitize_openai_responses_config_strips_reasoning_for_unsupported_models(model_id: str):
    settings = _settings_with(
        {
            "models": {
                "providers": {
                    "openai": {
                        "tiers": {
                            "balanced": {
                                "transport": "responses",
                                "reasoning": {"effort": "medium"},
                                "tooling": {"mode": "standard", "discovery": "off"},
                            }
                        }
                    }
                }
            }
        }
    )
    tier = settings.models.providers["openai"].tiers["balanced"]
    config = swarmee_river.utils.model_utils.default_model_config("openai", settings)
    config["model_id"] = model_id

    swarmee_river.utils.model_utils.sanitize_openai_responses_config(config, tier=tier, settings=settings)

    assert config["transport"] == "responses"
    assert "reasoning" not in config["params"]
    assert config["params"]["parallel_tool_calls"] is True
    assert config["params"]["tool_choice"] == "auto"


def test_openai_model_supports_responses_reasoning_flags_unsupported_variants():
    assert swarmee_river.utils.model_utils.openai_model_supports_responses_reasoning("gpt-5.2") is True
    assert swarmee_river.utils.model_utils.openai_model_supports_responses_reasoning("gpt-5-mini") is False
    assert swarmee_river.utils.model_utils.openai_model_supports_responses_reasoning("gpt-5-micro") is False


def test_probe_openai_responses_transport_available(monkeypatch):
    def _fake_version(name: str) -> str:
        if name == "strands-agents":
            return "1.29.0"
        if name == "openai":
            return "2.0.0"
        return ""

    monkeypatch.setattr(swarmee_river.utils.model_utils.importlib.metadata, "version", _fake_version)

    status = swarmee_river.utils.model_utils.probe_openai_responses_transport()

    assert status.available is True
    assert status.strands_version == "1.29.0"
    assert status.openai_version == "2.0.0"
    assert "available" in status.reason.lower()


def test_probe_openai_responses_transport_reports_unsupported_strands_version(monkeypatch):
    def _fake_version(name: str) -> str:
        if name == "strands-agents":
            return "1.26.0"
        if name == "openai":
            return "2.0.0"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(swarmee_river.utils.model_utils.importlib.metadata, "version", _fake_version)

    status = swarmee_river.utils.model_utils.probe_openai_responses_transport()

    assert status.available is False
    assert status.strands_version == "1.26.0"
    assert status.openai_version == "2.0.0"
    assert "strands-agents==1.26.0" in status.reason
    assert ">=1.29.0" in status.reason


def test_probe_openai_responses_transport_missing_package(monkeypatch):
    def _fake_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(swarmee_river.utils.model_utils.importlib.metadata, "version", _fake_version)

    status = swarmee_river.utils.model_utils.probe_openai_responses_transport()

    assert status.available is False
    assert status.strands_version is None
    assert status.openai_version is None
    assert "strands-agents is not installed" in status.reason


def test_probe_openai_responses_transport_reports_incompatible_openai_sdk(monkeypatch):
    def _fake_version(name: str) -> str:
        if name == "strands-agents":
            return "1.29.0"
        if name == "openai":
            return "1.109.1"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(swarmee_river.utils.model_utils.importlib.metadata, "version", _fake_version)

    status = swarmee_river.utils.model_utils.probe_openai_responses_transport()

    assert status.available is False
    assert status.strands_version == "1.29.0"
    assert status.openai_version == "1.109.1"
    assert "openai==1.109.1" in status.reason
    assert "openai>=2.0.0,<3.0.0" in status.reason


def test_ensure_openai_responses_transport_available_raises_for_incompatible_runtime(monkeypatch):
    monkeypatch.setattr(
        swarmee_river.utils.model_utils,
        "probe_openai_responses_transport",
        lambda: swarmee_river.utils.model_utils.OpenAIResponsesTransportStatus(
            available=False,
            strands_version="1.26.0",
            openai_version="2.0.0",
            reason="Installed strands-agents==1.26.0 is below Swarmee's supported runtime version.",
        ),
    )

    with pytest.raises(ImportError, match="Swarmee's OpenAI runtime is unavailable"):
        swarmee_river.utils.model_utils.ensure_openai_responses_transport_available()


def test_load_model(custom_model_dir):
    model_path = custom_model_dir / "test_model.py"
    model_path.write_text("def instance(**config): return config")

    tru_result = swarmee_river.utils.model_utils.load_model(model_path, {"k1": "v1"})
    exp_result = {"k1": "v1"}
    assert tru_result == exp_result


def test_load_model_openai_provider_uses_unique_dynamic_module_name():
    provider_path = swarmee_river.utils.model_utils.load_path("openai")

    model = swarmee_river.utils.model_utils.load_model(
        provider_path,
        {
            "model_id": "gpt-5-nano",
            "client_args": {"api_key": "sk-test"},
            "params": {"max_output_tokens": 32},
            "transport": "responses",
        },
    )

    assert type(model).__module__.startswith("_swarmee_dynamic_model_openai_")
