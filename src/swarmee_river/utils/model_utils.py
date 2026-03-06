"""Utilities for loading model providers in Swarmee."""

import importlib
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, cast

from botocore.config import Config
from strands.models import Model

from swarmee_river.config.env_policy import getenv_secret
from swarmee_river.settings import ModelTier, ProviderModels, SwarmeeSettings

_THINKING_ENABLE_TOKENS = {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}
_THINKING_DISABLE_TOKENS = {"0", "false", "f", "no", "n", "off", "disable", "disabled", ""}
_BEDROCK_THINKING_BUDGET_DEFAULT = 2048
_BEDROCK_THINKING_BUDGET_MAX = 32768
_BEDROCK_EXTENDED_BUDGETS = {"low": 1024, "medium": 4096, "high": 8192}
_BEDROCK_ADAPTIVE_EFFORTS = {"low": "low", "medium": "medium", "high": "high"}
_BEDROCK_INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
_OPENAI_RESPONSES_REASONING_UNSUPPORTED = (
    "gpt-5-mini",
    "gpt-5-micro",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BedrockModelCapabilities:
    family: str
    reasoning_mode: str
    supports_guardrails: bool
    supports_cache_tools: bool
    supports_forced_tool_with_reasoning: bool = False


def _as_int(value: Any, default: int, *, min_value: int | None = None) -> int:
    if value is None:
        out = default
    else:
        raw = str(value).strip()
        if not raw:
            out = default
        else:
            with_value = raw[1:] if raw.startswith(("+", "-")) else raw
            if not with_value.isdigit():
                return default
            out = int(raw)
    if min_value is not None and out < min_value:
        return default
    return out


def _as_float(value: Any, default: float, *, min_value: float | None = None) -> float:
    if value is None:
        out = default
    else:
        raw = str(value).strip()
        if not raw:
            out = default
        else:
            try:
                out = float(raw)
            except Exception:
                out = default
    if min_value is not None and out < min_value:
        return default
    return out


def _provider_extra(settings: SwarmeeSettings, provider: str) -> dict[str, Any]:
    pm = settings.models.providers.get(provider)
    if isinstance(pm, ProviderModels):
        return dict(pm.extra or {})
    return {}


def bedrock_model_capabilities(model_id: str | None) -> BedrockModelCapabilities:
    normalized = str(model_id or "").strip().lower()
    if "anthropic.claude-opus-4-6" in normalized:
        return BedrockModelCapabilities(
            family="claude_opus_4_6",
            reasoning_mode="adaptive",
            supports_guardrails=True,
            supports_cache_tools=True,
        )
    if "anthropic.claude-sonnet-4-5" in normalized:
        return BedrockModelCapabilities(
            family="claude_sonnet_4_5",
            reasoning_mode="extended",
            supports_guardrails=True,
            supports_cache_tools=True,
        )
    if "anthropic.claude-haiku-4-5" in normalized:
        return BedrockModelCapabilities(
            family="claude_haiku_4_5",
            reasoning_mode="extended",
            supports_guardrails=True,
            supports_cache_tools=True,
        )
    if "anthropic.claude" in normalized:
        return BedrockModelCapabilities(
            family="claude_generic",
            reasoning_mode="extended",
            supports_guardrails=True,
            supports_cache_tools=True,
        )
    return BedrockModelCapabilities(
        family="generic_bedrock",
        reasoning_mode="none",
        supports_guardrails=True,
        supports_cache_tools=False,
    )


def _default_bedrock_model_config(settings: SwarmeeSettings) -> dict[str, Any]:
    extra = _provider_extra(settings, "bedrock")
    max_tokens = settings.models.max_output_tokens if settings.models.max_output_tokens is not None else 32768
    max_tokens = _as_int(max_tokens, 32768, min_value=1)
    # Treat <= 0 as invalid; callers should use positive seconds.
    read_timeout = _as_float(extra.get("read_timeout_sec"), 45.0, min_value=0.01)
    connect_timeout = _as_float(extra.get("connect_timeout_sec"), 5.0, min_value=0.01)
    max_retries = _as_int(extra.get("max_retries"), 2, min_value=0)
    config: dict[str, Any] = {
        "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "max_tokens": max_tokens,
        "boto_client_config": Config(
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
            retries={
                "max_attempts": max_retries,
                "mode": "adaptive",
            },
        ),
    }
    return config


def _bedrock_reasoning_disabled(settings: SwarmeeSettings) -> bool:
    extra = _provider_extra(settings, "bedrock")
    raw_token = extra.get("thinking_type")
    if raw_token is None:
        return False
    return str(raw_token).strip().lower() in _THINKING_DISABLE_TOKENS


def _bedrock_beta_features(settings: SwarmeeSettings) -> list[str]:
    extra = _provider_extra(settings, "bedrock")
    raw = str(extra.get("anthropic_beta") or "").strip()
    return [part.strip() for part in raw.split(",") if part.strip()]


def _bedrock_extended_budget_for_tier(tier: ModelTier, settings: SwarmeeSettings) -> int:
    extra = _provider_extra(settings, "bedrock")
    override = _as_int(extra.get("thinking_budget_tokens"), 0, min_value=1)
    if override > 0:
        return min(override, _BEDROCK_THINKING_BUDGET_MAX)
    effort = tier.reasoning.effort if tier.reasoning is not None else "medium"
    return min(_BEDROCK_EXTENDED_BUDGETS.get(effort, _BEDROCK_THINKING_BUDGET_DEFAULT), _BEDROCK_THINKING_BUDGET_MAX)


def _bedrock_adaptive_effort_for_tier(tier: ModelTier) -> str:
    effort = tier.reasoning.effort if tier.reasoning is not None else "medium"
    return _BEDROCK_ADAPTIVE_EFFORTS.get(effort, "medium")


def _forced_bedrock_tool_choice(tool_choice: Any) -> bool:
    if not isinstance(tool_choice, dict):
        return False
    return "any" in tool_choice or "tool" in tool_choice


def sanitize_bedrock_converse_config(
    config: dict[str, Any],
    *,
    tier: ModelTier,
    settings: SwarmeeSettings,
    tool_choice: dict[str, Any] | None = None,
) -> BedrockModelCapabilities:
    """Normalize Bedrock request config from guided tier settings and model-family capabilities."""
    capabilities = bedrock_model_capabilities(str(config.get("model_id") or ""))
    additional = dict(config.get("additional_request_fields") or {}) if isinstance(
        config.get("additional_request_fields"), dict
    ) else {}

    additional.pop("thinking", None)
    additional.pop("output_config", None)
    additional.pop("anthropic_beta", None)

    if capabilities.supports_cache_tools:
        explicit_cache_tools = str(_provider_extra(settings, "bedrock").get("cache_tools") or "").strip()
        if explicit_cache_tools:
            config["cache_tools"] = explicit_cache_tools
        elif tier.context is not None and tier.context.strategy in {"cache_safe", "long_running"}:
            config["cache_tools"] = "default"
        else:
            config.pop("cache_tools", None)
    else:
        config.pop("cache_tools", None)

    if _forced_bedrock_tool_choice(tool_choice) or _bedrock_reasoning_disabled(settings):
        if additional:
            config["additional_request_fields"] = additional
        else:
            config.pop("additional_request_fields", None)
        return capabilities

    if capabilities.reasoning_mode == "adaptive":
        additional["thinking"] = {"type": "adaptive"}
        additional["output_config"] = {"effort": _bedrock_adaptive_effort_for_tier(tier)}
    elif capabilities.reasoning_mode == "extended":
        additional["thinking"] = {
            "type": "enabled",
            "budget_tokens": _bedrock_extended_budget_for_tier(tier, settings),
        }
        beta_features = _bedrock_beta_features(settings)
        if not beta_features and tier.tooling is not None and tier.tooling.mode != "minimal":
            beta_features = [_BEDROCK_INTERLEAVED_THINKING_BETA]
        if beta_features:
            additional["anthropic_beta"] = beta_features

    if additional:
        config["additional_request_fields"] = additional
    else:
        config.pop("additional_request_fields", None)
    return capabilities


def default_model_config(provider: str, settings: SwarmeeSettings) -> dict[str, Any]:
    provider = provider.strip().lower()
    if provider == "bedrock":
        return _default_bedrock_model_config(settings)

    if provider == "ollama":
        extra = _provider_extra(settings, "ollama")
        return {
            "host": str(extra.get("host") or "").strip() or None,
            "model_id": "llama3.1",
        }

    if provider == "openai":
        extra = _provider_extra(settings, "openai")
        model_id = "gpt-5-nano"
        max_tokens = settings.models.max_output_tokens
        params: dict[str, Any] = {}
        if isinstance(max_tokens, int) and max_tokens > 0:
            params["max_output_tokens"] = int(max_tokens)

        client_args: dict[str, Any] = {}
        client_args["max_retries"] = _as_int(extra.get("max_retries"), 0, min_value=0)
        api_key = getenv_secret("OPENAI_API_KEY")
        if api_key:
            client_args["api_key"] = api_key
        base_url = str(extra.get("base_url") or "").strip().rstrip("/")
        if base_url:
            # OpenAI Python SDK expects base_url to include `/v1` (default is https://api.openai.com/v1).
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
            client_args["base_url"] = base_url

        config: dict[str, Any] = {
            "model_id": model_id,
            "transport": "responses",
        }
        if params:
            config["params"] = params
        if client_args:
            config["client_args"] = client_args
        return config

    if provider == "github_copilot":
        extra = _provider_extra(settings, "github_copilot")
        model_id = "gpt-4o"
        max_tokens = settings.models.max_output_tokens
        params: dict[str, Any] = {}
        if isinstance(max_tokens, int) and max_tokens > 0:
            params["max_completion_tokens"] = int(max_tokens)

        client_args: dict[str, Any] = {}
        client_args["max_retries"] = _as_int(extra.get("max_retries"), 0, min_value=0)

        base_url = str(extra.get("base_url") or "https://api.githubcopilot.com").strip().rstrip("/")

        integration_id = str(extra.get("integration_id") or "").strip()
        if integration_id:
            client_args["default_headers"] = {"Copilot-Integration-Id": integration_id}

        api_key = (
            getenv_secret("SWARMEE_GITHUB_COPILOT_API_KEY")
            or getenv_secret("GITHUB_TOKEN")
            or getenv_secret("GH_TOKEN")
            or ""
        ).strip()
        if not api_key:
            try:
                from swarmee_river.auth.github_copilot import resolve_runtime_credentials

                creds = resolve_runtime_credentials(refresh=True)
                api_key = creds.access_token
                if not str(extra.get("base_url") or "").strip():
                    base_url = creds.base_url
                if creds.headers:
                    existing_headers = client_args.get("default_headers")
                    merged_headers: dict[str, Any] = {}
                    if isinstance(existing_headers, dict):
                        merged_headers.update(existing_headers)
                    merged_headers.update(creds.headers)
                    client_args["default_headers"] = merged_headers
            except Exception:
                pass

        if api_key:
            client_args["api_key"] = api_key
        if base_url:
            client_args["base_url"] = base_url

        config = {"model_id": model_id}
        if params:
            config["params"] = params
        if client_args:
            config["client_args"] = client_args
        return config

    return {}


def sanitize_openai_responses_config(
    config: dict[str, Any],
    *,
    tier: ModelTier,
    settings: SwarmeeSettings,
) -> None:
    """Normalize OpenAI configs to a single Responses-oriented shape."""
    _ = settings
    config["transport"] = "responses"

    params = dict(config.get("params") or {}) if isinstance(config.get("params"), dict) else {}
    params.pop("max_completion_tokens", None)

    max_tokens = config.get("max_output_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0 and "max_output_tokens" not in params:
        params["max_output_tokens"] = max_tokens

    reasoning = tier.reasoning
    if reasoning is not None and openai_model_supports_responses_reasoning(str(config.get("model_id") or "")):
        params["reasoning"] = {"effort": reasoning.effort}
    else:
        if "reasoning" in params:
            params.pop("reasoning", None)
        if reasoning is not None:
            logger.info(
                "OpenAI model '%s' uses Responses transport without reasoning because this model family does not support the reasoning argument.",
                str(config.get("model_id") or "").strip() or "(unknown)",
            )

    tooling = tier.tooling
    if tooling is not None:
        params["parallel_tool_calls"] = tooling.mode != "minimal"
        params["tool_choice"] = "auto"

    config["params"] = params


def openai_model_supports_responses_reasoning(model_id: str | None) -> bool:
    normalized = str(model_id or "").strip().lower()
    if not normalized:
        return True
    return not any(normalized == token or normalized.startswith(f"{token}-") for token in _OPENAI_RESPONSES_REASONING_UNSUPPORTED)


def load_path(name: str) -> pathlib.Path:
    """Locate the model provider module file path.

    First search "$CWD/.models". If the module file is not found, fall back to the built-in models directory.

    Args:
        name: Name of the model provider (e.g., bedrock).

    Returns:
        The file path to the model provider module.

    Raises:
        ImportError: If the model provider module cannot be found.
    """
    path = pathlib.Path.cwd() / ".models" / f"{name}.py"
    if not path.exists():
        path = pathlib.Path(__file__).parent / ".." / "models" / f"{name}.py"

    if not path.exists():
        raise ImportError(f"model_provider=<{name}> | does not exist")

    return path


def load_config(config: str) -> dict[str, Any] | None:
    """Load model configuration from a JSON string or file.

    Args:
        config: A JSON string or path to a JSON file containing model configuration.
            If empty string or '{}', the default config is used.

    Returns:
        The parsed configuration.
    """
    if not config or config == "{}":
        return None

    if config.endswith(".json"):
        with open(config, encoding="utf-8") as fp:
            payload = json.load(fp)
        return payload if isinstance(payload, dict) else None

    payload = json.loads(config)
    return payload if isinstance(payload, dict) else None


def load_model(path: pathlib.Path, config: dict[str, Any]) -> Model:
    """Dynamically load and instantiate a model provider from a Python module.

    Imports the module at the specified path and calls its 'instance' function
    with the provided configuration to create a model instance.

    Args:
        path: Path to the Python module containing the model provider implementation.
        config: Configuration to pass to the model provider's instance function.

    Returns:
        An instantiated model provider.
    """
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model provider module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    instance = getattr(module, "instance", None)
    if not callable(instance):
        raise AttributeError(f"Model provider module {path} has no callable 'instance'")

    return cast(Model, instance(**config))
