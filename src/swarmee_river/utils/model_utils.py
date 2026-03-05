"""Utilities for loading model providers in Swarmee."""

import importlib
import json
import pathlib
from typing import Any, cast

from botocore.config import Config
from strands.models import Model

from swarmee_river.config.env_policy import getenv_secret
from swarmee_river.settings import ProviderModels, SwarmeeSettings


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


def _default_bedrock_model_config(settings: SwarmeeSettings) -> dict[str, Any]:
    extra = _provider_extra(settings, "bedrock")
    max_tokens = settings.models.max_output_tokens if settings.models.max_output_tokens is not None else 32768
    max_tokens = _as_int(max_tokens, 32768, min_value=1)
    # Treat <= 0 as invalid; callers should use positive seconds.
    read_timeout = _as_float(extra.get("read_timeout_sec"), 45.0, min_value=0.01)
    connect_timeout = _as_float(extra.get("connect_timeout_sec"), 5.0, min_value=0.01)
    max_retries = _as_int(extra.get("max_retries"), 2, min_value=0)
    thinking_type = str(extra.get("thinking_type") or "enabled").strip() or "enabled"
    thinking_budget_tokens = _as_int(extra.get("thinking_budget_tokens"), 2048, min_value=1)
    cache_tools = str(extra.get("cache_tools") or "default").strip() or "default"
    anthropic_beta_features = str(extra.get("anthropic_beta") or "interleaved-thinking-2025-05-14").strip()

    config: dict[str, Any] = {
        "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "max_tokens": max_tokens,
        "boto_client_config": Config(
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
            retries={
                "max_attempts": max_retries,
                "mode": "adaptive",
            },
        ),
        "additional_request_fields": {
            "thinking": {
                "type": thinking_type,
                "budget_tokens": thinking_budget_tokens,
            },
        },
        "cache_tools": cache_tools,
    }
    if anthropic_beta_features:
        config["additional_request_fields"]["anthropic_beta"] = [
            part.strip() for part in anthropic_beta_features.split(",") if part.strip()
        ]
    return config


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
            # OpenAI Chat Completions uses `max_completion_tokens` for newer models.
            params["max_completion_tokens"] = int(max_tokens)

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

        config: dict[str, Any] = {"model_id": model_id}
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
