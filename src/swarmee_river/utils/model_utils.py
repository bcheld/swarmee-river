"""Utilities for loading model providers in Swarmee."""

import importlib
import json
import os
import pathlib
from typing import Any, cast

from botocore.config import Config
from strands.models import Model

_THINKING_ENABLE_TOKENS = {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}
_THINKING_DISABLE_TOKENS = {"0", "false", "f", "no", "n", "off", "disable", "disabled", ""}
_BEDROCK_THINKING_BUDGET_DEFAULT = 2048
_BEDROCK_THINKING_BUDGET_MAX = 32768


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        value = default
    else:
        with_value = raw
        if with_value.startswith(("+", "-")):
            with_value = with_value[1:]
        if not with_value.isdigit():
            return default
        value = int(raw)
    if min_value is not None and value < min_value:
        return default
    return value


def _bedrock_timeout_seconds(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def _thinking_mode_token() -> str:
    return str(os.getenv("STRANDS_THINKING_TYPE", "enabled") or "").strip().lower()


def _safe_thinking_budget(value: Any) -> int:
    try:
        budget = int(value)
    except Exception:
        budget = _BEDROCK_THINKING_BUDGET_DEFAULT
    if budget <= 0:
        budget = _BEDROCK_THINKING_BUDGET_DEFAULT
    if budget > _BEDROCK_THINKING_BUDGET_MAX:
        budget = _BEDROCK_THINKING_BUDGET_MAX
    return budget


def _normalized_bedrock_thinking_fields() -> dict[str, Any] | None:
    token = _thinking_mode_token()
    if token in _THINKING_DISABLE_TOKENS:
        return None
    if token not in _THINKING_ENABLE_TOKENS:
        token = "enabled"

    budget = _safe_thinking_budget(_env_int("STRANDS_BUDGET_TOKENS", _BEDROCK_THINKING_BUDGET_DEFAULT, min_value=1))
    return {
        "type": "enabled",
        "budget_tokens": budget,
    }


def sanitize_bedrock_thinking_config(config: dict[str, Any]) -> None:
    normalized = _normalized_bedrock_thinking_fields()
    extra = config.get("additional_request_fields")
    additional_request_fields = extra if isinstance(extra, dict) else {}
    existing = additional_request_fields.get("thinking")

    if normalized is None:
        additional_request_fields.pop("thinking", None)
        if additional_request_fields:
            config["additional_request_fields"] = additional_request_fields
        else:
            config.pop("additional_request_fields", None)
        return

    budget_value = normalized["budget_tokens"]
    if isinstance(existing, dict):
        existing_type = str(existing.get("type", "")).strip().lower()
        existing_budget = existing.get("budget_tokens")
        if existing_type in _THINKING_ENABLE_TOKENS and not isinstance(existing_budget, bool):
            budget_value = _safe_thinking_budget(existing_budget)

    additional_request_fields["thinking"] = {
        "type": "enabled",
        "budget_tokens": budget_value,
    }
    config["additional_request_fields"] = additional_request_fields


def _default_bedrock_model_config() -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_id": os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        "max_tokens": _env_int("STRANDS_MAX_TOKENS", 32768, min_value=1),
        "boto_client_config": Config(
            read_timeout=_bedrock_timeout_seconds("SWARMEE_BEDROCK_READ_TIMEOUT_SEC", 45.0),
            connect_timeout=_bedrock_timeout_seconds("SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC", 5.0),
            retries={
                "max_attempts": _env_int("SWARMEE_BEDROCK_MAX_RETRIES", 2, min_value=0),
                "mode": "adaptive",
            },
        ),
        "cache_tools": os.getenv("STRANDS_CACHE_TOOLS", "default"),
    }
    additional_request_fields: dict[str, Any] = {}
    thinking_fields = _normalized_bedrock_thinking_fields()
    if thinking_fields is not None:
        additional_request_fields["thinking"] = thinking_fields
    anthropic_beta_features = os.getenv("STRANDS_ANTHROPIC_BETA", "interleaved-thinking-2025-05-14")
    if anthropic_beta_features:
        additional_request_fields["anthropic_beta"] = anthropic_beta_features.split(",")
    if additional_request_fields:
        config["additional_request_fields"] = additional_request_fields
    return config


def default_model_config(provider: str) -> dict[str, Any]:
    provider = provider.strip().lower()
    if provider == "bedrock":
        return _default_bedrock_model_config()

    if provider == "ollama":
        return {
            "host": os.getenv("SWARMEE_OLLAMA_HOST"),
            "model_id": os.getenv("SWARMEE_OLLAMA_MODEL_ID", os.getenv("OLLAMA_MODEL", "llama3.1")),
        }

    if provider == "openai":
        model_id = os.getenv("SWARMEE_OPENAI_MODEL_ID", "gpt-5-nano")
        max_tokens = os.getenv("SWARMEE_MAX_TOKENS")
        params: dict[str, Any] = {}
        if max_tokens and max_tokens.isdigit():
            # OpenAI Chat Completions uses `max_completion_tokens` for newer models.
            params["max_completion_tokens"] = int(max_tokens)

        client_args: dict[str, Any] = {}
        max_retries = os.getenv("SWARMEE_OPENAI_MAX_RETRIES", "0").strip()
        if max_retries.isdigit():
            client_args["max_retries"] = int(max_retries)
        if os.getenv("OPENAI_API_KEY"):
            client_args["api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_BASE_URL"):
            base_url = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
            # OpenAI Python SDK expects base_url to include `/v1` (default is https://api.openai.com/v1).
            if base_url and not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
            client_args["base_url"] = base_url

        config: dict[str, Any] = {"model_id": model_id}
        if params:
            config["params"] = params
        if client_args:
            config["client_args"] = client_args
        return config

    if provider == "github_copilot":
        model_id = os.getenv("SWARMEE_GITHUB_COPILOT_MODEL_ID", "gpt-4o")
        max_tokens = os.getenv("SWARMEE_MAX_TOKENS")
        params: dict[str, Any] = {}
        if max_tokens and max_tokens.isdigit():
            params["max_completion_tokens"] = int(max_tokens)

        client_args: dict[str, Any] = {}
        max_retries = os.getenv("SWARMEE_GITHUB_COPILOT_MAX_RETRIES", "0").strip()
        if max_retries.isdigit():
            client_args["max_retries"] = int(max_retries)

        base_url = os.getenv("SWARMEE_GITHUB_COPILOT_BASE_URL", "https://api.githubcopilot.com").strip().rstrip("/")

        integration_id = os.getenv("SWARMEE_GITHUB_COPILOT_INTEGRATION_ID", "").strip()
        if integration_id:
            client_args["default_headers"] = {"Copilot-Integration-Id": integration_id}

        api_key = (
            os.getenv("SWARMEE_GITHUB_COPILOT_API_KEY") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""
        ).strip()
        if not api_key:
            try:
                from swarmee_river.auth.github_copilot import resolve_runtime_credentials

                creds = resolve_runtime_credentials(refresh=True)
                api_key = creds.access_token
                if not os.getenv("SWARMEE_GITHUB_COPILOT_BASE_URL"):
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
