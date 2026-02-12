"""Utilities for loading model providers in Swarmee."""

import importlib
import json
import os
import pathlib
from typing import Any

from botocore.config import Config
from strands.models import Model

# Default Bedrock model configuration
DEFAULT_BEDROCK_MODEL_CONFIG: dict[str, Any] = {
    "model_id": os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
    "max_tokens": int(os.getenv("STRANDS_MAX_TOKENS", "32768")),
    "boto_client_config": Config(
        read_timeout=900,
        connect_timeout=900,
        retries=dict(max_attempts=3, mode="adaptive"),
    ),
    "additional_request_fields": {
        "thinking": {
            "type": os.getenv("STRANDS_THINKING_TYPE", "enabled"),
            "budget_tokens": int(os.getenv("STRANDS_BUDGET_TOKENS", "2048")),
        },
    },
    "cache_tools": os.getenv("STRANDS_CACHE_TOOLS", "default"),
    "cache_prompt": os.getenv("STRANDS_CACHE_PROMPT", "default"),
}
ANTHROPIC_BETA_FEATURES = os.getenv("STRANDS_ANTHROPIC_BETA", "interleaved-thinking-2025-05-14")
if len(ANTHROPIC_BETA_FEATURES) > 0:
    DEFAULT_BEDROCK_MODEL_CONFIG["additional_request_fields"]["anthropic_beta"] = ANTHROPIC_BETA_FEATURES.split(",")


def default_model_config(provider: str) -> dict[str, Any]:
    provider = provider.strip().lower()
    if provider == "bedrock":
        return DEFAULT_BEDROCK_MODEL_CONFIG

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
        with open(config) as fp:
            return json.load(fp)

    return json.loads(config)


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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.instance(**config)
