"""Create instance of the Strands SDK OpenAI Responses model provider."""

from typing import Any

from strands.models import Model

try:
    from strands.models.openai_responses import OpenAIResponsesModel as _OpenAIProvider
except ImportError:
    from strands.models.openai import OpenAIModel as _OpenAIProvider


def instance(client_args: dict[str, Any] | None = None, **model_config: Any) -> Model:
    """Create instance of the Strands SDK OpenAI Responses model provider."""
    model_config.pop("transport", None)
    return _OpenAIProvider(client_args=client_args, **model_config)
