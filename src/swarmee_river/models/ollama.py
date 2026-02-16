"""Create instance of Strands SDK Ollama model provider."""

from typing import Any

from strands.models import Model
from strands.models.ollama import OllamaModel


def instance(
    host: str | None = None,
    model_id: str = "llama3.1",
    **model_config: Any,
) -> Model:
    """Create instance of SDK's Ollama model provider.

    Args:
        host: The address of the Ollama server hosting the model.
        model_id: Ollama model ID
        **model_config: Configuration options for the Ollama model.

    Returns:
        Ollama model provider.
    """

    return OllamaModel(host, model_id=model_id, **model_config)
