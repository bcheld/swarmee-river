"""Create instance of Strands SDK OpenAI model provider."""

from typing import Any

from strands.models import Model
from strands.models.openai import OpenAIModel
from typing_extensions import Unpack


def instance(client_args: dict[str, Any] | None = None, **model_config: Unpack[OpenAIModel.OpenAIConfig]) -> Model:
    """Create instance of Strands SDK OpenAI model provider."""
    return OpenAIModel(client_args=client_args, **model_config)
