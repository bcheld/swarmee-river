"""Create instance of Strands SDK Bedrock model provider."""

from typing import Any

from botocore.config import Config as BotocoreConfig
from strands.models import BedrockModel, Model
from typing_extensions import Unpack


def instance(**model_config: Unpack[BedrockModel.BedrockConfig]) -> Model:
    """Create instance of SDK's Bedrock model provider.

    Args:
        **model_config: Configuration options for the Bedrock model.

    Returns:
        Bedrock model provider.
    """
    # Handle conversion of boto_client_config from dict to BotocoreConfig
    config_dict: dict[str, Any] = dict(model_config)
    boto_client_config = config_dict.get("boto_client_config")
    if isinstance(boto_client_config, dict):
        config_dict["boto_client_config"] = BotocoreConfig(**boto_client_config)

    return BedrockModel(**config_dict)
