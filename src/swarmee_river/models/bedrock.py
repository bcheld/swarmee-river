"""Create instance of Strands SDK Bedrock model provider."""

import logging
from typing import Any

from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ParamValidationError
from botocore.validate import validate_parameters
from strands.models import BedrockModel, Model
from typing_extensions import Unpack

from swarmee_river.utils.provider_utils import resolve_aws_region_source

_LOGGER = logging.getLogger(__name__)
_BEDROCK_MODEL_PREFIX_REGION = {
    "us": "us-",
    "eu": "eu-",
    "jp": "ap-northeast-",
    "apac": "ap-",
    "au": "ap-southeast-",
}
_MISSING_REGION_WARNING_KEYS: set[str] = set()


class BedrockConverseStreamValidationError(ValueError):
    """Raised when Swarmee builds an invalid Bedrock ConverseStream request."""


def _resolve_region(config: dict[str, Any]) -> str:
    explicit = str(config.get("region_name") or "").strip()
    if explicit:
        return explicit
    env_or_inferred, _source = resolve_aws_region_source()
    if isinstance(env_or_inferred, str) and env_or_inferred.strip():
        return env_or_inferred.strip()
    return ""


def _warn_if_model_region_looks_mismatched(config: dict[str, Any]) -> None:
    model_id = str(config.get("model_id") or "").strip()
    if not model_id:
        return

    prefix = model_id.split(".", 1)[0].strip().lower()
    if prefix == "global":
        return

    region = _resolve_region(config).lower()
    if prefix in _BEDROCK_MODEL_PREFIX_REGION:
        if not region:
            warning_key = f"missing_region:{model_id}"
            if warning_key not in _MISSING_REGION_WARNING_KEYS:
                _MISSING_REGION_WARNING_KEYS.add(warning_key)
                _LOGGER.warning(
                    "Bedrock model_id '%s' is prefixed but AWS region is not set; set AWS_REGION/AWS_DEFAULT_REGION, "
                    "configure an AWS profile region, or set region_name explicitly.",
                    model_id,
                )
            return
        expected = _BEDROCK_MODEL_PREFIX_REGION[prefix]
        if not region.startswith(expected):
            _LOGGER.warning(
                "Bedrock model_id '%s' prefix '%s' may not match resolved region '%s'.",
                model_id,
                prefix,
                region,
            )
        return

    if ":" in model_id and model_id.startswith("arn:"):
        return

    _LOGGER.warning(
        "Bedrock model_id '%s' is unprefixed; prefixed IDs (for example 'us.<model-id>') are recommended.",
        model_id,
    )


def _validate_converse_stream_request(request: dict[str, Any], *, client: Any) -> None:
    service_model = getattr(getattr(client, "meta", None), "service_model", None)
    if service_model is None:
        return
    operation_model = service_model.operation_model("ConverseStream")
    input_shape = getattr(operation_model, "input_shape", None)
    if input_shape is None:
        return
    try:
        validate_parameters(request, input_shape)
    except ParamValidationError as exc:
        raise BedrockConverseStreamValidationError(
            "Bedrock ConverseStream request validation failed before sending the request. "
            f"{exc}"
        ) from exc


def _validating_bedrock_model_class(base_cls: type[BedrockModel]) -> type[BedrockModel]:
    class _ValidatingBedrockModel(base_cls):  # type: ignore[misc,valid-type]
        def _format_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            request = super()._format_request(*args, **kwargs)
            _validate_converse_stream_request(request, client=self.client)
            return request

    return _ValidatingBedrockModel


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

    _warn_if_model_region_looks_mismatched(config_dict)

    model_cls = _validating_bedrock_model_class(BedrockModel)
    return model_cls(**config_dict)
