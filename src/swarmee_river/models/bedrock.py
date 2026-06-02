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
_SWARMEE_CACHE_POLICY_KEY = "swarmee_cache_policy"


class BedrockConverseStreamValidationError(ValueError):
    """Raised when Swarmee builds an invalid Bedrock ConverseStream request."""


def _cache_point(*, cache_type: str, ttl: str | None) -> dict[str, Any]:
    point = {"type": cache_type}
    if ttl:
        point["ttl"] = ttl
    return {"cachePoint": point}


def _iter_cache_points(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        out: list[dict[str, Any]] = []
        if isinstance(value.get("cachePoint"), dict):
            out.append(value["cachePoint"])
        for nested in value.values():
            out.extend(_iter_cache_points(nested))
        return out
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for item in value:
            out.extend(_iter_cache_points(item))
        return out
    return []


def _strip_message_cache_points(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        message["content"] = [block for block in content if not (isinstance(block, dict) and "cachePoint" in block)]


def _message_text_chars(message: dict[str, Any]) -> int:
    total = 0
    content = message.get("content")
    if not isinstance(content, list):
        return total
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            total += len(text)
    return total


def _select_assistant_cache_indices(messages: list[dict[str, Any]], slots: int) -> list[int]:
    if slots <= 0:
        return []
    candidates = [
        idx
        for idx, message in enumerate(messages)
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
        and bool(message.get("content"))
    ]
    if len(candidates) <= slots:
        return candidates
    if slots == 1:
        return [candidates[-1]]

    # Spread checkpoints across assistant history so Bedrock can reuse both older
    # static context and the latest stable assistant boundary.
    selected: list[int] = []
    denominator = max(1, slots - 1)
    last_pos = len(candidates) - 1
    for slot in range(slots):
        pos = round((slot / denominator) * last_pos)
        idx = candidates[pos]
        if idx not in selected:
            selected.append(idx)
    while len(selected) < slots:
        for idx in reversed(candidates):
            if idx not in selected:
                selected.append(idx)
                break
        else:
            break
    return sorted(selected[:slots])


def _normalize_cache_policy(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"enabled": False}
    enabled = bool(raw.get("enabled"))
    cache_type = str(raw.get("cache_type") or "default").strip() or "default"
    ttl = str(raw.get("ttl") or "").strip() or None
    max_checkpoints_raw = raw.get("max_checkpoints")
    try:
        max_checkpoints = int(max_checkpoints_raw)
    except Exception:
        max_checkpoints = 4
    return {
        "enabled": enabled,
        "cache_type": cache_type,
        "ttl": ttl,
        "max_checkpoints": max(0, max_checkpoints),
        "min_tokens": raw.get("min_tokens"),
    }


def _apply_swarmee_cache_policy(request: dict[str, Any], *, policy: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "bedrock_cache_policy_enabled": bool(policy.get("enabled")),
        "bedrock_cache_checkpoint_count": 0,
        "bedrock_cache_checkpoint_locations": [],
        "bedrock_cache_ttl": policy.get("ttl"),
        "bedrock_cache_min_tokens": policy.get("min_tokens"),
    }
    if not policy.get("enabled"):
        return diagnostics

    max_checkpoints = max(0, int(policy.get("max_checkpoints") or 0))
    cache_type = str(policy.get("cache_type") or "default").strip() or "default"
    ttl = str(policy.get("ttl") or "").strip() or None
    locations: list[str] = []

    existing_tool_points = _iter_cache_points(request.get("toolConfig"))
    if ttl:
        for point in existing_tool_points:
            point["ttl"] = ttl
    if existing_tool_points:
        locations.extend(["tools"] * len(existing_tool_points))
    remaining = max(0, max_checkpoints - len(existing_tool_points))

    system_blocks = request.get("system")
    if remaining > 0 and isinstance(system_blocks, list) and system_blocks:
        system_blocks[:] = [block for block in system_blocks if not (isinstance(block, dict) and "cachePoint" in block)]
        if any(isinstance(block, dict) and "text" in block for block in system_blocks):
            system_blocks.append(_cache_point(cache_type=cache_type, ttl=ttl))
            locations.append("system")
            remaining -= 1

    messages = request.get("messages")
    if remaining > 0 and isinstance(messages, list):
        _strip_message_cache_points(messages)
        for idx in _select_assistant_cache_indices(messages, remaining):
            content = messages[idx].get("content")
            if not isinstance(content, list):
                continue
            content.append(_cache_point(cache_type=cache_type, ttl=ttl))
            locations.append(f"messages[{idx}]")

    diagnostics["bedrock_cache_checkpoint_count"] = len(locations)
    diagnostics["bedrock_cache_checkpoint_locations"] = locations
    diagnostics["bedrock_cache_message_text_chars"] = (
        sum(_message_text_chars(message) for message in messages) if isinstance(messages, list) else 0
    )
    return diagnostics


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
            f"Bedrock ConverseStream request validation failed before sending the request. {exc}"
        ) from exc


def _validating_bedrock_model_class(base_cls: type[BedrockModel]) -> type[BedrockModel]:
    class _SwarmeeBedrockModel(base_cls):  # type: ignore[misc,valid-type]
        def _format_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            request = super()._format_request(*args, **kwargs)
            policy = _normalize_cache_policy(getattr(self, "_swarmee_cache_policy", None))
            diagnostics = _apply_swarmee_cache_policy(request, policy=policy)
            self._swarmee_last_cache_diagnostics = diagnostics
            _validate_converse_stream_request(request, client=self.client)
            return request

    return _SwarmeeBedrockModel


def instance(**model_config: Unpack[BedrockModel.BedrockConfig]) -> Model:
    """Create instance of SDK's Bedrock model provider.

    Args:
        **model_config: Configuration options for the Bedrock model.

    Returns:
        Bedrock model provider.
    """
    # Handle conversion of boto_client_config from dict to BotocoreConfig
    config_dict: dict[str, Any] = dict(model_config)
    cache_policy = config_dict.pop(_SWARMEE_CACHE_POLICY_KEY, None)
    boto_client_config = config_dict.get("boto_client_config")
    if isinstance(boto_client_config, dict):
        config_dict["boto_client_config"] = BotocoreConfig(**boto_client_config)

    _warn_if_model_region_looks_mismatched(config_dict)

    model_cls = _validating_bedrock_model_class(BedrockModel)
    model = model_cls(**config_dict)
    model._swarmee_cache_policy = _normalize_cache_policy(cache_policy)
    model._swarmee_last_cache_diagnostics = {}
    return model
