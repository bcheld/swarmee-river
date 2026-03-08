"""Swarmee-owned OpenAI Responses model provider."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Protocol, TypedDict, TypeVar, cast

import openai
from pydantic import BaseModel
from strands.models import Model
from strands.models._validation import validate_config_keys
from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse
from typing_extensions import Unpack, override

from swarmee_river.utils.model_utils import ensure_openai_responses_transport_available

ensure_openai_responses_transport_available()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_MEDIA_SIZE_BYTES = 20 * 1024 * 1024
_MAX_MEDIA_SIZE_LABEL = "20MB"
_DEFAULT_MIME_TYPE = "application/octet-stream"
_CONTEXT_WINDOW_OVERFLOW_MSG = "OpenAI Responses API threw context window overflow error"
_RATE_LIMIT_MSG = "OpenAI Responses API threw rate limit error"


class _ToolCallInfo(TypedDict):
    name: str
    arguments: str
    call_id: str
    item_id: str


class Client(Protocol):
    @property
    def responses(self) -> Any:
        ...


def _encode_media_to_data_url(data: bytes, format_ext: str, media_type: str = "image") -> str:
    if len(data) > _MAX_MEDIA_SIZE_BYTES:
        raise ValueError(
            f"{media_type.capitalize()} size {len(data)} bytes exceeds maximum of "
            f"{_MAX_MEDIA_SIZE_BYTES} bytes ({_MAX_MEDIA_SIZE_LABEL})"
        )
    mime_type = mimetypes.types_map.get(f".{format_ext}", _DEFAULT_MIME_TYPE)
    encoded_data = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _has_assistant_history(messages: Messages) -> bool:
    for message in messages:
        if str(message.get("role", "")).strip().lower() == "assistant":
            return True
    return False


def _looks_like_assistant_history_format_error(detail: str) -> bool:
    lowered = detail.lower()
    return "input_text" in lowered and ("output_text" in lowered or "refusal" in lowered)


class OpenAIResponsesModel(Model):
    """Swarmee-owned OpenAI Responses model provider."""

    class OpenAIResponsesConfig(TypedDict, total=False):
        model_id: str
        params: dict[str, Any] | None

    def __init__(
        self,
        client: Client | None = None,
        client_args: dict[str, Any] | None = None,
        **model_config: Unpack[OpenAIResponsesConfig],
    ) -> None:
        validate_config_keys(model_config, self.OpenAIResponsesConfig)
        if client is not None and client_args:
            raise ValueError("Only one of 'client' or 'client_args' should be provided, not both.")
        self.config = dict(model_config)
        self._custom_client = client
        self.client_args = dict(client_args or {})

    @override
    def update_config(self, **model_config: Unpack[OpenAIResponsesConfig]) -> None:  # type: ignore[override]
        validate_config_keys(model_config, self.OpenAIResponsesConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> OpenAIResponsesConfig:
        return cast(OpenAIResponsesModel.OpenAIResponsesConfig, self.config)

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        del system_prompt_content, invocation_state, kwargs
        request = self._format_request(messages, tool_specs, system_prompt, tool_choice)
        logger.debug("formatted OpenAI Responses request=%s", request)

        async with self._client_context() as client:
            try:
                response = await client.responses.create(**request)
                yield self._format_chunk({"chunk_type": "message_start"})

                tool_calls: dict[str, _ToolCallInfo] = {}
                final_usage = None
                data_type: str | None = None
                stop_reason: str | None = None

                async for event in response:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.reasoning_text.delta":
                        chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
                        for chunk in chunks:
                            yield chunk
                        if isinstance(getattr(event, "delta", None), str):
                            yield self._format_chunk(
                                {
                                    "chunk_type": "content_delta",
                                    "data_type": "reasoning_content",
                                    "data": event.delta,
                                }
                            )
                        continue

                    if event_type == "response.output_text.delta":
                        chunks, data_type = self._stream_switch_content("text", data_type)
                        for chunk in chunks:
                            yield chunk
                        if isinstance(getattr(event, "delta", None), str):
                            yield self._format_chunk(
                                {"chunk_type": "content_delta", "data_type": "text", "data": event.delta}
                            )
                        continue

                    if event_type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if getattr(item, "type", None) == "function_call":
                            call_id = getattr(item, "call_id", "unknown")
                            tool_calls[call_id] = {
                                "name": getattr(item, "name", ""),
                                "arguments": "",
                                "call_id": call_id,
                                "item_id": getattr(item, "id", ""),
                            }
                        continue

                    if event_type == "response.function_call_arguments.delta":
                        item_id = getattr(event, "item_id", None)
                        delta = getattr(event, "delta", None)
                        if isinstance(item_id, str) and isinstance(delta, str):
                            for call_info in tool_calls.values():
                                if call_info["item_id"] == item_id:
                                    call_info["arguments"] += delta
                                    break
                        continue

                    if event_type == "response.function_call_arguments.done":
                        item_id = getattr(event, "item_id", None)
                        arguments = getattr(event, "arguments", None)
                        if isinstance(item_id, str) and isinstance(arguments, str):
                            for call_info in tool_calls.values():
                                if call_info["item_id"] == item_id:
                                    call_info["arguments"] = arguments
                                    break
                        continue

                    if event_type == "response.incomplete":
                        response_payload = getattr(event, "response", None)
                        usage = getattr(response_payload, "usage", None)
                        if usage is not None:
                            final_usage = usage
                        incomplete_details = getattr(response_payload, "incomplete_details", None)
                        if getattr(incomplete_details, "reason", None) == "max_output_tokens":
                            stop_reason = "length"
                        break

                    if event_type == "response.completed":
                        response_payload = getattr(event, "response", None)
                        usage = getattr(response_payload, "usage", None)
                        if usage is not None:
                            final_usage = usage
                        break
            except openai.BadRequestError as exc:
                raise self._normalize_bad_request(exc, messages=messages) from exc
            except openai.RateLimitError as exc:
                logger.warning(_RATE_LIMIT_MSG)
                raise ModelThrottledException(str(exc)) from exc

        if data_type:
            yield self._format_chunk({"chunk_type": "content_stop", "data_type": data_type})

        for call_info in tool_calls.values():
            tool_call = SimpleNamespace(
                function=SimpleNamespace(name=call_info["name"], arguments=call_info["arguments"]),
                id=call_info["call_id"],
            )
            yield self._format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_call})
            yield self._format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_call})
            yield self._format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

        finish_reason = "tool_calls" if tool_calls else ("length" if stop_reason == "length" else "stop")
        yield self._format_chunk({"chunk_type": "message_stop", "data": finish_reason})

        if final_usage:
            yield self._format_chunk({"chunk_type": "metadata", "data": final_usage})

    @override
    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        del kwargs
        request = self._format_request(prompt, system_prompt=system_prompt)
        parse_kwargs: dict[str, Any] = {
            "model": self.get_config()["model_id"],
            "input": request["input"],
            "text_format": output_model,
        }
        if isinstance(request.get("instructions"), str) and request["instructions"]:
            parse_kwargs["instructions"] = request["instructions"]

        async with self._client_context() as client:
            try:
                response = await client.responses.parse(**parse_kwargs)
            except openai.BadRequestError as exc:
                raise self._normalize_bad_request(exc, messages=prompt) from exc
            except openai.RateLimitError as exc:
                logger.warning(_RATE_LIMIT_MSG)
                raise ModelThrottledException(str(exc)) from exc

        if response.output_parsed:
            yield {"output": response.output_parsed}
            return
        raise ValueError("No valid parsed output found in the OpenAI Responses API response.")

    @asynccontextmanager
    async def _client_context(self) -> AsyncGenerator[Client, None]:
        if self._custom_client is not None:
            yield self._custom_client
            return
        async with openai.AsyncOpenAI(**self.client_args) as client:
            yield client

    def _normalize_bad_request(self, exc: openai.BadRequestError, *, messages: Messages) -> Exception:
        detail = str(exc).strip()
        if getattr(exc, "code", None) == "context_length_exceeded":
            logger.warning(_CONTEXT_WINDOW_OVERFLOW_MSG)
            return ContextWindowOverflowException(detail)
        if _has_assistant_history(messages) and _looks_like_assistant_history_format_error(detail):
            model_id = str(self.get_config().get("model_id") or "").strip() or "unknown"
            message = (
                f"OpenAI Responses request for model '{model_id}' was rejected while replaying assistant history. "
                "Swarmee expects assistant history to be serialized with output_text items for the Responses API. "
                f"Original error: {detail}"
            )
            logger.error(message)
            return RuntimeError(message)
        return exc

    def _format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        params = dict(self.config.get("params") or {}) if isinstance(self.config.get("params"), dict) else {}
        request: dict[str, Any] = {
            "model": self.config["model_id"],
            "input": self._format_request_messages(messages),
            "stream": True,
            **params,
        }
        if system_prompt:
            request["instructions"] = system_prompt
        if tool_specs:
            request["tools"] = [
                {
                    "type": "function",
                    "name": tool_spec["name"],
                    "description": tool_spec.get("description", ""),
                    "parameters": tool_spec["inputSchema"]["json"],
                }
                for tool_spec in tool_specs
            ]
            request.update(self._format_request_tool_choice(tool_choice))
        return request

    @classmethod
    def _format_request_tool_choice(cls, tool_choice: ToolChoice | None) -> dict[str, Any]:
        if not tool_choice:
            return {}
        match tool_choice:
            case {"auto": _}:
                return {"tool_choice": "auto"}
            case {"any": _}:
                return {"tool_choice": "required"}
            case {"tool": {"name": tool_name}}:
                return {"tool_choice": {"type": "function", "name": tool_name}}
            case _:
                return {"tool_choice": "auto"}

    @classmethod
    def _format_request_messages(cls, messages: Messages) -> list[dict[str, Any]]:
        formatted_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            contents = message["content"]

            formatted_contents = [
                cls._format_request_message_content_for_role(role, content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse"])
            ]
            formatted_tool_calls = [
                cls._format_request_message_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls._format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            if formatted_contents:
                formatted_messages.append({"role": role, "content": formatted_contents})
            formatted_messages.extend(formatted_tool_calls)
            formatted_messages.extend(formatted_tool_messages)

        return [
            message
            for message in formatted_messages
            if message.get("content") or message.get("type") in {"function_call", "function_call_output"}
        ]

    @classmethod
    def _format_request_message_content_for_role(cls, role: Any, content: ContentBlock) -> dict[str, Any]:
        normalized_role = str(role or "").strip().lower()
        if normalized_role == "assistant":
            if "image" in content or "document" in content:
                content_type = "image" if "image" in content else "document"
                raise ValueError(
                    "OpenAI Responses assistant history replay in Swarmee only supports text content and tool calls. "
                    f"Assistant {content_type} blocks are unsupported."
                )
            if "text" in content:
                return {"type": "output_text", "text": content["text"]}
        return cls._format_request_message_content(content)

    @classmethod
    def _format_request_message_content(cls, content: ContentBlock) -> dict[str, Any]:
        if "document" in content:
            doc = content["document"]
            return {
                "type": "input_file",
                "file_url": _encode_media_to_data_url(doc["source"]["bytes"], doc["format"], "document"),
            }
        if "image" in content:
            img = content["image"]
            return {
                "type": "input_image",
                "image_url": _encode_media_to_data_url(img["source"]["bytes"], img["format"], "image"),
            }
        if "text" in content:
            return {"type": "input_text", "text": content["text"]}
        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    @classmethod
    def _format_request_message_tool_call(cls, tool_use: ToolUse) -> dict[str, Any]:
        return {
            "type": "function_call",
            "call_id": tool_use["toolUseId"],
            "name": tool_use["name"],
            "arguments": json.dumps(tool_use["input"]),
        }

    @classmethod
    def _format_request_tool_message(cls, tool_result: ToolResult) -> dict[str, Any]:
        output_parts: list[dict[str, Any]] = []
        has_media = False
        for content in tool_result["content"]:
            if "json" in content:
                output_parts.append({"type": "input_text", "text": json.dumps(content["json"])})
            elif "text" in content:
                output_parts.append({"type": "input_text", "text": content["text"]})
            elif "image" in content:
                has_media = True
                img = content["image"]
                output_parts.append(
                    {
                        "type": "input_image",
                        "image_url": _encode_media_to_data_url(img["source"]["bytes"], img["format"], "image"),
                    }
                )
            elif "document" in content:
                has_media = True
                doc = content["document"]
                output_parts.append(
                    {
                        "type": "input_file",
                        "file_url": _encode_media_to_data_url(doc["source"]["bytes"], doc["format"], "document"),
                    }
                )

        output: list[dict[str, Any]] | str
        if has_media:
            output = output_parts
        else:
            output = "\n".join(part.get("text", "") for part in output_parts) if output_parts else ""
        return {"type": "function_call_output", "call_id": tool_result["toolUseId"], "output": output}

    def _stream_switch_content(self, data_type: str, prev_data_type: str | None) -> tuple[list[StreamEvent], str]:
        chunks: list[StreamEvent] = []
        if data_type != prev_data_type:
            if prev_data_type is not None:
                chunks.append(self._format_chunk({"chunk_type": "content_stop", "data_type": prev_data_type}))
            chunks.append(self._format_chunk({"chunk_type": "content_start", "data_type": data_type}))
        return chunks, data_type

    def _format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}
            case "content_start":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": event["data"].function.name,
                                    "toolUseId": event["data"].id,
                                }
                            }
                        }
                    }
                return {"contentBlockStart": {"start": {}}}
            case "content_delta":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments or ""}}}
                    }
                if event["data_type"] == "reasoning_content":
                    return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}}
                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}
            case "content_stop":
                return {"contentBlockStop": {}}
            case "message_stop":
                match event["data"]:
                    case "tool_calls":
                        return {"messageStop": {"stopReason": "tool_use"}}
                    case "length":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case _:
                        return {"messageStop": {"stopReason": "end_turn"}}
            case "metadata":
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": getattr(event["data"], "input_tokens", 0),
                            "outputTokens": getattr(event["data"], "output_tokens", 0),
                            "totalTokens": getattr(event["data"], "total_tokens", 0),
                        },
                        "metrics": {"latencyMs": 0},
                    }
                }
            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")


def instance(
    client: Client | None = None,
    client_args: dict[str, Any] | None = None,
    **model_config: Any,
) -> Model:
    """Create instance of Swarmee's OpenAI Responses provider."""
    model_config.pop("transport", None)
    return OpenAIResponsesModel(client=client, client_args=client_args, **model_config)
