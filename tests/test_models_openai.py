from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel
from strands.types.exceptions import ContextWindowOverflowException

from swarmee_river.models import openai as openai_model


class _FakeAsyncEvents:
    def __init__(self, events: list[object]) -> None:
        self._events = list(events)

    def __aiter__(self) -> _FakeAsyncEvents:
        return self

    async def __anext__(self) -> object:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class _FakeResponsesAPI:
    def __init__(self, *, stream_events: list[object] | None = None, create_error: Exception | None = None) -> None:
        self._stream_events = list(stream_events or [])
        self._create_error = create_error
        self.last_create_kwargs: dict[str, object] | None = None
        self.last_parse_kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> _FakeAsyncEvents:
        self.last_create_kwargs = dict(kwargs)
        if self._create_error is not None:
            raise self._create_error
        return _FakeAsyncEvents(self._stream_events)

    async def parse(self, **kwargs: object) -> object:
        self.last_parse_kwargs = dict(kwargs)
        return SimpleNamespace(output_parsed=SimpleNamespace(answer="ok"))


class _FakeClient:
    def __init__(self, responses: _FakeResponsesAPI) -> None:
        self.responses = responses


class _StructuredAnswer(BaseModel):
    answer: str


def _model(
    *,
    model_id: str = "gpt-5.2",
    client: _FakeClient | None = None,
) -> openai_model.OpenAIResponsesModel:
    return openai_model.instance(
        model_id=model_id,
        params={"max_output_tokens": 64},
        client=client,
    )


def test_openai_instance_uses_local_responses_wrapper() -> None:
    model = _model()

    assert isinstance(model, openai_model.OpenAIResponsesModel)


def test_openai_request_formats_assistant_text_as_output_text() -> None:
    model = _model()

    request = model._format_request(
        [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"text": "world"}]},
            {"role": "user", "content": [{"text": "repeat that"}]},
        ]
    )

    assert request["input"][0]["content"] == [{"type": "input_text", "text": "hello"}]
    assert request["input"][1]["role"] == "assistant"
    assert request["input"][1]["content"] == [{"type": "output_text", "text": "world"}]
    assert request["input"][2]["content"] == [{"type": "input_text", "text": "repeat that"}]
    assert not any(
        part.get("type") == "input_text"
        for item in request["input"]
        if item.get("role") == "assistant"
        for part in item.get("content", [])
    )


def test_openai_request_formats_tool_history_for_responses() -> None:
    model = _model()

    request = model._format_request(
        [
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "tool-1", "name": "shell", "input": {"command": "pwd"}}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tool-1",
                            "status": "success",
                            "content": [{"text": "/tmp"}],
                        }
                    }
                ],
            },
        ]
    )

    assert request["input"] == [
        {
            "type": "function_call",
            "call_id": "tool-1",
            "name": "shell",
            "arguments": '{"command": "pwd"}',
        },
        {
            "type": "function_call_output",
            "call_id": "tool-1",
            "output": "/tmp",
        },
    ]


@pytest.mark.parametrize(
    ("content", "label"),
    [
        ({"image": {"source": {"bytes": b"\x89PNG"}, "format": "png"}}, "image"),
        ({"document": {"source": {"bytes": b"%PDF"}, "format": "pdf"}}, "document"),
    ],
)
def test_openai_request_rejects_unsupported_assistant_media_history(content: dict, label: str) -> None:
    model = _model()

    with pytest.raises(ValueError, match=rf"Assistant {label} blocks are unsupported"):
        model._format_request([{"role": "assistant", "content": [content]}])


@pytest.mark.asyncio
async def test_openai_stream_wraps_assistant_history_bad_request() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(400, request=request)
    exc = openai.BadRequestError(
        (
            "Error code: 400 - {'error': {'message': "
            "\"Invalid value: 'input_text'. Supported values are: "
            "'output_text' and 'refusal'.\", 'param': "
            "'input[1].content[0]', 'code': 'invalid_value'}}"
        ),
        response=response,
        body={"type": "invalid_request_error", "param": "input[1].content[0]", "code": "invalid_value"},
    )
    model = _model(client=_FakeClient(_FakeResponsesAPI(create_error=exc)))

    with pytest.raises(RuntimeError, match="replaying assistant history"):
        async for _ in model.stream(
            [
                {"role": "user", "content": [{"text": "hello"}]},
                {"role": "assistant", "content": [{"text": "world"}]},
                {"role": "user", "content": [{"text": "repeat that"}]},
            ]
        ):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["gpt-5-nano", "gpt-5-mini", "gpt-5.2"])
async def test_openai_stream_normalizes_incremental_text_events(model_id: str) -> None:
    responses = _FakeResponsesAPI(
        stream_events=[
            SimpleNamespace(type="response.output_text.delta", delta="hello "),
            SimpleNamespace(type="response.output_text.delta", delta="world"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12)),
            ),
        ]
    )
    model = _model(model_id=model_id, client=_FakeClient(responses))

    events = [event async for event in model.stream([{"role": "user", "content": [{"text": "hi"}]}])]

    assert events == [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "hello "}}},
        {"contentBlockDelta": {"delta": {"text": "world"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}, "metrics": {"latencyMs": 0}}},
    ]
    assert responses.last_create_kwargs is not None
    assert responses.last_create_kwargs["model"] == model_id


@pytest.mark.asyncio
async def test_openai_stream_raises_context_window_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(400, request=request)
    exc = openai.BadRequestError(
        "context too large",
        response=response,
        body={"type": "invalid_request_error", "code": "context_length_exceeded"},
    )
    model = _model(client=_FakeClient(_FakeResponsesAPI(create_error=exc)))

    with pytest.raises(ContextWindowOverflowException, match="context too large"):
        async for _ in model.stream([{"role": "user", "content": [{"text": "hi"}]}]):
            pass


def test_openai_structured_output_passes_instructions_to_parse() -> None:
    responses = _FakeResponsesAPI()
    model = _model(client=_FakeClient(responses))

    async def _collect() -> list[dict[str, object]]:
        return [
            item
            async for item in model.structured_output(
                _StructuredAnswer,
                [{"role": "user", "content": [{"text": "hi"}]}],
                system_prompt="Return json",
            )
        ]

    output = asyncio.run(_collect())

    assert output and output[0]["output"].answer == "ok"
    assert responses.last_parse_kwargs is not None
    assert responses.last_parse_kwargs["instructions"] == "Return json"
