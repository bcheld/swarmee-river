from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import threading
import warnings
from typing import Any

from swarmee_river.interrupts import AgentInterruptedError

_STRANDS_KWARGS_DEPRECATION = r"`\*\*kwargs` parameter is deprecating, use `invocation_state` instead\."


def invoke_agent(
    agent: Any,
    query: str,
    *,
    callback_handler: Any,
    interrupt_event: threading.Event,
    invocation_state: dict[str, Any],
    structured_output_model: type[Any] | None = None,
    structured_output_prompt: str | None = None,
) -> Any:
    async def _invoke() -> Any:
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()

        def _exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            if interrupt_event.is_set():
                message = str(context.get("message") or "")
                exc = context.get("exception")
                if message.startswith("an error occurred during closing of asynchronous generator"):
                    return
                if isinstance(exc, RuntimeError):
                    if "athrow(): asynchronous generator is already running" in str(exc):
                        return
            if previous_handler:
                previous_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_exception_handler)

        class _OtelDetachFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
                message = str(record.getMessage())
                if interrupt_event.is_set() and message.startswith("Failed to detach context"):
                    return False
                return True

        otel_logger = logging.getLogger("opentelemetry.context")
        otel_filter = _OtelDetachFilter()
        otel_logger.addFilter(otel_filter)

        current_task = asyncio.current_task()

        async def _canceller() -> None:
            while not interrupt_event.is_set():
                await asyncio.sleep(0.05)
            callback_handler(force_stop=True)
            if current_task:
                current_task.cancel()

        canceller_task = asyncio.create_task(_canceller())
        try:
            invoke_kwargs: dict[str, Any] = {"invocation_state": invocation_state}
            invoke_query = query
            if structured_output_model is not None:
                invoke_kwargs["structured_output_model"] = structured_output_model

            supports_structured_output_prompt = True
            try:
                invoke_sig = inspect.signature(agent.invoke_async)
                params = invoke_sig.parameters
                supports_structured_output_prompt = "structured_output_prompt" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                supports_structured_output_prompt = True

            if structured_output_prompt is not None and supports_structured_output_prompt:
                invoke_kwargs["structured_output_prompt"] = structured_output_prompt
            elif structured_output_prompt is not None:
                prompt_text = structured_output_prompt.strip()
                if prompt_text:
                    invoke_query = f"{prompt_text}\n\nUser request:\n{query}"

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_STRANDS_KWARGS_DEPRECATION,
                    category=UserWarning,
                )
                return await agent.invoke_async(invoke_query, **invoke_kwargs)
        except asyncio.CancelledError as e:
            if interrupt_event.is_set():
                raise AgentInterruptedError("Interrupted by user (Esc)") from e
            raise
        finally:
            canceller_task.cancel()
            with contextlib.suppress(BaseException):
                await canceller_task
            otel_logger.removeFilter(otel_filter)
            loop.set_exception_handler(previous_handler)

    try:
        return asyncio.run(_invoke())
    except KeyboardInterrupt:
        raise AgentInterruptedError("Interrupted by user") from None
