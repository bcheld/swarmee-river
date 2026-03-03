from __future__ import annotations

import asyncio
import contextlib
import contextvars
import faulthandler
import inspect
import logging
import os
import sys
import threading
import time
import warnings
from collections.abc import Callable
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from swarmee_river.interrupts import AgentInterruptedError

_STRANDS_KWARGS_DEPRECATION = r"`\*\*kwargs` parameter is deprecating, use `invocation_state` instead\."

_OTEL_DETACH_FILTER_INSTALLED = False
_LOGGER = logging.getLogger(__name__)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def _resolve_windows_event_loop_policy() -> str:
    raw = str(os.getenv("SWARMEE_WINDOWS_EVENT_LOOP_POLICY", "auto")).strip().lower()
    if raw in {"auto", "selector", "proactor"}:
        return raw
    return "auto"


def _resolve_agent_invoke_mode() -> str:
    raw = str(os.getenv("SWARMEE_AGENT_INVOKE_MODE", "isolated")).strip().lower()
    if raw in {"isolated", "direct"}:
        return raw
    return "isolated"


def _run_coroutine_isolated(
    async_factory: Callable[[], Awaitable[Any]],
    *,
    invocation_state: dict[str, Any] | None = None,
) -> Any:
    def _execute() -> Any:
        _mark_invoke_stage(invocation_state, "invoke_worker_asyncio_run_start")
        return asyncio.run(async_factory())

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="swarmee-agent-invoke") as executor:
        context = contextvars.copy_context()
        future = executor.submit(context.run, _execute)
        return future.result()


def _ensure_invoke_diag(invocation_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(invocation_state, dict):
        return {}
    sw = invocation_state.get("swarmee")
    if not isinstance(sw, dict):
        sw = {}
        invocation_state["swarmee"] = sw
    diag = sw.get("invoke_diag")
    if not isinstance(diag, dict):
        diag = {}
        sw["invoke_diag"] = diag
    now = time.monotonic()
    diag.setdefault("invoke_start_mono", now)
    diag.setdefault("last_callback_mono", now)
    return diag


def _mark_invoke_stage(invocation_state: dict[str, Any] | None, stage: str) -> None:
    diag = _ensure_invoke_diag(invocation_state)
    if not diag:
        return
    now = time.monotonic()
    diag["stage"] = str(stage).strip() or "unknown"
    diag["stage_mono"] = now


def _is_bedrock_invocation(invocation_state: dict[str, Any] | None) -> bool:
    if not isinstance(invocation_state, dict):
        return False
    sw = invocation_state.get("swarmee")
    if not isinstance(sw, dict):
        return False
    return str(sw.get("provider", "")).strip().lower() == "bedrock"


@contextlib.contextmanager
def _windows_loop_policy_context() -> Any:
    if os.name != "nt":
        yield
        return

    desired = _resolve_windows_event_loop_policy()
    if desired == "auto":
        yield
        return

    previous_policy = asyncio.get_event_loop_policy()
    applied = False
    try:
        if desired == "selector":
            policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
            if policy_cls is not None:
                asyncio.set_event_loop_policy(policy_cls())
                applied = True
        elif desired == "proactor":
            policy_cls = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
            if policy_cls is not None:
                asyncio.set_event_loop_policy(policy_cls())
                applied = True
        if applied:
            _LOGGER.info("Applied Windows asyncio policy override: %s", desired)
        yield
    finally:
        if applied:
            with contextlib.suppress(Exception):
                asyncio.set_event_loop_policy(previous_policy)


def _start_stall_monitor(
    *,
    callback_handler: Callable[..., Any],
    invocation_state: dict[str, Any],
) -> tuple[threading.Event | None, threading.Thread | None]:
    warn_sec = _env_float("SWARMEE_BEDROCK_STALL_WARN_SEC", 90.0)
    if warn_sec is None or warn_sec <= 0:
        return None, None
    if not _is_bedrock_invocation(invocation_state):
        return None, None

    dump_enabled = _truthy_env("SWARMEE_BEDROCK_STALL_DIAG_DUMP", False)
    stop_event = threading.Event()
    check_interval = max(0.05, min(2.0, float(warn_sec) / 4.0))

    def _emit_warning(text: str, *, elapsed_s: float, stage: str) -> None:
        metadata = {
            "bedrock_stall_warn_sec": float(warn_sec),
            "bedrock_stall_elapsed_sec": round(elapsed_s, 2),
            "invoke_stage": stage,
        }
        with contextlib.suppress(Exception):
            callback_handler(warning_text=text, warning_metadata=metadata)

    def _run() -> None:
        last_warn_mono = -float(warn_sec)
        dumped = False
        while not stop_event.wait(check_interval):
            diag = _ensure_invoke_diag(invocation_state)
            now = time.monotonic()
            raw_last = diag.get("last_callback_mono", diag.get("invoke_start_mono", now))
            try:
                last_callback_mono = float(raw_last)
            except (TypeError, ValueError):
                last_callback_mono = now
            elapsed = now - last_callback_mono
            if elapsed < float(warn_sec):
                continue
            if (now - last_warn_mono) < float(warn_sec):
                continue
            stage = str(diag.get("stage", "unknown")).strip() or "unknown"
            warning_text = (
                f"Bedrock stream appears stalled: no callback events for {elapsed:.1f}s "
                f"(stage={stage})."
            )
            _LOGGER.warning(warning_text)
            _emit_warning(warning_text, elapsed_s=elapsed, stage=stage)
            if dump_enabled and not dumped:
                dumped = True
                with contextlib.suppress(Exception):
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            last_warn_mono = now

    thread = threading.Thread(
        target=_run,
        name="swarmee-bedrock-stall-monitor",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def invoke_agent(
    agent: Any,
    query: str,
    *,
    callback_handler: Any,
    interrupt_event: threading.Event,
    invocation_state: dict[str, Any],
    system_reminder: str | None = None,
    structured_output_model: type[Any] | None = None,
    structured_output_prompt: str | None = None,
) -> Any:
    _mark_invoke_stage(invocation_state, "pre_invoke_dispatch")
    monitor_stop, monitor_thread = _start_stall_monitor(
        callback_handler=callback_handler,
        invocation_state=invocation_state,
    )

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
                if isinstance(exc, ValueError) and "was created in a different Context" in str(exc):
                    return
            if previous_handler:
                previous_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_exception_handler)

        class _OtelDetachFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
                message = str(record.getMessage())
                # Opentelemetry can log noisy detach errors during cancellation.
                # Filter them unconditionally; these are not actionable for users.
                if message.startswith("Failed to detach context"):
                    return False
                return True

        global _OTEL_DETACH_FILTER_INSTALLED
        otel_filter = _OtelDetachFilter()
        if not _OTEL_DETACH_FILTER_INSTALLED:
            logging.getLogger("opentelemetry.context").addFilter(otel_filter)
            logging.getLogger("opentelemetry").addFilter(otel_filter)
            _OTEL_DETACH_FILTER_INSTALLED = True

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
            base_query = query
            reminder_prefix = (system_reminder or "").strip()
            invoke_query = f"{reminder_prefix}\n\n{base_query}".strip() if reminder_prefix else base_query
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
                    prefix_parts: list[str] = []
                    if reminder_prefix:
                        prefix_parts.append(reminder_prefix)
                    prefix_parts.append(prompt_text)
                    prefix = "\n\n".join(prefix_parts).strip()
                    invoke_query = f"{prefix}\n\nUser request:\n{base_query}".strip()

            _mark_invoke_stage(invocation_state, "invoke_async_enter")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_STRANDS_KWARGS_DEPRECATION,
                    category=UserWarning,
                )
                response = await agent.invoke_async(invoke_query, **invoke_kwargs)
                _mark_invoke_stage(invocation_state, "invoke_async_complete")
                return response
        except asyncio.CancelledError as e:
            _mark_invoke_stage(invocation_state, "invoke_async_cancelled")
            if interrupt_event.is_set():
                raise AgentInterruptedError("Interrupted by user (Esc)") from e
            raise
        except Exception:
            _mark_invoke_stage(invocation_state, "invoke_async_error")
            raise
        finally:
            canceller_task.cancel()
            with contextlib.suppress(BaseException):
                await canceller_task
            loop.set_exception_handler(previous_handler)

    try:
        with _windows_loop_policy_context():
            invoke_mode = _resolve_agent_invoke_mode()
            if invoke_mode == "direct":
                _mark_invoke_stage(invocation_state, "invoke_mode_direct_start")
                return asyncio.run(_invoke())
            _mark_invoke_stage(invocation_state, "invoke_mode_isolated_submit")
            return _run_coroutine_isolated(lambda: _invoke(), invocation_state=invocation_state)
    except KeyboardInterrupt:
        _mark_invoke_stage(invocation_state, "keyboard_interrupt")
        raise AgentInterruptedError("Interrupted by user") from None
    finally:
        if monitor_stop is not None:
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=0.5)
