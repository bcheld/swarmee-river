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
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from swarmee_river.interrupts import AgentInterruptedError

_STRANDS_KWARGS_DEPRECATION = r"`\*\*kwargs` parameter is deprecating, use `invocation_state` instead\."

_OTEL_DETACH_FILTER_INSTALLED = False
_LOGGER = logging.getLogger(__name__)

# Internal diagnostic guardrail (not end-user configurable).
_BEDROCK_STALL_WARN_SEC: float = 90.0
_BEDROCK_PRE_PROGRESS_WARN_SEC: float = 180.0


def _resolve_windows_event_loop_policy() -> str:
    # Environment-variable overrides are intentionally not supported; this is
    # internal behavior and should be deterministic.
    return "auto"


def _resolve_agent_invoke_mode(invocation_state: dict[str, Any] | None = None) -> str:
    if isinstance(invocation_state, dict):
        sw = invocation_state.get("swarmee")
        if isinstance(sw, dict):
            override = str(sw.get("invoke_mode", "")).strip().lower()
            if override in {"sync", "isolated", "direct"}:
                return override
    if _is_bedrock_invocation(invocation_state):
        return "sync"
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


def _build_invoke_request(
    *,
    agent: Any,
    query: str,
    invocation_state: dict[str, Any],
    system_reminder: str | None = None,
    structured_output_model: type[Any] | None = None,
    structured_output_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
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

    return invoke_query, invoke_kwargs


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
    # Fixed diagnostic guardrail (not end-user configurable).
    post_progress_warn_sec = float(_BEDROCK_STALL_WARN_SEC)
    pre_progress_warn_sec = max(float(_BEDROCK_PRE_PROGRESS_WARN_SEC), post_progress_warn_sec)
    if not _is_bedrock_invocation(invocation_state):
        return None, None
    sw = invocation_state.get("swarmee") if isinstance(invocation_state, dict) else None
    suppress_user_warnings = bool(sw.get("suppress_user_stall_warnings")) if isinstance(sw, dict) else False

    dump_enabled = False
    stop_event = threading.Event()
    check_interval = max(0.05, min(2.0, float(post_progress_warn_sec) / 4.0))

    def _emit_warning(text: str, *, elapsed_s: float, stage: str, phase: str, threshold_s: float) -> None:
        metadata = {
            "bedrock_stall_warn_sec": float(threshold_s),
            "bedrock_stall_elapsed_sec": round(elapsed_s, 2),
            "invoke_stage": stage,
            "invoke_phase": phase,
        }
        if suppress_user_warnings:
            return
        with contextlib.suppress(Exception):
            callback_handler(warning_text=text, warning_metadata=metadata)

    def _run() -> None:
        last_warn_mono = -float(pre_progress_warn_sec)
        dumped = False
        while not stop_event.wait(check_interval):
            diag = _ensure_invoke_diag(invocation_state)
            now = time.monotonic()
            stage = str(diag.get("stage", "unknown")).strip() or "unknown"
            if "first_progress_mono" in diag:
                raw_last = diag.get("last_progress_mono", diag.get("first_progress_mono", now))
            else:
                raw_last = diag.get("last_callback_mono", diag.get("invoke_start_mono", now))
            try:
                last_progress_mono = float(raw_last)
            except (TypeError, ValueError):
                last_progress_mono = now
            elapsed = now - last_progress_mono
            if "first_progress_mono" in diag:
                phase = "post_progress"
                threshold_s = post_progress_warn_sec
                warning_text = (
                    f"Bedrock stream appears stalled: no model/tool progress for {elapsed:.1f}s "
                    f"(stage={stage})."
                )
            else:
                phase = "pre_progress"
                threshold_s = pre_progress_warn_sec
                warning_text = (
                    f"Bedrock invocation is still thinking with no streamed progress for {elapsed:.1f}s "
                    f"(stage={stage}). This can be normal for long reasoning turns."
                )
            if elapsed < float(threshold_s):
                continue
            if (now - last_warn_mono) < float(threshold_s):
                continue
            _LOGGER.warning(warning_text)
            _emit_warning(
                warning_text,
                elapsed_s=elapsed,
                stage=stage,
                phase=phase,
                threshold_s=threshold_s,
            )
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
        cancel_wait_event = threading.Event()

        async def _canceller() -> None:
            def _wait_for_interrupt_or_cancel() -> bool:
                while not cancel_wait_event.is_set():
                    if interrupt_event.wait(0.1):
                        return True
                return False

            interrupted = await asyncio.to_thread(_wait_for_interrupt_or_cancel)
            if not interrupted:
                return
            callback_handler(force_stop=True)
            if current_task:
                current_task.cancel()

        canceller_task = asyncio.create_task(_canceller())
        try:
            invoke_query, invoke_kwargs = _build_invoke_request(
                agent=agent,
                query=query,
                invocation_state=invocation_state,
                system_reminder=system_reminder,
                structured_output_model=structured_output_model,
                structured_output_prompt=structured_output_prompt,
            )
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
            cancel_wait_event.set()
            canceller_task.cancel()
            with contextlib.suppress(BaseException):
                await canceller_task
            loop.set_exception_handler(previous_handler)

    try:
        with _windows_loop_policy_context():
            invoke_mode = _resolve_agent_invoke_mode(invocation_state)
            if invoke_mode == "sync":
                if interrupt_event.is_set():
                    callback_handler(force_stop=True)
                    raise AgentInterruptedError("Interrupted by user (Esc)")
                _mark_invoke_stage(invocation_state, "invoke_mode_sync_start")
                invoke_query, invoke_kwargs = _build_invoke_request(
                    agent=agent,
                    query=query,
                    invocation_state=invocation_state,
                    system_reminder=system_reminder,
                    structured_output_model=structured_output_model,
                    structured_output_prompt=structured_output_prompt,
                )
                stop_waiter = threading.Event()
                interrupted = threading.Event()
                force_stop_emitted = threading.Event()

                def _emit_force_stop_once() -> None:
                    if force_stop_emitted.is_set():
                        return
                    force_stop_emitted.set()
                    with contextlib.suppress(Exception):
                        callback_handler(force_stop=True)

                def _watch_interrupt() -> None:
                    while not stop_waiter.is_set():
                        if interrupt_event.wait(0.05):
                            interrupted.set()
                            _emit_force_stop_once()
                            return

                waiter = threading.Thread(target=_watch_interrupt, daemon=True, name="swarmee-sync-interrupt")
                waiter.start()
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=_STRANDS_KWARGS_DEPRECATION,
                            category=UserWarning,
                        )
                        response = agent(invoke_query, **invoke_kwargs)
                except AgentInterruptedError:
                    raise
                except Exception as exc:
                    if interrupted.is_set():
                        raise AgentInterruptedError("Interrupted by user (Esc)") from exc
                    raise
                finally:
                    stop_waiter.set()
                    waiter.join(timeout=0.2)
                if interrupted.is_set():
                    raise AgentInterruptedError("Interrupted by user (Esc)")
                return response
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
