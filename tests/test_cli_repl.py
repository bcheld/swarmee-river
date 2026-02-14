from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

from swarmee_river.cli.builtin_commands import register_builtin_commands
from swarmee_river.cli.commands import CLIContext, CommandRegistry
from swarmee_river.cli.repl import _prompt_for_context, run_repl
from swarmee_river.handlers.callback_handler import callback_handler_instance


def _make_ctx(outputs: list[str], *, generate_plan=None) -> CLIContext:
    plan_fn = generate_plan or (lambda _req: SimpleNamespace(confirmation_prompt="approve?"))
    return CLIContext(
        agent=mock.MagicMock(),
        agent_kwargs={},
        tools_dict={},
        model_manager=mock.MagicMock(),
        knowledge_base_id=None,
        effective_sop_paths=None,
        welcome_text="",
        auto_approve=False,
        selected_provider=None,
        settings=mock.MagicMock(),
        settings_path=mock.MagicMock(),
        refresh_system_prompt=lambda: None,
        generate_plan=plan_fn,
        execute_with_plan=lambda _req, _plan: None,
        render_plan=lambda _plan: "rendered plan",
        run_agent=lambda *_args, **_kwargs: None,
        store_conversation=lambda _req, _resp: None,
        output=lambda text: outputs.append(text),
        stop_spinners=lambda: None,
        build_session_meta=lambda: {},
        swap_agent=lambda _messages, _state: None,
    )


def test_prompt_for_context_shows_plan_next_indicator() -> None:
    ctx = _make_ctx([])
    ctx.force_plan_next = True

    assert _prompt_for_context(ctx) == "\n~ [plan-next] "


def test_cancel_clears_pending_plan_and_prompt_returns_to_normal() -> None:
    outputs: list[str] = []
    prompts: list[str] = []
    registry = CommandRegistry()
    register_builtin_commands(registry)
    ctx = _make_ctx(outputs)
    responses = iter(["implement the parser", ":cancel", "exit"])
    goodbye = mock.MagicMock()

    def _input(prompt: str, **_: object) -> str:
        prompts.append(prompt)
        return next(responses)

    run_repl(
        ctx=ctx,
        registry=registry,
        get_user_input=_input,
        classify_intent=lambda text: "work" if text == "implement the parser" else "info",
        render_goodbye_message=goodbye,
    )

    assert prompts == ["\n~ ", "\n~ [plan-pending] ", "\n~ "]
    assert ctx.pending_plan is None
    assert ctx.pending_request is None
    assert outputs[-1] == "Plan canceled."
    goodbye.assert_called_once()


def test_keyboard_interrupt_sets_interrupt_event_and_stops_spinners() -> None:
    outputs: list[str] = []
    registry = CommandRegistry()
    register_builtin_commands(registry)
    stop_spinners = mock.MagicMock()
    ctx = _make_ctx(outputs)
    ctx.stop_spinners = stop_spinners
    goodbye = mock.MagicMock()

    previous_event = callback_handler_instance.interrupt_event
    interrupt_event = threading.Event()
    callback_handler_instance.interrupt_event = interrupt_event
    try:
        run_repl(
            ctx=ctx,
            registry=registry,
            get_user_input=lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
            classify_intent=lambda _text: "info",
            render_goodbye_message=goodbye,
        )
    finally:
        callback_handler_instance.interrupt_event = previous_event

    assert interrupt_event.is_set()
    stop_spinners.assert_called_once()
    goodbye.assert_called_once()
