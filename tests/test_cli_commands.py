from __future__ import annotations

from unittest import mock

from swarmee_river.cli.builtin_commands import register_builtin_commands
from swarmee_river.cli.commands import CLIContext, CommandRegistry
from swarmee_river.cli.diagnostics import render_diagnostic_command_for_surface


def _make_ctx(outputs: list[str]) -> CLIContext:
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
        generate_plan=lambda _req: None,
        execute_with_plan=lambda _req, _plan: None,
        render_plan=lambda _plan: "",
        run_agent=lambda *_args, **_kwargs: None,
        store_conversation=lambda _req, _resp: None,
        output=lambda text: outputs.append(text),
        stop_spinners=lambda: None,
        build_session_meta=lambda: {},
        swap_agent=lambda _messages, _state: None,
    )


def test_cancel_command_is_registered_and_dispatches() -> None:
    outputs: list[str] = []
    registry = CommandRegistry()
    register_builtin_commands(registry)
    ctx = _make_ctx(outputs)
    ctx.pending_plan = object()
    ctx.pending_request = "implement tests"
    ctx.force_plan_next = True

    result = registry.dispatch(ctx, ":cancel")

    assert result.handled is True
    assert ctx.pending_plan is None
    assert ctx.pending_request is None
    assert ctx.force_plan_next is False
    assert outputs[-1] == "Plan canceled."


def test_help_lists_cancel_command() -> None:
    outputs: list[str] = []
    registry = CommandRegistry()
    register_builtin_commands(registry)
    ctx = _make_ctx(outputs)

    registry.dispatch(ctx, ":help")

    assert outputs
    assert ":cancel" in outputs[-1]


def test_render_diagnostic_usage_rewrite_is_generic(monkeypatch) -> None:
    monkeypatch.setattr(
        "swarmee_river.cli.diagnostics.render_diagnostic_command",
        lambda **_kwargs: "Usage: status | Usage: diff | Usage: artifact",
    )

    repl_text = render_diagnostic_command_for_surface(cmd="status", args=[], cwd=mock.MagicMock(), surface="repl")
    cli_text = render_diagnostic_command_for_surface(cmd="status", args=[], cwd=mock.MagicMock(), surface="cli")

    assert "Usage: :status" in repl_text
    assert "Usage: :diff" in repl_text
    assert "Usage: :artifact" in repl_text
    assert "Usage: swarmee status" in cli_text
    assert "Usage: swarmee diff" in cli_text
    assert "Usage: swarmee artifact" in cli_text
