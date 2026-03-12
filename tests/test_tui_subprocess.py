#!/usr/bin/env python3
"""
Tests for TUI subprocess helpers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from swarmee_river.handlers.callback_handler import TuiCallbackHandler
from swarmee_river.tui import app as tui_app
from swarmee_river.tui.mixins.agent_studio import AgentStudioMixin
from swarmee_river.tui.mixins.artifacts import ArtifactsMixin
from swarmee_river.tui.mixins.context_sources import ContextSourcesMixin
from swarmee_river.tui.mixins.daemon import DaemonMixin
from swarmee_river.tui.mixins.output import OutputMixin
from swarmee_river.tui.mixins.plan import PlanMixin
from swarmee_river.tui.mixins.session import SessionMixin
from swarmee_river.tui.mixins.settings import SettingsMixin
from swarmee_river.tui.mixins.thinking import ThinkingMixin
from swarmee_river.tui.mixins.tools import ToolsMixin
from swarmee_river.tui.mixins.transcript import TranscriptMixin
from swarmee_river.tui.state import AppState


def test_build_swarmee_cmd_run_mode():
    prompt = "show git status"
    command = tui_app.build_swarmee_cmd(prompt, auto_approve=True)
    assert command == [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--yes", prompt]


def test_build_swarmee_cmd_plan_mode():
    prompt = "show git status"
    command = tui_app.build_swarmee_cmd(prompt, auto_approve=False)
    assert command == [sys.executable, "-u", "-m", "swarmee_river.swarmee", prompt]


def test_build_swarmee_daemon_cmd():
    command = tui_app.build_swarmee_daemon_cmd()
    assert command == [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--tui-daemon"]


def test_model_select_options_lists_all_configured_provider_tiers(monkeypatch):
    class _Tier:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

    class _Provider:
        def __init__(self, tiers: dict[str, _Tier]) -> None:
            self.tiers = tiers

    class _Models:
        def __init__(self) -> None:
            self.provider = "openai"
            self.default_tier = "balanced"
            self.providers = {
                "openai": _Provider({"balanced": _Tier("gpt-5-mini")}),
                "bedrock": _Provider({"balanced": _Tier("anthropic.claude-sonnet")}),
            }
            self.tiers = {"balanced": _Tier("global-balanced"), "fast": _Tier("global-fast")}

    class _Settings:
        def __init__(self) -> None:
            self.models = _Models()

    import swarmee_river.settings as settings_module
    import swarmee_river.utils.provider_utils as provider_utils_module

    monkeypatch.setattr(settings_module, "load_settings", lambda: _Settings())
    monkeypatch.setattr(
        provider_utils_module,
        "resolve_model_provider",
        lambda **_kwargs: ("openai", None),
    )

    options, selected = tui_app.model_select_options()
    values = [value for _label, value in options]

    assert "__auto__" in values
    assert "openai|balanced" in values
    assert "bedrock|balanced" in values
    assert "bedrock|fast" in values
    assert "openai|fast" in values
    assert selected == "__auto__"


def test_model_select_options_keeps_explicit_bedrock_override_selected(monkeypatch):
    class _Tier:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

    class _Provider:
        def __init__(self, tiers: dict[str, _Tier]) -> None:
            self.tiers = tiers

    class _Models:
        def __init__(self) -> None:
            self.provider = None
            self.default_tier = "balanced"
            self.providers = {
                "openai": _Provider({"balanced": _Tier("gpt-5-mini")}),
                "bedrock": _Provider({"deep": _Tier("anthropic.claude-sonnet")}),
            }
            self.tiers = {}

    class _Settings:
        def __init__(self) -> None:
            self.models = _Models()

    import swarmee_river.settings as settings_module
    import swarmee_river.utils.provider_utils as provider_utils_module

    monkeypatch.setattr(settings_module, "load_settings", lambda: _Settings())
    monkeypatch.setattr(
        provider_utils_module,
        "resolve_model_provider",
        lambda **kwargs: (kwargs.get("cli_provider") or "openai", None),
    )

    options, selected = tui_app.model_select_options(provider_override="bedrock", tier_override="deep")
    values = [value for _label, value in options]

    assert "bedrock|deep" in values
    assert selected == "bedrock|deep"


def test_choose_daemon_model_select_value_prefers_pending_then_daemon():
    values = ["openai|fast", "openai|balanced"]
    selected = tui_app.choose_daemon_model_select_value(
        provider="openai",
        tier="balanced",
        option_values=values,
        pending_value="openai|fast",
    )
    assert selected == "openai|fast"

    selected_no_pending = tui_app.choose_daemon_model_select_value(
        provider="openai",
        tier="balanced",
        option_values=values,
    )
    assert selected_no_pending == "openai|balanced"


def test_handle_model_info_updates_runtime_context_budget():
    class _StatusBar:
        def __init__(self) -> None:
            self.context_calls: list[tuple[int | None, int | None]] = []
            self.model_calls: list[str] = []

        def set_context(self, *, prompt_tokens_est, budget_tokens) -> None:  # noqa: ANN001
            self.context_calls.append((prompt_tokens_est, budget_tokens))

        def set_model(self, summary: str) -> None:
            self.model_calls.append(summary)

    class _Harness(SettingsMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.daemon.last_prompt_tokens_est = 12_345
            self._status_bar = _StatusBar()
            self._model_select_target_until_mono = None
            self._model_select_target_value = None
            self._engage_orchestrator_status = None
            self._settings_general_view = None
            self._settings_models_view = None
            self._settings_advanced_view = None
            self._settings_view_general_button = None
            self._settings_view_models_button = None
            self._settings_view_advanced_button = None
            self.prompt_metric_refreshes = 0

        def _handle_connect_model_info_event(self, _event: dict[str, object]) -> None:
            return None

        def _refresh_agent_tool_catalog(self, _tool_names) -> None:  # noqa: ANN001
            return None

        def _refresh_model_select_from_daemon(self, **_kwargs) -> None:
            return None

        def _refresh_model_select(self) -> None:
            return None

        def _update_header_status(self) -> None:
            return None

        def _current_model_summary(self) -> str:
            return "bedrock/deep"

        def _refresh_agent_summary(self) -> None:
            return None

        def _render_agent_builder_panel(self) -> None:
            return None

        def _refresh_prompt_metrics(self) -> None:
            self.prompt_metric_refreshes += 1

    harness = _Harness()
    harness._handle_model_info(
        {
            "provider": "bedrock",
            "tier": "deep",
            "model_id": "us.anthropic.claude-opus-4-6-v1",
            "context_budget_tokens": 200000,
            "tiers": [
                {
                    "provider": "bedrock",
                    "name": "deep",
                    "context_budget_tokens": 200000,
                }
            ],
        }
    )

    assert harness.state.daemon.last_budget_tokens == 200000
    assert harness._status_bar.context_calls[-1] == (12_345, 200000)
    assert harness.prompt_metric_refreshes == 1


def test_set_model_tier_from_value_updates_budget_optimistically(monkeypatch):
    sent_commands: list[dict[str, object]] = []

    monkeypatch.setattr(
        "swarmee_river.tui.transport.send_daemon_command",
        lambda _proc, payload: sent_commands.append(dict(payload)) or True,
    )

    class _StatusBar:
        def __init__(self) -> None:
            self.context_calls: list[tuple[int | None, int | None]] = []
            self.model_calls: list[str] = []

        def set_context(self, *, prompt_tokens_est, budget_tokens) -> None:  # noqa: ANN001
            self.context_calls.append((prompt_tokens_est, budget_tokens))

        def set_model(self, summary: str) -> None:
            self.model_calls.append(summary)

    class _Proc:
        def poll(self) -> None:
            return None

    class _Harness(SettingsMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.daemon.ready = True
            self.state.daemon.proc = _Proc()
            self.state.daemon.last_prompt_tokens_est = 5000
            self.state.daemon.tiers = [
                {"provider": "openai", "name": "balanced", "context_budget_tokens": 400000},
                {"provider": "bedrock", "name": "deep", "context_budget_tokens": 200000},
            ]
            self._status_bar = _StatusBar()
            self.prompt_metric_refreshes = 0

        def _refresh_model_select(self) -> None:
            return None

        def _update_header_status(self) -> None:
            return None

        def _update_prompt_placeholder(self) -> None:
            return None

        def _write_transcript_line(self, _text: str) -> None:
            return None

        def _current_model_summary(self) -> str:
            return "bedrock/deep"

        def _refresh_prompt_metrics(self) -> None:
            self.prompt_metric_refreshes += 1

    harness = _Harness()
    harness._set_model_tier_from_value("bedrock|deep")

    assert harness.state.daemon.last_budget_tokens == 200000
    assert harness._status_bar.context_calls[-1] == (5000, 200000)
    assert sent_commands == [{"cmd": "set_model", "provider": "bedrock", "tier": "deep"}]


def test_parse_model_select_value_handles_valid_and_special_values():
    assert tui_app.parse_model_select_value("openai|deep") == ("openai", "deep")
    assert tui_app.parse_model_select_value(" OPENAI|Deep ") == ("openai", "deep")
    assert tui_app.parse_model_select_value("__auto__") is None
    assert tui_app.parse_model_select_value("__loading__") is None
    assert tui_app.parse_model_select_value("openai") is None


def test_classify_copy_command_matrix():
    cases = {
        "/copy": "transcript",
        ":copy": "transcript",
        "/copy plan": "plan",
        ":copy plan": "plan",
        "/copy issues": "issues",
        ":copy issues": "issues",
        "/copy artifacts": "artifacts",
        ":copy artifacts": "artifacts",
        "/copy last": "last",
        ":copy last": "last",
        "/copy all": "all",
        ":copy all": "all",
        "/copy nope": None,
        "copy": None,
    }
    for command, expected in cases.items():
        assert tui_app.classify_copy_command(command) == expected


def test_classify_model_command_matrix():
    cases = {
        "/model": ("help", None),
        "/model show": ("show", None),
        "/model list": ("list", None),
        "/model reset": ("reset", None),
        "/model provider openai": ("provider", "openai"),
        "/model tier deep": ("tier", "deep"),
        "/model provider": None,
        "/model tier": None,
        "/model unknown": None,
    }
    for command, expected in cases.items():
        assert tui_app.classify_model_command(command) == expected


def test_classify_pre_run_command_matrix():
    cases = {
        "/help": ("help", None),
        "/open 12": ("open", "12"),
        "/open": ("open_usage", None),
        "/search bug": ("search", "bug"),
        "/search": ("search_usage", None),
        "/text": ("text", None),
        "/text extra": ("text_usage", None),
        "/thinking": ("thinking", None),
        "/thinking full": ("thinking_usage", None),
        "/compact": ("compact", None),
        "/compact now": ("compact_usage", None),
        "/restore": ("restore", None),
        "/new": ("new", None),
        "/context": ("context_usage", None),
        "/context list": ("context", "list"),
        "/sop": ("sop_usage", None),
        "/sop list": ("sop", "list"),
        "/stop": ("stop", None),
        ":stop": ("stop", None),
        "/exit": ("exit", None),
        ":exit": ("exit", None),
        "/daemon restart": ("daemon_restart", None),
        "/restart-daemon": ("daemon_restart", None),
        "/daemon stop": ("daemon_stop", None),
        "/broker stop": ("daemon_stop", None),
        "/consent": ("consent_usage", None),
        "/consent y": ("consent", "y"),
        "/connect": ("connect", "github_copilot"),
        "/connect github_copilot": ("connect", "github_copilot"),
        "/connect aws": ("connect", "aws"),
        "/connect aws dev-profile": ("connect", "aws dev-profile"),
        "/auth": ("auth_usage", None),
        "/auth list": ("auth", "list"),
        "/auth logout github_copilot": ("auth", "logout github_copilot"),
        "/model show": ("model:show", None),
        "/model provider openai": ("model:provider", "openai"),
        "/approve": None,
        "hello": None,
    }
    for command, expected in cases.items():
        assert tui_app.classify_pre_run_command(command) == expected


def test_classify_post_run_command_matrix():
    cases = {
        "/approve": ("approve", None),
        "/replan": ("replan", None),
        "/clearplan": ("clearplan", None),
        "/plan": ("plan_mode", None),
        "/plan draft": ("plan_prompt", "draft"),
        "/run": ("run_mode", None),
        "/run now": ("run_prompt", "now"),
        "/model show": None,
        "hello": None,
    }
    for command, expected in cases.items():
        assert tui_app.classify_post_run_command(command) == expected


def test_plain_prompt_submit_forces_execute_mode():
    class _Harness:
        def __init__(self) -> None:
            self._default_auto_approve = True
            self.state = SimpleNamespace(daemon=SimpleNamespace(query_active=False))
            self.run_calls: list[tuple[str, bool, str | None]] = []
            self.transcript: list[str] = []
            self.user_inputs: list[str] = []

        def _write_user_input(self, text: str) -> None:
            self.user_inputs.append(text)

        def _handle_copy_command(self, _normalized: str) -> bool:
            return False

        def _handle_pre_run_command(self, _text: str) -> bool:
            return False

        def _handle_post_run_command(self, _text: str) -> bool:
            return False

        def _start_run(
            self,
            prompt: str,
            *,
            auto_approve: bool,
            mode: str | None = None,
            plan_context: dict[str, object] | None = None,
        ) -> None:
            self.run_calls.append((prompt, auto_approve, mode, plan_context))

        def _write_transcript_line(self, text: str) -> None:
            self.transcript.append(text)

    harness = _Harness()
    SwarmeeTUI = tui_app.get_swarmee_tui_class()
    SwarmeeTUI._handle_user_input(harness, "ping")

    assert harness.user_inputs == ["ping"]
    assert harness.run_calls == [("ping", True, "execute", None)]
    assert harness.transcript == []


def test_plan_prompt_command_still_starts_plan_mode():
    class _Harness:
        def __init__(self) -> None:
            self._default_auto_approve = True
            self.run_calls: list[tuple[str, bool, str | None]] = []

        def _dispatch_plan_action(self, _action: str) -> None:
            raise AssertionError("plan action dispatch should not be called for /plan <prompt>")

        def _update_prompt_placeholder(self) -> None:
            return None

        def _write_transcript_line(self, _text: str) -> None:
            return None

        def _start_run(
            self,
            prompt: str,
            *,
            auto_approve: bool,
            mode: str | None = None,
            plan_context: dict[str, object] | None = None,
        ) -> None:
            self.run_calls.append((prompt, auto_approve, mode, plan_context))

    harness = _Harness()
    handled = DaemonMixin._handle_post_run_command(harness, "/plan draft ping check")

    assert handled is True
    assert harness.run_calls == [("draft ping check", False, "plan", None)]


def test_plan_approve_dispatch_starts_execute_mode() -> None:
    from swarmee_river.planning import WorkPlan, new_pending_work_plan

    class _Harness:
        def __init__(self) -> None:
            pending = new_pending_work_plan(
                original_request="ship it",
                plan=WorkPlan(summary="Ship it", steps=[]),
                plan_run_id="plan-123",
            )
            self.state = SimpleNamespace(
                plan=SimpleNamespace(
                    pending_prompt="ship it",
                    pending_record=pending.model_dump(),
                    plan_run_id="plan-123",
                )
            )
            self.run_calls: list[tuple[str, bool, str | None, dict[str, object] | None]] = []
            self.transcript: list[str] = []
            self.saved = False

        def _pending_plan_record(self):
            return new_pending_work_plan(
                original_request="ship it",
                plan=WorkPlan(summary="Ship it", steps=[]),
                plan_run_id="plan-123",
            )

        def _clear_pending_plan_record(self) -> None:
            self.state.plan.pending_record = None
            self.state.plan.pending_prompt = None

        def _refresh_plan_actions_visibility(self) -> None:
            return None

        def _save_session(self) -> None:
            self.saved = True

        def _start_run(
            self,
            prompt: str,
            *,
            auto_approve: bool,
            mode: str | None = None,
            plan_context: dict[str, object] | None = None,
        ) -> None:
            self.run_calls.append((prompt, auto_approve, mode, plan_context))

        def _write_transcript_line(self, text: str) -> None:
            self.transcript.append(text)

    harness = _Harness()

    PlanMixin._dispatch_plan_action(harness, "approve")

    assert len(harness.run_calls) == 1
    prompt, auto_approve, mode, plan_context = harness.run_calls[0]
    assert (prompt, auto_approve, mode) == ("ship it", True, "execute")
    assert isinstance(plan_context, dict)
    approved_plan = plan_context.get("approved_plan")
    assert isinstance(approved_plan, dict)
    assert approved_plan.get("plan_run_id") == "plan-123"
    assert approved_plan.get("original_request") == "ship it"
    current_plan = approved_plan.get("current_plan")
    assert isinstance(current_plan, dict)
    assert current_plan.get("summary") == "Ship it"
    assert current_plan.get("steps") == []
    assert harness.transcript == []
    assert harness.state.plan.pending_record is None
    assert harness.saved is True


def test_plan_step_update_ignores_stale_plan_run_id() -> None:
    from swarmee_river.tui.event_router import _handle_plan_events

    class _Harness:
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                plan=SimpleNamespace(
                    plan_run_id="current-plan",
                    current_step_statuses=["pending"],
                    current_steps_total=1,
                    current_active_step=None,
                    step_counter=0,
                    updates_seen=False,
                    completion_announced=False,
                )
            )
            self.rendered = False
            self.transcript: list[str] = []

        def _render_plan_panel_from_status(self) -> None:
            self.rendered = True

        def _write_transcript_line(self, text: str) -> None:
            self.transcript.append(text)

    harness = _Harness()

    handled = _handle_plan_events(
        harness,
        "plan_step_update",
        {"event": "plan_step_update", "plan_run_id": "stale-plan", "step_index": 0, "status": "completed"},
    )

    assert handled is True
    assert harness.state.plan.current_step_statuses == ["pending"]
    assert harness.rendered is False


def test_start_fresh_session_rotates_session_id_and_restarts_daemon():
    class _Harness(SessionMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(
                    query_active=False,
                    session_id="old-session-id",
                    available_restore_session_id="old-session-id",
                    available_restore_turn_count=12,
                    last_restored_turn_count=4,
                    last_prompt_tokens_est=12345,
                    last_budget_tokens=200000,
                )
            )
            self.session_changes: list[tuple[str | None, str | None]] = []
            self.spawn_calls: list[bool] = []
            self.transcript: list[str] = []
            self.saved = False
            self.reset_calls: list[bool] = []

        def _on_active_session_changed(self, old_session_id: str | None, new_session_id: str | None) -> None:
            self.session_changes.append((old_session_id, new_session_id))
            self.state.daemon.session_id = new_session_id

        def _spawn_daemon(self, *, restart: bool = False) -> None:
            self.spawn_calls.append(restart)

        def _write_transcript_line(self, text: str) -> None:
            self.transcript.append(text)

        def _save_session(self) -> None:
            self.saved = True

        def _reset_run_local_ui_state(self, *, clear_prompt_context: bool) -> None:
            self.reset_calls.append(clear_prompt_context)
            if clear_prompt_context:
                self.state.daemon.last_prompt_tokens_est = 0

    harness = _Harness()
    harness._start_fresh_session()

    assert len(harness.session_changes) == 1
    old_sid, new_sid = harness.session_changes[0]
    assert old_sid == "old-session-id"
    assert isinstance(new_sid, str)
    assert re.fullmatch(r"[0-9a-f]{32}", new_sid or "")
    assert new_sid != "old-session-id"
    assert harness.state.daemon.session_id == new_sid
    assert harness.state.daemon.available_restore_session_id is None
    assert harness.state.daemon.available_restore_turn_count == 0
    assert harness.state.daemon.last_restored_turn_count == 0
    assert harness.state.daemon.last_prompt_tokens_est == 0
    assert harness.state.daemon.last_budget_tokens == 200000
    assert harness.spawn_calls == [True]
    assert harness.reset_calls == [True]
    assert harness.saved is True
    assert harness.transcript[-1] == "[session] starting fresh."


def test_reset_run_local_ui_state_clears_prompt_context_and_error_actions():
    class _StatusBar:
        def __init__(self) -> None:
            self.context_calls: list[tuple[int | None, int | None]] = []
            self.provider_usage_calls: list[tuple[int | None, int | None, int | None, float | None]] = []

        def set_state(self, _state: str) -> None:
            return None

        def set_tool_count(self, _count: int) -> None:
            return None

        def set_elapsed(self, _elapsed: float) -> None:
            return None

        def set_provider_usage(
            self,
            *,
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            cost_usd=None,
        ) -> None:  # noqa: ANN001
            self.provider_usage_calls.append((input_tokens, cached_input_tokens, output_tokens, cost_usd))

        def set_context(self, *, prompt_tokens_est, budget_tokens) -> None:  # noqa: ANN001
            self.context_calls.append((prompt_tokens_est, budget_tokens))

    class _Timer:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    class _Harness(OutputMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(
                    run_tool_count=3,
                    run_start_time=10.0,
                    query_active=False,
                    turn_output_chunks=["chunk"],
                    last_usage={"inputTokens": 1},
                    last_cost_usd=0.1,
                    last_prompt_tokens_est=180000,
                    last_budget_tokens=200000,
                    last_provider_input_tokens=50000,
                    last_provider_cached_input_tokens=10300,
                    last_provider_output_tokens=1200,
                    status_timer=_Timer(),
                )
            )
            self._pending_error_action = {"kind": "tool"}
            self._error_action_prompt_widget = None
            self._consent_prompt_widget = None
            self._consent_hide_timer = None
            self._consent_prompt_nonce = 0
            self._consent_active = False
            self._consent_buffer = []
            self._consent_tool_name = "tool"
            self._status_bar = _StatusBar()
            self._prompt_input_tokens_est = 42
            self._current_assistant_chunks = ["hello"]
            self._streaming_buffer = [" world"]
            self._current_assistant_model = "bedrock/deep"
            self._current_assistant_timestamp = "10:00 AM"
            self._assistant_completion_seen_turn = True
            self._assistant_placeholder_written = True
            self._stream_render_warning_emitted_turn = True
            self._structured_assistant_seen_turn = True
            self._raw_assistant_lines_suppressed_turn = 2
            self._last_structured_assistant_text_turn = "hello world"
            self._callback_event_trace_turn = ["text_delta"]
            self._active_assistant_message = None
            self._active_reasoning_block = object()
            self._last_thinking_text = "plan"
            self._thinking_seen_turn = True
            self._thinking_unavailable_notice_emitted_turn = True
            self._tool_blocks = {"tool-1": {"widget": object()}}
            self._tool_progress_pending_ids = {"tool-1"}
            self._refresh_count = 0
            self.flushed = 0
            self.finalized = 0
            self.dismissed = 0
            self.cleared_pending = 0
            self.cancelled_tool_progress = 0

        def _flush_all_streaming_buffers(self) -> None:
            self.flushed += 1

        def _finalize_assistant_message(self) -> None:
            self.finalized += 1

        def _dismiss_thinking(self, *, emit_summary: bool = False) -> None:
            del emit_summary
            self.dismissed += 1

        def _clear_pending_tool_starts(self) -> None:
            self.cleared_pending += 1

        def _cancel_tool_progress_flush_timer(self) -> None:
            self.cancelled_tool_progress += 1

        def _refresh_prompt_metrics(self) -> None:
            self._refresh_count += 1

        def _reset_consent_panel(self) -> None:
            return None

        def _reset_error_action_prompt(self) -> None:
            self._pending_error_action = None

    harness = _Harness()
    harness._reset_run_local_ui_state(clear_prompt_context=True)

    assert harness._pending_error_action is None
    assert harness.state.daemon.last_prompt_tokens_est == 0
    assert harness.state.daemon.last_budget_tokens == 200000
    assert harness._prompt_input_tokens_est == 0
    assert harness.state.daemon.run_tool_count == 0
    assert harness.state.daemon.turn_output_chunks == []
    assert harness.state.daemon.last_usage is None
    assert harness.state.daemon.last_cost_usd is None
    assert harness.state.daemon.last_provider_input_tokens is None
    assert harness.state.daemon.last_provider_cached_input_tokens is None
    assert harness.state.daemon.last_provider_output_tokens is None
    assert harness._tool_blocks == {}
    assert harness._tool_progress_pending_ids == set()
    assert harness._status_bar.context_calls[-1] == (0, 200000)
    assert harness._status_bar.provider_usage_calls[-1] == (None, None, None, None)
    assert harness.flushed == 1
    assert harness.finalized == 1
    assert harness.dismissed == 1
    assert harness.cleared_pending == 1
    assert harness.cancelled_tool_progress == 1
    assert harness._refresh_count == 1


def test_classify_tui_error_event_prefers_daemon_metadata() -> None:
    event = {
        "event": "error",
        "message": "rate limit exceeded",
        "category": "transient",
        "retryable": True,
        "retry_after_s": 4,
    }
    result = tui_app.classify_tui_error_event(event)
    assert result["category"] == "transient"
    assert result["retryable"] is True
    assert result["retry_after_s"] == 4


def test_summarize_error_for_toast_transient_message() -> None:
    message, severity, timeout = tui_app.summarize_error_for_toast(
        {"category": "transient", "message": "ThrottlingException", "retry_after_s": 5}
    )
    assert message == "Rate limited - retrying in 5s"
    assert severity == "warning"
    assert timeout == 5.0


def test_command_classification_precedence_ordering():
    # copy commands should be recognized before pre/post phases
    normalized_copy = "/copy all"
    assert tui_app.classify_copy_command(normalized_copy) == "all"
    assert tui_app.classify_pre_run_command(normalized_copy) is None
    assert tui_app.classify_post_run_command(normalized_copy) is None

    # pre-run commands should be recognized before post-run phase
    model_show = "/model show"
    assert tui_app.classify_copy_command(model_show) is None
    assert tui_app.classify_pre_run_command(model_show) == ("model:show", None)
    assert tui_app.classify_post_run_command(model_show) is None

    # post-run commands should not be mistaken for pre-run commands
    run_now = "/run now"
    assert tui_app.classify_copy_command(run_now) is None
    assert tui_app.classify_pre_run_command(run_now) is None
    assert tui_app.classify_post_run_command(run_now) == ("run_prompt", "now")


def test_should_skip_active_run_tier_warning_only_for_pending_match():
    assert (
        tui_app.should_skip_active_run_tier_warning(
            requested_provider="openai",
            requested_tier="deep",
            pending_value="openai|deep",
        )
        is True
    )
    assert (
        tui_app.should_skip_active_run_tier_warning(
            requested_provider="openai",
            requested_tier="deep",
            pending_value="openai|fast",
        )
        is False
    )


def test_choose_daemon_model_select_value_uses_override_when_available():
    values = ["openai|fast", "openai|balanced"]
    selected = tui_app.choose_daemon_model_select_value(
        provider="openai",
        tier="balanced",
        option_values=values,
        override_provider="openai",
        override_tier="fast",
    )
    assert selected == "openai|fast"


def test_choose_daemon_model_select_value_falls_back_to_first_option():
    values = ["openai|fast", "openai|balanced"]
    selected = tui_app.choose_daemon_model_select_value(
        provider="openai",
        tier="economy",
        option_values=values,
        pending_value="openai|missing",
    )
    assert selected == "openai|fast"


def test_daemon_model_select_options_injects_pending_when_missing_from_available():
    options, selected = tui_app.daemon_model_select_options(
        provider="openai",
        tier="balanced",
        tiers=[
            {"provider": "openai", "name": "balanced", "available": True, "model_id": "gpt-5-mini"},
        ],
        pending_value="openai|deep",
    )
    values = [value for _label, value in options]
    assert "openai|deep" in values
    assert selected == "openai|deep"
    assert options[0][0].endswith("(pending)")


def test_daemon_model_select_options_injects_override_when_missing_from_available():
    options, selected = tui_app.daemon_model_select_options(
        provider="openai",
        tier="balanced",
        tiers=[
            {"provider": "openai", "name": "balanced", "available": True, "model_id": "gpt-5-mini"},
        ],
        override_provider="openai",
        override_tier="deep",
    )
    values = [value for _label, value in options]
    assert "openai|deep" in values
    assert selected == "openai|deep"
    assert options[0][0].endswith("(selected)")


def test_daemon_model_select_options_includes_other_provider_unavailable_rows():
    options, selected = tui_app.daemon_model_select_options(
        provider="openai",
        tier="balanced",
        tiers=[
            {"provider": "openai", "name": "balanced", "available": True, "model_id": "gpt-5-mini"},
            {
                "provider": "bedrock",
                "name": "deep",
                "available": False,
                "model_id": "us.anthropic.claude-sonnet",
                "reason": "AWS credentials missing/expired",
            },
        ],
        override_provider="bedrock",
        override_tier="deep",
    )
    labels = [label for label, _value in options]
    values = [value for _label, value in options]
    assert "bedrock|deep" in values
    assert any("[unavailable:" in label for label in labels)
    assert selected == "bedrock|deep"


def test_choose_model_summary_parts_prefers_pending_until_confirmed():
    provider, tier, model_id = tui_app.choose_model_summary_parts(
        daemon_provider="openai",
        daemon_tier="balanced",
        daemon_model_id="gpt-5-mini",
        pending_value="openai|fast",
    )
    assert provider == "openai"
    assert tier == "fast"
    assert model_id is None


def test_choose_model_summary_parts_uses_model_id_when_pending_matches_daemon():
    provider, tier, model_id = tui_app.choose_model_summary_parts(
        daemon_provider="openai",
        daemon_tier="fast",
        daemon_model_id="gpt-5-mini",
        pending_value="openai|fast",
    )
    assert provider == "openai"
    assert tier == "fast"
    assert model_id == "gpt-5-mini"


def test_choose_model_summary_parts_falls_back_to_daemon_without_pending():
    provider, tier, model_id = tui_app.choose_model_summary_parts(
        daemon_provider="openai",
        daemon_tier="balanced",
        daemon_model_id="gpt-5-mini",
        daemon_tiers=[],
        pending_value=None,
    )
    assert provider == "openai"
    assert tier == "balanced"
    assert model_id == "gpt-5-mini"


def test_choose_model_summary_parts_prefers_override_over_stale_daemon():
    provider, tier, model_id = tui_app.choose_model_summary_parts(
        daemon_provider="openai",
        daemon_tier="balanced",
        daemon_model_id="gpt-5-mini",
        daemon_tiers=[
            {"provider": "openai", "name": "balanced", "available": True, "model_id": "gpt-5-mini"},
            {"provider": "openai", "name": "fast", "available": True, "model_id": "gpt-5-nano"},
        ],
        pending_value=None,
        override_provider="openai",
        override_tier="fast",
    )
    assert provider == "openai"
    assert tier == "fast"
    assert model_id == "gpt-5-nano"


def test_looks_like_plan_output():
    text = "Some output\nProposed plan:\n- Step 1\n- Step 2\nPlan generated. Re-run with --yes to execute."
    assert tui_app.looks_like_plan_output(text) is True
    assert tui_app.looks_like_plan_output("normal run output") is False


def test_extract_plan_section_returns_only_plan_block():
    output = (
        "preface\n"
        "Proposed plan:\n"
        "- Summary: Fix issue\n"
        "- Steps:\n"
        "  1. Reproduce\n"
        "\n"
        "Plan generated. Re-run with --yes (or set SWARMEE_AUTO_APPROVE=true) to execute.\n"
        "postface\n"
    )
    extracted = tui_app.extract_plan_section(output)
    assert extracted == "Proposed plan:\n- Summary: Fix issue\n- Steps:\n  1. Reproduce"


def test_extract_plan_section_returns_none_when_missing_marker():
    assert tui_app.extract_plan_section("plain execution output") is None


def test_extract_plan_section_from_output_ignores_jsonl_event_lines():
    output = '{"event":"plan","rendered":"Proposed plan:\\n- Summary: Json only"}\n'
    assert tui_app.extract_plan_section_from_output(output) is None


def test_extract_plan_section_from_output_extracts_plain_plan_lines():
    output = (
        '{"event":"thinking","text":"planning"}\n'
        "Proposed plan:\n"
        "- Summary: Fix issue\n"
        "- Steps:\n"
        "  1. Reproduce\n"
        "Plan generated. Re-run with --yes (or set SWARMEE_AUTO_APPROVE=true) to execute.\n"
    )
    extracted = tui_app.extract_plan_section_from_output(output)
    assert extracted == "Proposed plan:\n- Summary: Fix issue\n- Steps:\n  1. Reproduce"


def test_is_multiline_newline_key_detects_shift_enter():
    class _Event:
        key = "shift+enter"
        character = None
        aliases: list[str] = []

    assert tui_app.is_multiline_newline_key(_Event()) is True


def test_is_multiline_newline_key_detects_newline_alias():
    class _Event:
        key = "ctrl+j"
        character = "\n"
        aliases = ["ctrl+j", "newline"]

    assert tui_app.is_multiline_newline_key(_Event()) is True


def test_is_multiline_newline_key_rejects_plain_enter():
    class _Event:
        key = "enter"
        character = "\r"
        aliases = ["enter", "ctrl+m"]

    assert tui_app.is_multiline_newline_key(_Event()) is False


def test_is_multiline_newline_key_detects_shift_ctrl_m_variant():
    class _Event:
        key = "shift+ctrl+m"
        name = "shift_ctrl_m"
        character = "\r"
        aliases = ["shift+ctrl+m"]

    assert tui_app.is_multiline_newline_key(_Event()) is True


def test_is_multiline_newline_key_detects_shift_enter_alias():
    class _Event:
        key = "enter"
        name = "enter"
        character = "\r"
        aliases = ["enter", "ctrl+m", "shift+enter"]

    assert tui_app.is_multiline_newline_key(_Event()) is True


def test_resize_key_helpers_match_arrow_and_fallback_variants():
    class _Event:
        def __init__(self, key: str, aliases: list[str] | None = None, name: str = "") -> None:
            self.key = key
            self.aliases = aliases or []
            self.name = name

    assert tui_app.is_widen_side_key(_Event("ctrl+left")) is True
    assert tui_app.is_widen_side_key(_Event("left", aliases=["ctrl+left"])) is True
    assert tui_app.is_widen_side_key(_Event("alt+left")) is True
    assert tui_app.is_widen_side_key(_Event("ctrl+h")) is True
    assert tui_app.is_widen_side_key(_Event("f6")) is True

    assert tui_app.is_widen_transcript_key(_Event("ctrl+right")) is True
    assert tui_app.is_widen_transcript_key(_Event("right", aliases=["ctrl+right"])) is True
    assert tui_app.is_widen_transcript_key(_Event("alt+right")) is True
    assert tui_app.is_widen_transcript_key(_Event("ctrl+l")) is True
    assert tui_app.is_widen_transcript_key(_Event("f7")) is True

    assert tui_app.is_widen_side_key(_Event("enter")) is False
    assert tui_app.is_widen_transcript_key(_Event("enter")) is False


def test_should_ignore_programmatic_model_select_change_matches_marker():
    assert (
        tui_app.should_ignore_programmatic_model_select_change(
            value="openai|fast",
            programmatic_value="openai|fast",
        )
        is True
    )
    assert (
        tui_app.should_ignore_programmatic_model_select_change(
            value="OPENAI|FAST",
            programmatic_value="openai|fast",
        )
        is True
    )
    assert (
        tui_app.should_ignore_programmatic_model_select_change(
            value="openai|deep",
            programmatic_value="openai|fast",
        )
        is False
    )
    assert (
        tui_app.should_ignore_programmatic_model_select_change(
            value="openai|fast",
            programmatic_value=None,
        )
        is False
    )


def test_should_process_model_select_change_requires_focus_and_not_syncing():
    assert (
        tui_app.should_process_model_select_change(
            value="openai|fast",
            model_select_syncing=False,
            has_focus=True,
            programmatic_value=None,
        )
        is True
    )
    assert (
        tui_app.should_process_model_select_change(
            value="openai|fast",
            model_select_syncing=True,
            has_focus=True,
            programmatic_value=None,
        )
        is False
    )
    assert (
        tui_app.should_process_model_select_change(
            value="openai|fast",
            model_select_syncing=False,
            has_focus=False,
            programmatic_value=None,
        )
        is False
    )
    assert (
        tui_app.should_process_model_select_change(
            value="openai|fast",
            model_select_syncing=False,
            has_focus=True,
            programmatic_value="openai|fast",
        )
        is False
    )


def test_should_ignore_stale_model_info_update_only_within_target_window():
    assert (
        tui_app.should_ignore_stale_model_info_update(
            incoming_value="openai|balanced",
            target_value="openai|deep",
            target_until_mono=12.0,
            now_mono=10.0,
        )
        is True
    )
    assert (
        tui_app.should_ignore_stale_model_info_update(
            incoming_value="openai|deep",
            target_value="openai|deep",
            target_until_mono=12.0,
            now_mono=10.0,
        )
        is False
    )
    assert (
        tui_app.should_ignore_stale_model_info_update(
            incoming_value="openai|balanced",
            target_value="openai|deep",
            target_until_mono=9.0,
            now_mono=10.0,
        )
        is False
    )
    assert (
        tui_app.should_ignore_stale_model_info_update(
            incoming_value="",
            target_value="openai|deep",
            target_until_mono=12.0,
            now_mono=10.0,
        )
        is False
    )


def test_should_ignore_model_select_reversion_during_target_window():
    assert (
        tui_app.should_ignore_model_select_reversion_during_target(
            requested_value="openai|balanced",
            current_value="openai|balanced",
            target_value="openai|deep",
            target_until_mono=12.0,
            now_mono=10.0,
        )
        is True
    )
    assert (
        tui_app.should_ignore_model_select_reversion_during_target(
            requested_value="openai|deep",
            current_value="openai|balanced",
            target_value="openai|deep",
            target_until_mono=12.0,
            now_mono=10.0,
        )
        is False
    )
    assert (
        tui_app.should_ignore_model_select_reversion_during_target(
            requested_value="openai|balanced",
            current_value="openai|balanced",
            target_value="openai|deep",
            target_until_mono=9.0,
            now_mono=10.0,
        )
        is False
    )


def test_spawn_swarmee_configures_subprocess_run_mode(monkeypatch):
    captured: dict[str, object] = {}
    fake_proc = object()

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    proc = tui_app.spawn_swarmee("hello from tui", auto_approve=True)

    assert proc is fake_proc
    command = captured["command"]
    kwargs = captured["kwargs"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1:4] == ["-u", "-m", "swarmee_river.swarmee"]
    assert "--yes" in command
    assert "hello from tui" in command
    assert kwargs["stdin"] is tui_app.subprocess.PIPE
    assert kwargs["stdout"] is tui_app.subprocess.PIPE
    assert kwargs["stderr"] is tui_app.subprocess.STDOUT
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["bufsize"] == 1
    assert kwargs["start_new_session"] is True
    env = kwargs["env"]
    assert isinstance(env, dict)
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["SWARMEE_SPINNERS"] == "0"
    assert env["WATCHFILES_FORCE_POLLING"] == "1"
    assert env["WATCHDOG_USE_POLLING"] == "1"
    assert "PYTHONWARNINGS" in env
    assert "Http_requestTool" in str(env["PYTHONWARNINGS"])


def test_spawn_swarmee_configures_subprocess_plan_mode(monkeypatch):
    captured: dict[str, object] = {}
    fake_proc = object()

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    proc = tui_app.spawn_swarmee("hello from tui", auto_approve=False)

    assert proc is fake_proc
    command = captured["command"]
    kwargs = captured["kwargs"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1:4] == ["-u", "-m", "swarmee_river.swarmee"]
    assert "--yes" not in command
    assert "hello from tui" in command
    assert kwargs["stdin"] is tui_app.subprocess.PIPE
    assert kwargs["stderr"] is tui_app.subprocess.STDOUT
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["start_new_session"] is True
    env = kwargs["env"]
    assert isinstance(env, dict)
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["SWARMEE_SPINNERS"] == "0"
    assert env["WATCHFILES_FORCE_POLLING"] == "1"
    assert env["WATCHDOG_USE_POLLING"] == "1"
    assert "PYTHONWARNINGS" in env
    assert "Http_requestTool" in str(env["PYTHONWARNINGS"])


def test_spawn_swarmee_sets_session_id_env_when_provided(monkeypatch):
    captured: dict[str, object] = {}
    fake_proc = object()

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    proc = tui_app.spawn_swarmee("hello", auto_approve=False, session_id="abc123")

    assert proc is fake_proc
    env = captured["kwargs"]["env"]
    assert isinstance(env, dict)
    assert env["SWARMEE_SESSION_ID"] == "abc123"


def test_spawn_swarmee_applies_env_overrides(monkeypatch):
    captured: dict[str, object] = {}
    fake_proc = object()

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    proc = tui_app.spawn_swarmee(
        "hello",
        auto_approve=False,
        env_overrides={"SWARMEE_MODEL_PROVIDER": "openai", "SWARMEE_MODEL_TIER": "fast"},
    )

    assert proc is fake_proc
    env = captured["kwargs"]["env"]
    assert isinstance(env, dict)
    assert env["SWARMEE_MODEL_PROVIDER"] == "openai"
    assert env["SWARMEE_MODEL_TIER"] == "fast"


def test_write_to_proc_writes_newline_and_flushes():
    class FakeStdin:
        def __init__(self) -> None:
            self.payload = ""
            self.flush_calls = 0

        def write(self, text: str) -> None:
            self.payload += text

        def flush(self) -> None:
            self.flush_calls += 1

    class FakeProc:
        def __init__(self) -> None:
            self.stdin = FakeStdin()

    proc = FakeProc()
    assert tui_app.write_to_proc(proc, "y") is True
    assert proc.stdin.payload == "y\n"
    assert proc.stdin.flush_calls == 1


def test_write_to_proc_returns_false_when_stdin_missing():
    class FakeProc:
        stdin = None

    assert tui_app.write_to_proc(FakeProc(), "n") is False


def test_send_daemon_command_writes_jsonl_and_flushes():
    class FakeStdin:
        def __init__(self) -> None:
            self.payload = ""
            self.flush_calls = 0

        def write(self, text: str) -> None:
            self.payload += text

        def flush(self) -> None:
            self.flush_calls += 1

    class FakeProc:
        def __init__(self) -> None:
            self.stdin = FakeStdin()

    proc = FakeProc()
    assert tui_app.send_daemon_command(proc, {"cmd": "interrupt"}) is True
    assert proc.stdin.payload == '{"cmd": "interrupt"}\n'
    assert proc.stdin.flush_calls == 1


def test_send_daemon_command_writes_set_profile_payload():
    class FakeStdin:
        def __init__(self) -> None:
            self.payload = ""
            self.flush_calls = 0

        def write(self, text: str) -> None:
            self.payload += text

        def flush(self) -> None:
            self.flush_calls += 1

    class FakeProc:
        def __init__(self) -> None:
            self.stdin = FakeStdin()

    proc = FakeProc()
    payload = {
        "cmd": "set_profile",
        "profile": {"id": "qa", "name": "QA", "active_sops": ["review"]},
    }
    assert tui_app.send_daemon_command(proc, payload) is True
    assert proc.stdin.payload == (
        '{"cmd": "set_profile", "profile": {"id": "qa", "name": "QA", "active_sops": ["review"]}}\n'
    )
    assert proc.stdin.flush_calls == 1


def test_send_daemon_command_uses_transport_sender_when_available():
    class FakeTransport:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def send_command(self, payload):
            self.calls.append(payload)
            return True

    transport = FakeTransport()
    assert tui_app.send_daemon_command(transport, {"cmd": "interrupt"}) is True
    assert transport.calls == [{"cmd": "interrupt"}]


def test_socket_transport_send_command_forwards_set_profile():
    class FakeClient:
        def __init__(self) -> None:
            self.commands: list[dict[str, object]] = []
            self.closed = False

        def send_command(self, payload):
            self.commands.append(payload)

        def close(self) -> None:
            self.closed = True

        def read_event(self):
            return None

    client = FakeClient()
    transport = tui_app._SocketTransport(client=client, session_id="sess-1", broker_pid=123)
    payload = {"cmd": "set_profile", "profile": {"id": "qa", "name": "QA"}}
    assert transport.send_command(payload) is True
    assert client.commands == [payload]


def test_socket_transport_connect_forwards_attach_env_overrides(tmp_path):
    discovery_path = tmp_path / "runtime.json"
    discovery_path.write_text("{}", encoding="utf-8")

    class FakeClient:
        def __init__(self) -> None:
            self.attach_payload: dict[str, object] | None = None

        def connect(self) -> None:
            return None

        def hello(self, *, client_name: str, surface: str):
            return {"event": "hello_ack", "pid": 321}

        def attach(self, *, session_id: str, cwd: str, env_overrides: dict[str, str] | None = None):
            self.attach_payload = {
                "session_id": session_id,
                "cwd": cwd,
                "env_overrides": dict(env_overrides or {}),
            }
            return {"event": "attached", "session_id": session_id}

        def close(self) -> None:
            return None

    fake_client = FakeClient()
    transport = tui_app._SocketTransport.connect(
        session_id="sess-1",
        cwd=tmp_path,
        client_name="test",
        surface="tui",
        env_overrides={"AWS_PROFILE": "ds-pr"},
        runtime_discovery_path_fn=lambda *, cwd=None: discovery_path,
        client_from_discovery_fn=lambda _path: fake_client,
    )

    assert transport.pid == 321
    assert fake_client.attach_payload is not None
    assert fake_client.attach_payload["env_overrides"] == {"AWS_PROFILE": "ds-pr"}


def test_spawn_daemon_retries_windows_broker_attach_before_fallback(monkeypatch, tmp_path):
    import swarmee_river.tui.mixins.daemon as daemon_mixin
    import swarmee_river.tui.transport as transport

    class _FakeThread:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.started = False

        def start(self) -> None:
            self.started = True

    class _FakeClient:
        def read_event(self):
            return None

        def close(self) -> None:
            return None

        def send_command(self, _payload: dict[str, object]) -> None:
            return None

    class _Harness(DaemonMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(
                    is_shutting_down=False,
                    proc=None,
                    session_id="sess-1",
                    ready=False,
                    pending_model_select_value=None,
                    runner_thread=None,
                )
            )
            self._test_transport_factory = None
            self._context_sources = []
            self._active_sop_names = []
            self.messages: list[str] = []
            self.saved = False

        def _model_env_overrides(self) -> dict[str, str]:
            return {"AWS_PROFILE": "ds-pr"}

        def _write_transcript_line(self, text: str) -> None:
            self.messages.append(text)

        def _save_session(self) -> None:
            self.saved = True

        def _stream_daemon_output(self, proc) -> None:  # noqa: ANN001
            _ = proc
            return None

    ensure_calls: list[dict[str, object]] = []

    def _fake_ensure_runtime_broker(*, cwd=None, timeout_s=6.0, poll_interval_s=0.1):
        ensure_calls.append({"cwd": cwd, "timeout_s": timeout_s, "poll_interval_s": poll_interval_s})
        return tmp_path / "runtime.json"

    connect_calls: list[dict[str, object]] = []
    socket_transport = tui_app._SocketTransport(client=_FakeClient(), session_id="sess-1", broker_pid=321)

    def _fake_connect(cls, **kwargs):  # noqa: ANN001
        connect_calls.append(dict(kwargs))
        if len(connect_calls) < 3:
            raise ConnectionRefusedError("broker not ready")
        return socket_transport

    spawn_calls: list[dict[str, object]] = []

    def _fake_spawn_swarmee_daemon(*, session_id: str | None = None, env_overrides: dict[str, str] | None = None):
        spawn_calls.append({"session_id": session_id, "env_overrides": dict(env_overrides or {})})
        raise AssertionError("local daemon fallback should not run when broker attach succeeds during retries")

    shutdown_calls: list[dict[str, object]] = []

    def _fake_shutdown_runtime_broker(*, cwd=None, timeout_s=6.0):
        shutdown_calls.append({"cwd": cwd, "timeout_s": timeout_s})
        return True

    monkeypatch.setattr(daemon_mixin, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(daemon_mixin, "ensure_runtime_broker", _fake_ensure_runtime_broker)
    monkeypatch.setattr(daemon_mixin._SocketTransport, "connect", classmethod(_fake_connect))
    monkeypatch.setattr(daemon_mixin.threading, "Thread", _FakeThread)
    monkeypatch.setattr(daemon_mixin.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr("swarmee_river.runtime_service.client.shutdown_runtime_broker", _fake_shutdown_runtime_broker)
    monkeypatch.setattr(transport, "spawn_swarmee_daemon", _fake_spawn_swarmee_daemon)

    harness = _Harness()
    harness._spawn_daemon()

    assert ensure_calls
    assert ensure_calls[0]["timeout_s"] == daemon_mixin._BROKER_STARTUP_TIMEOUT_WINDOWS_S
    assert len(connect_calls) == 3
    assert connect_calls[0]["env_overrides"] == {"AWS_PROFILE": "ds-pr"}
    assert spawn_calls == []
    assert shutdown_calls == []
    assert isinstance(harness.state.daemon.proc, tui_app._SocketTransport)
    assert harness.saved is True


def test_spawn_daemon_windows_falls_back_once_after_retry_exhaustion(monkeypatch, tmp_path):
    import swarmee_river.tui.mixins.daemon as daemon_mixin
    import swarmee_river.tui.transport as transport

    class _FakeThread:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def start(self) -> None:
            return None

    class _FakeProc:
        pid = 777
        stdin = None
        stdout = None

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 0

    class _Harness(DaemonMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(
                    is_shutting_down=False,
                    proc=None,
                    session_id="sess-2",
                    ready=False,
                    pending_model_select_value=None,
                    runner_thread=None,
                )
            )
            self._test_transport_factory = None
            self._context_sources = []
            self._active_sop_names = []
            self.messages: list[str] = []
            self.saved = False

        def _model_env_overrides(self) -> dict[str, str]:
            return {"AWS_PROFILE": "ds-pr", "AWS_REGION": "us-east-1"}

        def _write_transcript_line(self, text: str) -> None:
            self.messages.append(text)

        def _save_session(self) -> None:
            self.saved = True

        def _stream_daemon_output(self, proc) -> None:  # noqa: ANN001
            _ = proc
            return None

    def _fake_ensure_runtime_broker(*, cwd=None, timeout_s=6.0, poll_interval_s=0.1):
        _ = (cwd, timeout_s, poll_interval_s)
        return tmp_path / "runtime.json"

    connect_calls = 0

    def _fake_connect(cls, **kwargs):  # noqa: ANN001
        _ = (cls, kwargs)
        nonlocal connect_calls
        connect_calls += 1
        raise ConnectionRefusedError("still booting")

    spawn_calls: list[dict[str, object]] = []

    def _fake_spawn_swarmee_daemon(*, session_id: str | None = None, env_overrides: dict[str, str] | None = None):
        spawn_calls.append({"session_id": session_id, "env_overrides": dict(env_overrides or {})})
        return _FakeProc()

    shutdown_calls: list[dict[str, object]] = []

    def _fake_shutdown_runtime_broker(*, cwd=None, timeout_s=6.0):
        shutdown_calls.append({"cwd": cwd, "timeout_s": timeout_s})
        return True

    monkeypatch.setattr(daemon_mixin, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(daemon_mixin, "ensure_runtime_broker", _fake_ensure_runtime_broker)
    monkeypatch.setattr(daemon_mixin._SocketTransport, "connect", classmethod(_fake_connect))
    monkeypatch.setattr(daemon_mixin.threading, "Thread", _FakeThread)
    monkeypatch.setattr(daemon_mixin.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr("swarmee_river.runtime_service.client.shutdown_runtime_broker", _fake_shutdown_runtime_broker)
    monkeypatch.setattr(transport, "spawn_swarmee_daemon", _fake_spawn_swarmee_daemon)

    harness = _Harness()
    harness._spawn_daemon()

    assert connect_calls == daemon_mixin._BROKER_ATTACH_ATTEMPTS_WINDOWS
    assert len(spawn_calls) == 1
    assert len(shutdown_calls) == 1
    assert spawn_calls[0]["env_overrides"] == {"AWS_PROFILE": "ds-pr", "AWS_REGION": "us-east-1"}
    assert isinstance(harness.state.daemon.proc, transport._SubprocessTransport)
    assert harness.saved is True


def test_spawn_daemon_windows_fallback_continues_when_broker_shutdown_raises(monkeypatch, tmp_path):
    import swarmee_river.tui.mixins.daemon as daemon_mixin
    import swarmee_river.tui.transport as transport

    class _FakeThread:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def start(self) -> None:
            return None

    class _FakeProc:
        pid = 778
        stdin = None
        stdout = None

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 0

    class _Harness(DaemonMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(
                    is_shutting_down=False,
                    proc=None,
                    session_id="sess-3",
                    ready=False,
                    pending_model_select_value=None,
                    runner_thread=None,
                )
            )
            self._test_transport_factory = None
            self._context_sources = []
            self._active_sop_names = []
            self.messages: list[str] = []
            self.saved = False

        def _model_env_overrides(self) -> dict[str, str]:
            return {"AWS_PROFILE": "ds-pr"}

        def _write_transcript_line(self, text: str) -> None:
            self.messages.append(text)

        def _save_session(self) -> None:
            self.saved = True

        def _stream_daemon_output(self, proc) -> None:  # noqa: ANN001
            _ = proc
            return None

    monkeypatch.setattr(daemon_mixin, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(daemon_mixin, "ensure_runtime_broker", lambda **_kwargs: tmp_path / "runtime.json")
    monkeypatch.setattr(
        daemon_mixin._SocketTransport,
        "connect",
        classmethod(lambda cls, **kwargs: (_ for _ in ()).throw(ConnectionRefusedError("still booting"))),
    )
    monkeypatch.setattr(daemon_mixin.threading, "Thread", _FakeThread)
    monkeypatch.setattr(daemon_mixin.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "swarmee_river.runtime_service.client.shutdown_runtime_broker",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("shutdown failed")),
    )
    monkeypatch.setattr(
        transport,
        "spawn_swarmee_daemon",
        lambda **_kwargs: _FakeProc(),
    )

    harness = _Harness()
    harness._spawn_daemon()

    assert isinstance(harness.state.daemon.proc, transport._SubprocessTransport)
    assert harness.saved is True


def test_detect_consent_prompt_matches_cli_prompt():
    assert tui_app.detect_consent_prompt("~ consent> ") is not None
    assert tui_app.detect_consent_prompt("Allow tool 'shell'? [y/n]") is not None
    assert tui_app.detect_consent_prompt("normal output line") is None


def test_parse_output_line_extracts_artifact_from_truncation_line():
    line = "[tool result truncated: kept 2000 chars of 9000; full output saved to .swarmee/artifacts/abc123.txt]"
    event = tui_app.parse_output_line(line)
    assert event is not None
    assert event.kind == "artifact"
    assert event.meta == {"path": ".swarmee/artifacts/abc123.txt"}


def test_parse_output_line_extracts_patch_path():
    event = tui_app.parse_output_line("patch: /tmp/swarmee/patches/plan.patch")
    assert event is not None
    assert event.kind == "artifact"
    assert event.meta == {"path": "/tmp/swarmee/patches/plan.patch"}


def test_parse_output_line_returns_none_for_unmatched_line():
    assert tui_app.parse_output_line("regular assistant output") is None


def test_parse_output_line_classifies_provider_fallback_as_noise():
    event = tui_app.parse_output_line(
        "[provider] No AWS credentials detected for Bedrock; falling back to OpenAI because OPENAI_API_KEY is set."
    )
    assert event is not None
    assert event.kind == "noise"


def test_parse_output_line_classifies_homebrew_ps_warning():
    event = tui_app.parse_output_line(
        "/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted"
    )
    assert event is not None
    assert event.kind == "warning"


def test_parse_output_line_classifies_traceback_and_fsevents_lines():
    head = tui_app.parse_output_line("Traceback (most recent call last):")
    frame = tui_app.parse_output_line('  File "/tmp/x.py", line 12, in <module>')
    fsevents = tui_app.parse_output_line("SystemError: Cannot start fsevents stream. Use polling observer.")

    assert head is not None
    assert head.kind == "warning"
    assert frame is not None
    assert frame.kind == "noise"
    assert fsevents is not None
    assert fsevents.kind == "warning"


def test_sanitize_output_text_strips_ansi_and_carriage_returns():
    text = "hello\r\n\x1b[31mred\x1b[0m"
    assert tui_app.sanitize_output_text(text) == "hello\nred"


def test_sanitize_output_text_strips_osc_title_sequences():
    text = "a\x1b]0;Python/git/rg\x07b\x1b]2;tool run\x1b\\c"
    assert tui_app.sanitize_output_text(text) == "abc"


def test_update_consent_capture_collects_recent_lines():
    consent_active = False
    consent_buffer: list[str] = []
    lines = [
        "some normal line",
        "Allow tool 'shell'? [y/n/a/v]",
        "context line 1",
        "~ consent> ",
        "context line 2",
    ]
    for line in lines:
        consent_active, consent_buffer = tui_app.update_consent_capture(
            consent_active,
            consent_buffer,
            line,
            max_lines=3,
        )

    assert consent_active is True
    assert consent_buffer == ["context line 1", "~ consent> ", "context line 2"]


def test_add_recent_artifacts_dedupes_and_caps():
    existing = ["a.txt", "b.txt", "c.txt"]
    updated = tui_app.add_recent_artifacts(existing, ["b.txt", "d.txt", "e.txt"], max_items=4)
    assert updated == ["c.txt", "b.txt", "d.txt", "e.txt"]


def test_normalize_artifact_index_entry_keeps_kind_id_timestamp_and_path():
    normalized = tui_app.normalize_artifact_index_entry(
        {
            "id": "abc123",
            "kind": "tui_transcript",
            "path": "/tmp/artifacts/a.txt",
            "created_at": "2026-02-23T10:00:00",
            "meta": {"name": "Run transcript"},
        }
    )
    assert normalized is not None
    assert normalized["id"] == "abc123"
    assert normalized["name"] == "Run transcript"
    assert normalized["kind"] == "tui_transcript"
    assert normalized["created_at"] == "2026-02-23T10:00:00"
    assert normalized["path"] == "/tmp/artifacts/a.txt"


def test_build_artifact_sidebar_items_renders_required_fields():
    items = tui_app.build_artifact_sidebar_items(
        [
            {
                "id": "abc123",
                "kind": "tool_result",
                "path": "/tmp/artifacts/tool.txt",
                "created_at": "2026-02-23T10:01:00",
                "meta": {"name": "Tool output"},
            }
        ]
    )
    assert len(items) == 1
    item = items[0]
    assert item["title"] == "Tool output"
    assert "tool_result" in item["subtitle"]
    assert "/tmp/artifacts/tool.txt" in item["subtitle"]


def test_artifact_context_source_payload_builds_file_source():
    payload = tui_app.artifact_context_source_payload("/tmp/artifacts/tool.txt", source_id="artifact-tool")
    assert payload == {
        "type": "file",
        "path": "/tmp/artifacts/tool.txt",
        "id": "artifact-tool",
    }


def test_build_session_issue_sidebar_items_marks_warning_and_error_states():
    items = tui_app.build_session_issue_sidebar_items(
        [
            {
                "id": "w1",
                "severity": "warning",
                "title": "Rate limit warning",
                "text": "WARN: throttling",
                "created_at": "2026-02-23 10:00:00",
            },
            {
                "id": "e1",
                "severity": "error",
                "title": "Tool failed",
                "text": "ERROR: tool shell failed (error) [tool-123]",
                "created_at": "2026-02-23 10:01:00",
            },
        ]
    )
    assert len(items) == 2
    assert items[0]["state"] == "warning"
    assert items[1]["state"] == "error"
    # Subtitle now uses relative time instead of raw timestamp
    assert "ago" in items[1]["subtitle"] or items[1]["subtitle"]


def test_render_session_issue_detail_text_includes_tool_metadata():
    text = tui_app.render_session_issue_detail_text(
        {
            "severity": "error",
            "title": "Tool Failed: shell",
            "created_at": "2026-02-23 10:02:00",
            "text": "ERROR: tool shell failed (error) [tool-123]",
            "tool_use_id": "tool-123",
            "tool_name": "shell",
            "next_tier": "deep",
        }
    )
    assert "Severity: error" in text
    assert "Tool Use ID: tool-123" in text
    assert "Suggested tier: deep" in text


def test_session_issue_actions_for_tool_failure_include_recovery_buttons():
    actions = tui_app.session_issue_actions(
        {
            "category": "tool_failure",
            "tool_use_id": "tool-123",
        }
    )
    action_ids = [item["id"] for item in actions]
    assert action_ids == [
        "session_issue_retry_tool",
        "session_issue_skip_tool",
        "session_issue_escalate_tier",
        "session_issue_interrupt",
    ]


def test_build_session_timeline_sidebar_items_populates_from_index_events():
    index = {
        "events": [
            {
                "id": "timeline-1",
                "event": "after_tool_call",
                "tool": "shell",
                "duration_s": 2.3,
                "success": False,
                "error": "exit 1",
                "ts": "2026-02-23T12:00:00",
            },
            {
                "id": "timeline-2",
                "event": "after_model_call",
                "duration_s": 0.9,
                "ts": "2026-02-23T12:00:01",
            },
        ]
    }

    items = tui_app.build_session_timeline_sidebar_items(index["events"])

    # Both tool calls and model calls are shown in the timeline.
    assert len(items) == 2
    assert items[0]["id"] == "timeline-1"
    assert "shell" in items[0]["title"]
    assert "error" in items[0]["title"]
    assert items[0]["state"] == "error"
    # Model call appears second
    assert items[1]["id"] == "timeline-2"
    assert "LLM call" in items[1]["title"]
    assert items[1]["state"] == "default"
    # Subtitle uses relative time
    assert "ago" in items[0]["subtitle"]


def test_model_call_with_usage_shows_token_summary_in_sidebar():
    events = [
        {
            "id": "m1",
            "event": "after_model_call",
            "duration_s": 1.2,
            "ts": "2026-02-23T12:00:00",
            "usage": {"input_tokens": 3200, "output_tokens": 800, "cache_read_input_tokens": 2100},
        },
    ]
    items = tui_app.build_session_timeline_sidebar_items(events)
    assert len(items) == 1
    assert "3.2k in" in items[0]["title"]
    assert "800 out" in items[0]["title"]
    assert "2.1k cached" in items[0]["title"]


def test_relative_time_treats_naive_timestamps_as_local_time():
    """Naive ISO timestamps (no TZ indicator) should be treated as local time,
    matching the jsonl_logger which uses time.localtime().  A timestamp generated
    'just now' in local time must show seconds-ago, not hours-ago."""
    import time as _time

    # Generate a timestamp the same way jsonl_logger does: local time, no TZ indicator
    local_ts = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime())
    events = [
        {
            "id": "m1",
            "event": "after_model_call",
            "duration_s": 0.5,
            "ts": local_ts,
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
    ]
    items = tui_app.build_session_timeline_sidebar_items(events)
    assert len(items) == 1
    subtitle = items[0]["subtitle"]
    # Should be "Xs ago" or "just now", definitely not hours/days ago
    assert subtitle.endswith("s ago") or subtitle == "just now", (
        f"Expected seconds-ago or 'just now' for a just-generated local timestamp, got: {subtitle!r}"
    )


def test_model_call_detail_renders_token_and_composition_sections():
    event = {
        "id": "m1",
        "event": "after_model_call",
        "duration_s": 1.5,
        "model_call_id": "abc123",
        "model_id": "anthropic.claude-3-5-sonnet",
        "usage": {
            "input_tokens": 5000,
            "output_tokens": 1200,
            "cache_read_input_tokens": 3500,
        },
        "system_prompt_chars": 8400,
        "tool_count": 25,
        "tool_schema_chars": 32000,
        "messages": 12,
        "message_breakdown": {"user": 4, "assistant": 3, "tool": 5},
    }
    detail = tui_app.render_session_timeline_detail_text(event)
    assert "Token Usage" in detail
    assert "5,000" in detail
    assert "1,200" in detail
    assert "3,500" in detail
    assert "70%" in detail  # 3500/5000 = 70%
    assert "Context Composition" in detail
    assert "8,400" in detail
    assert "32,000" in detail
    assert "25 tools" in detail
    assert "user=4" in detail
    assert "Metadata" in detail
    assert "anthropic.claude-3-5-sonnet" in detail


def test_model_call_detail_degrades_gracefully_without_usage():
    event = {
        "id": "m1",
        "event": "after_model_call",
        "duration_s": 0.8,
    }
    detail = tui_app.render_session_timeline_detail_text(event)
    assert "no usage data" in detail
    assert "LLM call" in detail


def test_model_call_with_camel_case_usage_shows_token_summary_in_sidebar():
    """Anthropic/Bedrock providers return camelCase usage keys (inputTokens, outputTokens)."""
    events = [
        {
            "id": "m1",
            "event": "after_model_call",
            "duration_s": 1.0,
            "ts": "2026-02-23T12:00:00",
            "usage": {"inputTokens": 4500, "outputTokens": 900, "cacheReadInputTokens": 3000},
        },
    ]
    items = tui_app.build_session_timeline_sidebar_items(events)
    assert len(items) == 1
    assert "4.5k in" in items[0]["title"]
    assert "900 out" in items[0]["title"]
    assert "3.0k cached" in items[0]["title"]


def test_model_call_detail_renders_camel_case_usage():
    """Anthropic/Bedrock camelCase usage keys render correctly in the detail view."""
    event = {
        "id": "m1",
        "event": "after_model_call",
        "duration_s": 1.5,
        "model_call_id": "abc456",
        "model_id": "anthropic.claude-sonnet-4-6",
        "usage": {
            "inputTokens": 5000,
            "outputTokens": 1200,
            "cacheReadInputTokens": 3500,
        },
        "system_prompt_chars": 8400,
        "tool_count": 25,
        "tool_schema_chars": 32000,
    }
    detail = tui_app.render_session_timeline_detail_text(event)
    assert "Token Usage" in detail
    assert "5,000" in detail
    assert "1,200" in detail
    assert "3,500" in detail
    assert "70%" in detail  # 3500/5000 = 70%
    assert "Context Composition" in detail
    assert "8,400" in detail


def test_session_timeline_detail_and_actions_render_without_crashing():
    event = {
        "id": "timeline-1",
        "event": "after_invocation",
        "duration_s": 4.2,
        "ts": "2026-02-23T12:03:00",
    }
    detail = tui_app.render_session_timeline_detail_text(event)
    actions = tui_app.session_timeline_actions(event)

    assert "Summary: invocation (4.2s)" in detail
    assert "Payload:" in detail
    assert [item["id"] for item in actions] == [
        "session_timeline_copy_json",
        "session_timeline_copy_summary",
    ]


def test_normalize_session_view_mode_switches_between_issues_and_timeline():
    assert tui_app.normalize_session_view_mode("timeline") == "timeline"
    assert tui_app.normalize_session_view_mode("issues") == "issues"
    # Invalid/blank modes should safely default to timeline.
    assert tui_app.normalize_session_view_mode("unknown") == "timeline"
    assert tui_app.normalize_session_view_mode("") == "timeline"


def test_normalize_agent_studio_view_mode_handles_known_and_unknown_values():
    assert tui_app.normalize_agent_studio_view_mode("overview") == "overview"
    assert tui_app.normalize_agent_studio_view_mode("builder") == "builder"
    assert tui_app.normalize_agent_studio_view_mode("BUILDER") == "builder"
    assert tui_app.normalize_agent_studio_view_mode("unknown") == "overview"


def test_activated_agent_helpers_filter_render_and_default():
    items = tui_app.build_activated_agent_sidebar_items(
        [
            {
                "id": "triage-research",
                "name": "Triage Research",
                "summary": "Investigates incoming issues",
                "prompt": "You triage incidents.",
                "provider": "openai",
                "tier": "balanced",
                "tool_names": ["file_read", "shell", "shell"],
                "sop_names": ["incident-triage"],
                "knowledge_base_id": "kb-123",
                "activated": True,
            },
            {
                "id": "draft-agent",
                "name": "Draft Agent",
                "activated": False,
            },
        ]
    )
    assert [item["id"] for item in items] == ["triage-research"]
    assert items[0]["state"] == "active"
    assert "Investigates incoming issues" in items[0]["subtitle"]

    detail = tui_app.render_activated_agent_detail_text(items[0])
    assert "Activated Agent" in detail
    assert "ID: triage-research" in detail
    assert "Tools: file_read, shell" in detail
    assert "SOPs: incident-triage" in detail

    empty_items = tui_app.build_activated_agent_sidebar_items([])
    assert [item["id"] for item in empty_items] == ["activated_agents_none"]
    assert "No agents are currently activated" in tui_app.render_activated_agent_detail_text(empty_items[0])


def test_build_swarm_agent_specs_and_run_prompt_from_activated_agents():
    from swarmee_river.tui.agent_studio import build_swarm_agent_specs

    agents = [
        {
            "id": "triage-research",
            "name": "Triage Research",
            "prompt": "You triage incidents.",
            "provider": "openai",
            "tier": "balanced",
            "tool_names": ["file_read", "shell"],
            "activated": True,
        },
        {
            "id": "inactive",
            "name": "Inactive Agent",
            "prompt": "ignored",
            "activated": False,
        },
    ]
    specs = build_swarm_agent_specs(agents)
    assert specs == [
        {
            "name": "Triage Research",
            "system_prompt": "You triage incidents.",
            "tools": ["file_read", "shell"],
            "model_provider": "openai",
            "model_settings": {"tier": "balanced"},
        }
    ]

    prompt = tui_app.build_activated_agents_run_prompt(agents, task="Handle this incident.")
    assert "Run activated agents with a single `swarm` tool call." in prompt
    assert "task: Handle this incident." in prompt
    assert '"name": "Triage Research"' in prompt
    assert tui_app.build_activated_agents_run_prompt([], task="ignored") == ""


def test_agent_tools_safety_policy_lens_helpers_render_effective_state():
    lens = tui_app.build_agent_policy_lens(
        tier_name="fast",
        overrides={
            "tool_consent": "deny",
            "tool_allowlist": ["shell", "shell", " file_read "],
            "tool_blocklist": ["editor"],
        },
    )
    tools_items = tui_app.build_agent_tools_safety_sidebar_items(lens)
    team_items = tui_app.build_agent_team_sidebar_items(
        [
            {
                "id": "triage-team",
                "name": "Triage Team",
                "description": "Incident response composition",
                "spec": {"swarm": {"agents": [{"id": "lead"}]}},
            }
        ]
    )

    assert [item["id"] for item in tools_items] == ["policy_lens", "session_overrides"]
    assert [item["id"] for item in team_items] == ["triage-team"]
    assert lens["effective"]["tool_consent"] == "deny"
    assert lens["effective"]["tool_allowlist"] == ["shell", "file_read"]
    assert "Policy Lens" in tui_app.render_agent_tools_safety_detail_text(tools_items[0], lens)
    assert "Triage Team" in tui_app.render_agent_team_detail_text(team_items[0])


def test_normalize_session_safety_overrides_discards_invalid_and_dedupes():
    normalized = tui_app.normalize_session_safety_overrides(
        {
            "tool_consent": "ALLOW",
            "tool_allowlist": ["shell", "shell", "bash", ""],
            "tool_blocklist": ["editor", " editor "],
            "extra": "ignored",
        }
    )
    assert normalized == {
        "tool_consent": "allow",
        "tool_allowlist": ["shell", "bash"],
        "tool_blocklist": ["editor"],
    }


def test_normalize_team_presets_dedupes_and_discards_invalid():
    normalized = tui_app.normalize_team_presets(
        [
            {"id": "alpha", "name": "Alpha", "description": "  Core ", "spec": {"swarm": {"workers": 2}}},
            {"name": "Bravo Team", "spec": {"agent_graph": {"nodes": 3}}},
            {"id": "alpha", "name": "Duplicate", "spec": {"ignored": True}},
            {"id": "bad", "name": "Bad", "spec": ["not", "dict"]},
        ]
    )
    assert [item["id"] for item in normalized] == ["alpha", "Bravo-Team"]
    assert normalized[0]["description"] == "Core"
    assert normalized[1]["spec"] == {"agent_graph": {"nodes": 3}}


def test_build_team_preset_run_prompt_is_deterministic():
    prompt = tui_app.build_team_preset_run_prompt(
        {
            "id": "triage-team",
            "name": "Triage Team",
            "description": "Incident response",
            "spec": {"swarm": {"agents": [{"id": "lead"}], "max_steps": 3}},
        }
    )

    assert "Run team preset 'Triage Team' (id: triage-team)." in prompt
    assert "Call the `swarm` tool exactly once" in prompt
    assert '"max_steps": 3' in prompt


def test_run_tui_compose_and_agent_studio_switch_smoke(monkeypatch):
    import textual.app as textual_app
    from textual.message_pump import active_app

    captured = {"composed": False, "switched": False}

    def _fake_run(self, *args, **kwargs):
        token = active_app.set(self)
        self._compose_stacks.append([])
        self._composed.append([])
        try:
            list(self.compose())
        finally:
            self._compose_stacks.pop()
            self._composed.pop()
            active_app.reset(token)
        captured["composed"] = True
        self._set_agent_studio_view_mode("overview")
        self._set_agent_studio_view_mode("builder")
        self._set_agent_studio_view_mode("invalid")
        captured["switched"] = True

    monkeypatch.setattr(textual_app.App, "run", _fake_run)
    result = tui_app.run_tui()

    assert result == 0
    assert captured["composed"] is True
    assert captured["switched"] is True


def test_run_tui_responsive_layout_class_switching(monkeypatch):
    import textual.app as textual_app
    from textual.message_pump import active_app

    captured = {"wide": False, "medium": False, "narrow": False}

    def _fake_run(self, *args, **kwargs):
        token = active_app.set(self)
        self._compose_stacks.append([])
        self._composed.append([])
        try:
            list(self.compose())
        finally:
            self._compose_stacks.pop()
            self._composed.pop()
            active_app.reset(token)

        self._sidebar_width = lambda: 50
        self._update_responsive_layout_classes()
        captured["wide"] = self.has_class("layout-wide")

        self._sidebar_width = lambda: 40
        self._update_responsive_layout_classes()
        captured["medium"] = self.has_class("layout-medium")

        self._sidebar_width = lambda: 30
        self._update_responsive_layout_classes()
        captured["narrow"] = self.has_class("layout-narrow")

    monkeypatch.setattr(textual_app.App, "run", _fake_run)
    result = tui_app.run_tui()

    assert result == 0
    assert captured["wide"] is True
    assert captured["medium"] is True
    assert captured["narrow"] is True


def test_tui_state_defaults_use_plan_and_tools_modes():
    state = AppState()
    assert state.engage_view_mode == "plan"
    assert state.tooling_view_mode == "tools"
    assert state.tooling.view_mode == "tools"


def test_set_engage_view_mode_maps_legacy_values_to_plan():
    class _Styles:
        def __init__(self) -> None:
            self.display = "none"

    class _View:
        def __init__(self) -> None:
            self.styles = _Styles()

    class _Button:
        def __init__(self) -> None:
            self.variant = "default"

    class _Dummy(ContextSourcesMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self._engage_plan_view = _View()
            self._engage_session_view = _View()
            self._engage_view_plan_button = _Button()
            self._engage_view_session_button = _Button()

    app = _Dummy()
    app._set_engage_view_mode("execution")
    assert app.state.engage_view_mode == "plan"
    assert app._engage_plan_view.styles.display == "block"
    assert app._engage_view_plan_button.variant == "primary"

    app._set_engage_view_mode("planning")
    assert app.state.engage_view_mode == "plan"
    assert app._engage_plan_view.styles.display == "block"
    assert app._engage_session_view.styles.display == "none"


def test_tooling_prompt_new_creates_selected_stub_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))

    class _Editor:
        def __init__(self) -> None:
            self.text = ""
            self.focused = False

        def focus(self) -> None:
            self.focused = True

    class _Dummy(ContextSourcesMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self._tooling_prompt_content_input = _Editor()
            self._tooling_prompts_table = None
            self._notify_messages: list[str] = []
            self._transcript_lines: list[str] = []

        def _notify(self, text: str, **_kwargs) -> None:
            self._notify_messages.append(text)

        def _write_transcript_line(self, text: str) -> None:
            self._transcript_lines.append(text)

    from swarmee_river.prompt_assets import load_prompt_assets

    app = _Dummy()
    app._tooling_prompt_new()

    assets = load_prompt_assets()
    created = [item for item in assets if item.id.startswith("new_prompt_")]
    assert len(created) == 1
    assert created[0].name == "New Prompt"
    assert created[0].content == ""
    assert app.state.tooling.prompt_selected_id == created[0].id
    assert app._tooling_prompt_content_input.focused is True


def test_tooling_prompt_save_updates_selected_content_only(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))

    class _Editor:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Dummy(ContextSourcesMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.tooling.prompt_selected_id = "existing_prompt"
            self._tooling_prompt_content_input = _Editor("Updated content")
            self._tooling_prompts_table = None
            self._notify_messages: list[str] = []
            self._transcript_lines: list[str] = []
            self._refresh_count = 0

        def _notify(self, text: str, **_kwargs) -> None:
            self._notify_messages.append(text)

        def _write_transcript_line(self, text: str) -> None:
            self._transcript_lines.append(text)

        def _refresh_tooling_prompts_list(self) -> None:
            self._refresh_count += 1

    from swarmee_river.prompt_assets import PromptAsset, load_prompt_assets, save_prompt_assets

    save_prompt_assets(
        [
            PromptAsset(
                id="existing_prompt",
                name="Existing Prompt",
                content="Initial",
                tags=["a"],
                source="project",
            )
        ]
    )
    app = _Dummy()
    app._tooling_prompt_save()

    assets = {item.id: item for item in load_prompt_assets()}
    assert assets["existing_prompt"].content == "Updated content"
    assert assets["existing_prompt"].name == "Existing Prompt"
    assert assets["existing_prompt"].tags == ["a"]
    assert app._refresh_count == 1


def test_tooling_prompt_metadata_edit_updates_id_tags_and_agent_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))

    class _Dummy(ContextSourcesMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.agent_studio.agents = [
                {
                    "id": "orchestrator",
                    "name": "Orchestrator",
                    "prompt_refs": ["legacy_prompt"],
                    "activated": False,
                }
            ]
            self._tooling_prompts_table = None
            self._notify_messages: list[str] = []
            self._refresh_count = 0
            self._draft_notes: list[str] = []
            self._builder_refresh_count = 0
            self._overview_refresh_count = 0

        def _notify(self, text: str, **_kwargs) -> None:
            self._notify_messages.append(text)

        def _refresh_tooling_prompts_list(self) -> None:
            self._refresh_count += 1

        def _set_agent_draft_dirty(self, _dirty: bool, *, note: str | None = None) -> None:
            if note:
                self._draft_notes.append(note)

        def _render_agent_builder_panel(self) -> None:
            self._builder_refresh_count += 1

        def _render_agent_overview_panel(self) -> None:
            self._overview_refresh_count += 1

    from swarmee_river.prompt_assets import PromptAsset, load_prompt_assets, save_prompt_assets

    save_prompt_assets(
        [
            PromptAsset(
                id="legacy_prompt",
                name="Legacy Prompt",
                content="Body",
                tags=[],
                source="project",
            )
        ]
    )
    app = _Dummy()
    app._tooling_prompt_apply_metadata_edit("legacy_prompt", "id", "Renamed Prompt")
    app._tooling_prompt_apply_metadata_edit("renamed-prompt", "tags", "alpha, beta, alpha")

    assets = {item.id: item for item in load_prompt_assets()}
    assert "renamed-prompt" in assets
    assert assets["renamed-prompt"].tags == ["alpha", "beta"]
    assert app.state.agent_studio.agents[0]["prompt_refs"] == ["renamed-prompt"]
    assert app.state.tooling.prompt_selected_id == "renamed-prompt"
    assert app._refresh_count == 2


def test_tooling_prompt_used_by_edit_toggles_prompt_refs():
    class _Dummy(ContextSourcesMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.agent_studio.agents = [
                {
                    "id": "orchestrator",
                    "name": "Orchestrator",
                    "prompt_refs": ["orchestrator_base"],
                    "activated": False,
                },
                {"id": "writer", "name": "Writer", "prompt_refs": [], "activated": True},
            ]
            self._refresh_count = 0
            self._dirty_notes: list[str] = []

        def _set_agent_draft_dirty(self, _dirty: bool, *, note: str | None = None) -> None:
            if note:
                self._dirty_notes.append(note)

        def _render_agent_builder_panel(self) -> None:
            return

        def _render_agent_overview_panel(self) -> None:
            return

        def _refresh_tooling_prompts_list(self) -> None:
            self._refresh_count += 1

    app = _Dummy()
    app._tooling_prompt_apply_used_by_edit("shared_prompt", ["writer", "orchestrator"])
    agents = {str(agent.get("id")): agent for agent in app.state.agent_studio.agents}
    assert "shared_prompt" in agents["orchestrator"]["prompt_refs"]
    assert "shared_prompt" in agents["writer"]["prompt_refs"]
    assert app._refresh_count == 1


def test_builder_tools_editor_uses_tooling_catalog_and_allows_custom():
    from swarmee_river.tui.widgets import CatalogMultiSelectScreen

    class _Dummy(AgentStudioMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.tooling.tool_catalog = [{"name": "shell"}, {"name": "git"}]
            self._agent_builder_tools_draft = ["shell", "custom_tool"]
            self._agent_builder_sops_draft = []
            self._agent_builder_kb_draft = ""
            self._agent_builder_tools_summary = None
            self._agent_builder_sops_summary = None
            self._agent_builder_kb_summary = None
            self.pushed_screen = None
            self.screen_callback = None
            self.dirty_values: list[bool] = []
            self.status_messages: list[str] = []

        def push_screen(self, screen, callback=None):  # noqa: ANN001
            self.pushed_screen = screen
            self.screen_callback = callback

        def _set_agent_builder_status(self, message: str) -> None:
            self.status_messages.append(message)

        def _set_agent_draft_dirty(self, dirty: bool, *, note: str | None = None) -> None:
            self.dirty_values.append(dirty)

    app = _Dummy()
    app._edit_agent_builder_capability("tools")
    assert isinstance(app.pushed_screen, CatalogMultiSelectScreen)
    assert app.pushed_screen._options == ["shell", "git"]
    assert app.pushed_screen._selected_values == ["shell", "custom_tool"]

    assert callable(app.screen_callback)
    app.screen_callback(["git", "custom_two", "git"])
    assert app._agent_builder_tools_draft == ["git", "custom_two"]
    assert app.dirty_values[-1] is True


def test_builder_sops_editor_uses_tooling_catalog_and_allows_custom():
    from swarmee_river.tui.widgets import CatalogMultiSelectScreen

    class _Dummy(AgentStudioMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.tooling.sop_catalog = [{"name": "incident_response"}, {"name": "qa_pass"}]
            self._agent_builder_tools_draft = []
            self._agent_builder_sops_draft = ["incident_response", "custom_sop"]
            self._agent_builder_kb_draft = ""
            self._agent_builder_tools_summary = None
            self._agent_builder_sops_summary = None
            self._agent_builder_kb_summary = None
            self.pushed_screen = None
            self.screen_callback = None
            self.dirty_values: list[bool] = []

        def push_screen(self, screen, callback=None):  # noqa: ANN001
            self.pushed_screen = screen
            self.screen_callback = callback

        def _set_agent_builder_status(self, _message: str) -> None:
            return

        def _set_agent_draft_dirty(self, dirty: bool, *, note: str | None = None) -> None:
            self.dirty_values.append(dirty)

    app = _Dummy()
    app._edit_agent_builder_capability("sops")
    assert isinstance(app.pushed_screen, CatalogMultiSelectScreen)
    assert app.pushed_screen._options == ["incident_response", "qa_pass"]
    assert app.pushed_screen._selected_values == ["incident_response", "custom_sop"]

    assert callable(app.screen_callback)
    app.screen_callback(["qa_pass", "custom_sop_2"])
    assert app._agent_builder_sops_draft == ["qa_pass", "custom_sop_2"]
    assert app.dirty_values[-1] is True


def test_builder_kb_editor_uses_tooling_kb_catalog_and_custom_seed():
    from swarmee_river.tui.widgets import CatalogSingleSelectScreen

    class _Dummy(AgentStudioMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.tooling.kb_entries = [{"id": "kb-primary", "name": "Primary KB"}]
            self._agent_builder_tools_draft = []
            self._agent_builder_sops_draft = []
            self._agent_builder_kb_draft = "kb-custom"
            self._agent_builder_tools_summary = None
            self._agent_builder_sops_summary = None
            self._agent_builder_kb_summary = None
            self.pushed_screen = None
            self.screen_callback = None
            self.dirty_values: list[bool] = []

        def push_screen(self, screen, callback=None):  # noqa: ANN001
            self.pushed_screen = screen
            self.screen_callback = callback

        def _set_agent_builder_status(self, _message: str) -> None:
            return

        def _set_agent_draft_dirty(self, dirty: bool, *, note: str | None = None) -> None:
            self.dirty_values.append(dirty)

    app = _Dummy()
    app._edit_agent_builder_capability("kb")
    assert isinstance(app.pushed_screen, CatalogSingleSelectScreen)
    assert app.pushed_screen._options == [("Primary KB (kb-primary)", "kb-primary")]
    assert app.pushed_screen._initial_custom_value == "kb-custom"

    assert callable(app.screen_callback)
    app.screen_callback("kb-primary")
    assert app._agent_builder_kb_draft == "kb-primary"
    assert app.dirty_values[-1] is True


def test_tools_row_selection_does_not_open_legacy_tag_editor():
    class _Table:
        pass

    class _Harness:
        def __init__(self) -> None:
            self._tooling_tools_table = _Table()
            self._tooling_sops_table = None
            self._tooling_prompts_table = None
            self._tooling_kbs_table = None
            self._session_timeline_table = None
            self._session_artifacts_table = None
            self._agent_overview_table = None
            self._agent_builder_table = None
            self._bundles_table = None
            self._settings_models_table = None
            self._settings_env_table = None
            self.selected: list[str] = []

        def _tooling_select_tool(self, selected_id: str) -> None:
            self.selected.append(selected_id)

        def _tooling_tool_open_tag_editor(self) -> None:
            raise AssertionError("Legacy row-driven tag editor should not open.")

    harness = _Harness()
    SwarmeeTUI = tui_app.get_swarmee_tui_class()
    event = SimpleNamespace(
        data_table=harness._tooling_tools_table,
        row_key=SimpleNamespace(value="shell"),
    )
    SwarmeeTUI.on_data_table_row_selected(harness, event)
    assert harness.selected == ["shell"]


def test_tag_manager_button_opens_manager():
    class _Harness:
        def __init__(self) -> None:
            self.called = 0

        def _tooling_tools_open_tag_manager(self) -> None:
            self.called += 1

    harness = _Harness()
    SwarmeeTUI = tui_app.get_swarmee_tui_class()
    event = SimpleNamespace(button=SimpleNamespace(id="tooling_tools_tag_manager"))
    SwarmeeTUI.on_button_pressed(harness, event)
    assert harness.called == 1


def test_settings_models_manage_button_opens_popup_manager():
    class _Harness:
        def __init__(self) -> None:
            self.called = 0

        def _open_settings_model_manager(self) -> None:
            self.called += 1

    harness = _Harness()
    SwarmeeTUI = tui_app.get_swarmee_tui_class()
    event = SimpleNamespace(button=SimpleNamespace(id="settings_models_open_manager"))
    SwarmeeTUI.on_button_pressed(harness, event)
    assert harness.called == 1


def test_agents_builder_header_uses_sidebar_header_action():
    import inspect

    from swarmee_river.tui.views import agents as agents_view

    source = inspect.getsource(agents_view.compose_agents_tab)
    assert 'SidebarHeader(\n                        "Agent Roster"' in source
    assert '"agent_builder_open_manager"' in source
    assert '"Agent Manager"' in source


def test_agent_builder_manager_button_opens_modal():
    class _Harness:
        def __init__(self) -> None:
            self.called = 0

        def _open_agent_manager(self) -> None:
            self.called += 1

    harness = _Harness()
    SwarmeeTUI = tui_app.get_swarmee_tui_class()
    SwarmeeTUI.on_button_pressed(harness, SimpleNamespace(button=SimpleNamespace(id="agent_builder_open_manager")))
    assert harness.called == 1


def test_settings_models_row_selection_updates_model_detail():
    class _Table:
        pass

    class _Harness:
        def __init__(self) -> None:
            self._tooling_tools_table = None
            self._tooling_sops_table = None
            self._tooling_prompts_table = None
            self._tooling_kbs_table = None
            self._session_timeline_table = None
            self._session_artifacts_table = None
            self._agent_overview_table = None
            self._agent_builder_table = None
            self._bundles_table = None
            self._settings_models_table = _Table()
            self._settings_env_table = None
            self._settings_models_selected_id = None
            self.detail_refreshes = 0

        def _refresh_settings_model_detail(self) -> None:
            self.detail_refreshes += 1

    harness = _Harness()
    SwarmeeTUI = tui_app.get_swarmee_tui_class()
    event = SimpleNamespace(
        data_table=harness._settings_models_table,
        row_key=SimpleNamespace(value="openai|coding"),
    )
    SwarmeeTUI.on_data_table_row_selected(harness, event)
    assert harness._settings_models_selected_id == "openai|coding"
    assert harness.detail_refreshes == 1


def test_tag_manager_result_updates_tags_and_preserves_non_tag_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))

    from swarmee_river.tui.tool_metadata import load_tool_metadata_overrides, save_tool_metadata_overrides

    class _Dummy(ContextSourcesMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.tooling.tool_catalog = [{"name": "shell"}, {"name": "git"}]
            self.state.tooling.tool_selected_id = "git"
            self.refresh_calls = 0
            self.selected_ids: list[str] = []
            self.transcript_lines: list[str] = []

        def _refresh_tooling_tools_list(self) -> None:
            self.refresh_calls += 1

        def _tooling_select_tool(self, selected_id: str | None) -> None:
            self.selected_ids.append(str(selected_id or ""))

        def _write_transcript_line(self, text: str) -> None:
            self.transcript_lines.append(text)

    save_tool_metadata_overrides(
        {
            "shell": {"tags": ["old"], "description": "Shell tool", "access_execute": True},
            "git": {"tags": ["legacy"], "access_execute": True},
            "unknown": {"tags": ["keep"]},
        }
    )

    app = _Dummy()
    app._on_tool_tag_manager_complete(
        {
            "tool_tags": {
                "shell": ["alpha", "beta", "alpha"],
                "git": [],
                "unknown": ["changed"],
            }
        }
    )

    overrides = load_tool_metadata_overrides()
    assert overrides["shell"]["tags"] == ["alpha", "beta"]
    assert overrides["shell"]["description"] == "Shell tool"
    assert overrides["shell"]["access_execute"] is True
    assert "tags" not in overrides["git"]
    assert overrides["git"]["access_execute"] is True
    assert overrides["unknown"]["tags"] == ["keep"]
    assert app.refresh_calls == 1
    assert app.selected_ids == ["git"]
    assert app.transcript_lines[-1] == "[tooling] saved tag manager changes."


def test_tag_manager_cancel_result_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(tmp_path / ".swarmee"))

    from swarmee_river.tui.tool_metadata import load_tool_metadata_overrides, save_tool_metadata_overrides

    class _Dummy(ContextSourcesMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.tooling.tool_catalog = [{"name": "shell"}]

        def _refresh_tooling_tools_list(self) -> None:
            raise AssertionError("refresh should not run for cancel/noop")

        def _tooling_select_tool(self, selected_id: str | None) -> None:
            raise AssertionError(f"selection should not run for cancel/noop: {selected_id}")

        def _write_transcript_line(self, text: str) -> None:
            raise AssertionError(f"transcript should not run for cancel/noop: {text}")

    save_tool_metadata_overrides({"shell": {"tags": ["ops"]}})
    app = _Dummy()
    app._on_tool_tag_manager_complete(None)
    assert load_tool_metadata_overrides()["shell"]["tags"] == ["ops"]


def test_tool_tag_manager_screen_tag_operations():
    from swarmee_river.tui.widgets import ToolTagManagerScreen

    screen = ToolTagManagerScreen(
        [
            {"name": "shell", "tags": ["ops"]},
            {"name": "git", "tags": ["Ops", "ci"]},
        ]
    )
    assert screen.add_tag("infra") is True
    assert screen.toggle_tool_for_selected_tag("shell") is True
    assert "infra" in screen.build_result_payload()["tool_tags"]["shell"]

    screen._selected_tag = "ops"
    assert screen.rename_selected_tag("CI") is True
    payload = screen.build_result_payload()["tool_tags"]
    assert payload["shell"] == ["CI", "infra"]
    assert payload["git"] == ["CI"]

    screen._selected_tag = "CI"
    assert screen.delete_selected_tag() is True
    payload_after_delete = screen.build_result_payload()["tool_tags"]
    assert payload_after_delete["shell"] == ["infra"]
    assert payload_after_delete["git"] == []


def test_agent_editor_payload_normalization_and_orchestrator_guards():
    from swarmee_river.profiles.models import ORCHESTRATOR_AGENT_ID
    from swarmee_river.tui.widgets import AgentEditorScreen

    payload = AgentEditorScreen._normalize_editor_payload(
        {
            "id": "writer-agent",
            "name": "Writer",
            "summary": "Writes drafts",
            "prompt_ref": "Draft_Prompt",
            "provider": "openai",
            "tier": "balanced",
            "tool_names": ["git", "shell", "git"],
            "sop_names": ["qa_pass", "qa_pass"],
            "knowledge_base_id": "kb-main",
            "activated": True,
        },
        is_orchestrator=False,
    )
    assert payload is not None
    assert payload["prompt_refs"] == ["draft_prompt"]
    assert payload["tool_names"] == ["git", "shell"]
    assert payload["sop_names"] == ["qa_pass"]
    assert payload["provider"] == "openai"
    assert payload["tier"] == "balanced"
    assert payload["activated"] is True

    orchestrator_payload = AgentEditorScreen._normalize_editor_payload(
        {
            "id": "something-else",
            "name": "Orchestrator",
            "activated": True,
        },
        is_orchestrator=True,
    )
    assert orchestrator_payload is not None
    assert orchestrator_payload["id"] == ORCHESTRATOR_AGENT_ID
    assert orchestrator_payload["activated"] is False


def test_agent_editor_cancel_and_escape_dismiss_none():
    from swarmee_river.tui.widgets import AgentEditorScreen

    class _KeyEvent:
        def __init__(self) -> None:
            self.key = "escape"
            self.stopped = False
            self.prevented = False

        def stop(self) -> None:
            self.stopped = True

        def prevent_default(self) -> None:
            self.prevented = True

    screen = AgentEditorScreen({"id": "writer", "name": "Writer"})
    dismissed: list[object] = []
    screen.dismiss = lambda value: dismissed.append(value)  # type: ignore[method-assign]
    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="agent_editor_cancel")))
    assert dismissed == [None]

    screen_esc = AgentEditorScreen({"id": "writer", "name": "Writer"})
    dismissed_esc: list[object] = []
    screen_esc.dismiss = lambda value: dismissed_esc.append(value)  # type: ignore[method-assign]
    event = _KeyEvent()
    screen_esc.on_key(event)
    assert dismissed_esc == [None]
    assert event.stopped is True
    assert event.prevented is True


def test_agent_editor_create_prompt_asset_record_generates_unique_selection():
    from swarmee_river.tui.widgets import AgentEditorScreen

    asset = AgentEditorScreen._create_prompt_asset_record(
        [{"id": "writer_prompt", "name": "Writer Prompt", "content": "Existing"}],
        name="Writer Prompt",
        content="New prompt body",
    )
    assert asset["id"] != "writer_prompt"
    assert asset["name"] == "Writer Prompt"
    assert asset["content"] == "New prompt body"


def test_agent_editor_nested_screens_use_app_push_screen():
    from textual.message_pump import active_app

    from swarmee_river.tui.widgets import AgentEditorScreen, CatalogMultiSelectScreen, CatalogSingleSelectScreen

    class _FakeApp:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def push_screen(self, screen, callback=None):  # noqa: ANN001
            self.calls.append((screen, callback))

    app = _FakeApp()
    token = active_app.set(app)
    try:
        screen = AgentEditorScreen(
            {"id": "writer", "name": "Writer"},
            tool_options=["git", "shell"],
            sop_options=["qa_pass"],
            kb_options=[("Primary KB (kb-primary)", "kb-primary")],
        )
        screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="agent_editor_tools_edit")))
        screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="agent_editor_sops_edit")))
        screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="agent_editor_kb_edit")))
    finally:
        active_app.reset(token)

    assert len(app.calls) == 3
    assert isinstance(app.calls[0][0], CatalogMultiSelectScreen)
    assert callable(app.calls[0][1])
    assert isinstance(app.calls[1][0], CatalogMultiSelectScreen)
    assert callable(app.calls[1][1])
    assert isinstance(app.calls[2][0], CatalogSingleSelectScreen)
    assert callable(app.calls[2][1])


def test_agent_manager_screen_updates_and_deletes_agents():
    from swarmee_river.tui.widgets import AgentManagerScreen

    screen = AgentManagerScreen(
        [
            {"id": "orchestrator", "name": "Orchestrator", "activated": False},
            {"id": "writer", "name": "Writer", "activated": True, "prompt_refs": ["draft_prompt"]},
        ],
        prompt_assets=[{"id": "draft_prompt", "name": "Draft Prompt", "content": "Draft well"}],
        tool_options=["git", "shell"],
        sop_options=["qa_pass"],
        kb_options=[("Primary KB (kb-primary)", "kb-primary")],
        selected_id="writer",
    )

    screen._apply_editor_result(
        {
            "id": "writer",
            "name": "Writer Updated",
            "summary": "Updated",
            "prompt_ref": "draft_prompt",
            "tool_names": ["git"],
            "sop_names": ["qa_pass"],
            "knowledge_base_id": "kb-primary",
            "activated": True,
        },
        source_id="writer",
    )
    updated = next(agent for agent in screen._agents if agent["id"] == "writer")
    assert updated["name"] == "Writer Updated"
    assert updated["prompt_refs"] == ["draft_prompt"]

    screen._delete_selected()
    assert all(agent["id"] != "writer" for agent in screen._agents)


def test_agent_manager_screen_new_and_edit_use_app_push_screen():
    from textual.message_pump import active_app

    from swarmee_river.tui.widgets import AgentEditorScreen, AgentManagerScreen

    class _FakeApp:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def push_screen(self, screen, callback=None):  # noqa: ANN001
            self.calls.append((screen, callback))

    app = _FakeApp()
    token = active_app.set(app)
    try:
        screen = AgentManagerScreen(
            [
                {"id": "orchestrator", "name": "Orchestrator", "activated": False},
                {"id": "writer", "name": "Writer", "activated": True},
            ],
            prompt_assets=[],
            tool_options=["git", "shell"],
            sop_options=["qa_pass"],
            kb_options=[("Primary KB (kb-primary)", "kb-primary")],
            selected_id="writer",
        )
        screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="agent_manager_new")))
        screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="agent_manager_edit")))
    finally:
        active_app.reset(token)

    assert len(app.calls) == 2
    assert all(isinstance(pushed_screen, AgentEditorScreen) for pushed_screen, _callback in app.calls)
    assert all(callable(callback) for _pushed_screen, callback in app.calls)


def test_agent_manager_screen_preserves_orchestrator_on_delete():
    from swarmee_river.tui.widgets import AgentManagerScreen

    screen = AgentManagerScreen(
        [{"id": "orchestrator", "name": "Orchestrator", "activated": False}],
        prompt_assets=[],
        tool_options=[],
        sop_options=[],
        kb_options=[],
        selected_id="orchestrator",
    )

    screen._delete_selected()
    assert len(screen._agents) == 1
    assert screen._agents[0]["id"] == "orchestrator"


def test_orchestrator_runtime_model_selection_applies_provider_tier_and_persists():
    class _StatusBar:
        def __init__(self) -> None:
            self.models: list[str] = []

        def set_model(self, value: str) -> None:
            self.models.append(value)

    class _Harness(AgentStudioMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self._status_bar = _StatusBar()
            self.pin_calls: list[tuple[str, str]] = []
            self.set_tier_calls: list[str] = []
            self.persist_calls: list[tuple[str | None, str | None]] = []
            self.refresh_summary_calls = 0

        def _pin_model_select_target(self, provider: str, tier: str, *, seconds: float = 2.5) -> None:
            del seconds
            self.pin_calls.append((provider, tier))

        def _set_model_tier_from_value(self, value: str) -> None:
            self.set_tier_calls.append(value)

        def _persist_quick_model_selection(self, *, provider: str | None, tier: str | None) -> None:
            self.persist_calls.append((provider, tier))

        def _refresh_agent_summary(self) -> None:
            self.refresh_summary_calls += 1

    harness = _Harness()
    harness._apply_orchestrator_runtime_model_selection({"id": "orchestrator", "provider": "openai", "tier": "deep"})
    assert harness.pin_calls == [("openai", "deep")]
    assert harness.set_tier_calls == ["openai|deep"]
    assert harness.persist_calls == [("openai", "deep")]
    assert harness.refresh_summary_calls == 1


def test_orchestrator_runtime_model_selection_inherit_clears_override_and_persists_auto():
    class _StatusBar:
        def __init__(self) -> None:
            self.models: list[str] = []

        def set_model(self, value: str) -> None:
            self.models.append(value)

    class _Harness(AgentStudioMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.daemon.model_provider_override = "openai"
            self.state.daemon.model_tier_override = "deep"
            self.state.daemon.pending_model_select_value = "openai|deep"
            self._status_bar = _StatusBar()
            self.pin_calls: list[tuple[str, str]] = []
            self.persist_calls: list[tuple[str | None, str | None]] = []
            self.refresh_model_select_calls = 0
            self.refresh_summary_calls = 0
            self.header_updates = 0
            self.placeholder_updates = 0

        def _pin_model_select_target(self, provider: str, tier: str, *, seconds: float = 2.5) -> None:
            del seconds
            self.pin_calls.append((provider, tier))

        def _persist_quick_model_selection(self, *, provider: str | None, tier: str | None) -> None:
            self.persist_calls.append((provider, tier))

        def _refresh_model_select(self) -> None:
            self.refresh_model_select_calls += 1

        def _update_header_status(self) -> None:
            self.header_updates += 1

        def _update_prompt_placeholder(self) -> None:
            self.placeholder_updates += 1

        def _current_model_summary(self) -> str:
            return "Model: auto"

        def _refresh_agent_summary(self) -> None:
            self.refresh_summary_calls += 1

    harness = _Harness()
    harness._apply_orchestrator_runtime_model_selection({"id": "orchestrator", "provider": None, "tier": None})
    assert harness.state.daemon.model_provider_override is None
    assert harness.state.daemon.model_tier_override is None
    assert harness.state.daemon.pending_model_select_value is None
    assert harness.pin_calls == [("", "")]
    assert harness.persist_calls == [(None, None)]
    assert harness.refresh_model_select_calls == 1
    assert harness.header_updates == 1
    assert harness.placeholder_updates == 1
    assert harness.refresh_summary_calls == 1


def test_apply_agent_builder_editor_result_updates_orchestrator_runtime_model_selection():
    class _Harness(AgentStudioMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.agent_studio.agents = [
                {"id": "orchestrator", "name": "Orchestrator", "summary": "", "prompt": "", "activated": False},
            ]
            self.state.agent_studio.builder_selected_item_id = None
            self.applied_payloads: list[dict[str, object]] = []
            self.render_calls = 0
            self.status_lines: list[str] = []
            self.dirty_notes: list[str] = []

        def _apply_orchestrator_runtime_model_selection(self, agent_def: dict[str, object]) -> None:
            self.applied_payloads.append(dict(agent_def))

        def _render_agent_builder_panel(self) -> None:
            self.render_calls += 1

        def _set_agent_builder_status(self, message: str) -> None:
            self.status_lines.append(message)

        def _set_agent_draft_dirty(self, _dirty: bool, *, note: str | None = None) -> None:
            self.dirty_notes.append(str(note or ""))

    harness = _Harness()
    harness._apply_agent_builder_editor_result(
        {
            "id": "orchestrator",
            "name": "Orchestrator",
            "summary": "Runtime control",
            "provider": "openai",
            "tier": "deep",
            "prompt_refs": ["orchestrator_base"],
            "activated": False,
        },
        source_id="orchestrator",
    )
    assert harness.render_calls == 1
    assert harness.state.agent_studio.builder_selected_item_id == "orchestrator"
    assert len(harness.applied_payloads) == 1
    assert harness.applied_payloads[0]["provider"] == "openai"
    assert harness.applied_payloads[0]["tier"] == "deep"
    assert any("Saved agent 'Orchestrator'" in line for line in harness.status_lines)
    assert any("Orchestrator" in note for note in harness.dirty_notes)


def test_model_manager_screen_stage_delete_and_default_pair():
    from swarmee_river.tui.widgets import ModelConfigManagerScreen

    payload = {
        "models": {
            "provider": "openai",
            "default_tier": "balanced",
            "default_selection": {"provider": "openai", "tier": "balanced"},
            "providers": {
                "openai": {
                    "tiers": {
                        "balanced": {"provider": "openai", "model_id": "gpt-5-mini"},
                        "coding": {"provider": "openai", "model_id": "gpt-5.3-codex"},
                    }
                }
            },
        },
        "env": {},
    }
    screen = ModelConfigManagerScreen(payload)
    rows = screen._model_rows()
    assert any(provider == "openai" and tier == "coding" for provider, tier, _ in rows)

    screen._set_default_pair("openai", "coding")
    result_before_delete = screen.build_result_payload()
    assert result_before_delete["models"]["default_selection"] == {"provider": "openai", "tier": "coding"}

    deleted = screen._delete_entry("openai", "coding")
    assert deleted is True
    result_after_delete = screen.build_result_payload()
    openai_tiers = result_after_delete["models"]["providers"]["openai"]["tiers"]
    assert "coding" not in openai_tiers


def test_model_manager_result_payload_preserves_guided_openai_fields():
    from swarmee_river.tui.widgets import ModelConfigManagerScreen

    payload = {
        "models": {
            "provider": "openai",
            "default_tier": "deep",
            "default_selection": {"provider": "openai", "tier": "deep"},
            "providers": {
                "openai": {
                    "tiers": {
                        "deep": {
                            "provider": "openai",
                            "model_id": "gpt-5.2",
                            "transport": "responses",
                            "reasoning": {"effort": "high"},
                            "tooling": {"mode": "tool-heavy", "discovery": "search"},
                            "context": {"strategy": "cache_safe", "compaction": "auto"},
                        }
                    }
                }
            },
        },
        "env": {},
    }

    screen = ModelConfigManagerScreen(payload)
    result = screen.build_result_payload()
    deep = result["models"]["providers"]["openai"]["tiers"]["deep"]

    assert deep["transport"] == "responses"
    assert deep["reasoning"]["effort"] == "high"
    assert deep["tooling"]["mode"] == "tool-heavy"
    assert deep["tooling"]["discovery"] == "search"
    assert deep["context"]["strategy"] == "cache_safe"


def test_model_manager_screen_accepts_raw_model_id_input():
    from swarmee_river.tui.widgets import ModelConfigManagerScreen

    class _Widget:
        def __init__(self, value: str = "") -> None:
            self.value = value
            self.text = ""

        def set_options(self, _options) -> None:
            return None

        def update(self, text: str) -> None:
            self.text = text

    screen = ModelConfigManagerScreen({"models": {"providers": {}}, "env": {}})
    widgets = {
        "#model_edit_provider": _Widget("bedrock"),
        "#model_edit_tier": _Widget("deep"),
        "#model_edit_model_id": _Widget("us.anthropic.claude-opus-4-6-v1"),
        "#model_edit_reasoning": _Widget("high"),
        "#model_edit_tool_mode": _Widget("tool-heavy"),
        "#model_edit_context_strategy": _Widget("cache_safe"),
        "#model_edit_context_compaction": _Widget("auto"),
        "#model_edit_display_name": _Widget("Claude Opus 4.6 (deep)"),
        "#model_edit_description": _Widget("Adaptive Claude reasoning for harder analytics tasks."),
        "#model_edit_price_input": _Widget(""),
        "#model_edit_price_output": _Widget(""),
        "#model_edit_price_cached": _Widget(""),
        "#model_manager_model_select": _Widget("__none__"),
        "#model_manager_preview": _Widget(""),
        "#model_manager_status": _Widget(""),
    }
    screen.query_one = lambda selector, _widget_type=None: widgets[selector]  # type: ignore[method-assign]

    assert screen._save_editor_entry() is True
    saved = screen.build_result_payload()["models"]["providers"]["bedrock"]["tiers"]["deep"]
    assert saved["model_id"] == "us.anthropic.claude-opus-4-6-v1"
    assert saved["tooling"]["mode"] == "tool-heavy"
    assert "Model ID: us.anthropic.claude-opus-4-6-v1" in widgets["#model_manager_preview"].text


def test_prompt_row_selected_opens_metadata_editor_for_editable_columns():
    class _Column:
        def __init__(self, key: str) -> None:
            self.key = key

    class _Table:
        def __init__(self) -> None:
            self.columns = [_Column("name"), _Column("id"), _Column("tags"), _Column("used_by")]
            self.cursor_coordinate = SimpleNamespace(column=1)

        def is_valid_column_index(self, index: int) -> bool:
            return 0 <= index < len(self.columns)

        def get_column_at(self, index: int) -> _Column:
            return self.columns[index]

    class _Harness:
        def __init__(self) -> None:
            self._tooling_tools_table = None
            self._tooling_sops_table = None
            self._tooling_prompts_table = _Table()
            self._tooling_kbs_table = None
            self._session_timeline_table = None
            self._session_artifacts_table = None
            self._agent_overview_table = None
            self._agent_builder_table = None
            self._settings_models_table = None
            self._settings_env_table = None
            self.selected: list[str] = []
            self.opened: list[tuple[str, str]] = []

        def _tooling_select_prompt(self, selected_id: str) -> None:
            self.selected.append(selected_id)

        def _tooling_prompt_open_table_cell_editor(self, row_id: str, column_key: str) -> None:
            self.opened.append((row_id, column_key))

    harness = _Harness()
    SwarmeeTUI = tui_app.get_swarmee_tui_class()
    event = SimpleNamespace(
        data_table=harness._tooling_prompts_table,
        row_key=SimpleNamespace(value="prompt_1"),
    )
    SwarmeeTUI.on_data_table_row_selected(harness, event)
    assert harness.selected == ["prompt_1"]
    assert harness.opened == [("prompt_1", "id")]


def test_prompt_row_selected_opens_used_by_editor_for_used_by_column():
    class _Column:
        def __init__(self, key: str) -> None:
            self.key = key

    class _Table:
        def __init__(self) -> None:
            self.columns = [_Column("name"), _Column("id"), _Column("tags"), _Column("used_by")]
            self.cursor_coordinate = SimpleNamespace(column=3)

        def is_valid_column_index(self, index: int) -> bool:
            return 0 <= index < len(self.columns)

        def get_column_at(self, index: int) -> _Column:
            return self.columns[index]

    class _Harness:
        def __init__(self) -> None:
            self._tooling_tools_table = None
            self._tooling_sops_table = None
            self._tooling_prompts_table = _Table()
            self._tooling_kbs_table = None
            self._session_timeline_table = None
            self._session_artifacts_table = None
            self._agent_overview_table = None
            self._agent_builder_table = None
            self._settings_models_table = None
            self._settings_env_table = None
            self.selected: list[str] = []
            self.opened_used_by: list[str] = []

        def _tooling_select_prompt(self, selected_id: str) -> None:
            self.selected.append(selected_id)

        def _tooling_prompt_open_table_cell_editor(self, row_id: str, column_key: str) -> None:
            raise AssertionError(f"Unexpected metadata editor open: {row_id}/{column_key}")

        def _tooling_prompt_open_used_by_editor(self, row_id: str) -> None:
            self.opened_used_by.append(row_id)

    harness = _Harness()
    SwarmeeTUI = tui_app.get_swarmee_tui_class()
    event = SimpleNamespace(
        data_table=harness._tooling_prompts_table,
        row_key=SimpleNamespace(value="prompt_1"),
    )
    SwarmeeTUI.on_data_table_row_selected(harness, event)

    assert harness.selected == ["prompt_1"]
    assert harness.opened_used_by == ["prompt_1"]


def test_set_planning_ui_mode_toggles_actions_row_visibility():
    class _Styles:
        def __init__(self) -> None:
            self.display = "none"

    class _Widget:
        def __init__(self) -> None:
            self.styles = _Styles()
            self.children: list[object] = []
            self.renderable = ""
            self.read_only = False
            self.show_cursor = True

        def update(self, text: str) -> None:
            self.renderable = text

    class _Dummy(PlanMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self._widgets = {
                "#plan": _Widget(),
                "#engage_plan_summary_scroll": _Widget(),
                "#engage_plan_summary": _Widget(),
                "#engage_plan_items": _Widget(),
                "#engage_plan_questions": _Widget(),
                "#engage_plan_actions_row": _Widget(),
                "#engage_start_plan": _Widget(),
                "#engage_continue_plan": _Widget(),
                "#engage_clear_plan": _Widget(),
                "#engage_cancel_plan": _Widget(),
                "#engage_planning_header": _Widget(),
            }

        def query_one(self, selector: str, _type=None):  # noqa: ANN001
            return self._widgets[selector]

    app = _Dummy()
    app.state.plan.current_summary = "Summary"
    app._widgets["#engage_plan_items"].children = [object()]
    app._widgets["#engage_plan_questions"].children = [object()]

    app._set_planning_ui_mode(pre_plan=True)
    assert app._widgets["#engage_start_plan"].styles.display == "block"
    assert app._widgets["#engage_plan_actions_row"].styles.display == "none"
    assert app._widgets["#engage_plan_summary_scroll"].styles.display == "none"

    app._set_planning_ui_mode(pre_plan=False)
    assert app._widgets["#engage_start_plan"].styles.display == "none"
    assert app._widgets["#engage_plan_actions_row"].styles.display == "block"
    assert app._widgets["#engage_plan_summary_scroll"].styles.display == "block"


def test_app_escape_dismisses_error_action_prompt() -> None:
    from swarmee_river.tui.app import get_swarmee_tui_class

    SwarmeeTUI = get_swarmee_tui_class()

    class _Widget:
        def __init__(self) -> None:
            self.styles = SimpleNamespace(display="block")

    class _Harness:
        def __init__(self) -> None:
            self._consent_active = False
            self._error_action_prompt_widget = _Widget()
            self.reset_calls = 0
            self.focus_calls = 0

        def _reset_error_action_prompt(self) -> None:
            self.reset_calls += 1

        def action_focus_prompt(self) -> None:
            self.focus_calls += 1

    stopped = {"value": False}
    prevented = {"value": False}

    event = SimpleNamespace(
        key="escape",
        stop=lambda: stopped.__setitem__("value", True),
        prevent_default=lambda: prevented.__setitem__("value", True),
    )

    harness = _Harness()
    SwarmeeTUI.on_key(harness, event)

    assert stopped["value"] is True
    assert prevented["value"] is True
    assert harness.reset_calls == 1
    assert harness.focus_calls == 1


def test_artifact_entries_are_scoped_to_active_session(monkeypatch):
    class _Dummy(ArtifactsMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.daemon.session_id = "session-a"

    app = _Dummy()

    class _Store:
        def list(self, *, limit: int = 50):
            return [
                {
                    "id": "a1",
                    "kind": "tui_transcript",
                    "path": "/tmp/a1.txt",
                    "created_at": "2026-02-01T10:00:00",
                    "meta": {"session_id": "session-a"},
                },
                {
                    "id": "b1",
                    "kind": "tui_transcript",
                    "path": "/tmp/b1.txt",
                    "created_at": "2026-02-01T11:00:00",
                    "meta": {"session_id": "session-b"},
                },
                {
                    "id": "legacy",
                    "kind": "tui_transcript",
                    "path": "/tmp/legacy.txt",
                    "created_at": "2026-02-01T12:00:00",
                },
            ]

    monkeypatch.setattr("swarmee_river.tui.mixins.artifacts.ArtifactStore", _Store)
    entries = app._load_indexed_artifact_entries(limit=20)

    assert [entry["id"] for entry in entries] == ["a1"]


def test_sync_artifact_session_scope_resets_on_session_change():
    class _Dummy(ArtifactsMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.daemon.session_id = "session-b"
            self.state.artifacts.session_id = "session-a"
            self.state.artifacts.recent_paths = ["/tmp/old.txt"]
            self.state.artifacts.entries = [{"id": "old"}]
            self.state.artifacts.selected_item_id = "old"

    app = _Dummy()
    app._sync_artifact_session_scope()
    assert app.state.artifacts.session_id == "session-b"
    assert app.state.artifacts.recent_paths == []
    assert app.state.artifacts.entries == []
    assert app.state.artifacts.selected_item_id is None


def test_artifact_preview_renderable_uses_diff_panel_for_file_diff(tmp_path: Path) -> None:
    class _Dummy(ArtifactsMixin):
        pass

    diff_path = tmp_path / "example.diff"
    diff_path.write_text("--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")
    app = _Dummy()

    renderable = app._artifact_preview_renderable(
        {
            "kind": "file_diff",
            "path": str(diff_path),
            "meta": {
                "tool": "editor",
                "changed_paths": ["a.txt"],
                "stats": {"files_changed": 1, "added_lines": 1, "removed_lines": 1},
            },
        }
    )

    assert type(renderable).__name__ == "Panel"


def test_handle_output_line_routes_file_diff_event_once_and_adds_artifact(tmp_path: Path) -> None:
    diff_path = tmp_path / "event.diff"
    diff_path.write_text("--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")

    class _Harness(OutputMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(query_active=False, turn_output_chunks=[]),
                session=SimpleNamespace(warning_count=0, error_count=0),
            )
            self.widgets: list[object] = []
            self.fallback: list[str] = []
            self.artifacts: list[list[str]] = []

        def _append_plain_text(self, text: str) -> None:
            self.fallback.append(text)

        def _apply_consent_capture(self, _line: str) -> None:
            return None

        def _write_issue(self, _text: str) -> None:
            return None

        def _update_header_status(self) -> None:
            return None

        def _add_artifact_paths(self, paths: list[str]) -> None:
            self.artifacts.append(paths)

        def _mount_transcript_widget(self, widget, *, plain_text=None) -> None:  # noqa: ANN001
            self.widgets.append(widget)
            if isinstance(plain_text, str):
                self.fallback.append(plain_text)

    app = _Harness()
    app._handle_output_line(
        json.dumps(
            {
                "event": "file_diff",
                "artifact_path": str(diff_path),
                "tool": "editor",
                "paths": ["a.txt"],
                "stats": {"files_changed": 1, "added_lines": 1, "removed_lines": 1},
            }
        )
    )

    assert len(app.widgets) == 1
    assert app.artifacts == [[str(diff_path)]]
    assert any("Δ editor changed a.txt" in line for line in app.fallback)


def test_handle_tool_events_routes_consent_prompt_with_diff_preview() -> None:
    from swarmee_river.tui.event_router import _handle_tool_events

    captured: dict[str, object] = {}

    class _App:
        def __init__(self) -> None:
            self._consent_buffer: list[str] = []

        def _show_consent_prompt(self, **kwargs) -> None:  # noqa: ANN003
            captured.update(kwargs)

    app = _App()
    handled = _handle_tool_events(
        app,
        "consent_prompt",
        {
            "event": "consent_prompt",
            "context": "Allow tool 'editor'?\n  Changed paths: notes.txt",
            "options": ["y", "n", "a", "v"],
            "changed_paths": ["notes.txt"],
            "diff_preview": "--- a/notes.txt\n+++ b/notes.txt\n@@ -1 +1 @@\n-old\n+new",
            "diff_hidden_lines": 3,
            "diff_stats": {"files_changed": 1, "added_lines": 1, "removed_lines": 1},
        },
    )

    assert handled is True
    assert app._consent_buffer == ["Allow tool 'editor'?\n  Changed paths: notes.txt"]
    assert captured["changed_paths"] == ["notes.txt"]
    assert str(captured["diff_preview"]).startswith("--- a/notes.txt")
    assert captured["diff_hidden_lines"] == 3


# ---------------------------------------------------------------------------
# parse_tui_event tests
# ---------------------------------------------------------------------------


def test_parse_tui_event_valid_json():
    result = tui_app.parse_tui_event('{"event":"text_delta","data":"hello"}')
    assert result == {"event": "text_delta", "data": "hello"}


def test_parse_tui_event_non_json_returns_none():
    assert tui_app.parse_tui_event("plain text output") is None
    assert tui_app.parse_tui_event("") is None
    assert tui_app.parse_tui_event("Error: something failed") is None


def test_parse_tui_event_ignores_osc_prefix():
    line = '\x1b]0;Python/git/rg\x07{"event":"text_delta","data":"hello"}'
    result = tui_app.parse_tui_event(line)
    assert result == {"event": "text_delta", "data": "hello"}


def test_extract_tui_text_chunk_prefers_data_then_falls_back_to_text():
    assert tui_app.extract_tui_text_chunk({"data": "hello", "text": "world"}) == "hello"
    assert tui_app.extract_tui_text_chunk({"text": "world"}) == "world"
    assert tui_app.extract_tui_text_chunk({"delta": "chunk"}) == "chunk"
    assert tui_app.extract_tui_text_chunk({"outputText": "done"}) == "done"
    assert tui_app.extract_tui_text_chunk({"delta": {"text": "nested"}}) == "nested"
    assert tui_app.extract_tui_text_chunk({"event": {"contentBlockDelta": {"delta": {"text": "bedrock"}}}}) == "bedrock"
    assert tui_app.extract_tui_text_chunk({"event": "text_delta"}) == ""


def test_parse_tui_event_malformed_json_returns_none():
    assert tui_app.parse_tui_event('{"event": "text_delta"') is None
    assert tui_app.parse_tui_event("{bad json}") is None


def test_parse_tui_event_rejects_non_object_json():
    assert tui_app.parse_tui_event("[]") is None
    assert tui_app.parse_tui_event('"event"') is None


def test_parse_tui_event_strips_whitespace():
    result = tui_app.parse_tui_event('  {"event":"thinking","text":"hmm"}  ')
    assert result == {"event": "thinking", "text": "hmm"}


def test_spawn_swarmee_sets_tui_events_env(monkeypatch):
    """spawn_swarmee should set SWARMEE_TUI_EVENTS=1 in the subprocess env."""
    captured_env = {}

    def fake_popen(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))

        class FakeProc:
            pid = 12345
            stdin = None
            stdout = None
            stderr = None

            def poll(self):
                return 0

        return FakeProc()

    monkeypatch.setattr(tui_app.subprocess, "Popen", fake_popen)
    tui_app.spawn_swarmee("test prompt", auto_approve=True, session_id="abc123")
    assert captured_env.get("SWARMEE_TUI_EVENTS") == "1"
    assert captured_env.get("SWARMEE_SPINNERS") == "0"


def test_spawn_swarmee_daemon_configures_subprocess(monkeypatch):
    captured: dict[str, object] = {}
    fake_proc = object()

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    proc = tui_app.spawn_swarmee_daemon(session_id="abc123", env_overrides={"SWARMEE_MODEL_TIER": "fast"})

    assert proc is fake_proc
    command = captured["command"]
    kwargs = captured["kwargs"]
    assert command == [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--tui-daemon"]
    assert kwargs["stdin"] is tui_app.subprocess.PIPE
    assert kwargs["stdout"] is tui_app.subprocess.PIPE
    assert kwargs["stderr"] is tui_app.subprocess.STDOUT
    assert kwargs["text"] is True
    assert kwargs["start_new_session"] is True
    env = kwargs["env"]
    assert isinstance(env, dict)
    assert env["SWARMEE_SESSION_ID"] == "abc123"
    assert env["SWARMEE_MODEL_TIER"] == "fast"
    assert env["SWARMEE_TUI_EVENTS"] == "1"
    assert env["SWARMEE_LOG_EVENTS"] == "1"


def test_spawn_swarmee_daemon_windows_sets_creationflags(monkeypatch):
    import swarmee_river.tui.transport as transport

    captured: dict[str, object] = {}
    fake_proc = object()
    monkeypatch.setattr(tui_app.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_proc

    class _OsModule:
        name = "nt"
        environ: dict[str, str] = {}

    proc = transport._spawn_swarmee_process(
        [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--tui-daemon"],
        session_id="abc123",
        popen=_fake_popen,
        subprocess_module=tui_app.subprocess,
        os_module=_OsModule,
    )

    assert proc is fake_proc
    kwargs = captured["kwargs"]
    assert kwargs["creationflags"] == 0x200
    assert "start_new_session" not in kwargs
    from swarmee_river.tui.widgets import _format_tool_input

    result = _format_tool_input("shell", {"command": "git status", "cwd": "/tmp"})
    assert "Command: git status" in result
    assert "CWD: /tmp" in result


def test_format_tool_input_generic():
    from swarmee_river.tui.widgets import _format_tool_input

    result = _format_tool_input("custom_tool", {"key": "value"})
    assert '"key"' in result
    assert '"value"' in result


def test_format_tool_input_oneliner_shell():
    from swarmee_river.tui.widgets import _format_tool_input_oneliner

    result = _format_tool_input_oneliner("shell", {"command": "git status"})
    assert result == "$ git status"


def test_format_tool_input_oneliner_file_read():
    from swarmee_river.tui.widgets import _format_tool_input_oneliner

    result = _format_tool_input_oneliner("file_read", {"path": "src/main.py"})
    assert result == "← src/main.py"


def test_render_tool_start_line_with_input_hides_tool_id():
    from swarmee_river.tui.widgets import render_tool_start_line_with_input

    rendered = render_tool_start_line_with_input(
        "shell",
        tool_input={"command": "git status"},
        tool_use_id="tool-abc123",
    )
    assert "tool-abc123" not in rendered.plain
    assert "$ git status" in rendered.plain


def test_render_tool_result_line_includes_input_summary():
    from swarmee_river.tui.widgets import render_tool_result_line

    rendered = render_tool_result_line(
        "shell",
        status="success",
        duration_s=2.3,
        tool_input={"command": "git status"},
        tool_use_id="tool-abc123",
    )
    assert rendered.plain.startswith("✓ shell (2.3s)")
    assert "$ git status" in rendered.plain
    assert "tool-abc123" not in rendered.plain


def test_plan_card_renders_steps():
    from swarmee_river.tui.widgets import PlanCard

    card = PlanCard(
        plan_json={
            "summary": "Fix login",
            "steps": ["Read auth module", "Add validation", "Update tests"],
        }
    )
    rendered = card._render_from_status()
    assert "Fix login" in rendered
    assert "1." in rendered
    assert "Read auth module" in rendered


def test_plan_card_mark_step_complete():
    from swarmee_river.tui.widgets import PlanCard

    card = PlanCard(
        plan_json={
            "summary": "Test",
            "steps": ["Step A", "Step B"],
        }
    )
    assert card._step_status == [False, False]
    card.mark_step_complete(0)
    assert card._step_status == [True, False]


def test_plan_actions_exposes_approve_replan_clear_buttons():
    from swarmee_river.tui.widgets import PlanActions

    actions = PlanActions()
    buttons = list(actions.compose())
    ids = [getattr(button, "id", None) for button in buttons]
    assert ids == ["plan_action_approve", "plan_action_replan", "plan_action_clear"]


def test_agent_profile_actions_exposes_new_save_delete_apply_buttons():
    from swarmee_river.tui.widgets import AgentProfileActions

    actions = AgentProfileActions()
    buttons = list(actions.compose())
    ids = [getattr(button, "id", None) for button in buttons]
    assert ids == ["agent_profile_new", "agent_profile_save", "agent_profile_delete", "agent_profile_apply"]


def test_sidebar_header_compose_supports_badges_and_actions():
    from textual.widgets import Button, Static

    from swarmee_river.tui.widgets import SidebarHeader

    header = SidebarHeader(
        "Saved Profiles",
        badges=["2 active"],
        actions=[{"id": "header_refresh", "label": "Refresh"}],
    )
    children = list(header.compose())
    static_children = [child for child in children if isinstance(child, Static)]
    assert any("Saved Profiles" in str(getattr(child, "_Static__content", "")) for child in static_children)
    assert any("2 active" in str(getattr(child, "_Static__content", "")) for child in static_children)
    buttons = [child for child in children if isinstance(child, Button)]
    assert len(buttons) == 1
    assert buttons[0].id == "header_refresh"


def test_sidebar_list_item_state_normalization():
    from swarmee_river.tui.widgets import SidebarListItem

    item = SidebarListItem(item_id="qa", title="QA", subtitle="Quality", state="warning")
    assert item.item_id == "qa"
    assert item.state == "warning"
    item.set_state("error")
    assert item.state == "error"
    item.set_state("not-a-state")
    assert item.state == "default"


def test_sidebar_list_selection_navigation():
    from swarmee_river.tui.widgets import SidebarList

    sidebar_list = SidebarList()
    sidebar_list.set_items(
        [
            {"id": "qa", "title": "QA", "subtitle": "openai/deep"},
            {"id": "ops", "title": "Ops", "subtitle": "openai/balanced"},
        ],
        selected_id="qa",
        emit=False,
    )
    assert sidebar_list.selected_id() == "qa"
    sidebar_list.move_selection(1, emit=False)
    assert sidebar_list.selected_id() == "ops"
    assert sidebar_list.select_by_id("qa", emit=False) is True
    assert sidebar_list.selected_id() == "qa"
    assert sidebar_list.select_by_id("missing", emit=False) is False


def test_sidebar_detail_compose_supports_preview_and_actions():
    from swarmee_river.tui.widgets import SidebarDetail

    detail = SidebarDetail(
        preview="Profile details",
        actions=[{"id": "apply_profile", "label": "Apply"}],
    )
    assert detail._preview == "Profile details"
    assert detail._actions[0]["id"] == "apply_profile"
    detail.set_preview("Updated preview")
    assert detail._preview == "Updated preview"
    detail.set_actions([{"id": "open_profile", "label": "Open"}])
    assert detail._actions[0]["id"] == "open_profile"


def test_render_agent_profile_summary_text_includes_core_sections():
    from swarmee_river.tui.widgets import render_agent_profile_summary_text

    summary = render_agent_profile_summary_text(
        {
            "id": "qa",
            "name": "QA",
            "provider": "openai",
            "tier": "deep",
            "system_prompt_snippets": ["Use strict validation."],
            "context_sources": [{"type": "kb", "id": "kb-123"}],
            "active_sops": ["review"],
            "knowledge_base_id": "kb-123",
        }
    )
    assert "Name: QA" in summary
    assert "Model: openai/deep" in summary
    assert "System snippets (1):" in summary
    assert "Context sources (1):" in summary
    assert "Active SOPs (1):" in summary


def test_command_palette_filter():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/pl")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/plan"


def test_command_palette_filter_no_match():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/zzz")
    assert len(palette._filtered) == 0
    assert palette.get_selected() is None


def test_command_palette_move_selection():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/co")
    assert len(palette._filtered) == 9
    palette.move_selection(1)
    assert palette._selected_index == 1


def test_assistant_message_accumulates_deltas():
    from swarmee_river.tui.widgets import AssistantMessage

    msg = AssistantMessage()
    msg._buffer.append("Hello ")
    msg._buffer.append("**world**")
    assert msg.full_text == "Hello **world**"
    assert msg.finalize() == "Hello **world**"


def test_assistant_message_finalize_renders_model_and_timestamp_metadata():
    from swarmee_river.tui.widgets import AssistantMessage

    msg = AssistantMessage(model="gpt-5-mini", timestamp="2:34 PM")
    msg._buffer.append("Hello")
    assert msg.finalize() == "Hello"
    content = msg._Static__content  # type: ignore[attr-defined]
    renderables = getattr(content, "renderables", [])
    assert len(renderables) == 2
    assert getattr(renderables[1], "plain", "") == "gpt-5-mini · 2:34 PM"


def test_user_message_renders_timestamp_when_provided():
    from swarmee_river.tui.widgets import UserMessage

    msg = UserMessage("hello", timestamp="2:34 PM")
    content = msg._Static__content  # type: ignore[attr-defined]
    assert "[dim]2:34 PM[/dim]" in str(content)


def test_status_bar_refresh_display():
    from swarmee_river.tui.widgets import StatusBar

    bar = StatusBar()

    # Access internal content set by update()
    def _get_text(widget):
        return widget._Static__content  # type: ignore[attr-defined]

    assert "idle" in _get_text(bar)

    bar.set_state("running")
    bar.set_model("Model: openai/balanced")
    bar.set_tool_count(3)
    bar.set_elapsed(12.4)
    text = _get_text(bar)
    assert "running" in text
    assert "tools 3" in text
    assert "12.4s" in text


def test_status_bar_counts():
    from swarmee_river.tui.widgets import StatusBar

    bar = StatusBar()
    bar.set_counts(warnings=2, errors=1)
    text = bar._Static__content  # type: ignore[attr-defined]
    assert "warn=2" in text
    assert "err=1" in text


def test_status_bar_shows_context_high_warning_over_90_percent():
    from swarmee_river.tui.widgets import StatusBar

    bar = StatusBar()
    bar.set_context(prompt_tokens_est=19_000, budget_tokens=20_000)
    text = bar._Static__content  # type: ignore[attr-defined]
    assert "CTX HIGH" in text


def test_status_bar_shows_estimated_and_last_provider_request_tokens() -> None:
    from swarmee_river.tui.widgets import StatusBar

    bar = StatusBar()
    bar.set_context(prompt_tokens_est=16_300, budget_tokens=200_000)
    bar.set_provider_usage(input_tokens=50_000, cached_input_tokens=10_300, output_tokens=1_200, cost_usd=0.0123)
    text = bar._Static__content  # type: ignore[attr-defined]
    assert "est 16.3k/200k" in text
    assert "cost $0.0123" in text
    assert "req 50.0k" in text
    assert "cache 10.3k" in text
    assert "out 1.20k" in text


def test_context_budget_bar_renders_warning_and_prompt_estimate():
    from swarmee_river.tui.widgets import ContextBudgetBar

    bar = ContextBudgetBar()
    bar.set_context(prompt_tokens_est=45_000, budget_tokens=50_000, animate=False)
    bar.set_provider_usage(input_tokens=50_000, cached_input_tokens=10_300, output_tokens=1_200, cost_usd=0.0123)
    bar.set_prompt_input_estimate(250)
    plain = bar.plain_text
    assert "Est: 45k / 50k (90%)" in plain
    assert "Req: 50k" in plain
    assert "Cache: 10.3k" in plain
    assert "Cost: $0.0123" in plain
    assert "Draft: ~250" in plain
    assert "⚠" in plain
    assert getattr(bar, "tooltip", None) == "Context nearly full. Consider /compact or /new."


def test_prompt_history_helpers_restore_cached_draft() -> None:
    from swarmee_river.tui.app import prompt_history_next, prompt_history_previous

    history = ["first", "second", "third"]
    index, draft, entry = prompt_history_previous(
        history,
        current_index=-1,
        draft_text=None,
        current_text="draft prompt",
    )
    assert (index, draft, entry) == (2, "draft prompt", "third")

    index, draft, entry = prompt_history_previous(
        history,
        current_index=index,
        draft_text=draft,
        current_text="ignored",
    )
    assert (index, draft, entry) == (1, "draft prompt", "second")

    index, draft, entry = prompt_history_next(
        history,
        current_index=index,
        draft_text=draft,
    )
    assert (index, draft, entry) == (2, "draft prompt", "third")

    index, draft, entry = prompt_history_next(
        history,
        current_index=index,
        draft_text=draft,
    )
    assert (index, draft, entry) == (-1, None, "draft prompt")


def test_command_palette_includes_copy_last():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/copy l")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/copy last"


def test_command_palette_includes_open_and_search():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/op")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/open"

    palette.filter("/se")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/search"


def test_command_palette_includes_sop():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/so")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/sop"


def test_command_palette_includes_text_toggle():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/te")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/text"


def test_command_palette_includes_thinking():
    from swarmee_river.tui.widgets import CommandPalette

    palette = CommandPalette()
    palette.filter("/th")
    assert len(palette._filtered) == 1
    assert palette._filtered[0][0] == "/thinking"


def test_render_thinking_indicator_stays_single_line_without_preview_text():
    from swarmee_river.tui.widgets import render_thinking_indicator

    rendered = render_thinking_indicator(char_count=1247, elapsed_s=8.1, preview="line one\nline two")
    plain = rendered.plain
    assert "thinking" in plain
    assert "1,247 chars" in plain
    assert "8s" in plain
    assert "line two" not in plain


def test_reasoning_unavailable_notice_emits_once_for_responses_or_bedrock_reasoning_modes():
    class _Harness(ThinkingMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.daemon.provider = "bedrock"
            self.state.daemon.tier = "deep"
            self.state.daemon.tiers = [
                {
                    "name": "deep",
                    "provider": "bedrock",
                    "model_id": "us.anthropic.claude-opus-4-6-v1",
                    "reasoning_effort": "high",
                    "reasoning_mode": "adaptive",
                }
            ]
            self._thinking_seen_turn = False
            self._thinking_unavailable_notice_emitted_turn = False
            self.lines: list[str] = []

        def _write_transcript_line(self, line: str) -> None:
            self.lines.append(line)

    app = _Harness()
    app._maybe_emit_reasoning_unavailable_notice()
    app._maybe_emit_reasoning_unavailable_notice()
    assert app.lines == ["[thinking] no reasoning stream was emitted by the model for this turn."]


def test_openai_responses_without_reasoning_effort_does_not_emit_unavailable_notice():
    class _Harness(ThinkingMixin):
        def __init__(self) -> None:
            self.state = AppState()
            self.state.daemon.provider = "openai"
            self.state.daemon.tier = "balanced"
            self.state.daemon.tiers = [
                {
                    "name": "balanced",
                    "provider": "openai",
                    "model_id": "gpt-5-mini",
                    "transport": "responses",
                    "reasoning_effort": None,
                    "reasoning_mode": "none",
                }
            ]
            self._thinking_seen_turn = False
            self._thinking_unavailable_notice_emitted_turn = False
            self.lines: list[str] = []

        def _write_transcript_line(self, line: str) -> None:
            self.lines.append(line)

    app = _Harness()
    app._maybe_emit_reasoning_unavailable_notice()
    assert app.lines == []


def test_action_sheet_selection_wraps():
    from swarmee_river.tui.widgets import ActionSheet

    sheet = ActionSheet()
    sheet.set_actions(
        title="Actions",
        actions=[
            {"id": "one", "icon": "1", "label": "One", "shortcut": "A"},
            {"id": "two", "icon": "2", "label": "Two", "shortcut": "B"},
        ],
    )
    assert sheet.selected_action_id() == "one"
    sheet.move_selection(1)
    assert sheet.selected_action_id() == "two"
    sheet.move_selection(1)
    assert sheet.selected_action_id() == "one"


def test_session_save_load(tmp_path, monkeypatch):
    """Session save/load round-trips prompt history and settings."""
    import swarmee_river.tui.app as app_mod

    monkeypatch.setattr(app_mod, "sessions_dir", lambda: tmp_path)

    session_file = tmp_path / "tui_session.json"
    import json

    data = {
        "prompt_history": ["hello", "world"],
        "last_prompt": "world",
        "plan_text": "my plan",
        "artifacts": ["/tmp/a.txt"],
        "model_provider_override": "openai",
        "model_tier_override": "fast",
        "default_auto_approve": True,
        "split_ratio": 3,
    }
    session_file.write_text(json.dumps(data))

    loaded = json.loads(session_file.read_text())
    assert loaded["prompt_history"] == ["hello", "world"]
    assert loaded["split_ratio"] == 3
    assert loaded["model_provider_override"] == "openai"


def test_stop_process_escalates(monkeypatch):
    class FakeProc:
        def __init__(self) -> None:
            self.send_signal_calls: list[int] = []
            self.terminate_called = False
            self.kill_called = False
            self.wait_calls = 0

        def poll(self):
            return None

        def send_signal(self, sig: int) -> None:
            self.send_signal_calls.append(sig)

        def terminate(self) -> None:
            self.terminate_called = True

        def kill(self) -> None:
            self.kill_called = True

        def wait(self, timeout: float):
            self.wait_calls += 1
            raise tui_app.subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

    proc = FakeProc()
    monkeypatch.setattr(tui_app.os, "name", "posix")
    killpg_calls: list[tuple[int, int]] = []

    def _fake_killpg(pid: int, sig: int) -> None:
        killpg_calls.append((pid, sig))

    monkeypatch.setattr(tui_app.os, "killpg", _fake_killpg)
    monkeypatch.setattr(proc, "pid", 4242, raising=False)

    tui_app.stop_process(proc, timeout_s=0.01)

    if hasattr(tui_app.signal, "SIGINT"):
        assert proc.send_signal_calls == []
        expected_signals = [tui_app.signal.SIGINT, tui_app.signal.SIGTERM, tui_app.signal.SIGKILL]
        assert [sig for (_, sig) in killpg_calls] == expected_signals
        assert all(pid == 4242 for (pid, _) in killpg_calls)
    else:
        assert proc.send_signal_calls == []
    assert proc.terminate_called is False
    assert proc.kill_called is False


# ── event_router direct-import tests ──


def test_event_router_exports_classify_and_summarize() -> None:
    from swarmee_river.tui.event_router import classify_tui_error_event, summarize_error_for_toast

    event = {"message": "rate limit exceeded", "category": "transient", "retryable": True, "retry_after_s": 3}
    result = classify_tui_error_event(event)
    assert result["category"] == "transient"
    assert result["retry_after_s"] == 3

    msg, sev, _timeout = summarize_error_for_toast(result)
    assert "3s" in msg
    assert sev == "warning"


def test_summarize_error_for_toast_auth_error() -> None:
    msg, sev, timeout = tui_app.summarize_error_for_toast({"category": "auth_error", "message": "bad creds"})
    assert sev == "error"
    assert timeout == 10.0
    assert "credentials" in msg.lower()


def test_summarize_error_for_toast_escalatable() -> None:
    msg, sev, timeout = tui_app.summarize_error_for_toast({"category": "escalatable", "message": "context exceeded"})
    assert sev == "warning"
    assert timeout == 8.0


def test_summarize_error_for_toast_tool_error_with_id() -> None:
    msg, sev, _timeout = tui_app.summarize_error_for_toast(
        {"category": "tool_error", "message": "failed", "tool_use_id": "tu_123"}
    )
    assert "tu_123" in msg
    assert sev == "error"


def test_summarize_error_for_toast_fatal_fallback() -> None:
    msg, sev, _timeout = tui_app.summarize_error_for_toast({"category": "fatal", "message": ""})
    assert msg == "Fatal error"
    assert sev == "error"


def test_classify_tui_error_event_string_retry_after() -> None:
    result = tui_app.classify_tui_error_event({"message": "throttled", "retry_after_s": "7"})
    assert result["retry_after_s"] == 7


def test_classify_tui_error_event_missing_message_uses_text() -> None:
    result = tui_app.classify_tui_error_event({"text": "something broke"})
    assert result["message"] == "something broke"


def test_warning_event_handler_calls_notify() -> None:
    from swarmee_river.tui.event_router import _handle_error_warning_events

    class FakeState:
        class session:
            warning_count = 0
            error_count = 0

    notifications = []
    issues = []

    class FakeApp:
        state = FakeState()

        def _write_issue(self, text):
            issues.append(text)

        def _update_header_status(self):
            pass

        def _notify(self, msg, *, severity="information", timeout=2.5):
            notifications.append((msg, severity, timeout))

    app = FakeApp()
    event = {"text": "some warning"}
    result = _handle_error_warning_events(app, "warning", event)
    assert result is True
    assert app.state.session.warning_count == 1
    assert len(notifications) == 1
    assert notifications[0][1] == "warning"
    assert "WARN:" in issues[0]


def test_handle_output_line_suppresses_duplicate_raw_assistant_echo() -> None:
    class _Harness(OutputMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(query_active=True, turn_output_chunks=[]),
                session=SimpleNamespace(warning_count=0, error_count=0),
            )
            self._structured_assistant_seen_turn = True
            self._last_structured_assistant_text_turn = "Hello from the model."
            self._current_assistant_chunks = []
            self._streaming_buffer = []
            self._active_assistant_message = None
            self._raw_assistant_lines_suppressed_turn = 0
            self._callback_event_trace_turn = ["llm_start", "text_delta(first)"]
            self.plain: list[str] = []
            self.issues: list[str] = []

        def _append_plain_text(self, text: str) -> None:
            self.plain.append(text)

        def _apply_consent_capture(self, _line: str) -> None:
            return None

        def _write_issue(self, text: str) -> None:
            self.issues.append(text)

        def _update_header_status(self) -> None:
            return None

        def _add_artifact_paths(self, _paths: list[str]) -> None:
            return None

    app = _Harness()
    app._handle_output_line("Hello from the model.")

    assert app.plain == []
    assert app._raw_assistant_lines_suppressed_turn == 1
    assert app._callback_event_trace_turn[-1] == "raw_suppressed"
    assert app.state.daemon.turn_output_chunks == ["Hello from the model.\n"]


def test_tui_output_path_does_not_duplicate_assistant_text_for_dual_strands_callbacks() -> None:
    class _Timer:
        def stop(self) -> None:
            return None

    class _Harness(OutputMixin, ToolsMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(query_active=True, turn_output_chunks=[], current_model="gpt-5-nano"),
                session=SimpleNamespace(warning_count=0, error_count=0),
            )
            self._structured_assistant_seen_turn = False
            self._last_structured_assistant_text_turn = ""
            self._current_assistant_chunks: list[str] = []
            self._streaming_buffer: list[str] = []
            self._active_assistant_message = None
            self._raw_assistant_lines_suppressed_turn = 0
            self._callback_event_trace_turn: list[str] = []
            self._assistant_completion_seen_turn = False
            self._current_assistant_model = None
            self._current_assistant_timestamp = None
            self._assistant_placeholder_written = False
            self._streaming_flush_timer = None
            self._streaming_last_flush_mono = 0.0
            self._stream_render_warning_emitted_turn = False
            self._tool_progress_flush_timer = None
            self._tool_progress_pending_ids: set[str] = set()
            self._tool_blocks: dict[str, dict[str, object]] = {}
            self._last_transcript_dedup_line = ""
            self._last_transcript_dedup_count = 0
            self.widgets: list[object] = []
            self.fallback: list[str] = []

        def _append_plain_text(self, text: str) -> None:
            self.fallback.append(text)

        def _apply_consent_capture(self, _line: str) -> None:
            return None

        def _write_issue(self, _text: str) -> None:
            return None

        def _update_header_status(self) -> None:
            return None

        def _add_artifact_paths(self, _paths: list[str]) -> None:
            return None

        def _record_thinking_event(self, _text: str) -> None:
            return None

        def _dismiss_thinking(self, *, emit_summary: bool) -> None:
            del emit_summary
            return None

        def _trace_turn_event(self, label: str) -> None:
            self._callback_event_trace_turn.append(label)

        def _turn_timestamp(self) -> str:
            return "12:34 PM"

        def _is_transcript_following_tail(self, *, threshold: float = 0.95) -> bool:
            del threshold
            return True

        def _sync_live_transcript_after_append(self, *, follow_tail: bool) -> None:
            del follow_tail
            return None

        def _record_transcript_fallback(self, text: str) -> None:
            self.fallback.append(text)

        def _mount_transcript_widget(self, widget, *, plain_text=None) -> None:  # noqa: ANN001
            del plain_text
            self.widgets.append(widget)

        def set_timer(self, _delay: float, callback):  # noqa: ANN001
            callback()
            return _Timer()

    handler = TuiCallbackHandler()
    app = _Harness()

    lines: list[str] = []

    original_stdout = sys.stdout
    try:
        import io

        buf = io.StringIO()
        sys.stdout = buf
        handler.callback_handler(event={"contentBlockDelta": {"delta": {"text": "hel"}}})
        handler.callback_handler(data="hel", delta={"text": "hel"})
        handler.callback_handler(event={"contentBlockDelta": {"delta": {"text": "lo"}}})
        handler.callback_handler(data="lo", delta={"text": "lo"})
        handler.callback_handler(message={"role": "assistant", "content": [{"text": "hello"}]})
        lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    finally:
        sys.stdout = original_stdout

    for line in lines:
        app._handle_output_line(line, raw_line=f"{line}\n")
    app._finalize_assistant_message()

    assert len(app.widgets) == 1
    assistant_widget = app.widgets[0]
    assert getattr(assistant_widget, "full_text", "") == "hello"
    assert app._current_assistant_chunks == []
    assert app.fallback.count("hel") == 1
    assert app.fallback.count("lo") == 1


def test_handle_output_line_preserves_warning_during_structured_turn() -> None:
    class _Harness(OutputMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(query_active=True, turn_output_chunks=[]),
                session=SimpleNamespace(warning_count=0, error_count=0),
            )
            self._structured_assistant_seen_turn = True
            self._last_structured_assistant_text_turn = "Hello from the model."
            self._current_assistant_chunks = []
            self._streaming_buffer = []
            self._active_assistant_message = None
            self._raw_assistant_lines_suppressed_turn = 0
            self._callback_event_trace_turn = []
            self.issues: list[str] = []
            self.plain: list[str] = []

        def _append_plain_text(self, text: str) -> None:
            self.plain.append(text)

        def _apply_consent_capture(self, _line: str) -> None:
            return None

        def _write_issue(self, text: str) -> None:
            self.issues.append(text)

        def _update_header_status(self) -> None:
            return None

        def _add_artifact_paths(self, _paths: list[str]) -> None:
            return None

    app = _Harness()
    app._handle_output_line("UserWarning: callback path was slow")

    assert app._raw_assistant_lines_suppressed_turn == 0
    assert app.state.session.warning_count == 1
    assert app.plain == []
    assert any("WARN:" in issue for issue in app.issues)


def test_warning_event_handler_uses_connect_popup_instead_of_toast() -> None:
    from swarmee_river.tui.event_router import _handle_error_warning_events

    class FakeState:
        class session:
            warning_count = 0
            error_count = 0

    notifications = []
    issues = []
    popup_lines = []

    class FakeApp:
        state = FakeState()

        def _write_issue(self, text):
            issues.append(text)

        def _update_header_status(self):
            pass

        def _notify(self, msg, *, severity="information", timeout=2.5):
            notifications.append((msg, severity, timeout))

        def _handle_connect_status_warning(self, text):
            popup_lines.append(text)
            return True

    app = FakeApp()
    event = {"text": "visit URL and enter code ABCD-EFGH"}
    result = _handle_error_warning_events(app, "warning", event)
    assert result is True
    assert app.state.session.warning_count == 1
    assert len(notifications) == 0
    assert popup_lines == ["visit URL and enter code ABCD-EFGH"]
    assert "WARN:" in issues[0]


def test_connect_popup_model_info_keeps_warning_capture_enabled() -> None:
    class _PopupScreen:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def append_line(self, line: str) -> None:
            self.lines.append(line)

    class _Harness(DaemonMixin):
        def __init__(self) -> None:
            self._auth_connect_screen = _PopupScreen()
            self._auth_connect_capture_warnings = True
            self._auth_connect_completion_announced = False

    harness = _Harness()
    harness._handle_connect_model_info_event({})
    harness._handle_connect_model_info_event({})

    assert harness._auth_connect_screen.lines == [
        "Authentication status refreshed.",
        "You can close this popup.",
    ]
    assert harness._auth_connect_capture_warnings is True
    assert harness._auth_connect_completion_announced is True


def test_warning_after_model_info_stays_in_popup_until_close() -> None:
    from swarmee_river.tui.event_router import _handle_error_warning_events

    class _PopupScreen:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def append_line(self, line: str) -> None:
            self.lines.append(line)

    class _State:
        class session:
            warning_count = 0
            error_count = 0

    class _Harness(DaemonMixin):
        state = _State()

        def __init__(self) -> None:
            self._auth_connect_screen = _PopupScreen()
            self._auth_connect_provider = "bedrock"
            self._auth_connect_capture_warnings = True
            self._auth_connect_completion_announced = False
            self.issues: list[str] = []
            self.notifications: list[tuple[str, str, float]] = []

        def _write_issue(self, text: str) -> None:
            self.issues.append(text)

        def _update_header_status(self) -> None:
            return None

        def _notify(self, msg: str, *, severity: str = "information", timeout: float = 2.5) -> None:
            self.notifications.append((msg, severity, timeout))

    harness = _Harness()
    harness._handle_connect_model_info_event({})
    handled = _handle_error_warning_events(harness, "warning", {"text": "Open this URL to continue."})

    assert handled is True
    assert harness.notifications == []
    assert harness._auth_connect_screen.lines == [
        "Authentication status refreshed.",
        "You can close this popup.",
        "Open this URL to continue.",
    ]


def test_warning_after_popup_close_uses_toast_fallback() -> None:
    from swarmee_river.tui.event_router import _handle_error_warning_events

    class _PopupScreen:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def append_line(self, line: str) -> None:
            self.lines.append(line)

    class _State:
        class session:
            warning_count = 0
            error_count = 0

    class _Harness(DaemonMixin):
        state = _State()

        def __init__(self) -> None:
            self._auth_connect_screen = _PopupScreen()
            self._auth_connect_provider = "bedrock"
            self._auth_connect_capture_warnings = True
            self._auth_connect_completion_announced = False
            self.issues: list[str] = []
            self.notifications: list[tuple[str, str, float]] = []

        def _write_issue(self, text: str) -> None:
            self.issues.append(text)

        def _update_header_status(self) -> None:
            return None

        def _notify(self, msg: str, *, severity: str = "information", timeout: float = 2.5) -> None:
            self.notifications.append((msg, severity, timeout))

    harness = _Harness()
    popup = harness._auth_connect_screen
    harness._on_auth_connect_popup_closed(popup)

    handled = _handle_error_warning_events(harness, "warning", {"text": "Credentials refreshed."})

    assert handled is True
    assert popup.lines == []
    assert len(harness.notifications) == 1
    assert harness.notifications[0][0] == "WARN: Credentials refreshed."


def test_bedrock_region_warning_shows_guidance_once_without_toast() -> None:
    from swarmee_river.tui.event_router import _handle_error_warning_events

    class _FakeApp:
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                session=SimpleNamespace(
                    warning_count=0,
                    error_count=0,
                    bedrock_setup_guidance_shown=False,
                )
            )
            self.issues: list[str] = []
            self.notifications: list[tuple[str, str, float]] = []
            self.guidance: list[str] = []
            self.tabs: list[str] = []
            self.views: list[str] = []

        def _write_issue(self, text: str) -> None:
            self.issues.append(text)

        def _update_header_status(self) -> None:
            return None

        def _notify(self, msg: str, *, severity: str = "information", timeout: float = 2.5) -> None:
            self.notifications.append((msg, severity, timeout))

        def _handle_connect_status_warning(self, _text: str) -> bool:
            return False

        def render_system_message(self, text: str):  # noqa: ANN201
            return text

        def _mount_transcript_widget(self, _widget, *, plain_text: str) -> None:  # noqa: ANN001
            self.guidance.append(plain_text)

        def _switch_side_tab(self, tab: str) -> None:
            self.tabs.append(tab)

        def _set_settings_view_mode(self, mode: str) -> None:
            self.views.append(mode)

        def _refresh_settings_models(self) -> None:
            return None

    app = _FakeApp()
    warning = (
        "Bedrock model_id 'us.anthropic.claude-haiku-4-5-20251001-v1:0' is prefixed but AWS region is not set; "
        "set AWS_REGION/AWS_DEFAULT_REGION, configure an AWS profile region, or set region_name explicitly."
    )
    handled = _handle_error_warning_events(app, "warning", {"text": warning})

    assert handled is True
    assert app.state.session.warning_count == 1
    assert len(app.issues) == 1
    assert app.notifications == []
    assert len(app.guidance) == 1
    assert "Configure AWS profile and region" in app.guidance[0]
    assert app.state.session.bedrock_setup_guidance_shown is True
    assert app.tabs == ["tab_settings"]
    assert app.views == ["models"]


def test_bedrock_region_warning_guidance_is_deduped_per_session() -> None:
    from swarmee_river.tui.event_router import _handle_error_warning_events

    class _FakeApp:
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                session=SimpleNamespace(
                    warning_count=0,
                    error_count=0,
                    bedrock_setup_guidance_shown=False,
                )
            )
            self.guidance: list[str] = []

        def _write_issue(self, _text: str) -> None:
            return None

        def _update_header_status(self) -> None:
            return None

        def _notify(self, _msg: str, *, severity: str = "information", timeout: float = 2.5) -> None:
            _ = (severity, timeout)
            return None

        def _handle_connect_status_warning(self, _text: str) -> bool:
            return False

        def render_system_message(self, text: str):  # noqa: ANN201
            return text

        def _mount_transcript_widget(self, _widget, *, plain_text: str) -> None:  # noqa: ANN001
            self.guidance.append(plain_text)

        def _switch_side_tab(self, _tab: str) -> None:
            return None

        def _set_settings_view_mode(self, _mode: str) -> None:
            return None

        def _refresh_settings_models(self) -> None:
            return None

    app = _FakeApp()
    warning = (
        "Bedrock model_id 'us.anthropic.claude-haiku-4-5-20251001-v1:0' is prefixed but AWS region is not set; "
        "set AWS_REGION/AWS_DEFAULT_REGION, configure an AWS profile region, or set region_name explicitly."
    )

    _handle_error_warning_events(app, "warning", {"text": warning})
    _handle_error_warning_events(app, "warning", {"text": warning})

    assert len(app.guidance) == 1


def test_usage_event_handler_computes_fallback_cost_when_event_omits_cost(
    monkeypatch, tmp_path: Path
) -> None:
    from swarmee_river.tui.event_router import _handle_usage_and_compaction_events

    # Pricing overrides are settings-driven (env knobs removed).
    (tmp_path / ".swarmee").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".swarmee" / "settings.json").write_text(
        '{"pricing":{"providers":{"openai":{"input_per_1m":2,"cached_input_per_1m":2,"output_per_1m":8}}}}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    status_calls: list[tuple[int | None, int | None, int | None, float | None]] = []

    class _StatusBar:
        def set_provider_usage(
            self,
            *,
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            cost_usd=None,
        ) -> None:  # noqa: ANN001
            status_calls.append((input_tokens, cached_input_tokens, output_tokens, cost_usd))

    class FakeApp:
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(
                    last_usage=None,
                    last_cost_usd=None,
                    last_prompt_tokens_est=None,
                    last_budget_tokens=None,
                    last_provider_input_tokens=None,
                    last_provider_cached_input_tokens=None,
                    last_provider_output_tokens=None,
                )
            )
            self._status_bar = _StatusBar()
            self.refresh_calls = 0

        def _refresh_prompt_metrics(self) -> None:
            self.refresh_calls += 1

    app = FakeApp()
    event = {
        "usage": {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 250},
        "provider": "openai",
        "model_id": "custom-openai-model",
    }
    handled = _handle_usage_and_compaction_events(app, "usage", event)
    assert handled is True
    assert app.state.daemon.last_usage == {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 250}
    assert app.state.daemon.last_cost_usd == 0.006
    assert app.state.daemon.last_provider_input_tokens == 1250
    assert app.state.daemon.last_provider_cached_input_tokens == 250
    assert app.state.daemon.last_provider_output_tokens == 500
    assert status_calls == [(1250, 250, 500, 0.006)]
    assert app.refresh_calls == 1


def test_usage_event_handler_prefers_event_cost_over_fallback(monkeypatch, tmp_path: Path) -> None:
    from swarmee_river.tui.event_router import _handle_usage_and_compaction_events

    (tmp_path / ".swarmee").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".swarmee" / "settings.json").write_text(
        '{"pricing":{"providers":{"openai":{"input_per_1m":999,"output_per_1m":999}}}}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    status_costs: list[tuple[int | None, int | None, int | None, float | None]] = []

    class _StatusBar:
        def set_provider_usage(
            self,
            *,
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            cost_usd=None,
        ) -> None:
            status_costs.append((input_tokens, cached_input_tokens, output_tokens, cost_usd))

    class FakeApp:
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(
                    last_usage=None,
                    last_cost_usd=None,
                    last_prompt_tokens_est=None,
                    last_budget_tokens=None,
                    last_provider_input_tokens=None,
                    last_provider_cached_input_tokens=None,
                    last_provider_output_tokens=None,
                )
            )
            self._status_bar = _StatusBar()

        def _refresh_prompt_metrics(self) -> None:
            return None

    app = FakeApp()
    handled = _handle_usage_and_compaction_events(
        app,
        "usage",
        {"usage": {"input_tokens": 1000, "output_tokens": 500}, "provider": "openai", "cost_usd": 0.1234},
    )
    assert handled is True
    assert app.state.daemon.last_cost_usd == 0.1234
    assert status_costs == [(1000, 0, 500, 0.1234)]


def test_compact_complete_event_writes_transcript_and_updates_budget() -> None:
    from swarmee_river.tui.event_router import _handle_usage_and_compaction_events

    notifications: list[tuple[str, str, float]] = []
    transcript_lines: list[str] = []

    class _StatusBar:
        def __init__(self) -> None:
            self.context_calls: list[tuple[int | None, int | None]] = []

        def set_context(self, *, prompt_tokens_est, budget_tokens) -> None:  # noqa: ANN001
            self.context_calls.append((prompt_tokens_est, budget_tokens))

    class FakeApp:
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(
                    last_usage=None,
                    last_cost_usd=None,
                    last_prompt_tokens_est=180000,
                    last_budget_tokens=200000,
                )
            )
            self._status_bar = _StatusBar()
            self.refresh_calls = 0

        def _refresh_prompt_metrics(self) -> None:
            self.refresh_calls += 1

        def _notify(self, text: str, severity: str = "information", timeout: float = 0.0) -> None:
            notifications.append((text, severity, timeout))

        def _write_transcript_line(self, text: str) -> None:
            transcript_lines.append(text)

    app = FakeApp()
    handled = _handle_usage_and_compaction_events(
        app,
        "compact_complete",
        {
            "automatic": True,
            "compacted": True,
            "before_tokens_est": 230000,
            "after_tokens_est": 180000,
            "budget_tokens": 200000,
            "summary_passes": 1,
            "trimmed_messages": 2,
            "compacted_read_results": 3,
        },
    )

    assert handled is True
    assert app.state.daemon.last_budget_tokens == 200000
    assert app.state.daemon.last_prompt_tokens_est == 180000
    assert "auto-compacted 180,000/200,000 tokens" in transcript_lines[0]
    assert "summarized 1 pass" in transcript_lines[0]
    assert notifications == [("Context compacted.", "information", 4.0)]


def test_flush_streaming_buffer_keeps_transcript_follow_state() -> None:
    class _AssistantMessage:
        def __init__(self) -> None:
            self.chunks: list[str] = []

        def append_delta(self, text: str) -> None:
            self.chunks.append(text)

    class _ToolsHarness(ToolsMixin):
        def __init__(self) -> None:
            self._streaming_buffer = ["hello"]
            self._current_assistant_chunks: list[str] = []
            self._active_assistant_message = _AssistantMessage()
            self._current_assistant_model = None
            self._current_assistant_timestamp = None
            self._assistant_placeholder_written = False
            self.follow_calls: list[bool] = []
            self.fallback: list[str] = []

        def _is_transcript_following_tail(self, *, threshold: float = 0.95) -> bool:
            del threshold
            return True

        def _sync_live_transcript_after_append(self, *, follow_tail: bool) -> None:
            self.follow_calls.append(follow_tail)

        def _mount_transcript_widget(self, _renderable, *, plain_text=None) -> None:  # noqa: ANN001
            del plain_text
            return None

        def _record_transcript_fallback(self, text: str) -> None:
            self.fallback.append(text)

    app = _ToolsHarness()
    app._flush_streaming_buffer()
    assert app._current_assistant_chunks == ["hello"]
    assert app._active_assistant_message.chunks == ["hello"]
    assert app.follow_calls == [True]
    assert app.fallback == ["hello"]


def test_flush_tool_progress_render_keeps_transcript_follow_state() -> None:
    class _ToolWidget:
        def __init__(self) -> None:
            self.outputs: list[tuple[str, str]] = []

        def append_output(self, content: str, *, stream: str = "stdout") -> None:
            self.outputs.append((content, stream))

    class _ToolsHarness(ToolsMixin):
        def __init__(self) -> None:
            self.follow_calls: list[bool] = []
            self.fallback: list[str] = []
            self._tool_blocks = {
                "tool-1": {
                    "tool_use_id": "tool-1",
                    "tool": "shell",
                    "start_rendered": True,
                    "widget": _ToolWidget(),
                    "pending_output": "line\n",
                    "pending_stream": "stdout",
                    "last_progress_render_mono": 0.0,
                }
            }

        def _is_transcript_following_tail(self, *, threshold: float = 0.95) -> bool:
            del threshold
            return False

        def _sync_live_transcript_after_append(self, *, follow_tail: bool) -> None:
            self.follow_calls.append(follow_tail)

        def _mount_transcript_widget(self, _renderable, *, plain_text=None) -> None:  # noqa: ANN001
            del plain_text
            return None

        def _record_transcript_fallback(self, text: str) -> None:
            self.fallback.append(text)

        def _emit_tool_start_line(self, _tool_use_id: str) -> bool:
            return True

    app = _ToolsHarness()
    rendered = app._flush_tool_progress_render("tool-1", force=True)
    assert rendered is True
    record = app._tool_blocks["tool-1"]
    assert record["widget"].outputs == [("line\n", "stdout")]
    assert record["pending_output"] == ""
    assert app.follow_calls == [False]
    assert app.fallback == ["line\n"]


def test_call_from_thread_safe_retries_backlog_and_eventually_delivers_callback() -> None:
    class _Harness(TranscriptMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(daemon=SimpleNamespace(is_shutting_down=False))
            self.failures_remaining = 1
            self.values: list[str] = []
            self.lines: list[str] = []

        def call_from_thread(self, callback, *args, **kwargs) -> None:  # noqa: ANN001
            if self.failures_remaining > 0:
                self.failures_remaining -= 1
                raise RuntimeError("busy")
            callback(*args, **kwargs)

        def _write_transcript_line(self, line: str) -> None:
            self.lines.append(line)

    app = _Harness()
    app._call_from_thread_safe(app.values.append, "first")
    assert app.values == ["first"]
    app._call_from_thread_safe(app.values.append, "second")
    assert app.values == ["first", "second"]
    assert app._thread_dispatch_dropped_total == 0


def test_call_from_thread_safe_drops_overflow_and_emits_throttled_warning() -> None:
    class _Harness(TranscriptMixin):
        _THREAD_DISPATCH_QUEUE_MAX = 1
        _THREAD_DISPATCH_MAX_ATTEMPTS = 2
        _THREAD_DISPATCH_WARN_INTERVAL_S = 0.0

        def __init__(self) -> None:
            self.state = SimpleNamespace(daemon=SimpleNamespace(is_shutting_down=False))
            self.failures_remaining = 99
            self.values: list[str] = []
            self.lines: list[str] = []

        def call_from_thread(self, callback, *args, **kwargs) -> None:  # noqa: ANN001
            if self.failures_remaining > 0:
                self.failures_remaining -= 1
                raise RuntimeError("still busy")
            callback(*args, **kwargs)

        def _write_transcript_line(self, line: str) -> None:
            self.lines.append(line)

    app = _Harness()
    app._call_from_thread_safe(app.values.append, "one")
    app._call_from_thread_safe(app.values.append, "two")
    app._call_from_thread_safe(app.values.append, "three")
    assert app._thread_dispatch_dropped_total >= 1

    app.failures_remaining = 0
    app._call_from_thread_safe(app.values.append, "four")
    assert "four" in app.values
    assert any("dropped" in line for line in app.lines)


def test_flush_streaming_buffer_append_failure_degrades_to_finalize_path() -> None:
    class _BrokenMessage:
        def append_delta(self, _text: str) -> None:
            raise RuntimeError("render failed")

        def remove(self) -> None:
            return None

    class _ToolsHarness(ToolsMixin):
        def __init__(self) -> None:
            self._streaming_buffer = ["hello"]
            self._current_assistant_chunks: list[str] = []
            self._active_assistant_message = _BrokenMessage()
            self._current_assistant_model = "openai/fast"
            self._current_assistant_timestamp = "10:00 AM"
            self._assistant_placeholder_written = False
            self._streaming_last_flush_mono = 0.0
            self._stream_render_warning_emitted_turn = False
            self.fallback: list[str] = []
            self.lines: list[str] = []

        def _is_transcript_following_tail(self, *, threshold: float = 0.95) -> bool:
            del threshold
            return True

        def _sync_live_transcript_after_append(self, *, follow_tail: bool) -> None:
            del follow_tail
            return None

        def _mount_transcript_widget(self, _renderable, *, plain_text=None) -> None:  # noqa: ANN001
            del plain_text
            raise RuntimeError("mount failed")

        def _record_transcript_fallback(self, text: str) -> None:
            self.fallback.append(text)

        def _write_transcript_line(self, line: str) -> None:
            self.lines.append(line)

    app = _ToolsHarness()
    app._flush_streaming_buffer()
    assert app._current_assistant_chunks == ["hello"]
    assert app._active_assistant_message is None
    assert app._assistant_placeholder_written is False
    assert app.fallback == ["hello"]
    assert any("degraded" in line for line in app.lines)


def test_schedule_streaming_flush_immediate_first_chunk_then_debounced(monkeypatch) -> None:
    class _Timer:
        def stop(self) -> None:
            return None

    class _ToolsHarness(ToolsMixin):
        def __init__(self) -> None:
            self._streaming_flush_timer = None
            self._streaming_last_flush_mono = 0.0
            self.flush_calls = 0
            self.timer_delays: list[float] = []

        def _flush_streaming_buffer(self) -> None:
            self.flush_calls += 1
            self._streaming_last_flush_mono = 100.0

        def set_timer(self, delay: float, _callback):  # noqa: ANN001
            self.timer_delays.append(delay)
            return _Timer()

    app = _ToolsHarness()
    monkeypatch.setattr("swarmee_river.tui.mixins.tools.time.monotonic", lambda: 100.0)
    app._schedule_streaming_flush()
    assert app.flush_calls == 1
    assert app.timer_delays == []

    monkeypatch.setattr("swarmee_river.tui.mixins.tools.time.monotonic", lambda: 100.02)
    app._schedule_streaming_flush()
    assert app.flush_calls == 1
    assert len(app.timer_delays) == 1
    assert app.timer_delays[0] > 0.0


def test_streaming_handler_llm_start_triggers_thinking_indicator() -> None:
    from swarmee_river.tui.event_router import _handle_streaming_events

    calls: list[str] = []

    class FakeApp:
        def _record_thinking_event(self, text: str) -> None:
            calls.append(text)

    app = FakeApp()
    handled = _handle_streaming_events(app, "llm_start", {"event": "llm_start"})
    assert handled is True
    assert calls == [""]


def test_streaming_handler_text_delta_accepts_nested_bedrock_payload() -> None:
    from swarmee_river.tui.event_router import _handle_streaming_events

    class FakeApp:
        def __init__(self) -> None:
            self._current_assistant_chunks: list[str] = []
            self._streaming_buffer: list[str] = []
            self._current_assistant_model: str | None = None
            self._current_assistant_timestamp: str | None = None
            self._structured_assistant_seen_turn = False
            self._last_structured_assistant_text_turn = ""
            self._callback_event_trace_turn: list[str] = []
            self.state = SimpleNamespace(daemon=SimpleNamespace(current_model="us.anthropic.claude-sonnet-4"))
            self.dismiss_calls = 0
            self.flush_scheduled = 0

        def _dismiss_thinking(self, *, emit_summary: bool) -> None:
            del emit_summary
            self.dismiss_calls += 1

        def _turn_timestamp(self) -> str:
            return "2026-03-03T00:00:00Z"

        def _schedule_streaming_flush(self) -> None:
            self.flush_scheduled += 1

        def _trace_turn_event(self, label: str) -> None:
            self._callback_event_trace_turn.append(label)

    app = FakeApp()
    handled = _handle_streaming_events(
        app,
        "text_delta",
        {"event": "text_delta", "payload": {"contentBlockDelta": {"delta": {"text": "hello"}}}},
    )
    assert handled is True
    assert app._streaming_buffer == ["hello"]
    assert app._structured_assistant_seen_turn is True
    assert app._last_structured_assistant_text_turn == "hello"
    assert app._callback_event_trace_turn == ["text_delta(first)"]
    assert app.dismiss_calls == 1
    assert app.flush_scheduled == 1


def test_streaming_handler_complete_events_finalize_once_per_turn() -> None:
    from swarmee_river.tui.event_router import _handle_streaming_events

    class FakeApp:
        def __init__(self) -> None:
            self._assistant_completion_seen_turn = False
            self.finalize_calls = 0
            self.flush_calls = 0
            self.cancel_calls = 0
            self.tail_calls = 0

        def _cancel_streaming_flush_timer(self) -> None:
            self.cancel_calls += 1

        def _flush_streaming_buffer(self) -> None:
            self.flush_calls += 1

        def _finalize_assistant_message(self) -> None:
            self.finalize_calls += 1

        def _force_transcript_tail_after_refresh(self) -> None:
            self.tail_calls += 1

    app = FakeApp()
    handled_first = _handle_streaming_events(app, "text_complete", {"event": "text_complete"})
    handled_second = _handle_streaming_events(app, "complete", {"event": "complete"})

    assert handled_first is True
    assert handled_second is True
    assert app.cancel_calls == 1
    assert app.flush_calls == 1
    assert app.finalize_calls == 1
    assert app.tail_calls == 1


def test_suppressed_raw_echo_stays_suppressed_through_complete_and_turn_complete() -> None:
    from swarmee_river.tui.event_router import _handle_streaming_events, handle_daemon_event

    class FakeApp(OutputMixin):
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                daemon=SimpleNamespace(
                    query_active=True,
                    turn_output_chunks=[],
                    current_model="openai/gpt-5.2",
                    last_restored_turn_count=0,
                ),
                session=SimpleNamespace(warning_count=0, error_count=0),
            )
            self._assistant_completion_seen_turn = False
            self._structured_assistant_seen_turn = True
            self._last_structured_assistant_text_turn = "hello world"
            self._current_assistant_chunks = []
            self._streaming_buffer = []
            self._active_assistant_message = None
            self._callback_event_trace_turn = ["llm_start", "text_delta(first)"]
            self._raw_assistant_lines_suppressed_turn = 0
            self.plain: list[str] = []
            self.assistant_finalize_calls = 0
            self.turn_finalize_calls = 0
            self.flush_calls = 0
            self.cancel_calls = 0
            self.timeline_refreshes = 0
            self.tail_calls = 0

        def _append_plain_text(self, text: str) -> None:
            self.plain.append(text)

        def _apply_consent_capture(self, _line: str) -> None:
            return None

        def _write_issue(self, _text: str) -> None:
            return None

        def _update_header_status(self) -> None:
            return None

        def _add_artifact_paths(self, _paths: list[str]) -> None:
            return None

        def _cancel_streaming_flush_timer(self) -> None:
            self.cancel_calls += 1

        def _flush_streaming_buffer(self) -> None:
            self.flush_calls += 1

        def _finalize_assistant_message(self) -> None:
            self.assistant_finalize_calls += 1

        def _force_transcript_tail_after_refresh(self) -> None:
            self.tail_calls += 1

        def _finalize_turn(self, *, exit_status: str) -> None:
            assert exit_status == "ok"
            self.turn_finalize_calls += 1

        def _set_planning_controls_enabled(self, *, enabled: bool) -> None:
            assert enabled is True

        def _reset_error_action_prompt(self) -> None:
            return None

        def _schedule_session_timeline_refresh(self) -> None:
            self.timeline_refreshes += 1

        def _trace_turn_event(self, label: str) -> None:
            self._callback_event_trace_turn.append(label)

    app = FakeApp()
    app._handle_output_line("hello world")
    handled_complete = _handle_streaming_events(app, "text_complete", {"event": "text_complete"})
    handle_daemon_event(app, {"event": "turn_complete", "exit_status": "ok"})

    assert handled_complete is True
    assert app.plain == []
    assert app._raw_assistant_lines_suppressed_turn == 1
    assert app.assistant_finalize_calls == 1
    assert app.turn_finalize_calls == 1
    assert app.timeline_refreshes == 1
    assert app.tail_calls == 1
    assert "text_complete" in app._callback_event_trace_turn
    assert app._callback_event_trace_turn[-1] == "turn_complete"
