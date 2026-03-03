#!/usr/bin/env python3
"""
Unit tests for the swarmee.py module using pytest
"""

import asyncio
import contextlib
import io
import json
import os
import sys
import warnings
from unittest import mock

import pytest

from swarmee_river import swarmee


def test_help_includes_runtime_broker_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["swarmee", "--help"])
    with pytest.raises(SystemExit) as exc:
        swarmee.main()
    assert int(exc.value.code or 0) == 0
    stdout = capsys.readouterr().out
    assert "swarmee daemon stop" in stdout
    assert "swarmee daemon stop all" in stdout
    assert "swarmee broker stop" in stdout


def test_broker_alias_dispatches_to_daemon_command(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_run_daemon_command(raw_args: list[str]) -> int:
        captured["args"] = list(raw_args)
        return 0

    monkeypatch.setattr(swarmee, "_run_daemon_command", _fake_run_daemon_command)
    monkeypatch.setattr(sys, "argv", ["swarmee", "broker", "stop"])
    with pytest.raises(SystemExit) as exc:
        swarmee.main()
    assert int(exc.value.code or 0) == 0
    assert captured.get("args") == ["stop"]


def test_daemon_stop_all_invokes_global_shutdown_helper(monkeypatch, capsys) -> None:
    monkeypatch.setattr(swarmee, "_stop_all_runtime_brokers", lambda timeout_s=6.0: (2, 0))
    result = swarmee._run_daemon_command(["stop", "all"])
    assert result == 0
    assert "stopped 2 broker(s)." in capsys.readouterr().out


def test_runtime_broker_pids_windows_parses_powershell_csv(monkeypatch) -> None:
    monkeypatch.setattr(swarmee.os, "name", "nt", raising=False)
    monkeypatch.setattr(swarmee.os, "getpid", lambda: 1111)
    monkeypatch.setattr(
        swarmee.subprocess,
        "check_output",
        lambda *_args, **_kwargs: (
            '"ProcessId","CommandLine"\n'
            '"1111","python -m swarmee_river.swarmee serve"\n'
            '"2222","python -m swarmee_river.swarmee serve --port 0"\n'
            '"3333","python -m some_other_module serve"\n'
        ),
    )

    assert swarmee._runtime_broker_pids() == [2222]


def test_runtime_broker_pids_windows_falls_back_to_wmic(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _check_output(command: list[str], **_kwargs) -> str:
        calls.append(list(command))
        if command and command[0].lower() == "powershell":
            raise RuntimeError("powershell unavailable")
        return (
            "Node,CommandLine,ProcessId\n"
            'HOST,"python -m swarmee_river.swarmee serve",4444\n'
            'HOST,"python -m unrelated.module run",5555\n'
        )

    monkeypatch.setattr(swarmee.os, "name", "nt", raising=False)
    monkeypatch.setattr(swarmee.os, "getpid", lambda: 9999)
    monkeypatch.setattr(swarmee.subprocess, "check_output", _check_output)

    assert swarmee._runtime_broker_pids() == [4444]
    assert calls and calls[0][0].lower() == "powershell"
    assert any(call and call[0].lower() == "wmic" for call in calls)


def test_stop_all_runtime_brokers_windows_without_sigkill(monkeypatch) -> None:
    import signal as _signal

    monkeypatch.setattr(swarmee, "_runtime_broker_pids", lambda: [7001])
    monkeypatch.delattr(_signal, "SIGKILL", raising=False)

    term_signal = getattr(_signal, "SIGTERM")
    kill_calls: list[tuple[int, int]] = []
    state = {"forced": False, "term_count": 0}

    def _fake_kill(pid: int, sig: int) -> None:
        kill_calls.append((pid, sig))
        if sig == 0:
            if state["forced"]:
                raise OSError("process exited")
            return
        if sig == term_signal:
            state["term_count"] += 1
            if state["term_count"] >= 2:
                state["forced"] = True
            return
        raise AssertionError(f"unexpected signal {sig}")

    monotonic_value = {"t": 0.0}

    def _fake_monotonic() -> float:
        monotonic_value["t"] += 0.3
        return monotonic_value["t"]

    monkeypatch.setattr(swarmee.os, "kill", _fake_kill)
    monkeypatch.setattr(swarmee.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(swarmee.time, "monotonic", _fake_monotonic)

    stopped, failed = swarmee._stop_all_runtime_brokers(timeout_s=0.1)

    assert (stopped, failed) == (1, 0)
    non_probe_signals = [sig for _pid, sig in kill_calls if sig != 0]
    assert non_probe_signals.count(term_signal) >= 2


def test_daemon_start_with_all_target_is_rejected(capsys) -> None:
    result = swarmee._run_daemon_command(["start", "all"])
    assert result == 1
    assert "only valid with 'stop'" in capsys.readouterr().out


def test_build_resolved_invocation_state_includes_session_safety_overrides() -> None:
    from swarmee_river.settings import default_settings_template

    class _Tier:
        def __init__(self) -> None:
            self.name = "balanced"
            self.model_id = "mock-model"

    class _ModelManager:
        current_tier = "balanced"

        @staticmethod
        def list_tiers():
            return [_Tier()]

    resolved = swarmee._build_resolved_invocation_state(
        invocation_state={"swarmee": {"mode": "execute"}},
        runtime_environment={"os": "darwin"},
        model_manager=_ModelManager(),
        selected_provider="bedrock",
        settings=default_settings_template(),
        structured_output_model=None,
        session_safety_overrides={"tool_consent": "deny", "tool_allowlist": ["file_read"]},
    )
    sw_state = resolved["swarmee"]
    assert sw_state["session_safety_overrides"] == {
        "tool_consent": "deny",
        "tool_allowlist": ["file_read"],
    }


def test_write_stdout_jsonl_uses_default_shared_lock(monkeypatch) -> None:
    writer_spy = mock.Mock()
    monkeypatch.setattr(swarmee, "write_stdout_jsonl", writer_spy)

    swarmee._write_stdout_jsonl({"event": "warning", "text": "test"})

    writer_spy.assert_called_once_with({"event": "warning", "text": "test"})


class TestInteractiveMode:
    """Test cases for interactive mode functionality"""

    def test_interactive_mode(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Test the interactive mode of swarmee"""
        # Setup mocks
        mock_user_input.side_effect = ["test query", "exit"]

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        swarmee.main()

        # Verify welcome message was rendered
        mock_welcome_message.assert_called_once()

        # Verify user input was called with the correct parameters
        mock_user_input.assert_called_with("\n~ ", default="", keyboard_interrupt_return_default=False)

        # Verify user input was processed
        call = mock_agent.invoke_async.call_args
        assert call.args[0] == "test query"
        assert call.kwargs["invocation_state"]["swarmee"]["mode"] == "execute"
        assert "structured_output_model" not in call.kwargs
        assert "structured_output_prompt" not in call.kwargs

        # Verify goodbye message was rendered
        mock_goodbye_message.assert_called_once()

    def test_shell_command_shortcut(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Test the shell command shortcut with ! prefix"""
        # Setup mocks
        mock_user_input.side_effect = ["!ls -la", "exit"]

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        swarmee.main()

        # Verify shell was called with the command
        mock_agent.tool.shell.assert_called_with(
            command="ls -la",
            user_message_override="!ls -la",
            non_interactive_mode=True,
            record_direct_tool_call=False,
        )

    def test_keyboard_interrupt(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Test handling of keyboard interrupt (Ctrl+C)"""
        # Setup mocks - simulate keyboard interrupt
        mock_user_input.side_effect = KeyboardInterrupt()

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function - should exit gracefully
        swarmee.main()

        # Verify goodbye message was rendered
        mock_goodbye_message.assert_called_once()

    def test_empty_input(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Test handling of empty input"""
        # Setup mocks - empty input followed by exit
        mock_user_input.side_effect = ["", "   ", "\t", "exit"]

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        swarmee.main()

        # Verify agent's methods were not called for the empty input
        mock_agent.assert_not_called()

    def test_invoke_async_compat_without_structured_output_prompt(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        """Ensure compatibility with SDK variants lacking structured_output_prompt kwarg."""

        async def invoke_async_no_prompt(
            prompt: str,
            *,
            invocation_state: dict[str, object],
            structured_output_model: type[object] | None = None,
        ):
            del invocation_state
            del structured_output_model
            return mock.MagicMock(structured_output=None, message=[{"role": "assistant", "content": [{"text": "ok"}]}])

        mock_agent.invoke_async = invoke_async_no_prompt
        mock_user_input.side_effect = ["test query", "exit"]
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        swarmee.main()

    def test_suppresses_strands_kwargs_deprecation_warning(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        async def invoke_async_with_warning(
            prompt: str,
            *,
            invocation_state: dict[str, object],
            structured_output_model: type[object] | None = None,
            structured_output_prompt: str | None = None,
        ):
            del prompt
            del invocation_state
            del structured_output_model
            del structured_output_prompt
            warnings.warn(
                "`**kwargs` parameter is deprecating, use `invocation_state` instead.",
                UserWarning,
                stacklevel=2,
            )
            return mock.MagicMock(structured_output=None, message=[{"role": "assistant", "content": [{"text": "ok"}]}])

        mock_agent.invoke_async = invoke_async_with_warning
        mock_user_input.side_effect = ["test query", "exit"]
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            swarmee.main()

        assert not any(
            "`**kwargs` parameter is deprecating, use `invocation_state` instead." in str(w.message) for w in caught
        )

    def test_plan_generation_sets_structured_output_tool_allowlist(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        from swarmee_river.planning import PlanStep, WorkPlan

        plan = WorkPlan(
            summary="Fix a bug",
            steps=[PlanStep(description="Inspect failing test", tools_expected=["file_read"])],
        )
        mock_agent.invoke_async = mock.AsyncMock(return_value=mock.MagicMock(structured_output=plan, message=[]))

        monkeypatch.setattr(sys, "argv", ["swarmee", "fix", "the", "bug"])

        swarmee.main()

        call = mock_agent.invoke_async.call_args
        sw_state = call.kwargs["invocation_state"]["swarmee"]
        assert sw_state["mode"] == "plan"
        assert "WorkPlan" in sw_state.get("plan_allowed_tools", [])

    def test_plan_generation_fallback_injects_prompt_when_sdk_lacks_structured_output_prompt(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        monkeypatch,
    ):
        from swarmee_river.planning import WorkPlan

        captured: dict[str, object] = {}

        async def invoke_async_no_structured_prompt(
            prompt: str,
            *,
            invocation_state: dict[str, object],
            structured_output_model: type[object] | None = None,
        ):
            captured["prompt"] = prompt
            captured["invocation_state"] = invocation_state
            del structured_output_model
            return mock.MagicMock(structured_output=WorkPlan(summary="Plan summary", steps=[]), message=[])

        mock_agent.invoke_async = invoke_async_no_structured_prompt
        mock_user_input.side_effect = ["fix bug in runtime", ":cancel", "exit"]
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        swarmee.main()

        prompt_text = str(captured.get("prompt", ""))
        assert "Do NOT produce any text output" in prompt_text
        assert "User request:\nfix bug in runtime" in prompt_text
        sw_state = captured["invocation_state"]["swarmee"]  # type: ignore[index]
        assert sw_state["mode"] == "plan"  # type: ignore[index]

    @mock.patch.object(swarmee, "get_user_input")
    @mock.patch.object(swarmee, "Agent")
    @mock.patch.object(swarmee, "render_goodbye_message")
    def test_keyboard_interrupt_exception(self, mock_goodbye, mock_agent, mock_input):
        """Test handling of KeyboardInterrupt exception in interactive mode"""
        # Setup mocks
        mock_agent_instance = mock.MagicMock()
        mock_agent.return_value = mock_agent_instance

        # Simulate KeyboardInterrupt when getting input
        mock_input.side_effect = KeyboardInterrupt()

        # Run main
        with mock.patch.object(sys, "argv", ["swarmee"]):
            swarmee.main()

        # Verify goodbye message was called
        mock_goodbye.assert_called_once()

    @mock.patch.object(swarmee, "get_user_input")
    @mock.patch.object(swarmee, "Agent")
    @mock.patch.object(swarmee, "render_goodbye_message")
    def test_eof_error_exception(self, mock_goodbye, mock_agent, mock_input):
        """Test handling of EOFError exception in interactive mode"""
        # Setup mocks
        mock_agent_instance = mock.MagicMock()
        mock_agent.return_value = mock_agent_instance

        # Simulate EOFError when getting input
        mock_input.side_effect = EOFError()

        # Run main
        with mock.patch.object(sys, "argv", ["swarmee"]):
            swarmee.main()

        # Verify goodbye message was called
        mock_goodbye.assert_called_once()

    @mock.patch.object(swarmee, "get_user_input")
    @mock.patch.object(swarmee, "Agent")
    @mock.patch.object(swarmee, "print")
    @mock.patch.object(swarmee, "callback_handler")
    def test_general_exception_handling(self, mock_callback_handler, mock_print, mock_agent, mock_input):
        """Test handling of general exceptions in interactive mode"""
        # Setup mocks
        mock_agent_instance = mock.MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke_async = mock.AsyncMock(
            return_value=mock.MagicMock(structured_output=None, message=[])
        )

        # First return valid input, then cause exception, then exit
        mock_input.side_effect = ["test input", Exception("Test error"), "exit"]

        # Run main
        with mock.patch.object(sys, "argv", ["swarmee"]), mock.patch.object(swarmee, "render_goodbye_message"):
            swarmee.main()

        # Verify error was printed
        mock_print.assert_any_call("\nError: Test error")
        mock_callback_handler.assert_called_once_with(force_stop=True)


class TestTuiDaemonMode:
    """Daemon mode tests for long-running TUI subprocess protocol."""

    def test_tui_daemon_emits_ready_and_model_info_on_startup(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"shutdown"}\n'))

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        assert len(events) >= 2
        assert events[0]["event"] == "ready"
        assert isinstance(events[0].get("session_id"), str)
        assert events[1]["event"] == "model_info"
        assert "provider" in events[1]
        assert "tier" in events[1]
        assert "tiers" in events[1]
        assert "tool_names" in events[1]
        assert isinstance(events[1]["tiers"], list)
        assert isinstance(events[1]["tool_names"], list)
        assert events[1]["tool_names"] == sorted(events[1]["tool_names"])

    def test_tui_daemon_set_tier_emits_updated_model_info(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"set_tier","tier":"deep"}\n{"cmd":"shutdown"}\n'))

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        model_events = [event for event in events if event.get("event") == "model_info"]
        assert len(model_events) >= 2
        assert any(event.get("tier") == "deep" for event in model_events)

    def test_tui_daemon_applies_project_env_overrides_before_loading_settings(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"shutdown"}\n'))
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        observed: dict[str, str] = {}

        def _fake_apply(path, overwrite=True):
            del path
            assert overwrite is True
            os.environ["AWS_PROFILE"] = "ds-pr"
            return {"AWS_PROFILE": "ds-pr"}

        def _fake_load_settings(path=None):
            observed["aws_profile_during_load"] = os.getenv("AWS_PROFILE", "")
            from swarmee_river.settings import default_settings_template

            return default_settings_template()

        monkeypatch.setattr(swarmee, "apply_project_env_overrides", _fake_apply)
        monkeypatch.setattr(swarmee, "load_settings", _fake_load_settings)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        assert observed.get("aws_profile_during_load") == "ds-pr"
        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        assert any(event.get("event") == "ready" for event in events)
        assert any(event.get("event") == "model_info" for event in events)

    def test_tui_daemon_connect_bedrock_rebuilds_active_model(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(swarmee, "_connect_aws_credentials", lambda **_kwargs: "profile")
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO('{"cmd":"connect","provider":"bedrock","profile":"dev"}\n{"cmd":"shutdown"}\n'),
        )
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        assert mock_bedrock.call_count == 2
        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        model_events = [event for event in events if event.get("event") == "model_info"]
        assert len(model_events) >= 2

    def test_tui_daemon_query_missing_text_emits_classified_error(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"query"}\n{"cmd":"shutdown"}\n'))

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        error_events = [event for event in events if event.get("event") == "error"]
        assert error_events
        assert error_events[0].get("message") == "query.text is required"
        assert error_events[0].get("category") == "fatal"
        assert error_events[0].get("retryable") is False

    def test_tui_daemon_query_bedrock_without_credentials_fails_fast(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(swarmee, "has_aws_credentials", lambda: False)
        run_query_spy = mock.Mock(return_value=(None, "ok", True))
        monkeypatch.setattr(swarmee, "_run_query_with_optional_plan", run_query_spy)
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"query","text":"hello"}\n{"cmd":"shutdown"}\n'))

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        assert run_query_spy.call_count == 0
        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        error_events = [event for event in events if event.get("event") == "error"]
        assert error_events
        assert error_events[0].get("category") == "auth_error"
        assert any(event.get("event") == "turn_complete" and event.get("exit_status") == "error" for event in events)

    def test_tui_daemon_retry_tool_without_history_emits_tool_error(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"retry_tool","tool_use_id":"t-1"}\n{"cmd":"shutdown"}\n'))

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        error_events = [event for event in events if event.get("event") == "error"]
        assert error_events
        assert error_events[0].get("category") == "tool_error"
        assert error_events[0].get("tool_use_id") == "t-1"

    def test_tui_daemon_set_sop_updates_session_meta(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
        tmp_path,
    ):
        saved_meta: list[dict[str, object]] = []

        class _FakeStore:
            def save(self, session_id, *, meta=None, messages=None, state=None, last_plan=None):
                del session_id, messages, state, last_plan
                if isinstance(meta, dict):
                    saved_meta.append(dict(meta))
                return None

            def list(self, *, limit=50):
                del limit
                return []

            def load_messages(self, session_id, *, max_messages=200, expected_version=1):
                del session_id, max_messages, expected_version
                return []

            def save_messages(self, session_id, messages, *, max_messages=200, version=1):
                del session_id, messages, max_messages, version
                return {"version": 1, "message_count": 0, "turn_count": 0}

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(swarmee, "SessionStore", _FakeStore)
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO('{"cmd":"set_sop","name":"bugfix","content":"Use bugfix SOP"}\n{"cmd":"shutdown"}\n'),
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        assert any("bugfix" in list(meta.get("active_sops", [])) for meta in saved_meta)
        assert any(meta.get("active_sop") == "bugfix" for meta in saved_meta)

    def test_tui_daemon_set_profile_invalid_payload_emits_error(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"set_profile"}\n{"cmd":"shutdown"}\n'))

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        error_events = [event for event in events if event.get("event") == "error"]
        assert error_events
        assert any(event.get("message") == "set_profile.profile is required" for event in error_events)
        assert not any(event.get("event") == "profile_applied" for event in events)

    def test_tui_daemon_set_profile_emits_profile_applied(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(
                '{"cmd":"set_profile","profile":{"id":"qa","name":"QA","tier":"deep",'
                '"system_prompt_snippets":["Keep answers short"],'
                '"context_sources":[{"type":"note","text":"release checklist"},{"type":"kb","id":"kb-old"}],'
                '"knowledge_base_id":"kb-new","active_sops":[],'
                '"auto_delegate_assistive":"false",'
                '"agents":[{"id":"triage-research","name":"Triage Research","summary":"Investigates incoming issues",'
                '"prompt":"You triage incidents.","provider":"openai","tier":"balanced",'
                '"tool_names":["file_read","shell","shell"],"sop_names":["incident-triage"],'
                '"knowledge_base_id":"kb-123","activated":true}]}}\n{"cmd":"shutdown"}\n'
            ),
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        applied_events = [event for event in events if event.get("event") == "profile_applied"]
        assert applied_events
        applied = applied_events[-1]["profile"]
        assert applied["id"] == "qa"
        assert applied["name"] == "QA"
        assert applied["tier"] == "deep"
        assert applied["knowledge_base_id"] == "kb-new"
        assert applied["system_prompt_snippets"] == ["Keep answers short"]
        assert {"type": "note", "text": "release checklist", "id": "release-checklist"} in applied["context_sources"]
        assert {"type": "kb", "id": "kb-new"} in applied["context_sources"]
        assert not any(item.get("id") == "kb-old" for item in applied["context_sources"] if item.get("type") == "kb")
        assert applied["auto_delegate_assistive"] is False
        assert applied["agents"] == [
            {
                "id": "orchestrator",
                "name": "Orchestrator",
                "summary": "",
                "prompt": "",
                "prompt_refs": ["orchestrator_base"],
                "provider": None,
                "tier": None,
                "tool_names": [],
                "sop_names": [],
                "knowledge_base_id": None,
                "activated": False,
            },
            {
                "id": "triage-research",
                "name": "Triage Research",
                "summary": "Investigates incoming issues",
                "prompt": "You triage incidents.",
                "prompt_refs": [],
                "provider": "openai",
                "tier": "balanced",
                "tool_names": ["file_read", "shell"],
                "sop_names": ["incident-triage"],
                "knowledge_base_id": "kb-123",
                "activated": True,
            },
        ]

    def test_tui_daemon_set_profile_replaces_stale_sop_overrides(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
        tmp_path,
    ):
        saved_meta: list[dict[str, object]] = []

        class _FakeStore:
            def save(self, session_id, *, meta=None, messages=None, state=None, last_plan=None):
                del session_id, messages, state, last_plan
                if isinstance(meta, dict):
                    saved_meta.append(dict(meta))
                return None

            def list(self, *, limit=50):
                del limit
                return []

            def load_messages(self, session_id, *, max_messages=200, expected_version=1):
                del session_id, max_messages, expected_version
                return []

            def save_messages(self, session_id, messages, *, max_messages=200, version=1):
                del session_id, messages, max_messages, version
                return {"version": 1, "message_count": 0, "turn_count": 0}

        def _fake_run_sop(*, action, name, sop_paths=None):
            del sop_paths
            if action == "get" and name == "review":
                return {"status": "success", "content": [{"text": "Use review SOP"}]}
            return {"status": "error", "content": []}

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(swarmee, "SessionStore", _FakeStore)
        monkeypatch.setattr(swarmee, "run_sop", _fake_run_sop)
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(
                '{"cmd":"set_sop","name":"bugfix","content":"Use bugfix SOP"}\n'
                '{"cmd":"set_profile","profile":{"id":"ops","name":"Ops","active_sops":["review"]}}\n'
                '{"cmd":"shutdown"}\n'
            ),
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        assert saved_meta
        final_meta = saved_meta[-1]
        assert final_meta.get("active_sop") == "review"
        assert final_meta.get("active_sops") == ["review"]
        assert "bugfix" not in list(final_meta.get("active_sops", []))

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        applied_events = [event for event in events if event.get("event") == "profile_applied"]
        assert applied_events
        assert applied_events[-1]["profile"]["active_sops"] == ["review"]

    def test_tui_daemon_set_safety_overrides_emits_safety_overrides_event(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(
                '{"cmd":"set_safety_overrides","tool_consent":"deny","tool_allowlist":["file_read"],'
                '"tool_blocklist":["shell"]}\n{"cmd":"shutdown"}\n'
            ),
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        safety_events = [event for event in events if event.get("event") == "safety_overrides"]
        assert safety_events
        latest = safety_events[-1]
        assert latest["overrides"] == {
            "tool_consent": "deny",
            "tool_allowlist": ["file_read"],
            "tool_blocklist": ["shell"],
        }

    def test_tui_daemon_set_safety_overrides_invalid_payload_emits_error(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO('{"cmd":"set_safety_overrides","tool_consent":"sometimes"}\n{"cmd":"shutdown"}\n'),
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        error_events = [event for event in events if event.get("event") == "error"]
        assert error_events
        assert any(
            "set_safety_overrides.tool_consent must be ask|allow|deny" in str(event.get("message", ""))
            for event in error_events
        )

    def test_tui_daemon_compact_emits_completion_event(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"compact"}\n{"cmd":"shutdown"}\n'))
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"text": "world"}]},
        ]

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        compact_events = [event for event in events if event.get("event") == "compact_complete"]
        assert compact_events
        assert "compacted" in compact_events[0]

    def test_tui_daemon_emits_session_available_for_same_cwd(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
        tmp_path,
    ):
        class _FakeStore:
            def save(self, session_id, *, meta=None, messages=None, state=None, last_plan=None):
                return None

            def list(self, *, limit=50):
                del limit
                return [{"id": "sid-prev", "cwd": str(tmp_path)}]

            def load_messages(self, session_id, *, max_messages=200, expected_version=1):
                del max_messages, expected_version
                if session_id != "sid-prev":
                    return []
                return [
                    {"role": "user", "content": [{"text": "hello"}]},
                    {"role": "assistant", "content": [{"text": "world"}]},
                ]

            def save_messages(self, session_id, messages, *, max_messages=200, version=1):
                del session_id, messages, max_messages, version
                return {"version": 1, "message_count": 2, "turn_count": 1}

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(swarmee, "SessionStore", _FakeStore)
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"shutdown"}\n'))

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        assert any(
            event.get("event") == "session_available"
            and event.get("session_id") == "sid-prev"
            and int(event.get("turn_count", 0)) == 1
            for event in events
        )

    def test_tui_daemon_restore_session_emits_replay_events(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
        tmp_path,
    ):
        class _FakeStore:
            def save(self, session_id, *, meta=None, messages=None, state=None, last_plan=None):
                del session_id, meta, messages, state, last_plan
                return None

            def list(self, *, limit=50):
                del limit
                return []

            def load_messages(self, session_id, *, max_messages=200, expected_version=1):
                del max_messages, expected_version
                if session_id != "restore-me":
                    return []
                return [
                    {
                        "role": "user",
                        "content": [{"text": "What changed?"}],
                        "timestamp": "10:01 AM",
                    },
                    {
                        "role": "assistant",
                        "content": [{"text": "I updated the config."}],
                        "model": "openai/deep",
                        "timestamp": "10:02 AM",
                    },
                ]

            def save_messages(self, session_id, messages, *, max_messages=200, version=1):
                del session_id, messages, max_messages, version
                return {"version": 1, "message_count": 2, "turn_count": 1}

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(swarmee, "SessionStore", _FakeStore)
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO('{"cmd":"restore_session","session_id":"restore-me"}\n{"cmd":"shutdown"}\n'),
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip().startswith("{")]
        assert any(event.get("event") == "session_restored" and event.get("turn_count") == 1 for event in events)
        assert any(
            event.get("event") == "replay_turn" and event.get("role") == "user" and event.get("text") == "What changed?"
            for event in events
        )
        assert any(
            event.get("event") == "replay_turn"
            and event.get("role") == "assistant"
            and event.get("model") == "openai/deep"
            for event in events
        )
        assert any(event.get("event") == "replay_complete" and event.get("turn_count") == 1 for event in events)

    def test_tui_daemon_query_success_persists_messages(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
        tmp_path,
    ):
        saved_calls: list[tuple[str, list[dict[str, object]]]] = []

        class _FakeStore:
            def save(self, session_id, *, meta=None, messages=None, state=None, last_plan=None):
                del session_id, meta, messages, state, last_plan
                return None

            def list(self, *, limit=50):
                del limit
                return []

            def load_messages(self, session_id, *, max_messages=200, expected_version=1):
                del session_id, max_messages, expected_version
                return []

            def save_messages(self, session_id, messages, *, max_messages=200, version=1):
                del max_messages, version
                saved_calls.append((session_id, list(messages) if isinstance(messages, list) else []))
                return {"version": 1, "message_count": len(messages), "turn_count": 1}

        real_thread = swarmee.threading.Thread

        def _thread_factory(*args, **kwargs):
            if kwargs.get("name") == "swarmee-session-save":

                class _ImmediateThread:
                    def start(self_inner):
                        target = kwargs.get("target")
                        if callable(target):
                            target()

                    def is_alive(self_inner):
                        return False

                    def join(self_inner, timeout=None):
                        del timeout
                        return None

                return _ImmediateThread()
            return real_thread(*args, **kwargs)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(swarmee, "SessionStore", _FakeStore)
        monkeypatch.setattr(swarmee.threading, "Thread", _thread_factory)
        monkeypatch.setattr(swarmee, "resolve_model_provider", lambda **_kwargs: ("bedrock", None))
        monkeypatch.setattr(swarmee, "has_aws_credentials", lambda: True)
        monkeypatch.setattr(swarmee, "_run_query_with_optional_plan", lambda **_kwargs: (None, "ok", True))
        monkeypatch.setattr(sys, "argv", ["swarmee", "--tui-daemon"])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"query","text":"hello"}\n{"cmd":"shutdown"}\n'))

        mock_agent.messages = [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"text": "ok"}]},
        ]

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            swarmee.main()

        assert saved_calls
        assert saved_calls[0][1][0]["role"] == "user"


class TestCommandLine:
    """Test cases for command line mode functionality"""

    def test_command_line_query(self, mock_agent, mock_bedrock, mock_load_prompt, monkeypatch):
        """Test processing a query from command line arguments"""
        # Mock sys.argv with a test query
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        # Call the main function
        swarmee.main()

        # Verify agent was called with the query
        call = mock_agent.invoke_async.call_args
        assert call.args[0] == "test query"
        assert call.kwargs["invocation_state"]["swarmee"]["mode"] == "execute"
        assert "runtime_environment" in call.kwargs["invocation_state"]["swarmee"]
        assert "structured_output_model" not in call.kwargs
        assert "structured_output_prompt" not in call.kwargs

    def test_command_line_query_with_kb(
        self, mock_agent, mock_bedrock, mock_load_prompt, mock_store_conversation, monkeypatch
    ):
        """Test processing a query with knowledge base from command line"""
        # Mock sys.argv with a test query and KB ID
        monkeypatch.setattr(sys, "argv", ["swarmee", "--kb", "test-kb-id", "test", "query"])

        # Call the main function
        swarmee.main()

        # Verify retrieve was called
        mock_agent.tool.retrieve.assert_called_with(text="test query", knowledgeBaseId="test-kb-id")

        # Verify conversation was stored
        mock_store_conversation.assert_called_with(mock_agent, "test query", mock.ANY, "test-kb-id")

    @mock.patch.object(swarmee, "Agent")
    @mock.patch.object(swarmee, "store_conversation_in_kb")
    def test_command_line_with_kb_environment(self, mock_store, mock_agent):
        """Test command line mode with KB from environment variable"""
        # Setup mocks
        mock_agent_instance = mock.MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke_async = mock.AsyncMock(
            return_value=mock.MagicMock(structured_output=None, message=[])
        )

        # Run main with test query and environment variable
        with (
            mock.patch.object(sys, "argv", ["swarmee", "test", "query"]),
            mock.patch.dict(os.environ, {"SWARMEE_KNOWLEDGE_BASE_ID": "env-kb-id"}),
        ):
            swarmee.main()

        # Verify retrieve was called with the right KB ID
        mock_agent_instance.tool.retrieve.assert_called_once_with(text="test query", knowledgeBaseId="env-kb-id")

        # Verify store_conversation_in_kb was called
        mock_store.assert_called_once_with(mock_agent_instance, "test query", mock.ANY, "env-kb-id")


class TestConfiguration:
    """Test cases for configuration handling"""

    def test_environment_variables(self, mock_agent, mock_bedrock, monkeypatch):
        """Test handling of environment variables"""
        # Mock sys.argv with a test query
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        # Call the main function
        swarmee.main()

        # Verify agent was called with the correct prompt
        call = mock_agent.invoke_async.call_args
        assert call.args[0] == "test query"
        assert call.kwargs["invocation_state"]["swarmee"]["mode"] == "execute"
        assert "structured_output_model" not in call.kwargs
        assert "structured_output_prompt" not in call.kwargs

    def test_kb_environment_variable(
        self, mock_agent, mock_bedrock, mock_load_prompt, mock_store_conversation, monkeypatch
    ):
        """Test handling of knowledge base environment variable"""
        # Set environment variables
        monkeypatch.setenv("SWARMEE_KNOWLEDGE_BASE_ID", "env-kb-id")

        # Mock sys.argv with a test query
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        # Call the main function
        swarmee.main()

        # Verify retrieve was called with the right KB ID
        mock_agent.tool.retrieve.assert_called_with(text="test query", knowledgeBaseId="env-kb-id")

        # Verify conversation was stored
        mock_store_conversation.assert_called_with(mock_agent, "test query", mock.ANY, "env-kb-id")


class TestErrorHandling:
    """Test cases for error handling"""

    def test_general_exception(self, mock_agent, mock_bedrock, mock_load_prompt, monkeypatch, capfd):
        """Test handling of general exceptions"""
        # Make agent raise an exception
        mock_agent.invoke_async.side_effect = Exception("Test error")

        # Mock sys.argv with a test query
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        # Call the main function
        with pytest.raises(Exception, match="Test error"):
            swarmee.main()

        # Ensure the test passes without checking stderr
        assert True


class TestShellCommandError:
    """Test shell command error handling"""

    @mock.patch("builtins.print")
    def test_shell_command_exception(
        self, mock_print, mock_agent, mock_bedrock, mock_load_prompt, mock_user_input, mock_welcome_message, monkeypatch
    ):
        """Test handling exceptions when executing shell commands"""
        # Setup mocks
        mock_user_input.side_effect = ["!failing-command", "exit"]

        # Configure shell command to raise an exception
        mock_agent_instance = mock_agent
        mock_agent_instance.tool.shell.side_effect = Exception("Shell command failed")

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        with mock.patch.object(swarmee, "render_goodbye_message"):
            swarmee.main()

        # Verify error was printed
        mock_print.assert_any_call("Shell command execution error: Shell command failed")


class TestKnowledgeBaseIntegration:
    """Test cases for knowledge base integration"""

    def test_interactive_mode_with_kb(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        mock_welcome_message,
        mock_goodbye_message,
        mock_store_conversation,
        monkeypatch,
    ):
        """Test interactive mode with knowledge base"""
        # Setup mocks
        mock_user_input.side_effect = ["test query", "exit"]

        # Configure environment
        monkeypatch.setenv("SWARMEE_KNOWLEDGE_BASE_ID", "test-kb-id")

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee"])

        # Call the main function
        swarmee.main()

        # Verify retrieve was called with knowledge base ID
        mock_agent.tool.retrieve.assert_called_with(text="test query", knowledgeBaseId="test-kb-id")

        # Verify store_conversation_in_kb was called
        mock_store_conversation.assert_called_once()

    def test_welcome_message_with_kb(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        monkeypatch,
    ):
        """Test that welcome text is passed via a cache-friendly system reminder when enabled."""
        # Setup mocks
        mock_user_input.side_effect = ["test query", "exit"]

        monkeypatch.setattr(swarmee, "read_welcome_text", lambda: "Custom welcome text")

        # Mock load_system_prompt
        base_system_prompt = "Base system prompt"
        mock_load_prompt.return_value = base_system_prompt

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee", "--include-welcome-in-prompt"])

        # Call the main function
        with mock.patch.object(swarmee, "render_welcome_message"), mock.patch.object(swarmee, "render_goodbye_message"):
            swarmee.main()

        # Verify the Agent was constructed with the base system prompt (stable prefix).
        agent_kwargs = swarmee.Agent.call_args.kwargs  # type: ignore[attr-defined]
        assert base_system_prompt in str(agent_kwargs.get("system_prompt", ""))

        # Verify the welcome text is injected as a system reminder (not via system prompt refresh).
        call = mock_agent.invoke_async.call_args
        prompt = call.args[0]
        assert "<system-reminder>" in prompt
        assert "Welcome Text Reference:" in prompt
        assert "Custom welcome text" in prompt

    def test_welcome_message_failure(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        mock_user_input,
        monkeypatch,
    ):
        """Test handling of welcome message retrieval failure"""
        # Setup mocks
        mock_user_input.side_effect = ["test query", "exit"]

        monkeypatch.setattr(swarmee, "read_welcome_text", lambda: "")

        # Mock load_system_prompt
        base_system_prompt = "Base system prompt"
        mock_load_prompt.return_value = base_system_prompt

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["swarmee", "--include-welcome-in-prompt"])

        # Call the main function
        with mock.patch.object(swarmee, "render_welcome_message"), mock.patch.object(swarmee, "render_goodbye_message"):
            swarmee.main()

        # Verify the Agent was constructed with the base system prompt (stable prefix).
        agent_kwargs = swarmee.Agent.call_args.kwargs  # type: ignore[attr-defined]
        assert base_system_prompt in str(agent_kwargs.get("system_prompt", ""))

        # Verify the welcome reminder is omitted when no welcome text is available.
        call = mock_agent.invoke_async.call_args
        prompt = call.args[0]
        assert "Welcome Text Reference:" not in prompt
        assert "<system-reminder>" not in prompt


class TestToolConsentPrompt:
    """Tests for consent prompt UX wiring."""

    def test_consent_prompt_uses_repl_style_input(
        self,
        mock_agent,
        mock_bedrock,
        mock_load_prompt,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        class FakeToolConsentHooks:
            def __init__(self, *args, **kwargs):
                captured["prompt"] = kwargs["prompt"]

        monkeypatch.setattr(swarmee, "ToolConsentHooks", FakeToolConsentHooks)
        monkeypatch.setattr(sys, "argv", ["swarmee", "test", "query"])

        swarmee.main()

        prompt_fn = captured.get("prompt")
        assert callable(prompt_fn)

        consent_text = "Allow tool 'shell'? [y]es/[n]o/[a]lways/[v]never:"
        with (
            mock.patch.object(swarmee, "callback_handler") as mock_callback_handler,
            mock.patch.object(swarmee, "_render_tool_consent_message") as mock_render_consent,
            mock.patch.object(
                swarmee,
                "pause_active_interrupt_watcher_for_input",
                return_value=contextlib.nullcontext(),
            ) as mock_pause,
            mock.patch.object(swarmee, "_get_user_input_compat", return_value="y") as mock_user_input_compat,
        ):
            response = prompt_fn(consent_text)  # type: ignore[operator]

        assert response == "y"
        mock_callback_handler.assert_called_once_with(force_stop=True)
        mock_render_consent.assert_called_once_with(consent_text)
        mock_pause.assert_called_once_with()
        mock_user_input_compat.assert_called_once_with(
            "\n~ consent> ",
            default="",
            keyboard_interrupt_return_default=True,
            prefer_prompt_toolkit_in_async=False,
        )

    def test_get_user_input_compat_uses_stdin_when_event_loop_running(self):
        async def _run() -> str:
            with (
                mock.patch.object(swarmee, "_prompt_input_with_stdin", return_value="y") as mock_stdin,
                mock.patch.object(swarmee, "get_user_input", side_effect=AssertionError("should not be called")),
            ):
                result = swarmee._get_user_input_compat(
                    "\n~ consent> ", default="", keyboard_interrupt_return_default=True
                )

            mock_stdin.assert_called_once_with(
                "\n~ consent> ",
                default="",
                keyboard_interrupt_return_default=True,
            )
            return result

        assert asyncio.run(_run()) == "y"

    def test_get_user_input_compat_can_use_prompt_toolkit_when_requested(self):
        async def _run() -> str:
            with (
                mock.patch.object(
                    swarmee, "_prompt_input_with_prompt_toolkit", return_value="y"
                ) as mock_prompt_toolkit,
                mock.patch.object(
                    swarmee, "_prompt_input_with_stdin", side_effect=AssertionError("should not be called")
                ),
            ):
                result = swarmee._get_user_input_compat(
                    "\n~ consent> ",
                    default="",
                    keyboard_interrupt_return_default=True,
                    prefer_prompt_toolkit_in_async=True,
                )
            mock_prompt_toolkit.assert_called_once_with(
                "\n~ consent> ",
                default="",
                keyboard_interrupt_return_default=True,
            )
            return result

        assert asyncio.run(_run()) == "y"

    def test_plan_json_for_execution_excludes_confirmation_prompt(self):
        from swarmee_river.planning import PlanStep, WorkPlan
        from swarmee_river.utils.agent_runtime_utils import plan_json_for_execution

        plan = WorkPlan(
            summary="Refactor hooks",
            steps=[PlanStep(description="Update tests")],
            confirmation_prompt="Approve with :y",
        )

        rendered = plan_json_for_execution(plan)

        assert "confirmation_prompt" not in rendered
        assert '"summary": "Refactor hooks"' in rendered


class TestOverflowErrorClassification:
    def test_context_window_overflow_markers(self):
        assert swarmee._is_context_window_overflow_error(RuntimeError("OpenAI threw context window overflow error"))
        assert swarmee._is_context_window_overflow_error(RuntimeError("maximum context length exceeded"))
        assert swarmee._is_context_window_overflow_error(RuntimeError("too many tokens in request"))

    def test_context_window_overflow_negative(self):
        assert not swarmee._is_context_window_overflow_error(RuntimeError("rate limit exceeded"))

    def test_build_tui_error_event_includes_metadata(self):
        event = swarmee._build_tui_error_event(
            "ThrottlingException: slow down",
            category_hint="transient",
            retry_after_s=4,
        )
        assert event["event"] == "error"
        assert event["message"] == "ThrottlingException: slow down"
        assert event["category"] == "transient"
        assert event["retryable"] is True
        assert event["retry_after_s"] == 4

    def test_build_tui_error_event_extracts_tool_use_id(self):
        event = swarmee._build_tui_error_event(
            "tool execution failed: tool_use_id=t-99",
            category_hint="tool_error",
        )
        assert event["category"] == "tool_error"
        assert event["retryable"] is False
        assert event["tool_use_id"] == "t-99"

    def test_render_tool_consent_message_preserves_choice_brackets(self):
        if swarmee.Console is None:
            pytest.skip("rich is not available")

        from rich.console import Console

        console = Console(record=True, width=160)
        consent_text = "Allow tool 'shell'? [y]es/[n]o/[a]lways/[v]never:"

        with mock.patch.object(swarmee, "_consent_console", console):
            swarmee._render_tool_consent_message(consent_text)

        rendered = console.export_text()
        assert "[y]es/[n]o/[a]lways/[v]never:" in rendered


class TestStructuredPlanPrompt:
    """Tests for the structured plan prompt and PlanStep model."""

    def test_structured_plan_prompt_suppresses_text_output(self):
        from swarmee_river.planning import structured_plan_prompt

        prompt = structured_plan_prompt()
        assert "Do NOT produce any text output" in prompt, (
            "Plan prompt must instruct the model not to produce text before WorkPlan"
        )
        assert "WorkPlan tool call" in prompt

    def test_structured_plan_prompt_does_not_ban_all_tools(self):
        from swarmee_river.planning import structured_plan_prompt

        prompt = structured_plan_prompt()
        assert "Do NOT execute tools" not in prompt, (
            "Plan prompt should not blanket-ban tools since read-only tools are allowed"
        )
        assert "read-only tools" in prompt

    def test_plan_step_coerces_null_list_fields_to_empty_list(self):
        from swarmee_river.planning import PlanStep

        step = PlanStep(
            description="Test step",
            files_to_read=None,
            files_to_edit=None,
            tools_expected=None,
            commands_expected=None,
            risks=None,
        )
        assert step.files_to_read == []
        assert step.files_to_edit == []
        assert step.tools_expected == []
        assert step.commands_expected == []
        assert step.risks == []

    def test_plan_step_preserves_valid_list_fields(self):
        from swarmee_river.planning import PlanStep

        step = PlanStep(
            description="Test step",
            files_to_read=["src/foo.py"],
            files_to_edit=["src/bar.py"],
            tools_expected=["bash"],
            commands_expected=["pytest"],
            risks=["data loss"],
        )
        assert step.files_to_read == ["src/foo.py"]
        assert step.files_to_edit == ["src/bar.py"]
        assert step.tools_expected == ["bash"]
        assert step.commands_expected == ["pytest"]
        assert step.risks == ["data loss"]

    def test_plan_step_omitted_fields_default_to_empty_list(self):
        from swarmee_river.planning import PlanStep

        step = PlanStep(description="Minimal step")
        assert step.files_to_read == []
        assert step.files_to_edit == []
        assert step.tools_expected == []
        assert step.commands_expected == []
        assert step.risks == []
