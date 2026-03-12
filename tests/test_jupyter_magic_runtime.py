from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from swarmee_river.jupyter import magic
from swarmee_river.planning import PendingWorkPlan, WorkPlan
from swarmee_river.settings import SwarmeeSettings, default_settings_template


class _FakeRuntimeClient:
    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        fork_event: dict[str, Any] | None = None,
        hello_event: dict[str, Any] | None = None,
    ) -> None:
        self._events = list(events)
        self.sent_commands: list[dict[str, Any]] = []
        self.attached_session_id: str | None = None
        self.attached_cwd: str | None = None
        self.fork_calls: list[dict[str, Any]] = []
        self._fork_event = fork_event or {"event": "surface_session_forked", "session_id": "branch-session"}
        self._hello_event = hello_event or {
            "event": "hello_ack",
            "schema_version": "2",
            "capabilities": {"fork_surface_session": True},
        }

    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def hello(self, *, client_name: str, surface: str) -> dict[str, Any]:
        assert client_name == "swarmee-notebook"
        assert surface == "jupyter"
        return dict(self._hello_event)

    def attach(self, *, session_id: str, cwd: str) -> dict[str, Any]:
        self.attached_session_id = session_id
        self.attached_cwd = cwd
        return {"event": "attached", "session_id": session_id}

    def fork_surface_session(
        self,
        *,
        cwd: str,
        surface: str,
        session_id: str | None = None,
        source_session_id: str | None = None,
    ) -> dict[str, Any]:
        self.fork_calls.append(
            {
                "cwd": cwd,
                "surface": surface,
                "session_id": session_id,
                "source_session_id": source_session_id,
            }
        )
        return dict(self._fork_event)

    def send_command(self, payload: dict[str, Any]) -> None:
        self.sent_commands.append(dict(payload))

    def read_event(self) -> dict[str, Any] | None:
        if not self._events:
            return None
        return self._events.pop(0)


def test_run_swarmee_uses_runtime_when_enabled(monkeypatch, tmp_path: Path):
    discovery = tmp_path / "runtime.json"
    discovery.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SWARMEE_NOTEBOOK_USE_RUNTIME", "true")
    monkeypatch.setenv("SWARMEE_SESSION_ID", "session-from-env")
    monkeypatch.setattr(magic, "ensure_runtime_broker", lambda *, cwd=None: discovery)

    fake_client = _FakeRuntimeClient(
        [
            {"event": "text_delta", "text": "hello "},
            {"event": "text_delta", "text": "world"},
            {"event": "turn_complete", "exit_status": "ok"},
        ]
    )
    monkeypatch.setattr(
        magic.RuntimeServiceClient,
        "from_discovery_file",
        staticmethod(lambda _path: fake_client),
    )

    text = magic._run_swarmee(
        ipython=None,
        user_prompt="test prompt",
        include_context=False,
        force_plan=True,
        auto_approve=False,
    )

    assert text == "hello world"
    assert fake_client.attached_session_id == "session-from-env"
    assert fake_client.attached_cwd == str(Path.cwd().resolve())
    assert fake_client.fork_calls == []
    assert fake_client.sent_commands == [
        {
            "cmd": "query",
            "text": "test prompt",
            "mode": "plan",
            "auto_approve": False,
        }
    ]


def test_run_swarmee_runtime_deduplicates_complete_and_replay_text(monkeypatch, tmp_path: Path) -> None:
    discovery = tmp_path / "runtime.json"
    discovery.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SWARMEE_NOTEBOOK_USE_RUNTIME", "1")
    monkeypatch.setenv("SWARMEE_SESSION_ID", "session-from-env")
    monkeypatch.setattr(magic, "ensure_runtime_broker", lambda *, cwd=None: discovery)

    fake_client = _FakeRuntimeClient(
        [
            {"event": "text_delta", "text": "hello "},
            {"event": "text_delta", "text": "world"},
            {"event": "text_complete", "text": "hello world"},
            {"event": "replay_turn", "role": "assistant", "text": "hello world"},
            {"event": "turn_complete", "exit_status": "ok"},
        ]
    )
    monkeypatch.setattr(
        magic.RuntimeServiceClient,
        "from_discovery_file",
        staticmethod(lambda _path: fake_client),
    )

    text = magic._run_swarmee(
        ipython=None,
        user_prompt="test prompt",
        include_context=False,
        force_plan=False,
        auto_approve=False,
    )

    assert text == "hello world"


def test_run_swarmee_runtime_returns_plan_when_no_text(monkeypatch, tmp_path: Path):
    discovery = tmp_path / "runtime.json"
    discovery.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SWARMEE_NOTEBOOK_USE_RUNTIME", "1")
    monkeypatch.delenv("SWARMEE_SESSION_ID", raising=False)
    monkeypatch.setattr(magic, "ensure_runtime_broker", lambda *, cwd=None: discovery)
    monkeypatch.setattr(magic, "default_session_id_for_cwd", lambda _cwd: "cwd-derived-session")

    fake_client = _FakeRuntimeClient(
        [
            {"event": "plan", "rendered": "Proposed plan:\n1. do work"},
            {"event": "turn_complete", "exit_status": "ok"},
        ],
        fork_event={"event": "error", "code": "no_active_parent_session"},
    )
    monkeypatch.setattr(
        magic.RuntimeServiceClient,
        "from_discovery_file",
        staticmethod(lambda _path: fake_client),
    )

    text = magic._run_swarmee(
        ipython=None,
        user_prompt="plan this",
        include_context=False,
        force_plan=True,
        auto_approve=False,
    )

    assert text == "Proposed plan:\n1. do work"
    assert fake_client.attached_session_id == "cwd-derived-session"


def test_run_swarmee_runtime_branches_and_applies_read_only_overrides(monkeypatch, tmp_path: Path) -> None:
    discovery = tmp_path / "runtime.json"
    discovery.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SWARMEE_NOTEBOOK_USE_RUNTIME", "1")
    monkeypatch.delenv("SWARMEE_SESSION_ID", raising=False)
    monkeypatch.setattr(magic, "ensure_runtime_broker", lambda *, cwd=None: discovery)
    monkeypatch.setattr(
        magic,
        "load_settings",
        lambda *_args, **_kwargs: type(
            "_Settings",
            (),
            {"runtime": type("_Runtime", (), {"enable_project_context_tool": True})()},
        )(),
    )

    fake_client = _FakeRuntimeClient(
        [
            {"event": "text_delta", "text": "branched"},
            {"event": "turn_complete", "exit_status": "ok"},
        ],
        fork_event={
            "event": "surface_session_forked",
            "session_id": "branch-123",
            "source_session_id": "parent-1",
            "source_freshness": "live",
            "surface": "jupyter",
        },
    )
    monkeypatch.setattr(
        magic.RuntimeServiceClient,
        "from_discovery_file",
        staticmethod(lambda _path: fake_client),
    )

    text = magic._run_swarmee(
        ipython=None,
        user_prompt="inspect this repo",
        include_context=False,
        force_plan=False,
        auto_approve=False,
    )

    assert text == "branched"
    assert fake_client.fork_calls == [
        {
            "cwd": str(Path.cwd().resolve()),
            "surface": "jupyter",
            "session_id": None,
            "source_session_id": None,
        }
    ]
    assert fake_client.attached_session_id == "branch-123"
    assert fake_client.sent_commands[0] == {
        "cmd": "set_safety_overrides",
        "overrides": {
            "tool_consent": "deny",
            "tool_allowlist": [
                "list",
                "glob",
                "file_list",
                "file_search",
                "file_read",
                "notebook_read",
                "project_context",
            ],
            "tool_blocklist": [],
        },
    }
    assert fake_client.sent_commands[1] == {
        "cmd": "query",
        "text": "inspect this repo",
        "auto_approve": False,
    }


def test_run_swarmee_runtime_legacy_broker_falls_back_and_warns(monkeypatch, tmp_path: Path) -> None:
    discovery = tmp_path / "runtime.json"
    discovery.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SWARMEE_NOTEBOOK_USE_RUNTIME", "1")
    monkeypatch.delenv("SWARMEE_SESSION_ID", raising=False)
    monkeypatch.setattr(magic, "ensure_runtime_broker", lambda *, cwd=None: discovery)
    monkeypatch.setattr(magic, "default_session_id_for_cwd", lambda _cwd: "legacy-cwd-session")

    fake_client = _FakeRuntimeClient(
        [
            {"event": "text_delta", "text": "legacy"},
            {"event": "turn_complete", "exit_status": "ok"},
        ],
        hello_event={"event": "hello_ack"},
    )
    monkeypatch.setattr(
        magic.RuntimeServiceClient,
        "from_discovery_file",
        staticmethod(lambda _path: fake_client),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = magic._run_swarmee(
            ipython=None,
            user_prompt="inspect this repo",
            include_context=False,
            force_plan=False,
            auto_approve=False,
        )

    assert text == "legacy"
    assert fake_client.fork_calls == []
    assert fake_client.attached_session_id == "legacy-cwd-session"
    assert any("does not support cache-aware surface branching" in str(item.message) for item in caught)


def test_install_notebook_warning_filters_suppresses_http_request_tool_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        magic._install_notebook_warning_filters()
        warnings.warn_explicit(
            magic._HTTP_REQUEST_TOOL_WARNING,
            category=UserWarning,
            filename="test_warning.py",
            lineno=1,
            module="pydantic.main",
        )

    assert caught == []


def test_execute_with_plan_uses_execute_mode() -> None:
    captured: dict[str, Any] = {}

    def _fake_invoke(_runtime: Any, _query: str, *, invocation_state: dict[str, Any]) -> str:
        captured["invocation_state"] = invocation_state
        return "ok"

    runtime = object()
    plan = WorkPlan(summary="Inspect the issue", steps=[])
    original_invoke = magic._invoke_agent
    try:
        magic._invoke_agent = _fake_invoke  # type: ignore[assignment]
        result = magic._execute_with_plan(runtime, "investigate", plan, auto_approve=True)
    finally:
        magic._invoke_agent = original_invoke  # type: ignore[assignment]

    assert result == "ok"
    assert captured["invocation_state"]["swarmee"]["mode"] == "execute"
    assert captured["invocation_state"]["swarmee"]["enforce_plan"] is True


def test_generate_plan_uses_child_agent_without_mutating_parent_messages(monkeypatch) -> None:
    parent_agent = SimpleNamespace(messages=[{"role": "user", "content": [{"text": "old"}]}])
    runtime = magic._NotebookRuntime(
        agent=parent_agent,
        tools_dict={},
        settings=SimpleNamespace(),
        model_manager=SimpleNamespace(current_tier="deep"),
        runtime_environment={},
        base_system_prompt="system",
        artifact_store=SimpleNamespace(write_text=lambda **_kwargs: None),
        knowledge_base_id=None,
    )

    child_messages: list[dict[str, Any]] = []

    def _fake_create_shared_prefix_child_agent(*, parent_agent: Any, **_kwargs: Any):
        child = SimpleNamespace(messages=[dict(item) for item in parent_agent.messages])
        return child, SimpleNamespace()

    def _fake_invoke(runtime_obj: Any, _query: str, **_kwargs: Any):
        runtime_obj.agent.messages.append({"role": "assistant", "content": [{"text": "planning"}]})
        child_messages.extend(runtime_obj.agent.messages)
        return SimpleNamespace(structured_output=WorkPlan(summary="Inspect", steps=[]))

    monkeypatch.setattr(magic, "create_shared_prefix_child_agent", _fake_create_shared_prefix_child_agent)
    monkeypatch.setattr(magic, "_invoke_agent", _fake_invoke)

    pending = magic._generate_plan(runtime, "investigate", auto_approve=False)

    assert isinstance(pending, PendingWorkPlan)
    assert parent_agent.messages == [{"role": "user", "content": [{"text": "old"}]}]
    assert child_messages[-1] == {"role": "assistant", "content": [{"text": "planning"}]}


def test_run_swarmee_falls_back_to_local_runtime_when_runtime_not_configured(monkeypatch):
    monkeypatch.setenv("SWARMEE_NOTEBOOK_USE_RUNTIME", "0")
    monkeypatch.setattr(magic, "_get_or_create_runtime", lambda: object())
    monkeypatch.setattr(magic, "classify_intent", lambda _prompt: "info")
    monkeypatch.setattr(
        magic,
        "_invoke_agent",
        lambda _runtime, _query, **_kwargs: "local-result",
    )

    text = magic._run_swarmee(
        ipython=None,
        user_prompt="hello",
        include_context=False,
        force_plan=False,
        auto_approve=False,
    )

    assert text == "local-result"


def test_notebook_runtime_settings_use_notebook_default_selection() -> None:
    payload = default_settings_template().to_dict()
    payload["models"]["provider"] = "openai"
    payload["models"]["default_tier"] = "deep"
    payload["models"]["default_selection"] = {"provider": "openai", "tier": "deep"}
    payload["notebook"] = {"default_selection": {"provider": "openai", "tier": "fast"}}

    settings = SwarmeeSettings.from_dict(payload)

    notebook_settings, warning = magic._notebook_runtime_settings(settings)

    assert warning is None
    assert notebook_settings.models.provider == "openai"
    assert notebook_settings.models.default_tier == "fast"
    assert notebook_settings.models.default_selection.provider == "openai"
    assert notebook_settings.models.default_selection.tier == "fast"


def test_notebook_runtime_settings_falls_back_when_tier_is_unavailable() -> None:
    payload = default_settings_template().to_dict()
    payload["models"]["provider"] = "bedrock"
    payload["models"]["default_tier"] = "balanced"
    payload["models"]["default_selection"] = {"provider": "bedrock", "tier": "balanced"}
    payload["notebook"] = {"default_selection": {"provider": "bedrock", "tier": "coding"}}

    settings = SwarmeeSettings.from_dict(payload)

    notebook_settings, warning = magic._notebook_runtime_settings(settings)

    assert notebook_settings.models.provider == "bedrock"
    assert notebook_settings.models.default_tier == "balanced"
    assert notebook_settings.models.default_selection.provider == "bedrock"
    assert notebook_settings.models.default_selection.tier == "balanced"
    assert warning is not None
    assert "configured fallback tier 'coding'" in warning
    assert "using 'balanced'" in warning


def test_load_ipython_extension_prints_once_and_returns_none(monkeypatch) -> None:
    captured_magics: list[type[Any]] = []
    printed: list[str] = []

    class _FakeIPython:
        def register_magics(self, magics_cls: type[Any]) -> None:
            captured_magics.append(magics_cls)

    monkeypatch.setattr(magic, "_run_swarmee", lambda *_args, **_kwargs: "notebook output")
    monkeypatch.setattr("builtins.print", lambda text: printed.append(str(text)))

    fake_ipython = _FakeIPython()
    magic.load_ipython_extension(fake_ipython)

    assert captured_magics
    magics = captured_magics[0](fake_ipython)
    result = magics.swarmee("", "hello")

    assert result is None
    assert printed == ["notebook output"]
