from __future__ import annotations

from pathlib import Path
from typing import Any

from swarmee_river.jupyter import magic


class _FakeRuntimeClient:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = list(events)
        self.sent_commands: list[dict[str, Any]] = []
        self.attached_session_id: str | None = None
        self.attached_cwd: str | None = None

    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def hello(self, *, client_name: str, surface: str) -> dict[str, Any]:
        assert client_name == "swarmee-notebook"
        assert surface == "jupyter"
        return {"event": "hello_ack"}

    def attach(self, *, session_id: str, cwd: str) -> dict[str, Any]:
        self.attached_session_id = session_id
        self.attached_cwd = cwd
        return {"event": "attached", "session_id": session_id}

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
    assert fake_client.sent_commands == [
        {
            "cmd": "query",
            "text": "test prompt",
            "mode": "plan",
            "auto_approve": False,
        }
    ]


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
        ]
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
