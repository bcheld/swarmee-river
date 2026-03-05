from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from swarmee_river.hooks.session_s3 import SessionS3Hooks
from swarmee_river.settings import default_settings_template


class _ImmediateThread:
    def __init__(self, *, target: Any, kwargs: dict[str, Any] | None = None, **_: Any) -> None:
        self._target = target
        self._kwargs = dict(kwargs or {})

    def start(self) -> None:
        self._target(**self._kwargs)


def test_session_s3_hook_auto_export_debounced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWARMEE_SESSION_ID", "sid-1")
    monkeypatch.setattr("swarmee_river.hooks.session_s3.threading.Thread", _ImmediateThread)

    settings = default_settings_template()
    settings = replace(
        settings,
        runtime=replace(settings.runtime, session_s3_bucket="bucket", session_s3_auto_export=True),
    )

    calls: list[str] = []

    def _fake_export(*, session_id: str, s3_bucket: str, s3_prefix: str) -> dict[str, Any]:
        assert s3_bucket == "bucket"
        assert session_id == "sid-1"
        calls.append(s3_prefix)
        return {"ok": True}

    monkeypatch.setattr("swarmee_river.hooks.session_s3.export_session_to_s3", _fake_export)

    hook = SessionS3Hooks(settings=settings, debounce_seconds=30)
    event = SimpleNamespace(invocation_state={"swarmee": {}}, result="ok")

    hook.after_invocation(event)
    hook.after_invocation(event)

    assert len(calls) == 1


def test_session_s3_hook_promotes_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWARMEE_SESSION_ID", "sid-2")
    monkeypatch.setattr("swarmee_river.hooks.session_s3.threading.Thread", _ImmediateThread)

    settings = default_settings_template()
    settings = replace(
        settings,
        runtime=replace(
            settings.runtime,
            session_s3_bucket="bucket",
            session_s3_auto_export=False,
            session_kb_promote_on_complete=True,
            knowledge_base_id="kb-1",
        ),
    )

    promoted: list[str] = []

    monkeypatch.setattr(
        "swarmee_river.hooks.session_s3.export_session_to_s3",
        lambda **kwargs: {"ok": True},
    )

    def _fake_promote(*, session_id: str, knowledge_base_id: str, content_filter: str) -> dict[str, Any]:
        promoted.append(f"{session_id}:{knowledge_base_id}:{content_filter}")
        return {"promoted_count": 1}

    monkeypatch.setattr("swarmee_river.hooks.session_s3.promote_session_to_kb", _fake_promote)

    hook = SessionS3Hooks(settings=settings, debounce_seconds=30, promote_debounce_seconds=10)
    event = SimpleNamespace(invocation_state={"swarmee": {"mode": "execute"}}, result="ok")

    hook.after_invocation(event)

    assert promoted == ["sid-2:kb-1:all"]
