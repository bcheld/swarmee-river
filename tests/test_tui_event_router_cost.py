"""Tests for the usage-cost fallback's settings caching (doc 12 F1).

The fallback runs on the UI thread per usage event; it must not re-read
settings.json from disk on every event, but must notice external edits.
"""

from __future__ import annotations

import json
from pathlib import Path

import swarmee_river.tui.event_router as event_router


def _usage_event() -> dict:
    return {
        "usage": {"input_tokens": 1000, "output_tokens": 500},
        "provider": "openai",
        "model_id": "gpt-test",
    }


def test_usage_cost_fallback_reuses_cached_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    event_router._settings_cache = None

    load_calls = []
    real_load = event_router.load_settings

    def counting_load():
        load_calls.append(1)
        return real_load()

    monkeypatch.setattr(event_router, "load_settings", counting_load)

    for _ in range(5):
        event_router._compute_usage_cost_fallback(_usage_event())

    assert len(load_calls) == 1, "settings must be loaded once, not per event"


def test_usage_cost_fallback_reloads_after_external_edit(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    event_router._settings_cache = None

    load_calls = []
    real_load = event_router.load_settings

    def counting_load():
        load_calls.append(1)
        return real_load()

    monkeypatch.setattr(event_router, "load_settings", counting_load)

    event_router._compute_usage_cost_fallback(_usage_event())
    assert len(load_calls) == 1

    settings_dir = Path(tmp_path) / ".swarmee"
    settings_dir.mkdir(exist_ok=True)
    (settings_dir / "settings.json").write_text(json.dumps({}), encoding="utf-8")

    event_router._compute_usage_cost_fallback(_usage_event())
    assert len(load_calls) == 2, "an external settings edit must invalidate the cache"
