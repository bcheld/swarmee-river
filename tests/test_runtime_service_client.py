from __future__ import annotations

import json
from pathlib import Path

from swarmee_river.runtime_service.client import (
    default_session_id_for_cwd,
    load_runtime_discovery,
)


def test_default_session_id_for_cwd_is_stable(tmp_path: Path) -> None:
    session_a = default_session_id_for_cwd(tmp_path)
    session_b = default_session_id_for_cwd(tmp_path)
    assert session_a == session_b
    assert session_a.startswith("cwd-")
    assert len(session_a) > 8


def test_load_runtime_discovery_round_trip(tmp_path: Path) -> None:
    discovery_path = tmp_path / "runtime.json"
    payload = {
        "host": "127.0.0.1",
        "port": 7342,
        "token": "abc123",
        "pid": 999,
        "started_at": "2026-02-23T00:00:00Z",
    }
    discovery_path.write_text(json.dumps(payload), encoding="utf-8")

    discovery = load_runtime_discovery(discovery_path)
    assert discovery.host == "127.0.0.1"
    assert discovery.port == 7342
    assert discovery.token == "abc123"
    assert discovery.pid == 999
    assert discovery.started_at == "2026-02-23T00:00:00Z"
