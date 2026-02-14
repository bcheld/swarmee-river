from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _iso_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _safe_json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"


@dataclass(frozen=True)
class SessionPaths:
    session_id: str
    dir: Path
    meta: Path
    messages: Path
    state: Path
    last_plan: Path


class SessionStore:
    """
    Project-local session persistence under `.swarmee/sessions/<session_id>/`.

    Files:
    - meta.json
    - messages.json
    - state.json
    - last_plan.json
    """

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or (Path.cwd() / ".swarmee" / "sessions")

    def _paths(self, session_id: str) -> SessionPaths:
        sid = (session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        base = self.root_dir / sid
        return SessionPaths(
            session_id=sid,
            dir=base,
            meta=base / "meta.json",
            messages=base / "messages.json",
            state=base / "state.json",
            last_plan=base / "last_plan.json",
        )

    def create(self, *, meta: dict[str, Any] | None = None, session_id: str | None = None) -> str:
        sid = session_id.strip() if isinstance(session_id, str) and session_id.strip() else uuid.uuid4().hex
        paths = self._paths(sid)
        paths.dir.mkdir(parents=True, exist_ok=False)

        payload = dict(meta or {})
        payload.setdefault("id", sid)
        payload.setdefault("created_at", _iso_ts())
        payload.setdefault("updated_at", payload.get("created_at"))
        paths.meta.write_text(_safe_json_dump(payload), encoding="utf-8")
        return sid

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not self.root_dir.exists():
            return []
        entries: list[dict[str, Any]] = []
        for child in sorted(self.root_dir.iterdir()):
            if not child.is_dir():
                continue
            meta_path = child / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {"id": child.name}
            if isinstance(meta, dict):
                meta.setdefault("id", child.name)
                entries.append(meta)

        def _sort_key(item: dict[str, Any]) -> str:
            return str(item.get("updated_at") or item.get("created_at") or "")

        entries.sort(key=_sort_key, reverse=True)
        return entries[: max(0, int(limit))]

    def read_meta(self, session_id: str) -> dict[str, Any]:
        paths = self._paths(session_id)
        if not paths.meta.exists():
            raise FileNotFoundError(f"Session meta not found: {session_id}")
        return json.loads(paths.meta.read_text(encoding="utf-8"))

    def load(self, session_id: str) -> tuple[dict[str, Any], Any | None, Any | None, Any | None]:
        paths = self._paths(session_id)
        meta = self.read_meta(session_id)

        messages: Any | None = None
        state: Any | None = None
        last_plan: Any | None = None

        if paths.messages.exists():
            try:
                messages = json.loads(paths.messages.read_text(encoding="utf-8"))
            except Exception:
                messages = None
        if paths.state.exists():
            try:
                state = json.loads(paths.state.read_text(encoding="utf-8"))
            except Exception:
                state = None
        if paths.last_plan.exists():
            try:
                last_plan = json.loads(paths.last_plan.read_text(encoding="utf-8"))
            except Exception:
                last_plan = None

        return meta, messages, state, last_plan

    def save(
        self,
        session_id: str,
        *,
        meta: dict[str, Any] | None = None,
        messages: Any | None = None,
        state: Any | None = None,
        last_plan: Any | None = None,
    ) -> SessionPaths:
        paths = self._paths(session_id)
        paths.dir.mkdir(parents=True, exist_ok=True)

        next_meta = dict(meta or {})
        next_meta.setdefault("id", session_id)
        next_meta.setdefault("updated_at", _iso_ts())
        if not next_meta.get("created_at") and paths.meta.exists():
            try:
                prev = json.loads(paths.meta.read_text(encoding="utf-8"))
                if isinstance(prev, dict) and prev.get("created_at"):
                    next_meta["created_at"] = prev.get("created_at")
            except Exception:
                pass
        if not next_meta.get("created_at"):
            next_meta["created_at"] = _iso_ts()

        paths.meta.write_text(_safe_json_dump(next_meta), encoding="utf-8")

        if messages is not None:
            paths.messages.write_text(_safe_json_dump(messages), encoding="utf-8")
        if state is not None:
            paths.state.write_text(_safe_json_dump(state), encoding="utf-8")
        if last_plan is not None:
            paths.last_plan.write_text(_safe_json_dump(last_plan), encoding="utf-8")

        return paths

    def delete(self, session_id: str) -> None:
        paths = self._paths(session_id)
        if not paths.dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        shutil.rmtree(paths.dir)
