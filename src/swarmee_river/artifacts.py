from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from swarmee_river.state_paths import artifacts_dir as _default_artifacts_dir


def _iso_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _compact_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _safe_name(value: str) -> str:
    keep = []
    for ch in value or "":
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("._")
    return cleaned or "artifact"


_INDEX_LOCK = threading.Lock()


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    path: Path


class ArtifactStore:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or _default_artifacts_dir()
        self.index_path = self.artifacts_dir / "index.jsonl"
        self._lock = _INDEX_LOCK

    def write_text(
        self,
        *,
        kind: str,
        text: str,
        suffix: str = "txt",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_id = uuid.uuid4().hex
        filename = f"{_compact_ts()}_{_safe_name(kind)}_{artifact_id}.{_safe_name(suffix)}"
        path = self.artifacts_dir / filename
        path.write_text(text, encoding="utf-8", errors="replace")

        entry: dict[str, Any] = {
            "id": artifact_id,
            "kind": kind,
            "path": str(path),
            "created_at": _iso_ts(),
            "bytes": path.stat().st_size,
            "chars": len(text),
        }
        if metadata:
            entry["meta"] = metadata

        self.append_index(entry)
        return ArtifactRef(artifact_id=artifact_id, path=path)

    def append_index(self, entry: dict[str, Any]) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with self.index_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")

    def list(self, *, limit: int = 50, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if kind and str(data.get("kind")) != kind:
                    continue
                entries.append(data)

        # newest first
        entries.reverse()
        return entries[: max(0, int(limit))]

    def get_by_id(self, artifact_id: str) -> dict[str, Any] | None:
        if not self.index_path.exists():
            return None
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if isinstance(data, dict) and str(data.get("id")) == artifact_id:
                    return data
        return None

    def read_text(self, path: str | Path, *, max_chars: int | None = None) -> str:
        p = Path(path).expanduser()
        text = p.read_text(encoding="utf-8", errors="replace")
        if max_chars is not None and max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + f"\n… (truncated to {max_chars} chars) …"
        return text


def tools_expected_from_plan(plan: Any) -> set[str]:
    """
    Extract a conservative tool allowlist from a WorkPlan-like object.

    This is intentionally defensive and accepts either a Pydantic model instance
    or a plain dict.
    """
    steps: Iterable[Any] = []
    if plan is None:
        return set()
    if isinstance(plan, dict):
        steps = plan.get("steps") or []
    else:
        steps = getattr(plan, "steps", []) or []

    allowed: set[str] = set()
    for step in steps:
        tools = step.get("tools_expected") if isinstance(step, dict) else getattr(step, "tools_expected", None)
        if not tools:
            continue
        if isinstance(tools, (list, tuple)):
            for t in tools:
                if isinstance(t, str) and t.strip():
                    allowed.add(t.strip())
    return allowed
