from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

_ALLOWED_CONTEXT_SOURCE_TYPES = {"file", "note", "sop", "kb", "url"}


def _normalize_text(value: object | None, *, lower: bool = False) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text.lower() if lower else text


def _normalize_text_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _normalize_text(item)
        if text is None or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _sanitize_context_source_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


def normalize_context_source(source: Any) -> dict[str, str] | None:
    if not isinstance(source, dict):
        return None

    source_type = str(source.get("type", "")).strip().lower()
    if source_type not in _ALLOWED_CONTEXT_SOURCE_TYPES:
        return None

    normalized: dict[str, str] = {"type": source_type}
    if source_type == "file":
        path = str(source.get("path", "")).strip()
        if not path:
            return None
        normalized["path"] = path
        normalized["id"] = _sanitize_context_source_token(str(source.get("id", path)))
        return normalized
    if source_type == "note":
        text = str(source.get("text", "")).strip()
        if not text:
            return None
        normalized["text"] = text
        normalized["id"] = _sanitize_context_source_token(str(source.get("id", text[:64])))
        return normalized
    if source_type == "sop":
        name = str(source.get("name", "")).strip()
        if not name:
            return None
        normalized["name"] = name
        normalized["id"] = _sanitize_context_source_token(str(source.get("id", name)))
        return normalized
    if source_type == "kb":
        kb_id = str(source.get("id", source.get("kb_id", ""))).strip()
        if not kb_id:
            return None
        normalized["id"] = kb_id
        return normalized
    if source_type == "url":
        url = str(source.get("url", source.get("path", ""))).strip()
        if not url:
            return None
        normalized["url"] = url
        normalized["id"] = _sanitize_context_source_token(str(source.get("id", url)))
        return normalized
    return None


def normalize_context_sources(raw_sources: Any) -> list[dict[str, str]]:
    if not isinstance(raw_sources, list):
        return []

    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_sources:
        source = normalize_context_source(item)
        if source is None:
            continue
        source_type = source.get("type", "")
        value_key = (
            source.get("path")
            or source.get("text")
            or source.get("name")
            or source.get("url")
            or source.get("id")
            or ""
        ).strip()
        dedupe_key = (source_type, value_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(source)
    return normalized


def _normalize_team_preset_spec(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    try:
        encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        decoded = json.loads(encoded)
    except Exception:
        return None
    return decoded if isinstance(decoded, dict) else None


def normalize_team_preset(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    name = _normalize_text(raw.get("name"))
    if name is None:
        return None

    preset_id = _normalize_text(raw.get("id"))
    if preset_id is None:
        preset_id = _sanitize_context_source_token(name)

    spec = _normalize_team_preset_spec(raw.get("spec"))
    if spec is None:
        return None

    return {
        "id": preset_id,
        "name": name,
        "description": str(raw.get("description", "")).strip(),
        "spec": spec,
    }


def normalize_team_presets(raw_presets: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_presets, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw_presets:
        preset = normalize_team_preset(item)
        if preset is None:
            continue
        preset_id = str(preset.get("id", "")).strip()
        if not preset_id or preset_id in seen_ids:
            continue
        seen_ids.add(preset_id)
        normalized.append(preset)
    return normalized


@dataclass
class AgentProfile:
    id: str
    name: str
    provider: str | None = None
    tier: str | None = None
    system_prompt_snippets: list[str] = field(default_factory=list)
    context_sources: list[dict[str, str]] = field(default_factory=list)
    active_sops: list[str] = field(default_factory=list)
    knowledge_base_id: str | None = None
    team_presets: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Any) -> AgentProfile:
        if not isinstance(raw, dict):
            raise ValueError("profile payload must be an object")

        profile_id = _normalize_text(raw.get("id"))
        if profile_id is None:
            raise ValueError("profile id is required")

        name = _normalize_text(raw.get("name"))
        if name is None:
            raise ValueError("profile name is required")

        return cls(
            id=profile_id,
            name=name,
            provider=_normalize_text(raw.get("provider"), lower=True),
            tier=_normalize_text(raw.get("tier"), lower=True),
            system_prompt_snippets=_normalize_text_list(raw.get("system_prompt_snippets")),
            context_sources=normalize_context_sources(raw.get("context_sources")),
            active_sops=_normalize_text_list(raw.get("active_sops")),
            knowledge_base_id=_normalize_text(raw.get("knowledge_base_id")),
            team_presets=normalize_team_presets(raw.get("team_presets")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "tier": self.tier,
            "system_prompt_snippets": list(self.system_prompt_snippets),
            "context_sources": [dict(source) for source in self.context_sources],
            "active_sops": list(self.active_sops),
            "knowledge_base_id": self.knowledge_base_id,
            "team_presets": normalize_team_presets(self.team_presets),
        }
