from __future__ import annotations

import contextlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from swarmee_river.state_paths import state_dir

_PROMPTS_FILENAME = "prompts.json"
_MIGRATION_SENTINEL = "prompt_assets_migrated_v1"
_PROMPT_FILE_SUFFIX = ".prompt.md"
_ORCHESTRATOR_DEFAULT_ID = "orchestrator_base"
_DEFAULT_PROMPT_TEXT = "You are a helpful assistant."


@dataclass
class PromptAsset:
    id: str
    name: str
    content: str
    tags: list[str] = field(default_factory=list)
    source: str = "project"
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptAsset:
        prompt_id = _sanitize_prompt_id(str(data.get("id", "")))
        name = str(data.get("name", "")).strip() or prompt_id
        content = str(data.get("content", "")).strip()
        tags = [str(tag).strip() for tag in (data.get("tags") or []) if str(tag).strip()]
        source = str(data.get("source", "project")).strip() or "project"
        created_at = str(data.get("created_at", "")).strip() or None
        updated_at = str(data.get("updated_at", "")).strip() or None
        return cls(
            id=prompt_id,
            name=name,
            content=content,
            tags=tags,
            source=source,
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass(frozen=True)
class BootstrapResult:
    migrated: bool
    migrated_count: int
    prompt_paths: list[str]


def _prompts_path() -> Path:
    return state_dir() / _PROMPTS_FILENAME


def _sentinel_path() -> Path:
    return state_dir() / _MIGRATION_SENTINEL


def _legacy_templates_path() -> Path:
    return state_dir() / "prompt_templates.json"


def _legacy_dot_prompt_path() -> Path:
    return Path.cwd() / ".prompt"


def _sanitize_prompt_id(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (value or "").strip().lower())
    token = token.strip("_")
    return token or _ORCHESTRATOR_DEFAULT_ID


def _safe_read_text(path: Path) -> str:
    with contextlib.suppress(Exception):
        return path.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def _strip_prompt_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    text = (markdown or "").lstrip("\ufeff")
    if not text.startswith("---\n"):
        return {}, text.strip()
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text.strip()
    header = text[4:end].strip()
    body = text[end + len("\n---\n") :].strip()
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        k = key.strip().lower()
        v = value.strip()
        if k and v:
            meta[k] = v
    return meta, body


def load_prompt_assets() -> list[PromptAsset]:
    path = _prompts_path()
    if not path.exists() or not path.is_file():
        return []
    with contextlib.suppress(Exception):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            assets: list[PromptAsset] = []
            seen: set[str] = set()
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                asset = PromptAsset.from_dict(entry)
                if asset.id in seen:
                    continue
                seen.add(asset.id)
                assets.append(asset)
            return sorted(assets, key=lambda item: item.name.lower())
    return []


def save_prompt_assets(assets: list[PromptAsset]) -> None:
    path = _prompts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped: dict[str, PromptAsset] = {}
    for asset in assets:
        deduped[asset.id] = asset
    serialized = [asset.to_dict() for asset in sorted(deduped.values(), key=lambda item: item.name.lower())]
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def _discover_legacy_prompt_markdown_files() -> list[Path]:
    discovered: list[Path] = []
    search_roots: list[Path] = []
    for candidate in (Path.cwd() / "sops", Path(__file__).resolve().parent / "sops"):
        if candidate.exists() and candidate.is_dir():
            resolved = candidate.resolve()
            if str(resolved) not in {str(path) for path in search_roots}:
                search_roots.append(resolved)
    for root in search_roots:
        for path in sorted(root.glob(f"*{_PROMPT_FILE_SUFFIX}")):
            if path.exists() and path.is_file():
                discovered.append(path)
    return discovered


def _load_legacy_template_assets() -> list[PromptAsset]:
    assets: dict[str, PromptAsset] = {}

    legacy_templates = _legacy_templates_path()
    if legacy_templates.exists() and legacy_templates.is_file():
        with contextlib.suppress(Exception):
            raw = json.loads(legacy_templates.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for entry in raw:
                    if not isinstance(entry, dict):
                        continue
                    prompt_id = _sanitize_prompt_id(str(entry.get("id", "")) or str(entry.get("name", "")))
                    name = str(entry.get("name", "")).strip() or prompt_id
                    content = str(entry.get("content", "")).strip()
                    tags = [str(tag).strip() for tag in (entry.get("tags") or []) if str(tag).strip()]
                    if not content:
                        continue
                    assets[prompt_id] = PromptAsset(
                        id=prompt_id,
                        name=name,
                        content=content,
                        tags=tags,
                        source="migrated_template",
                    )

    for path in _discover_legacy_prompt_markdown_files():
        raw_text = _safe_read_text(path)
        if not raw_text:
            continue
        meta, body = _strip_prompt_frontmatter(raw_text)
        if not body:
            continue
        file_name = path.name
        derived_name = file_name[: -len(_PROMPT_FILE_SUFFIX)] if file_name.endswith(_PROMPT_FILE_SUFFIX) else path.stem
        prompt_id = _sanitize_prompt_id(str(meta.get("id", "")) or derived_name)
        name = str(meta.get("name", "")).strip() or derived_name
        tags_raw = str(meta.get("tags", "")).strip()
        tags = [token.strip() for token in tags_raw.split(",") if token.strip()] if tags_raw else []
        assets[prompt_id] = PromptAsset(
            id=prompt_id,
            name=name,
            content=body,
            tags=tags,
            source="migrated_template",
        )

    return sorted(assets.values(), key=lambda item: item.name.lower())


def ensure_prompt_assets_bootstrapped() -> BootstrapResult:
    prompt_paths: list[str] = []
    for path in (_legacy_dot_prompt_path(), _legacy_templates_path()):
        if path.exists() and path.is_file():
            prompt_paths.append(str(path))

    sentinel = _sentinel_path()
    current_assets = load_prompt_assets()
    if sentinel.exists() and current_assets:
        return BootstrapResult(migrated=False, migrated_count=0, prompt_paths=prompt_paths)

    assets_by_id: dict[str, PromptAsset] = {asset.id: asset for asset in current_assets}
    migrated_count = 0

    if _ORCHESTRATOR_DEFAULT_ID not in assets_by_id:
        dot_prompt_text = _safe_read_text(_legacy_dot_prompt_path())
        orchestrator_content = dot_prompt_text or _DEFAULT_PROMPT_TEXT
        source = "migrated_dotprompt" if dot_prompt_text else "project"
        assets_by_id[_ORCHESTRATOR_DEFAULT_ID] = PromptAsset(
            id=_ORCHESTRATOR_DEFAULT_ID,
            name="Orchestrator Base",
            content=orchestrator_content,
            tags=["orchestrator", "base"],
            source=source,
        )
        migrated_count += 1

    for asset in _load_legacy_template_assets():
        if asset.id in assets_by_id:
            continue
        assets_by_id[asset.id] = asset
        migrated_count += 1

    try:
        save_prompt_assets(list(assets_by_id.values()))
    except Exception:
        return BootstrapResult(migrated=False, migrated_count=0, prompt_paths=prompt_paths)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        sentinel.write_text("migrated\n", encoding="utf-8")

    return BootstrapResult(migrated=True, migrated_count=migrated_count, prompt_paths=prompt_paths)


def resolve_prompt_asset(prompt_id: str, *, assets: list[PromptAsset] | None = None) -> PromptAsset | None:
    target = _sanitize_prompt_id(prompt_id)
    for asset in assets or load_prompt_assets():
        if asset.id == target:
            return asset
    return None


def resolve_orchestrator_prompt_from_agent(
    orchestrator_agent: dict[str, Any] | None,
    assets_by_id: dict[str, PromptAsset] | None = None,
    *,
    default_id: str = _ORCHESTRATOR_DEFAULT_ID,
) -> str:
    ensure_prompt_assets_bootstrapped()
    assets = assets_by_id if isinstance(assets_by_id, dict) else {asset.id: asset for asset in load_prompt_assets()}
    if isinstance(orchestrator_agent, dict):
        composed = resolve_agent_prompt_text(orchestrator_agent, assets).strip()
        if composed:
            return composed
    prompt_id = _sanitize_prompt_id(default_id)
    asset = assets.get(prompt_id)
    if asset is None:
        fallback = assets.get(_ORCHESTRATOR_DEFAULT_ID) or resolve_prompt_asset(_ORCHESTRATOR_DEFAULT_ID)
        if fallback is not None and fallback.content.strip():
            return fallback.content.strip()
        return _DEFAULT_PROMPT_TEXT
    content = asset.content.strip()
    return content or _DEFAULT_PROMPT_TEXT


def resolve_orchestrator_prompt(_settings: Any | None = None) -> str:
    """Backwards-compatible orchestrator resolver (settings are ignored)."""
    return resolve_orchestrator_prompt_from_agent(None)


def resolve_agent_prompt_text(agent_def: dict[str, Any], assets_by_id: dict[str, PromptAsset]) -> str:
    refs = agent_def.get("prompt_refs") or []
    sections: list[str] = []
    if isinstance(refs, list):
        for prompt_id in refs:
            token = _sanitize_prompt_id(str(prompt_id))
            asset = assets_by_id.get(token)
            if asset is None:
                continue
            text = asset.content.strip()
            if text:
                sections.append(text)
    inline_text = str(agent_def.get("prompt", "")).strip()
    if inline_text:
        sections.append(inline_text)
    return "\n\n".join(section for section in sections if section).strip()


__all__ = [
    "BootstrapResult",
    "PromptAsset",
    "ensure_prompt_assets_bootstrapped",
    "load_prompt_assets",
    "resolve_agent_prompt_text",
    "resolve_orchestrator_prompt_from_agent",
    "resolve_orchestrator_prompt",
    "resolve_prompt_asset",
    "save_prompt_assets",
]
