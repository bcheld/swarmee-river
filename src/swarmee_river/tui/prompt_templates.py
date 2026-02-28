"""Prompt template discovery, persistence, and CRUD helpers for the TUI."""

from __future__ import annotations

import contextlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from swarmee_river.state_paths import state_dir

_PROMPT_FILE_SUFFIX = ".prompt.md"
_TEMPLATES_FILENAME = "prompt_templates.json"


@dataclass
class PromptTemplate:
    id: str
    name: str
    content: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = "local"  # "local", "default", "s3"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptTemplate:
        return cls(
            id=str(data.get("id", "")).strip() or uuid.uuid4().hex[:12],
            name=str(data.get("name", "")).strip(),
            content=str(data.get("content", "")).strip(),
            tags=[str(t).strip() for t in (data.get("tags") or []) if str(t).strip()],
            source=str(data.get("source", "local")).strip(),
        )


def _templates_path() -> Path:
    return state_dir() / _TEMPLATES_FILENAME


def _strip_prompt_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    """Extract YAML-style frontmatter metadata and body from a prompt file."""
    text = (markdown or "").lstrip("\ufeff")
    if not text.startswith("---\n"):
        return {}, text.strip()
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text.strip()
    header = text[4:end].strip()
    body = text[end + len("\n---\n"):].strip()
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


def load_prompt_templates() -> list[PromptTemplate]:
    """Load user-created prompt templates from .swarmee/prompt_templates.json."""
    path = _templates_path()
    if not path.exists():
        return []
    with contextlib.suppress(Exception):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            templates: list[PromptTemplate] = []
            for entry in raw:
                if isinstance(entry, dict):
                    templates.append(PromptTemplate.from_dict(entry))
            return templates
    return []


def save_prompt_templates(templates: list[PromptTemplate]) -> None:
    """Persist prompt templates to .swarmee/prompt_templates.json."""
    path = _templates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([t.to_dict() for t in templates], indent=2, default=str),
        encoding="utf-8",
    )


def discover_prompt_templates() -> list[PromptTemplate]:
    """Discover prompt templates from filesystem and user storage.

    Sources (in order):
    1. ``*.prompt.md`` files in ``<cwd>/sops/`` and package ``sops/`` dirs.
    2. User-saved templates in ``.swarmee/prompt_templates.json``.
    """
    records: dict[str, PromptTemplate] = {}

    # 1. Discover *.prompt.md files from sops directories (same dirs SOPs use).
    local_dirs: list[Path] = []
    for candidate in (Path.cwd() / "sops", Path(__file__).resolve().parents[1] / "sops"):
        if candidate.exists() and candidate.is_dir():
            resolved = candidate.resolve()
            if str(resolved) not in {str(d) for d in local_dirs}:
                local_dirs.append(resolved)

    for directory in local_dirs:
        for file_path in sorted(directory.glob(f"*{_PROMPT_FILE_SUFFIX}")):
            if not file_path.is_file():
                continue
            with contextlib.suppress(Exception):
                raw_text = file_path.read_text(encoding="utf-8", errors="replace")
                meta, body = _strip_prompt_frontmatter(raw_text)
                file_name = file_path.name
                derived = file_name[:-len(_PROMPT_FILE_SUFFIX)] if file_name.endswith(_PROMPT_FILE_SUFFIX) else file_path.stem
                name = str(meta.get("name", derived)).strip() or derived
                template_id = str(meta.get("id", derived)).strip() or derived
                tags_raw = str(meta.get("tags", "")).strip()
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
                records[template_id] = PromptTemplate(
                    id=template_id,
                    name=name,
                    content=body,
                    tags=tags,
                    source="default",
                )

    # 2. User-saved templates override/extend file-based ones.
    for template in load_prompt_templates():
        records[template.id] = template

    return sorted(records.values(), key=lambda t: t.name.lower())


__all__ = [
    "PromptTemplate",
    "discover_prompt_templates",
    "load_prompt_templates",
    "save_prompt_templates",
]
