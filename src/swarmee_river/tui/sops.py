"""Pure SOP discovery and parsing helpers for the TUI."""

from __future__ import annotations

import contextlib
from pathlib import Path

_SOP_FILE_SUFFIX = ".sop.md"
_SOP_SOURCE_LOCAL = "local"
_SOP_SOURCE_STRANDS = "strands-sops"
_SOP_SOURCE_PRIORITY: dict[str, int] = {
    _SOP_SOURCE_LOCAL: 0,
    "pack": 1,
    _SOP_SOURCE_STRANDS: 2,
}


def _strip_sop_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
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


def _first_sop_paragraph(markdown: str) -> str:
    _meta, body = _strip_sop_frontmatter(markdown)
    lines = [line.rstrip() for line in body.splitlines()]
    paragraph_lines: list[str] = []
    started = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if started:
                break
            continue
        if line.startswith("#") and not started:
            continue
        paragraph_lines.append(line)
        started = True
    preview = " ".join(paragraph_lines).strip()
    if not preview:
        return "(no preview available)"
    if len(preview) > 220:
        return preview[:219].rstrip() + "…"
    return preview


def _load_sop_file(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, body = _strip_sop_frontmatter(raw)
    file_name = path.name
    derived = file_name[: -len(_SOP_FILE_SUFFIX)] if file_name.endswith(_SOP_FILE_SUFFIX) else path.stem
    name = str(meta.get("name", derived)).strip() or derived
    return name, body.strip()


def discover_available_sops() -> list[dict[str, str]]:
    records: dict[str, dict[str, str]] = {}

    def _record_source_priority(source: str) -> int:
        if source == _SOP_SOURCE_LOCAL:
            return _SOP_SOURCE_PRIORITY[_SOP_SOURCE_LOCAL]
        if source.startswith("pack:"):
            return _SOP_SOURCE_PRIORITY["pack"]
        return _SOP_SOURCE_PRIORITY[_SOP_SOURCE_STRANDS]

    def _add_record(*, name: str, path: str, source: str, content: str) -> None:
        sop_name = name.strip()
        if not sop_name:
            return
        existing = records.get(sop_name)
        priority = _record_source_priority(source)
        if existing is not None:
            existing_priority = _record_source_priority(existing.get("source", ""))
            if existing_priority <= priority:
                return
        body = content.strip()
        preview = _first_sop_paragraph(body)
        records[sop_name] = {
            "name": sop_name,
            "path": path.strip(),
            "source": source.strip(),
            "first_paragraph_preview": preview,
            "content": body,
        }

    local_dirs: list[Path] = []
    for candidate in (Path.cwd() / "sops", Path(__file__).resolve().parents[1] / "sops"):
        if candidate.exists() and candidate.is_dir():
            local_dirs.append(candidate.resolve())
    seen_local: set[str] = set()
    for directory in local_dirs:
        key = str(directory)
        if key in seen_local:
            continue
        seen_local.add(key)
        for file_path in sorted(directory.glob(f"*{_SOP_FILE_SUFFIX}")):
            if not file_path.is_file():
                continue
            with contextlib.suppress(Exception):
                name, content = _load_sop_file(file_path)
                _add_record(name=name, path=str(file_path.resolve()), source=_SOP_SOURCE_LOCAL, content=content)

    try:
        from swarmee_river.packs import iter_packs
        from swarmee_river.settings import load_settings

        settings = load_settings()
        for pack in iter_packs(settings):
            if not pack.enabled:
                continue
            sop_dir = pack.sops_dir
            if not sop_dir.exists() or not sop_dir.is_dir():
                continue
            source_label = f"pack:{pack.name}"
            for file_path in sorted(sop_dir.glob(f"*{_SOP_FILE_SUFFIX}")):
                if not file_path.is_file():
                    continue
                with contextlib.suppress(Exception):
                    name, content = _load_sop_file(file_path)
                    _add_record(name=name, path=str(file_path.resolve()), source=source_label, content=content)
    except Exception:
        pass

    try:
        import strands_agents_sops as strands_sops

        module_path = Path(getattr(strands_sops, "__file__", "")).resolve().parent
        candidate_dirs = [module_path / "sops", module_path]
        saw_file = False
        for directory in candidate_dirs:
            if not directory.exists() or not directory.is_dir():
                continue
            for file_path in sorted(directory.glob(f"*{_SOP_FILE_SUFFIX}")):
                if not file_path.is_file():
                    continue
                saw_file = True
                with contextlib.suppress(Exception):
                    name, content = _load_sop_file(file_path)
                    _add_record(name=name, path=str(file_path.resolve()), source=_SOP_SOURCE_STRANDS, content=content)
        if not saw_file:
            for attr_name in dir(strands_sops):
                if attr_name.startswith("_"):
                    continue
                value = getattr(strands_sops, attr_name, None)
                if not isinstance(value, str) or len(value.strip()) < 40:
                    continue
                meta, body = _strip_sop_frontmatter(value)
                name = str(meta.get("name", attr_name)).strip() or attr_name
                _add_record(
                    name=name,
                    path=f"{strands_sops.__name__}.{attr_name}",
                    source=_SOP_SOURCE_STRANDS,
                    content=body.strip(),
                )
    except Exception:
        pass

    return [records[name] for name in sorted(records.keys())]


def discover_available_sop_names() -> list[str]:
    return [record["name"] for record in discover_available_sops()]


__all__ = [
    "_SOP_FILE_SUFFIX",
    "_SOP_SOURCE_LOCAL",
    "_SOP_SOURCE_PRIORITY",
    "_SOP_SOURCE_STRANDS",
    "_first_sop_paragraph",
    "_load_sop_file",
    "_strip_sop_frontmatter",
    "discover_available_sop_names",
    "discover_available_sops",
]
