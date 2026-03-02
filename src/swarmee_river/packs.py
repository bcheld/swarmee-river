from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from strands.tools.tools import AgentTool

from swarmee_river.settings import PackEntry, SwarmeeSettings

PATH_PACK_TYPE = "path_pack"
AGENT_BUNDLE_TYPE = "agent_bundle"


@dataclass(frozen=True)
class Pack:
    name: str
    path: Path
    enabled: bool
    meta: dict[str, Any]

    @property
    def tools_dir(self) -> Path:
        return self.path / "tools"

    @property
    def sops_dir(self) -> Path:
        return self.path / "sops"

    @property
    def system_prompt_path(self) -> Path:
        return self.path / "prompts" / "system.md"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_pack_meta(pack_path: Path) -> dict[str, Any]:
    meta_path = pack_path / "pack.json"
    if not meta_path.exists():
        return {}
    return _load_json(meta_path)


def iter_packs(settings: SwarmeeSettings) -> list[Pack]:
    packs: list[Pack] = []
    for entry in settings.packs.installed:
        if str(getattr(entry, "type", PATH_PACK_TYPE)).strip().lower() != PATH_PACK_TYPE:
            continue
        if not entry.path:
            continue
        pack_path = Path(entry.path).expanduser()
        meta = _load_pack_meta(pack_path)
        name = entry.name or str(meta.get("name") or pack_path.name)
        packs.append(Pack(name=name, path=pack_path, enabled=entry.enabled, meta=meta))
    return packs


def enabled_sop_paths(settings: SwarmeeSettings) -> list[Path]:
    paths: list[Path] = []
    for pack in iter_packs(settings):
        if not pack.enabled:
            continue
        if pack.sops_dir.exists() and pack.sops_dir.is_dir():
            paths.append(pack.sops_dir)
    return paths


def enabled_system_prompts(settings: SwarmeeSettings) -> list[str]:
    prompts: list[str] = []
    for pack in iter_packs(settings):
        if not pack.enabled:
            continue
        p = pack.system_prompt_path
        if p.exists() and p.is_file():
            try:
                text = p.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if text:
                prompts.append(f"Pack System Prompt ({pack.name}):\n{text}")
    return prompts


def _load_module_from_path(path: Path, *, module_name: str, sys_path_root: Path | None = None) -> ModuleType:
    import importlib.util

    if sys_path_root is not None:
        sys.path.insert(0, str(sys_path_root))

    try:
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if not spec or not spec.loader:
            raise ImportError(f"Failed to load tool module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys_path_root is not None:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(sys_path_root))


def _extract_tools(module: ModuleType) -> dict[str, Any]:
    tools: dict[str, Any] = {}

    for value in module.__dict__.values():
        if isinstance(value, AgentTool):
            tools[value.tool_name] = value

    spec = getattr(module, "TOOL_SPEC", None)
    if isinstance(spec, dict):
        name = spec.get("name")
        if isinstance(name, str) and name.strip():
            fn = getattr(module, name.strip(), None)
            if callable(fn):
                tools[name.strip()] = fn

    return tools


def load_enabled_pack_tools(settings: SwarmeeSettings) -> dict[str, Any]:
    """
    Load tools from enabled packs without copying them into ./tools.

    Conventions supported:
    - `@tool` decorated functions (instances of `strands.tools.AgentTool`)
    - Strands Tools-style modules that expose `TOOL_SPEC` and a callable matching TOOL_SPEC["name"]
    """
    tools: dict[str, Any] = {}

    for pack in iter_packs(settings):
        if not pack.enabled:
            continue
        if not pack.tools_dir.exists() or not pack.tools_dir.is_dir():
            continue

        for file_path in sorted(pack.tools_dir.glob("*.py")):
            if file_path.name.startswith("_"):
                continue
            raw_name = f"swarmee_pack_{pack.name}_{file_path.stem}"
            module_name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in raw_name)
            try:
                module = _load_module_from_path(file_path, module_name=module_name, sys_path_root=pack.path)
            except Exception:
                continue

            for tool_name, tool_obj in _extract_tools(module).items():
                # Do not override core tools implicitly.
                if tool_name in tools:
                    continue
                tools[tool_name] = tool_obj

    return tools


def with_installed_pack(settings: SwarmeeSettings, entry: PackEntry) -> SwarmeeSettings:
    installed = [
        p
        for p in settings.packs.installed
        if not (
            str(getattr(p, "type", PATH_PACK_TYPE)).strip().lower() == PATH_PACK_TYPE
            and str(p.name).strip().lower() == str(entry.name).strip().lower()
        )
    ]
    installed.append(entry)
    from swarmee_river.settings import PacksConfig
    from swarmee_river.settings import SwarmeeSettings as Settings

    return Settings(
        models=settings.models,
        safety=settings.safety,
        packs=PacksConfig(installed=installed),
        harness=settings.harness,
        raw=settings.raw,
    )


def list_agent_bundles(settings: SwarmeeSettings) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    for entry in settings.packs.installed:
        if str(getattr(entry, "type", "")).strip().lower() != AGENT_BUNDLE_TYPE:
            continue
        bundle_id = str(getattr(entry, "id", "")).strip()
        if not bundle_id:
            continue
        bundles.append(
            {
                "id": bundle_id,
                "name": str(getattr(entry, "name", "")).strip() or bundle_id,
                "provider": getattr(entry, "provider", None),
                "tier": getattr(entry, "tier", None),
                "system_prompt_snippets": list(getattr(entry, "system_prompt_snippets", []) or []),
                "context_sources": [dict(item) for item in (getattr(entry, "context_sources", []) or [])],
                "active_sops": list(getattr(entry, "active_sops", []) or []),
                "knowledge_base_id": getattr(entry, "knowledge_base_id", None),
                "agents": [dict(item) for item in (getattr(entry, "agents", []) or [])],
                "auto_delegate_assistive": bool(getattr(entry, "auto_delegate_assistive", True)),
                "team_presets": [dict(item) for item in (getattr(entry, "team_presets", []) or [])],
                "enabled": bool(getattr(entry, "enabled", True)),
            }
        )
    bundles.sort(key=lambda item: str(item.get("name", "")).lower())
    return bundles


def list_pack_catalog(settings: SwarmeeSettings) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for entry in settings.packs.installed:
        entry_type = str(getattr(entry, "type", PATH_PACK_TYPE)).strip().lower()
        if entry_type == AGENT_BUNDLE_TYPE:
            bundle_id = str(getattr(entry, "id", "")).strip()
            if not bundle_id:
                continue
            catalog.append(
                {
                    "type": AGENT_BUNDLE_TYPE,
                    "id": bundle_id,
                    "name": str(getattr(entry, "name", "")).strip() or bundle_id,
                    "provider": getattr(entry, "provider", None),
                    "tier": getattr(entry, "tier", None),
                    "path": "",
                    "enabled": bool(getattr(entry, "enabled", True)),
                    "agents": [dict(item) for item in (getattr(entry, "agents", []) or [])],
                    "bundle": {
                        "id": bundle_id,
                        "name": str(getattr(entry, "name", "")).strip() or bundle_id,
                        "provider": getattr(entry, "provider", None),
                        "tier": getattr(entry, "tier", None),
                        "system_prompt_snippets": list(getattr(entry, "system_prompt_snippets", []) or []),
                        "context_sources": [dict(item) for item in (getattr(entry, "context_sources", []) or [])],
                        "active_sops": list(getattr(entry, "active_sops", []) or []),
                        "knowledge_base_id": getattr(entry, "knowledge_base_id", None),
                        "agents": [dict(item) for item in (getattr(entry, "agents", []) or [])],
                        "auto_delegate_assistive": bool(getattr(entry, "auto_delegate_assistive", True)),
                        "team_presets": [dict(item) for item in (getattr(entry, "team_presets", []) or [])],
                        "enabled": bool(getattr(entry, "enabled", True)),
                    },
                }
            )
            continue
        catalog.append(
            {
                "type": PATH_PACK_TYPE,
                "id": str(getattr(entry, "name", "")).strip(),
                "name": str(getattr(entry, "name", "")).strip(),
                "provider": None,
                "tier": None,
                "path": str(getattr(entry, "path", "")).strip(),
                "enabled": bool(getattr(entry, "enabled", True)),
                "agents": [],
            }
        )
    catalog.sort(
        key=lambda item: (
            0 if item.get("type") == AGENT_BUNDLE_TYPE else 1,
            str(item.get("name", "")).lower(),
        )
    )
    return catalog


def find_agent_bundle(settings: SwarmeeSettings, bundle_id: str) -> dict[str, Any] | None:
    target = str(bundle_id or "").strip().lower()
    if not target:
        return None
    for bundle in list_agent_bundles(settings):
        if str(bundle.get("id", "")).strip().lower() == target:
            return bundle
    return None


def with_upserted_agent_bundle(settings: SwarmeeSettings, bundle: dict[str, Any]) -> SwarmeeSettings:
    bundle_id = str(bundle.get("id", "")).strip()
    if not bundle_id:
        raise ValueError("bundle.id is required")
    bundle_name = str(bundle.get("name", "")).strip() or bundle_id
    normalized = PackEntry.from_dict(
        {
            "type": AGENT_BUNDLE_TYPE,
            "id": bundle_id,
            "name": bundle_name,
            "provider": bundle.get("provider"),
            "tier": bundle.get("tier"),
            "system_prompt_snippets": bundle.get("system_prompt_snippets") or [],
            "context_sources": bundle.get("context_sources") or [],
            "active_sops": bundle.get("active_sops") or [],
            "knowledge_base_id": bundle.get("knowledge_base_id"),
            "agents": bundle.get("agents") or [],
            "auto_delegate_assistive": bundle.get("auto_delegate_assistive", True),
            "team_presets": bundle.get("team_presets") or [],
            "enabled": bool(bundle.get("enabled", True)),
        }
    )
    installed: list[PackEntry] = []
    target = bundle_id.lower()
    replaced = False
    for entry in settings.packs.installed:
        is_target = (
            str(getattr(entry, "type", "")).strip().lower() == AGENT_BUNDLE_TYPE
            and str(getattr(entry, "id", "")).strip().lower() == target
        )
        if is_target:
            installed.append(normalized)
            replaced = True
            continue
        installed.append(entry)
    if not replaced:
        installed.append(normalized)
    from swarmee_river.settings import PacksConfig
    from swarmee_river.settings import SwarmeeSettings as Settings

    return Settings(
        models=settings.models,
        safety=settings.safety,
        packs=PacksConfig(installed=installed),
        harness=settings.harness,
        raw=settings.raw,
    )


def with_deleted_agent_bundle(settings: SwarmeeSettings, bundle_id: str) -> SwarmeeSettings:
    target = str(bundle_id or "").strip().lower()
    from swarmee_river.settings import PacksConfig
    from swarmee_river.settings import SwarmeeSettings as Settings

    installed = [
        entry
        for entry in settings.packs.installed
        if not (
            str(getattr(entry, "type", "")).strip().lower() == AGENT_BUNDLE_TYPE
            and str(getattr(entry, "id", "")).strip().lower() == target
        )
    ]
    return Settings(
        models=settings.models,
        safety=settings.safety,
        packs=PacksConfig(installed=installed),
        harness=settings.harness,
        raw=settings.raw,
    )
