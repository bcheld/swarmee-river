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
        return json.loads(path.read_text(encoding="utf-8"))
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
            tools[value.name] = value

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
    installed = [p for p in settings.packs.installed if p.name != entry.name]
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
