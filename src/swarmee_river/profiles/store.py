from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from swarmee_river.profiles.models import AgentProfile
from swarmee_river.state_paths import state_dir

_CATALOG_VERSION = 1
_LEGACY_DELETE_SENTINEL = "legacy_profiles_deleted_v1"


def _profiles_root(*, root_dir: Path | None = None) -> Path:
    return root_dir if root_dir is not None else state_dir() / "profiles"


def _catalog_path(*, root_dir: Path | None = None) -> Path:
    return _profiles_root(root_dir=root_dir) / "profiles.json"


def _safe_json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _load_catalog(path: Path) -> list[AgentProfile]:
    if not path.exists() or not path.is_file():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    raw_profiles: Any
    if isinstance(payload, dict):
        raw_profiles = payload.get("profiles")
    elif isinstance(payload, list):
        raw_profiles = payload
    else:
        return []

    if not isinstance(raw_profiles, list):
        return []

    profiles: list[AgentProfile] = []
    for item in raw_profiles:
        try:
            profiles.append(AgentProfile.from_dict(item))
        except Exception:
            continue
    return profiles


def _write_catalog(path: Path, profiles: list[AgentProfile]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _CATALOG_VERSION,
        "profiles": [item.to_dict() for item in profiles],
    }
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(_safe_json_dump(payload), encoding="utf-8")
    temp_path.replace(path)
    return path


def list_profiles(*, root_dir: Path | None = None) -> list[AgentProfile]:
    return _load_catalog(_catalog_path(root_dir=root_dir))


def load_profile(profile_id: str, *, root_dir: Path | None = None) -> AgentProfile:
    profile_key = str(profile_id or "").strip()
    if not profile_key:
        raise ValueError("profile_id is required")

    for profile in list_profiles(root_dir=root_dir):
        if profile.id == profile_key:
            return profile
    raise FileNotFoundError(f"Profile not found: {profile_key}")


def save_profile(profile: AgentProfile | dict[str, Any], *, root_dir: Path | None = None) -> AgentProfile:
    normalized = AgentProfile.from_dict(profile.to_dict() if isinstance(profile, AgentProfile) else profile)
    profiles = list_profiles(root_dir=root_dir)

    replaced = False
    for idx, existing in enumerate(profiles):
        if existing.id == normalized.id:
            profiles[idx] = normalized
            replaced = True
            break
    if not replaced:
        profiles.append(normalized)

    _write_catalog(_catalog_path(root_dir=root_dir), profiles)
    return normalized


def delete_profile(profile_id: str, *, root_dir: Path | None = None) -> bool:
    profile_key = str(profile_id or "").strip()
    if not profile_key:
        raise ValueError("profile_id is required")

    profiles = list_profiles(root_dir=root_dir)
    kept = [profile for profile in profiles if profile.id != profile_key]
    if len(kept) == len(profiles):
        return False

    _write_catalog(_catalog_path(root_dir=root_dir), kept)
    return True


def delete_legacy_profiles_on_first_launch(*, root_dir: Path | None = None) -> bool:
    """
    Delete legacy profile catalog once and drop a sentinel marker.

    Returns True when a legacy profile file existed and was removed.
    """
    base = root_dir if root_dir is not None else state_dir()
    sentinel = base / _LEGACY_DELETE_SENTINEL
    if sentinel.exists():
        return False
    catalog = _catalog_path(root_dir=root_dir)
    removed = False
    if catalog.exists() and catalog.is_file():
        with contextlib.suppress(Exception):
            catalog.unlink()
            removed = True
    with contextlib.suppress(Exception):
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("deleted\n", encoding="utf-8")
    return removed
