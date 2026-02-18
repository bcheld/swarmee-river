from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.state_paths import project_map_path as _default_project_map_path
from swarmee_river.utils.env_utils import truthy_env
from tools.project_context import run_project_context


def _run(cmd: list[str], *, cwd: Path, timeout_s: int = 5) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except Exception as e:
        return 1, str(e)
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    return p.returncode, out if out else err


def _detect_languages(root: Path) -> list[str]:
    exts: dict[str, int] = {}
    skip_dirs = {".git", ".venv", "venv", "dist", "build", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
    file_limit = 5000
    seen_files = 0
    for _dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            suffix = Path(fn).suffix.lower().lstrip(".")
            if not suffix:
                continue
            exts[suffix] = exts.get(suffix, 0) + 1
            seen_files += 1
            if seen_files >= file_limit:
                break
        if seen_files >= file_limit:
            break

    # rough mapping
    mapping = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "jsx": "javascript",
        "go": "go",
        "rs": "rust",
        "java": "java",
        "kt": "kotlin",
        "rb": "ruby",
        "php": "php",
        "cs": "csharp",
        "cpp": "cpp",
        "c": "c",
        "h": "c",
        "md": "markdown",
        "toml": "toml",
        "yaml": "yaml",
        "yml": "yaml",
    }
    langs: dict[str, int] = {}
    for ext, count in exts.items():
        lang = mapping.get(ext)
        if not lang:
            continue
        langs[lang] = langs.get(lang, 0) + count

    return [k for k, _v in sorted(langs.items(), key=lambda kv: kv[1], reverse=True)]


def _detect_package_managers(root: Path) -> list[str]:
    found: list[str] = []
    if (root / "pyproject.toml").exists():
        found.append("python (pyproject.toml)")
    if (root / "requirements.txt").exists():
        found.append("python (requirements.txt)")
    if (root / "package.json").exists():
        found.append("node (package.json)")
    if (root / "go.mod").exists():
        found.append("go (go.mod)")
    if (root / "Cargo.toml").exists():
        found.append("rust (Cargo.toml)")
    return found


def _detect_test_commands(root: Path) -> list[str]:
    cmds: list[str] = []
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        cmds.append("pytest")
        cmds.append("hatch test")
    if (root / "package.json").exists():
        cmds.append("npm test")
    if (root / "Makefile").exists():
        cmds.append("make test")
    # de-dup while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for c in cmds:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def build_project_map(cwd: Path | None = None) -> dict[str, Any]:
    base = (cwd or Path.cwd()).expanduser().resolve()
    code, git_root = _run(["git", "rev-parse", "--show-toplevel"], cwd=base)
    root = Path(git_root).expanduser().resolve() if code == 0 and git_root else base

    project: dict[str, Any] = {
        "cwd": str(base),
        "git_root": str(root) if root else None,
        "languages": _detect_languages(root),
        "package_managers": _detect_package_managers(root),
        "test_commands": _detect_test_commands(root),
    }
    return project


def save_project_map(project_map: dict[str, Any], path: Path | None = None) -> Path:
    out_path = path or _default_project_map_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(project_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def render_project_map_summary(project_map: dict[str, Any]) -> str:
    lines: list[str] = ["Project map (cached):"]
    if project_map.get("git_root"):
        lines.append(f"- git_root: {project_map.get('git_root')}")
    langs = project_map.get("languages") or []
    if isinstance(langs, list) and langs:
        lines.append(f"- languages: {', '.join(str(x) for x in langs[:8])}")
    pms = project_map.get("package_managers") or []
    if isinstance(pms, list) and pms:
        lines.append(f"- package: {', '.join(str(x) for x in pms[:8])}")
    tests = project_map.get("test_commands") or []
    if isinstance(tests, list) and tests:
        lines.append(f"- tests: {', '.join(str(x) for x in tests[:6])}")
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class ContextSnapshot:
    preflight_prompt_section: str | None
    project_map_prompt_section: str | None


def build_context_snapshot(
    *,
    artifact_store: ArtifactStore,
    interactive: bool,
    default_preflight_level: str | None = None,
) -> ContextSnapshot:
    """
    Build a lightweight repo context snapshot (preflight + project map) and return prompt sections.

    Controlled by existing env vars:
    - SWARMEE_PREFLIGHT=enabled|disabled
    - SWARMEE_PREFLIGHT_LEVEL=summary|summary+tree|summary+files
    - SWARMEE_PREFLIGHT_MAX_CHARS
    - SWARMEE_PROJECT_MAP=enabled|disabled
    """
    preflight_prompt_section: str | None = None
    project_map_prompt_section: str | None = None

    if truthy_env("SWARMEE_PREFLIGHT", True):
        level = (os.getenv("SWARMEE_PREFLIGHT_LEVEL") or default_preflight_level or "summary").strip().lower()
        max_chars = int(os.getenv("SWARMEE_PREFLIGHT_MAX_CHARS", "8000"))
        actions = ["summary"]
        if level == "summary+tree":
            actions.append("tree")
        elif level == "summary+files":
            actions.append("files")

        preflight_parts: list[str] = []
        for action in actions:
            try:
                result = run_project_context(action=action, max_chars=max_chars)
                if result.get("status") == "success":
                    preflight_parts.append(result.get("content", [{"text": ""}])[0].get("text", ""))
            except Exception:
                continue
        preflight_text = "\n\n".join([p for p in preflight_parts if p]).strip()
        if preflight_text:
            artifact_store.write_text(
                kind="context_snapshot",
                text=preflight_text,
                suffix="txt",
                metadata={"source": "project_context", "level": level},
            )
            preflight_prompt_section = f"Project context snapshot:\n{preflight_text}"
            should_print_preflight = truthy_env("SWARMEE_PREFLIGHT_PRINT", False)
            if interactive and should_print_preflight:
                print("\n[preflight]\n" + preflight_text + "\n")

    if truthy_env("SWARMEE_PROJECT_MAP", True):
        try:
            pm = build_project_map()
            pm_path = save_project_map(pm)
            project_map_prompt_section = render_project_map_summary(pm) + f"\n(project_map: {pm_path})"
        except Exception:
            project_map_prompt_section = None
    else:
        project_map_prompt_section = None

    return ContextSnapshot(
        preflight_prompt_section=preflight_prompt_section,
        project_map_prompt_section=project_map_prompt_section,
    )
