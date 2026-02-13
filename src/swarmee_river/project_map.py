from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, cwd: Path, timeout_s: int = 5) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_s, check=False)
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
    for dirpath, dirnames, filenames in os.walk(root):
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
    out_path = path or (Path.cwd() / ".swarmee" / "project_map.json")
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
