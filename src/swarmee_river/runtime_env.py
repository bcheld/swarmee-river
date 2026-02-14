from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any


def _normalized_os_name() -> str:
    raw = (platform.system() or "").strip().lower()
    if raw.startswith("win"):
        return "windows"
    if raw == "darwin":
        return "macos"
    if raw == "linux":
        return "linux"
    return raw or "unknown"


def _detect_shell_program() -> str:
    shell = os.getenv("SHELL")
    if shell and shell.strip():
        return shell.strip()

    comspec = os.getenv("COMSPEC")
    if comspec and comspec.strip():
        return comspec.strip()

    return ""


def _detect_shell_family(shell_program: str) -> str:
    name = Path(shell_program).name.strip().lower()
    if "powershell" in name or name in {"pwsh", "pwsh.exe"}:
        return "powershell"
    if name in {"cmd", "cmd.exe"}:
        return "cmd"
    if "zsh" in name:
        return "zsh"
    if "bash" in name:
        return "bash"
    if "fish" in name:
        return "fish"
    if name in {"sh", "dash", "ksh"}:
        return "sh"
    return "unknown"


def detect_runtime_environment(*, cwd: Path | None = None) -> dict[str, Any]:
    os_name = _normalized_os_name()
    release = (platform.release() or "").strip()
    version = (platform.version() or "").strip()
    shell_program = _detect_shell_program()
    shell_family = _detect_shell_family(shell_program)

    is_wsl = bool(os_name == "linux" and "microsoft" in release.lower())
    is_ci = bool(os.getenv("CI", "").strip().lower() in {"1", "true", "t", "yes", "y", "on"})

    return {
        "os": os_name,
        "platform": platform.platform(),
        "release": release,
        "version": version,
        "shell_program": shell_program,
        "shell_family": shell_family,
        "is_wsl": is_wsl,
        "is_ci": is_ci,
        "python_version": platform.python_version(),
        "cwd": str((cwd or Path.cwd()).resolve()),
        "encoding": str(sys.getfilesystemencoding() or ""),
    }


def render_runtime_environment_section(runtime_env: dict[str, Any]) -> str:
    os_name = str(runtime_env.get("os") or "unknown")
    shell_family = str(runtime_env.get("shell_family") or "unknown")
    shell_program = str(runtime_env.get("shell_program") or "(unset)")
    is_wsl = bool(runtime_env.get("is_wsl"))
    python_version = str(runtime_env.get("python_version") or "unknown")

    command_guidance = (
        "Use PowerShell/CMD-compatible commands unless the command explicitly invokes bash."
        if os_name == "windows" and shell_family in {"powershell", "cmd"}
        else "Use commands compatible with this OS and shell."
    )

    lines = [
        "Runtime Environment (tool and command compatibility):",
        f"- os: {os_name}",
        f"- shell_family: {shell_family}",
        f"- shell_program: {shell_program}",
        f"- is_wsl: {is_wsl}",
        f"- python_version: {python_version}",
        f"- command_guidance: {command_guidance}",
    ]
    return "\n".join(lines)
