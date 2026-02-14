from __future__ import annotations

from swarmee_river.runtime_env import detect_runtime_environment, render_runtime_environment_section


def test_detect_runtime_environment_contains_core_fields() -> None:
    runtime = detect_runtime_environment()

    assert isinstance(runtime, dict)
    assert "os" in runtime
    assert "shell_family" in runtime
    assert "python_version" in runtime
    assert "cwd" in runtime


def test_render_runtime_environment_section_windows_guidance() -> None:
    text = render_runtime_environment_section(
        {
            "os": "windows",
            "shell_family": "powershell",
            "shell_program": "powershell.exe",
            "is_wsl": False,
            "python_version": "3.12.0",
        }
    )

    assert "Runtime Environment" in text
    assert "PowerShell/CMD-compatible commands" in text
