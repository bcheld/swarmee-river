from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

from strands import tool

from swarmee_river.tool_permissions import set_permissions
from swarmee_river.utils.tool_interrupts import (
    current_interrupt_event,
    run_subprocess_capture_interruptible,
)

_ORIGINAL_INTERRUPTIBLE_RUN = run_subprocess_capture_interruptible


@tool
def shell(
    command: str,
    cwd: Optional[str] = None,
    timeout_s: int = 600,
    env: Optional[dict[str, str]] = None,
    non_interactive_mode: bool = True,
    user_message_override: Optional[str] = None,
) -> dict[str, Any]:
    """
    Cross-platform shell execution tool (Windows/macOS/Linux).

    Notes:
    - Uses `subprocess.run(..., shell=True)` to allow typical shell syntax.
    - Intended for enterprise environments where `strands_tools.shell` may be unavailable on Windows.
    - In non-interactive mode, stdin is detached to avoid blocking on prompts.
    """
    if not command or not command.strip():
        return {"status": "error", "content": [{"text": "Command is required."}]}

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        helper_overridden = run_subprocess_capture_interruptible is not _ORIGINAL_INTERRUPTIBLE_RUN
        if current_interrupt_event() is not None or helper_overridden:
            completed = run_subprocess_capture_interruptible(
                command,
                shell=True,
                cwd=cwd,
                env=run_env,
                stdin=subprocess.DEVNULL if non_interactive_mode else None,
                timeout_s=timeout_s,
            )
        else:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=run_env,
                stdin=subprocess.DEVNULL if non_interactive_mode else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Shell execution error: {str(e)}"}]}

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    exit_code = completed.returncode
    if getattr(completed, "interrupted", False):
        stderr = stderr or "Command interrupted."
    elif getattr(completed, "timed_out", False):
        stderr = stderr or f"Command timed out after {timeout_s}s."

    status = (
        "success"
        if exit_code == 0
        and not getattr(completed, "interrupted", False)
        and not getattr(completed, "timed_out", False)
        else "error"
    )
    combined = ""
    if stdout:
        combined += f"STDOUT:\n{stdout}\n"
    if stderr:
        combined += f"STDERR:\n{stderr}\n"
    combined = combined.strip()

    return {
        "status": status,
        "content": [
            {"text": f"exit_code: {exit_code}"},
            {"text": combined or "(no output)"},
        ],
    }


set_permissions(shell, "execute")
