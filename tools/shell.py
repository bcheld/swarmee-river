from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

from strands import tool


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
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=run_env,
            stdin=subprocess.DEVNULL if non_interactive_mode else None,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "status": "error",
            "content": [
                {"text": f"Command timed out after {timeout_s}s."},
                {"text": str(e)},
            ],
        }
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Shell execution error: {str(e)}"}]}

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    exit_code = completed.returncode

    # Print for interactive `!` usage where the caller ignores the return value.
    if non_interactive_mode:
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, end="" if stderr.endswith("\n") else "\n")

    status = "success" if exit_code == 0 else "error"
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
