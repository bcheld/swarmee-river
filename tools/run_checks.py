from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from strands import tool

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.utils.text_utils import truncate


def _recommended_commands(root: Path) -> list[str]:
    cmds: list[str] = []
    if (root / "pyproject.toml").exists():
        cmds.extend(
            [
                "hatch fmt --formatter --check",
                "hatch fmt --linter --check",
                "hatch test --cover",
            ]
        )
    elif (root / "requirements.txt").exists():
        cmds.append("pytest")
    return cmds


@tool
def run_checks(
    action: str = "run",
    *,
    commands: list[str] | None = None,
    cwd: Optional[str] = None,
    timeout_s: int = 1800,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Run project checks (tests/lint/format) and capture output as an artifact if large.

    Actions:
    - recommend: return recommended commands for the repo
    - run: run commands (or recommended defaults)
    """
    action = (action or "").strip().lower()
    root = Path(cwd or os.getcwd()).expanduser().resolve()

    if action == "recommend":
        cmds = _recommended_commands(root)
        return {"status": "success", "content": [{"text": "\n".join(cmds) if cmds else "(no recommendations)"}]}

    if action != "run":
        return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}

    to_run = list(commands) if commands else _recommended_commands(root)
    if not to_run:
        return {"status": "error", "content": [{"text": "No commands provided and no defaults detected."}]}

    store = ArtifactStore()
    results: list[str] = []
    overall_ok = True

    for cmd in to_run:
        if not cmd or not str(cmd).strip():
            continue
        t0 = time.time()
        try:
            completed = subprocess.run(
                str(cmd),
                shell=True,
                cwd=str(root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            overall_ok = False
            results.append(f"$ {cmd}\nexit_code: timeout\n{e}")
            continue
        except Exception as e:
            overall_ok = False
            results.append(f"$ {cmd}\nexit_code: error\n{e}")
            continue

        duration_s = round(time.time() - t0, 3)
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        combined = ""
        if stdout:
            combined += f"STDOUT:\n{stdout}\n"
        if stderr:
            combined += f"STDERR:\n{stderr}\n"
        combined = combined.strip()

        artifact_path = None
        if max_chars > 0 and len(combined) > max_chars:
            ref = store.write_text(
                kind="run_checks",
                text=combined,
                suffix="txt",
                metadata={"command": cmd, "exit_code": exit_code, "duration_s": duration_s},
            )
            artifact_path = str(ref.path)

        ok = exit_code == 0
        overall_ok = overall_ok and ok
        output = truncate(combined, max_chars) if combined else "(no output)"
        if artifact_path:
            output += f"\n(full output: {artifact_path})"
        results.append(f"$ {cmd}\nexit_code: {exit_code} (duration_s={duration_s})\n{output}")

    return {
        "status": "success" if overall_ok else "error",
        "content": [{"text": "\n\n".join(results)}],
    }
