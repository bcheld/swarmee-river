from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ACTIVE_INTERRUPT_EVENT: threading.Event | None = None


@dataclass(frozen=True)
class SubprocessCaptureResult:
    returncode: int
    stdout: str
    stderr: str
    interrupted: bool = False
    timed_out: bool = False


def current_interrupt_event() -> threading.Event | None:
    return _ACTIVE_INTERRUPT_EVENT


def set_active_interrupt_event(event: threading.Event | None) -> None:
    global _ACTIVE_INTERRUPT_EVENT
    _ACTIVE_INTERRUPT_EVENT = event


def interrupt_requested() -> bool:
    event = current_interrupt_event()
    return bool(event is not None and event.is_set())


def sleep_with_interrupt(seconds: float, *, poll_interval_s: float = 0.1) -> bool:
    remaining = max(0.0, float(seconds))
    interval = max(0.01, float(poll_interval_s))
    event = current_interrupt_event()
    if event is None:
        time.sleep(remaining)
        return False
    while remaining > 0:
        if event.wait(min(interval, remaining)):
            return True
        remaining -= interval
    return bool(event.is_set())


def terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
    else:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        with contextlib.suppress(Exception):
            proc.wait(timeout=2)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    with contextlib.suppress(Exception):
        if proc.poll() is None:
            proc.kill()


def run_subprocess_capture_interruptible(
    command: str | list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout_s: int | float = 600,
    shell: bool = False,
    stdin: Any = None,
) -> SubprocessCaptureResult:
    event = current_interrupt_event()
    if event is None:
        completed = subprocess.run(
            command,
            shell=shell,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(0.0, float(timeout_s)),
            check=False,
        )
        return SubprocessCaptureResult(
            returncode=int(completed.returncode or 0),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd) if cwd is not None else None,
        "env": env,
        "stdin": stdin,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(command, shell=shell, **popen_kwargs)
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    stdout = ""
    stderr = ""

    while True:
        try:
            stdout, stderr = proc.communicate(timeout=0.1)
            return SubprocessCaptureResult(
                returncode=int(proc.returncode or 0),
                stdout=stdout or "",
                stderr=stderr or "",
            )
        except subprocess.TimeoutExpired:
            if event is not None and event.is_set():
                terminate_process_tree(proc)
                with contextlib.suppress(Exception):
                    stdout, stderr = proc.communicate(timeout=1.0)
                return SubprocessCaptureResult(
                    returncode=1,
                    stdout=stdout or "",
                    stderr=(stderr or "").strip() or "Command interrupted.",
                    interrupted=True,
                )
            if time.monotonic() >= deadline:
                terminate_process_tree(proc)
                with contextlib.suppress(Exception):
                    stdout, stderr = proc.communicate(timeout=1.0)
                return SubprocessCaptureResult(
                    returncode=1,
                    stdout=stdout or "",
                    stderr=(stderr or "").strip() or f"Command timed out after {timeout_s}s.",
                    timed_out=True,
                )
