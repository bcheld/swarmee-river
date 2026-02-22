from __future__ import annotations

import contextlib
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

from strands import tool

from swarmee_river.utils.text_utils import truncate


def _run(cmd: list[str], *, cwd: Path, timeout_s: int) -> tuple[int, str, str]:
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
        return 1, "", str(e)
    return p.returncode, p.stdout or "", p.stderr or ""


def _run_streaming(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_s: int,
    heartbeat_s: int = 15,
) -> tuple[int, str, str]:
    started = time.monotonic()
    output_lines: list[str] = []

    try:
        popen_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            cmd,
            **popen_kwargs,
        )
    except Exception as e:
        return 1, "", str(e)

    line_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    last_heartbeat = started
    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed > timeout_s:
                _terminate_process_tree(proc)
                return 1, "".join(output_lines), f"Command timed out after {timeout_s}s."

            if _interrupt_requested():
                _terminate_process_tree(proc)
                return 1, "".join(output_lines), "Command interrupted."

            try:
                item = line_queue.get(timeout=1.0)
            except queue.Empty:
                # Stop immediately if the command process has exited, even if a descendant still holds stdout open.
                if proc.poll() is not None:
                    break
                if now - last_heartbeat >= heartbeat_s:
                    print(f"[git] still running ({int(elapsed)}s elapsed)...")
                    last_heartbeat = now
                continue

            if item is None:
                if proc.poll() is not None:
                    break
                continue

            line = item if item.endswith("\n") else f"{item}\n"
            print(line, end="")
            output_lines.append(line)
            last_heartbeat = time.monotonic()
    except KeyboardInterrupt:
        _terminate_process_tree(proc)
        raise
    finally:
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            reader.join(timeout=0.2)

    return int(proc.returncode or 0), "".join(output_lines), ""


def _interrupt_requested() -> bool:
    try:
        from swarmee_river.handlers.callback_handler import callback_handler_instance

        event = callback_handler_instance.interrupt_event
        return bool(event is not None and event.is_set())
    except Exception:
        return False


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
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


@tool
def git(
    action: str = "status",
    *,
    cwd: Optional[str] = None,
    paths: list[str] | None = None,
    ref: str | None = None,
    message: str | None = None,
    stash_action: str | None = None,
    max_lines: int = 50,
    timeout_s: int = 60,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Git tool for common repo workflows.

    Actions:
    - status
    - diff
    - diff_staged
    - log
    - branch
    - checkout
    - add
    - commit
    - restore
    - stash (sub-actions: list|push|pop)
    """
    action = (action or "").strip().lower()
    base = Path(cwd or ".").expanduser().resolve()
    path_args = ["--", *(paths or [])] if paths else []

    if action == "status":
        code, out, err = _run(["git", "status", "--porcelain=v1", "-b"], cwd=base, timeout_s=timeout_s)
        text = out.strip() if out.strip() else err.strip()
        return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}

    if action == "diff":
        code, out, err = _run(["git", "diff", *path_args], cwd=base, timeout_s=timeout_s)
        text = out if out else err
        return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}

    if action == "diff_staged":
        code, out, err = _run(["git", "diff", "--staged", *path_args], cwd=base, timeout_s=timeout_s)
        text = out if out else err
        return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}

    if action == "log":
        n = max(1, int(max_lines))
        code, out, err = _run(["git", "log", "--oneline", "-n", str(n)], cwd=base, timeout_s=timeout_s)
        text = out if out else err
        return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}

    if action == "branch":
        code1, out1, err1 = _run(["git", "branch", "--show-current"], cwd=base, timeout_s=timeout_s)
        code2, out2, err2 = _run(["git", "branch", "--list"], cwd=base, timeout_s=timeout_s)
        text = ""
        if out1.strip():
            text += f"current: {out1.strip()}\n"
        text += out2 if out2 else (err2 or err1)
        code = 0 if (code1 == 0 and code2 == 0) else 1
        return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}

    if action == "checkout":
        if not ref or not ref.strip():
            return {"status": "error", "content": [{"text": "ref is required for action=checkout"}]}
        code, out, err = _run(["git", "checkout", ref.strip()], cwd=base, timeout_s=timeout_s)
        text = out.strip() if out.strip() else err.strip()
        return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}

    if action == "add":
        if not paths:
            return {"status": "error", "content": [{"text": "paths is required for action=add"}]}
        code, out, err = _run(["git", "add", "--", *paths], cwd=base, timeout_s=timeout_s)
        text = out.strip() if out.strip() else err.strip()
        return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}

    if action == "commit":
        if not message or not message.strip():
            return {"status": "error", "content": [{"text": "message is required for action=commit"}]}
        commit_timeout_s = max(int(timeout_s), 1800)
        print(f"[git] running commit (timeout={commit_timeout_s}s). pre-commit hooks may take a while...")
        t0 = time.monotonic()
        code, out, err = _run_streaming(["git", "commit", "-m", message.strip()], cwd=base, timeout_s=commit_timeout_s)
        duration_s = round(time.monotonic() - t0, 2)
        print(f"[git] commit finished in {duration_s}s (exit_code={code})")
        text = out.strip() if out.strip() else err.strip()
        return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}

    if action == "restore":
        if not paths:
            return {"status": "error", "content": [{"text": "paths is required for action=restore"}]}
        code, out, err = _run(["git", "restore", "--", *paths], cwd=base, timeout_s=timeout_s)
        text = out.strip() if out.strip() else err.strip()
        return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}

    if action == "stash":
        sub = (stash_action or "list").strip().lower()
        if sub == "list":
            code, out, err = _run(["git", "stash", "list"], cwd=base, timeout_s=timeout_s)
            text = out.strip() if out.strip() else err.strip()
            return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}
        if sub == "push":
            args = ["git", "stash", "push"]
            if message and message.strip():
                args.extend(["-m", message.strip()])
            code, out, err = _run(args, cwd=base, timeout_s=timeout_s)
            text = out.strip() if out.strip() else err.strip()
            return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}
        if sub == "pop":
            code, out, err = _run(["git", "stash", "pop"], cwd=base, timeout_s=timeout_s)
            text = out.strip() if out.strip() else err.strip()
            return {"status": "success" if code == 0 else "error", "content": [{"text": truncate(text, max_chars)}]}
        return {"status": "error", "content": [{"text": "Unknown stash_action. Use list|push|pop."}]}

    return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}
