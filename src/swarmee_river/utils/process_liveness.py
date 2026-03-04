from __future__ import annotations

import os
import subprocess


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    except Exception:
        pass

    try:
        output = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return False

    lowered = output.strip().lower()
    if not lowered:
        return False
    if "no tasks are running" in lowered:
        return False
    return f'"{pid}"' in output


__all__ = ["is_process_running"]
