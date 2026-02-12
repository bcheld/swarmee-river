from __future__ import annotations

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from strands import tool

_REPL_GLOBALS: dict[str, Any] = {"__name__": "__swarmee_python_repl__"}


@tool
def python_repl(code: str) -> dict[str, Any]:
    """
    Cross-platform Python REPL tool.

    Executes Python code and returns captured stdout/stderr. State is persisted for the lifetime
    of the process via a module-level globals dict.
    """
    if not code or not code.strip():
        return {"status": "error", "content": [{"text": "code is required."}]}

    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            exec(code, _REPL_GLOBALS, _REPL_GLOBALS)  # noqa: S102
    except Exception:
        err.write(traceback.format_exc())

    stdout = out.getvalue()
    stderr = err.getvalue()

    status = "success" if not stderr else "error"
    combined = ""
    if stdout:
        combined += f"STDOUT:\n{stdout}\n"
    if stderr:
        combined += f"STDERR:\n{stderr}\n"

    return {"status": status, "content": [{"text": combined.strip() or "(no output)"}]}

