from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Iterator
from threading import Lock


@contextlib.contextmanager
def _lock_context(lock: Lock | None) -> Iterator[None]:
    if lock is None:
        yield
        return
    with lock:
        yield


def configure_stdio_for_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")


def write_stdout_jsonl(event: dict[str, object], *, lock: Lock | None = None) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with _lock_context(lock):
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
            return
        except UnicodeEncodeError:
            pass

        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            with contextlib.suppress(Exception):
                buffer.write(line.encode("utf-8", errors="replace"))
                buffer.flush()
                return

        with contextlib.suppress(Exception):
            sys.stdout.write(line.encode("ascii", errors="replace").decode("ascii"))
            sys.stdout.flush()
