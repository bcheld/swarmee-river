from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from strands import tool

from swarmee_river.artifacts import ArtifactStore


def _extract_target_files(patch_text: str) -> list[str]:
    targets: list[str] = []
    for line in (patch_text or "").splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].strip()
        if raw == "/dev/null":
            continue
        if raw.startswith("b/") or raw.startswith("a/"):
            raw = raw[2:]
        raw = raw.strip()
        if raw and raw not in targets:
            targets.append(raw)
    return targets


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… (truncated to {max_chars} chars) …"


@tool
def patch_apply(
    patch: str,
    *,
    cwd: Optional[str] = None,
    dry_run: bool = False,
    timeout_s: int = 60,
    max_chars: int = 12000,
    max_backup_bytes: int = 200_000,
) -> dict[str, Any]:
    """
    Apply a unified diff patch to the working tree using `git apply`.

    Writes:
    - patch artifact
    - per-file backup artifacts for files touched by the patch (best-effort)
    """
    if not patch or not patch.strip():
        return {"status": "error", "content": [{"text": "patch is required"}]}

    base = Path(cwd or ".").expanduser().resolve()
    store = ArtifactStore()

    patch_ref = store.write_text(kind="patch_apply", text=patch, suffix="diff", metadata={"dry_run": dry_run})
    touched = _extract_target_files(patch)

    backups: list[str] = []
    for rel in touched:
        try:
            p = (base / rel).resolve()
        except Exception:
            continue
        if base not in p.parents and p != base:
            return {"status": "error", "content": [{"text": f"Refusing to touch path outside cwd: {rel}"}]}
        if not p.exists() or not p.is_file():
            continue
        try:
            if p.stat().st_size > max_backup_bytes:
                continue
            backups.append(
                str(
                    store.write_text(
                        kind="patch_backup",
                        text=p.read_text(encoding="utf-8", errors="replace"),
                        suffix=(p.suffix.lstrip(".") or "txt"),
                        metadata={"file": rel},
                    ).path
                )
            )
        except Exception:
            continue

    cmd = ["git", "apply", "--whitespace=nowarn"]
    if dry_run:
        cmd.append("--check")
    cmd.append(str(patch_ref.path))

    t0 = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return {"status": "error", "content": [{"text": f"Timed out after {timeout_s}s"}, {"text": str(e)}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"patch_apply error: {e}"}]}

    duration_s = round(time.time() - t0, 3)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    exit_code = completed.returncode

    if exit_code != 0:
        store.write_text(
            kind="patch_apply_error",
            text=f"exit_code={exit_code}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}".strip(),
            suffix="txt",
            metadata={"patch": str(patch_ref.path)},
        )
        return {
            "status": "error",
            "content": [
                {"text": f"exit_code: {exit_code} (duration_s={duration_s})"},
                {"text": _truncate((stderr or stdout or "").strip(), max_chars)},
                {"text": f"patch: {patch_ref.path}"},
            ],
        }

    summary_lines = [
        f"Applied patch (dry_run={dry_run}) in {duration_s}s",
        f"patch: {patch_ref.path}",
    ]
    if backups:
        summary_lines.append(f"backups: {len(backups)} files")
    if touched:
        summary_lines.append("touched:\n" + "\n".join(f"- {p}" for p in touched))

    return {"status": "success", "content": [{"text": "\n".join(summary_lines)}]}
