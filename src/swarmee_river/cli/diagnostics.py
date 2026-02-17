from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.state_paths import artifacts_dir as _default_artifacts_dir
from swarmee_river.state_paths import logs_dir as _default_logs_dir


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… (truncated to {max_chars} chars) …"


def _resolve_under_cwd(path: str | Path, cwd: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (cwd / p).resolve()


def _run_git(args: list[str], *, cwd: Path, timeout_s: int = 10) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return 1, "", f"git timed out after {timeout_s}s: {e}"
    except Exception as e:
        return 1, "", f"git error: {e}"
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def render_git_status(*, cwd: Path, max_chars: int = 12000) -> str:
    code, out, err = _run_git(["status", "--porcelain=v1", "-b"], cwd=cwd, timeout_s=10)
    text = out.strip() if out.strip() else err.strip()
    if not text:
        text = "(no output)"
    if code != 0:
        text = "git status failed:\n" + text
    return _truncate(text, max_chars)


def render_git_diff(
    *,
    cwd: Path,
    staged: bool = False,
    paths: list[str] | None = None,
    max_chars: int = 12000,
) -> str:
    args = ["diff"]
    if staged:
        args.append("--staged")
    if paths:
        args.extend(["--", *paths])
    code, out, err = _run_git(args, cwd=cwd, timeout_s=20)
    text = out if out else err
    if not (text or "").strip():
        text = "(no diff)"
    if code != 0 and err.strip():
        text = "git diff failed:\n" + err.strip()
    return _truncate(text, max_chars)


_SECRET_KEY_SUBSTRINGS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
_SECRET_ENV_EXACT = {
    "OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
}


def _is_secret_env_key(name: str) -> bool:
    if name in _SECRET_ENV_EXACT:
        return True
    upper = name.upper()
    return any(part in upper for part in _SECRET_KEY_SUBSTRINGS)


def _format_env_value(key: str, value: str) -> str:
    if _is_secret_env_key(key):
        return "<redacted>"
    return value


def _compact_json(value: Any, *, max_chars: int = 160) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        text = str(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def render_effective_config(
    *,
    cwd: Path,
    settings_path: Path,
    settings: Any,
    selected_provider: str | None,
    model_manager: Any,
    knowledge_base_id: str | None,
    effective_sop_paths: str | None,
    auto_approve: bool,
    max_chars: int = 12000,
) -> str:
    tier = getattr(model_manager, "current_tier", None)
    tier = str(tier) if tier else "(unknown)"

    tier_status = None
    try:
        for t in model_manager.list_tiers():
            if getattr(t, "name", None) == tier:
                tier_status = t
                break
    except Exception:
        tier_status = None

    model_bits: list[str] = []
    if tier_status is not None:
        if getattr(tier_status, "display_name", None):
            model_bits.append(str(tier_status.display_name))
        if getattr(tier_status, "model_id", None):
            model_bits.append(f"model_id={tier_status.model_id}")
        if getattr(tier_status, "provider", None):
            model_bits.append(f"provider={tier_status.provider}")

    packs = []
    try:
        packs = list(getattr(getattr(settings, "packs", None), "installed", []) or [])
    except Exception:
        packs = []

    lines: list[str] = [
        "# Config (effective)",
        "",
        f"- cwd: {cwd}",
        f"- settings_path: {settings_path} ({'exists' if settings_path.exists() else 'missing'})",
        f"- provider: {selected_provider or '(unknown)'}",
        f"- tier: {tier}",
        f"- model: {', '.join(model_bits) if model_bits else '(unknown)'}",
        f"- knowledge_base_id: {knowledge_base_id or '(unset)'}",
        f"- auto_approve: {auto_approve}",
        f"- sop_paths: {effective_sop_paths or '(unset)'}",
    ]

    if packs:
        lines.append("- packs:")
        for p in packs:
            try:
                name = getattr(p, "name", None) or ""
                enabled = getattr(p, "enabled", True)
                path = getattr(p, "path", None) or ""
                lines.append(f"  - {name} ({'enabled' if enabled else 'disabled'}) -> {path}")
            except Exception:
                continue

    safety = getattr(settings, "safety", None)
    if safety is not None:
        lines.append("- safety:")
        try:
            lines.append(f"  - tool_consent: {getattr(safety, 'tool_consent', '(unknown)')}")
        except Exception:
            lines.append("  - tool_consent: (unknown)")
        rules: Iterable[Any] = getattr(safety, "tool_rules", []) or []
        if rules:
            lines.append("  - tool_rules:")
            for r in rules:
                try:
                    tool_name = getattr(r, "tool", None) or ""
                    default = getattr(r, "default", None) or ""
                    remember = getattr(r, "remember", None)
                    lines.append(f"    - {tool_name}: {default} (remember={bool(remember)})")
                except Exception:
                    continue
        permission_rules: Iterable[Any] = getattr(safety, "permission_rules", []) or []
        if permission_rules:
            lines.append("  - permission_rules:")
            for r in permission_rules:
                try:
                    tool_name = getattr(r, "tool", None) or ""
                    action = getattr(r, "action", None) or ""
                    remember = getattr(r, "remember", None)
                    when = getattr(r, "when", None)
                    lines.append(
                        f"    - {tool_name}: {action} (remember={bool(remember)}) when={_compact_json(when or {})}"
                    )
                except Exception:
                    continue

    env_keys = [
        "SWARMEE_MODEL_PROVIDER",
        "SWARMEE_MODEL_TIER",
        "SWARMEE_TIER_AUTO",
        "SWARMEE_AUTO_APPROVE",
        "SWARMEE_ENABLE_TOOLS",
        "SWARMEE_DISABLE_TOOLS",
        "BYPASS_TOOL_CONSENT",
        "SWARMEE_PREFLIGHT",
        "SWARMEE_PREFLIGHT_LEVEL",
        "SWARMEE_PREFLIGHT_MAX_CHARS",
        "SWARMEE_PROJECT_MAP",
        "SWARMEE_LOG_EVENTS",
        "SWARMEE_LOG_DIR",
        "SWARMEE_LOG_MAX_FIELD_CHARS",
        "SWARMEE_OPENAI_REASONING_EFFORT",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "STRANDS_MODEL_ID",
        "STRANDS_MAX_TOKENS",
        "STRANDS_BUDGET_TOKENS",
        "STRANDS_THINKING_TYPE",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ]
    lines.append("- env:")
    for k in env_keys:
        v = os.getenv(k)
        if v is None:
            lines.append(f"  - {k}: <unset>")
        else:
            vv = _format_env_value(k, v)
            lines.append(f"  - {k}: {_truncate(vv, 200)}")

    return _truncate("\n".join(lines).strip(), max_chars)


def render_artifact_list(*, cwd: Path, kind: str | None = None, limit: int = 25) -> str:
    store = ArtifactStore(artifacts_dir=_default_artifacts_dir(cwd=cwd))
    entries = store.list(limit=limit, kind=kind)
    if not entries:
        return "No artifacts found."
    lines: list[str] = ["# Artifacts", ""]
    for e in entries:
        lines.append(f"- `{e.get('id')}` ({e.get('kind')}) -> {e.get('path')}")
    return "\n".join(lines)


def render_artifact_get(*, cwd: Path, artifact_id: str | None, path: str | None, max_chars: int = 12000) -> str:
    store = ArtifactStore(artifacts_dir=_default_artifacts_dir(cwd=cwd))
    resolved_path: Path | None = None

    if artifact_id:
        meta = store.get_by_id(artifact_id)
        if meta and meta.get("path"):
            resolved_path = Path(str(meta["path"]))

    if resolved_path is None and path:
        resolved_path = _resolve_under_cwd(path, cwd)

    if resolved_path is None:
        return "artifact_id or path is required."
    if not resolved_path.exists() or not resolved_path.is_file():
        return f"File not found: {resolved_path}"
    try:
        text = store.read_text(resolved_path, max_chars=max_chars)
    except Exception as e:
        return f"Failed to read artifact: {e}"
    return text


def _iter_jsonl_files(log_dir: Path) -> list[Path]:
    if not log_dir.exists() or not log_dir.is_dir():
        return []
    files = [p for p in log_dir.glob("*.jsonl") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def render_log_tail(*, cwd: Path, lines: int = 50) -> str:
    log_dir = _resolve_under_cwd(os.getenv("SWARMEE_LOG_DIR", str(_default_logs_dir(cwd=cwd))), cwd)
    files = _iter_jsonl_files(log_dir)
    if not files:
        return f"No logs found in {log_dir}"

    path = files[0]
    try:
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"Failed to read log: {e}"

    tail = raw_lines[-max(1, int(lines)) :]
    return "\n".join([f"# Log tail: {path}", "", *tail]).strip()


def render_replay_invocation(
    *,
    cwd: Path,
    invocation_id: str,
    max_events: int = 500,
    max_chars: int = 12000,
) -> str:
    inv = (invocation_id or "").strip()
    if not inv:
        return "invocation_id is required."

    log_dir = _resolve_under_cwd(os.getenv("SWARMEE_LOG_DIR", str(_default_logs_dir(cwd=cwd))), cwd)
    files = _iter_jsonl_files(log_dir)
    if not files:
        return f"No logs found in {log_dir}"

    events: list[dict[str, Any]] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if inv not in line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if str(data.get("invocation_id") or "") != inv:
                        continue
                    events.append(data)
                    if len(events) >= max_events:
                        break
        except Exception:
            continue
        if len(events) >= max_events:
            break

    if not events:
        return f"No events found for invocation_id={inv} in {log_dir}"

    lines: list[str] = [f"# Replay: invocation_id={inv}", ""]
    session_id = events[0].get("session_id")
    if session_id:
        lines.append(f"(session_id={session_id})\n")

    for e in events:
        ts = e.get("ts") or ""
        ev = e.get("event") or ""
        if ev == "before_invocation":
            lines.append(f"[{ts}] before_invocation input_items={e.get('input_items')}")
            continue
        if ev == "after_invocation":
            result_preview = _truncate(str(e.get("result") or ""), 200)
            lines.append(f"[{ts}] after_invocation duration_s={e.get('duration_s')} result={result_preview}")
            continue
        if ev == "before_model_call":
            lines.append(
                f"[{ts}] before_model_call model_call_id={e.get('model_call_id')} messages={e.get('messages')}"
            )
            continue
        if ev == "after_model_call":
            usage = e.get("usage")
            resp = e.get("response") or ""
            lines.append(
                f"[{ts}] after_model_call model_call_id={e.get('model_call_id')} duration_s={e.get('duration_s')} "
                f"usage={_truncate(str(usage or ''), 200)} response={_truncate(str(resp), 200)}"
            )
            continue
        if ev == "before_tool_call":
            tool = e.get("tool")
            tool_use_id = e.get("toolUseId")
            inp = e.get("input") or ""
            lines.append(
                f"[{ts}] before_tool_call tool={tool} toolUseId={tool_use_id} input={_truncate(str(inp), 200)}"
            )
            continue
        if ev == "after_tool_call":
            tool = e.get("tool")
            tool_use_id = e.get("toolUseId")
            res = e.get("result") or ""
            lines.append(
                f"[{ts}] after_tool_call tool={tool} toolUseId={tool_use_id} duration_s={e.get('duration_s')} "
                f"result={_truncate(str(res), 200)}"
            )
            continue

        lines.append(f"[{ts}] {ev} " + _truncate(json.dumps(e, ensure_ascii=False), 200))

    return _truncate("\n".join(lines).strip(), max_chars)
