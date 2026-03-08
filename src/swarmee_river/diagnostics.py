from __future__ import annotations

import json
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any

from swarmee_river.settings import load_settings
from swarmee_river.state_paths import logs_dir, state_dir

_DIAG_LEVELS = {"baseline", "verbose"}
_DEFAULT_DIAG_LEVEL = "baseline"
_DEFAULT_RETENTION_DAYS = 7
_DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_MIN_RETENTION_DAYS = 1
_MIN_MAX_BYTES = 1024 * 1024
_MAX_TAIL_LINES = 2000

_REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(Bearer)\s+([A-Za-z0-9._\-]{20,})"), r"\1 <redacted>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<redacted-aws-access-key-id>"),
    (re.compile(r"ASIA[0-9A-Z]{16}"), "<redacted-aws-access-key-id>"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "<redacted-openai-key>"),
]


def diagnostics_level() -> str:
    level = str(load_settings().diagnostics.level or _DEFAULT_DIAG_LEVEL).strip().lower()
    return level if level in _DIAG_LEVELS else _DEFAULT_DIAG_LEVEL


def diagnostics_events_enabled() -> bool:
    return bool(load_settings().diagnostics.log_events)


def diagnostics_redact_enabled() -> bool:
    return bool(load_settings().diagnostics.redact)


def diagnostics_retention_days() -> int:
    return _DEFAULT_RETENTION_DAYS


def diagnostics_max_bytes() -> int:
    return _DEFAULT_MAX_BYTES


def diagnostics_dir(*, cwd: Path | None = None) -> Path:
    return state_dir(cwd=cwd) / "diagnostics"


def broker_log_path(*, cwd: Path | None = None) -> Path:
    return diagnostics_dir(cwd=cwd) / "broker.log"


def session_events_dir(*, cwd: Path | None = None) -> Path:
    return diagnostics_dir(cwd=cwd) / "sessions"


def session_issues_dir(*, cwd: Path | None = None) -> Path:
    return diagnostics_dir(cwd=cwd) / "issues"


def _session_token(session_id: str | None) -> str:
    token = str(session_id or "").strip()
    if not token:
        return "unknown"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", token).strip("-")
    return safe or "unknown"


def session_events_path(session_id: str | None, *, cwd: Path | None = None) -> Path:
    return session_events_dir(cwd=cwd) / f"{_session_token(session_id)}.jsonl"


def session_issues_path(session_id: str | None, *, cwd: Path | None = None) -> Path:
    return session_issues_dir(cwd=cwd) / f"{_session_token(session_id)}.log"


def session_events_template(*, cwd: Path | None = None) -> str:
    return str(session_events_dir(cwd=cwd) / "{session_id}.jsonl")


def _redact_text(text: str) -> str:
    output = text
    for key in (
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "SWARMEE_GITHUB_COPILOT_API_KEY",
    ):
        value = str(os.getenv(key, "")).strip()
        if value:
            output = output.replace(value, "<redacted>")
    for pattern, replacement in _REDACTION_PATTERNS:
        output = pattern.sub(replacement, output)
    return output


def redact_payload(payload: Any) -> Any:
    if not diagnostics_redact_enabled():
        return payload
    if isinstance(payload, str):
        return _redact_text(payload)
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            normalized_key = str(key)
            if any(marker in normalized_key.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                out[normalized_key] = "<redacted>"
            else:
                out[normalized_key] = redact_payload(value)
        return out
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(redact_payload(item) for item in payload)
    return payload


def _append_text_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip("\n"))
        handle.write("\n")


def append_session_issue(*, session_id: str | None, line: str, cwd: Path | None = None) -> Path:
    path = session_issues_path(session_id, cwd=cwd)
    _append_text_line(path, line)
    return path


def append_session_event(
    *,
    session_id: str | None,
    event: dict[str, Any],
    cwd: Path | None = None,
) -> Path:
    path = session_events_path(session_id, cwd=cwd)
    payload = dict(event)
    payload.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
    safe_payload = redact_payload(payload) if diagnostics_redact_enabled() else payload
    _append_text_line(path, json.dumps(safe_payload, ensure_ascii=False))
    return path


def _iter_diag_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    files = [path for path in root.rglob("*") if path.is_file()]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return files


def sweep_retention(*, cwd: Path | None = None) -> None:
    root = diagnostics_dir(cwd=cwd)
    files = _iter_diag_files(root)
    if not files:
        return

    max_age_s = diagnostics_retention_days() * 24 * 60 * 60
    now = time.time()
    for file_path in list(files):
        with_age = now - file_path.stat().st_mtime
        if with_age <= max_age_s:
            continue
        try:
            file_path.unlink()
        except OSError:
            pass

    files = _iter_diag_files(root)
    total_bytes = sum(path.stat().st_size for path in files)
    max_bytes = diagnostics_max_bytes()
    for file_path in reversed(files):
        if total_bytes <= max_bytes:
            break
        try:
            size = file_path.stat().st_size
            file_path.unlink()
            total_bytes -= size
        except OSError:
            continue


def _tail_path(path: Path, *, lines: int) -> str:
    if not path.exists():
        return f"File not found: {path}"
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Failed to read {path}: {exc}"
    tail = raw[-max(1, min(_MAX_TAIL_LINES, int(lines))):]
    return "\n".join(tail)


def render_diagnostics_tail(*, cwd: Path, lines: int = 100) -> str:
    candidates: list[Path] = []
    candidates.extend(_iter_diag_files(session_events_dir(cwd=cwd))[:1])
    broker = broker_log_path(cwd=cwd)
    if broker.exists():
        candidates.append(broker)
    candidates.extend(sorted(logs_dir(cwd=cwd).glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:1])
    if not candidates:
        return f"No diagnostics files found under {diagnostics_dir(cwd=cwd)}"
    chosen = candidates[0]
    body = _tail_path(chosen, lines=lines)
    return f"# Diagnostics tail: {chosen}\n\n{body}".strip()


def render_diagnostics_doctor(*, cwd: Path) -> str:
    from swarmee_river.runtime_service.client import runtime_discovery_path
    from swarmee_river.session.models import SessionModelManager
    from swarmee_river.utils.model_utils import probe_openai_responses_transport
    from swarmee_river.utils.provider_utils import resolve_aws_auth_source, resolve_model_provider

    discovery = runtime_discovery_path(cwd=cwd)
    discovery_exists = discovery.exists()
    broker_log = broker_log_path(cwd=cwd)
    events = _iter_diag_files(session_events_dir(cwd=cwd))
    issues = _iter_diag_files(session_issues_dir(cwd=cwd))
    has_aws, aws_source = resolve_aws_auth_source()
    settings_path = cwd / ".swarmee" / "settings.json"
    settings = load_settings(settings_path)
    selected_provider, provider_notice = resolve_model_provider(
        cli_provider=None,
        env_provider=None,
        settings_provider=settings.models.provider,
    )
    model_manager = SessionModelManager(settings, fallback_provider=selected_provider)
    current_tier = str(model_manager.current_tier or "").strip().lower() or "balanced"
    selected_tier = next((item for item in model_manager.list_tiers() if item.name == current_tier), None)
    transport_status = probe_openai_responses_transport()

    lines = [
        "# Diagnostics doctor",
        "",
        f"- scope: {cwd}",
        f"- diagnostics_dir: {diagnostics_dir(cwd=cwd)}",
        f"- runtime_discovery: {'present' if discovery_exists else 'missing'} ({discovery})",
        f"- broker_log: {'present' if broker_log.exists() else 'missing'} ({broker_log})",
        f"- session_events_files: {len(events)}",
        f"- session_issue_files: {len(issues)}",
        f"- aws_auth_source: {aws_source}{'' if has_aws else ' (unavailable)'}",
        f"- selected_provider: {selected_provider or '(unknown)'}",
        f"- selected_tier: {current_tier}",
        f"- strands_agents_version: {transport_status.strands_version or 'missing'}",
        f"- openai_sdk_version: {transport_status.openai_version or 'missing'}",
        (
            "- openai_responses_transport: available"
            if transport_status.available
            else f"- openai_responses_transport: unavailable ({transport_status.reason})"
        ),
    ]
    if provider_notice:
        lines.append(f"- provider_notice: {provider_notice}")
    if selected_tier is not None:
        lines.append(
            f"- selected_tier_available: {'yes' if selected_tier.available else 'no'}"
            + (f" ({selected_tier.reason})" if selected_tier.reason else "")
        )
    if events:
        lines.append(f"- latest_session_events: {events[0]}")
    if issues:
        lines.append(f"- latest_session_issues: {issues[0]}")
    return "\n".join(lines)


def _bundle_name() -> str:
    return f"swarmee-diagnostics-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}.zip"


def _collect_bundle_inputs(*, cwd: Path) -> list[Path]:
    inputs: list[Path] = []
    root = state_dir(cwd=cwd)
    for path in (
        root / "runtime.json",
        broker_log_path(cwd=cwd),
        root / "settings.json",
        cwd / ".swarmee" / "settings.json",
    ):
        if path.exists() and path.is_file():
            inputs.append(path)
    inputs.extend(sorted(logs_dir(cwd=cwd).glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:5])
    inputs.extend(_iter_diag_files(session_events_dir(cwd=cwd))[:10])
    inputs.extend(_iter_diag_files(session_issues_dir(cwd=cwd))[:10])
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def create_support_bundle(*, cwd: Path) -> Path:
    sweep_retention(cwd=cwd)
    bundle_root = diagnostics_dir(cwd=cwd) / "bundles"
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_root / _bundle_name()
    inputs = _collect_bundle_inputs(cwd=cwd)
    redact = diagnostics_redact_enabled()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest: dict[str, Any] = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "cwd": str(cwd),
            "inputs": [str(path) for path in inputs],
            "redacted": redact,
        }
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        for path in inputs:
            arcname = path.name
            if path.suffix.lower() in {".json", ".jsonl", ".log", ".txt", ".md"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if redact:
                    text = _redact_text(text)
                archive.writestr(arcname, text)
            else:
                try:
                    archive.write(path, arcname=arcname)
                except OSError:
                    continue
    return bundle_path


__all__ = [
    "append_session_event",
    "append_session_issue",
    "broker_log_path",
    "create_support_bundle",
    "diagnostics_dir",
    "diagnostics_events_enabled",
    "diagnostics_level",
    "diagnostics_max_bytes",
    "diagnostics_redact_enabled",
    "diagnostics_retention_days",
    "redact_payload",
    "render_diagnostics_doctor",
    "render_diagnostics_tail",
    "session_events_path",
    "session_events_template",
    "session_issues_path",
    "sweep_retention",
]
