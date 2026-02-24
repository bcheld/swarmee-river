"""Optional Textual app scaffold for `swarmee tui`."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json as _json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from swarmee_river.artifacts import ArtifactStore
from swarmee_river.error_classification import (
    ERROR_CATEGORY_AUTH_ERROR,
    ERROR_CATEGORY_ESCALATABLE,
    ERROR_CATEGORY_FATAL,
    ERROR_CATEGORY_TOOL_ERROR,
    ERROR_CATEGORY_TRANSIENT,
    classify_error_message,
    normalize_error_category,
)
from swarmee_river.profiles import AgentProfile, delete_profile, list_profiles, save_profile
from swarmee_river.runtime_service.client import RuntimeServiceClient, runtime_discovery_path
from swarmee_river.session.graph_index import (
    build_session_graph_index,
    load_session_graph_index,
    write_session_graph_index,
)
from swarmee_river.settings import load_settings
from swarmee_river.state_paths import logs_dir, sessions_dir

_CONSENT_CHOICES = {"y", "n", "a", "v"}
_MODEL_AUTO_VALUE = "__auto__"
_MODEL_LOADING_VALUE = "__loading__"
_TRUNCATED_ARTIFACT_RE = re.compile(r"full output saved to (?P<path>[^\]]+)")
_PATH_TOKEN_RE = re.compile(r"[A-Za-z]:\\[^\s,;]+|/(?:[^\s,;]+)|\./[^\s,;]+|\.\./[^\s,;]+")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC_ESCAPE_RE = re.compile(r"\x1b\][^\x1b\x07]*(?:\x07|\x1b\\)")
_PROVIDER_NOISE_PREFIX = "[provider]"
_PROVIDER_FALLBACK_PHRASE = "falling back to"
_MODEL_USAGE_TEXT = "Usage: /model show | /model list | /model provider <name> | /model tier <name> | /model reset"
_CONSENT_USAGE_TEXT = "Usage: /consent <y|n|a|v>"
_CONNECT_USAGE_TEXT = "Usage: /connect [github_copilot]"
_AUTH_USAGE_TEXT = "Usage: /auth list | /auth logout [provider]"
_SEARCH_USAGE_TEXT = "Usage: /search <term>"
_OPEN_USAGE_TEXT = "Usage: /open <number>"
_EXPAND_USAGE_TEXT = "Usage: /expand <tool_use_id>"
_COMPACT_USAGE_TEXT = "Usage: /compact"
_TEXT_USAGE_TEXT = "Usage: /text"
_THINKING_USAGE_TEXT = "Usage: /thinking"
_CONTEXT_USAGE_TEXT = (
    "Usage: /context add file <path> | /context add note <text> | /context add sop <name> | "
    "/context add kb <id> | /context remove <index> | /context list | /context clear"
)
_SOP_USAGE_TEXT = "Usage: /sop list | /sop activate <name> | /sop deactivate <name> | /sop preview <name>"
_RUN_ACTIVE_TIER_WARNING = "[model] cannot change tier while a run is active."
_AGENT_PROFILE_SELECT_NONE = "__agent_profile_none__"
_AGENT_TOOL_CONSENT_VALUES = {"ask", "allow", "deny"}
_CONTEXT_SOURCE_ICONS: dict[str, str] = {
    "file": "📄",
    "url": "🌐",
    "kb": "📚",
    "sop": "📋",
    "note": "📝",
}
_CONTEXT_SOURCE_TYPES = {"file", "note", "sop", "kb", "url"}
_CONTEXT_SOURCE_MAX_LABEL = 72
_SOP_FILE_SUFFIX = ".sop.md"
_CONTEXT_SELECT_PLACEHOLDER = "__context_select_none__"
_CONTEXT_INPUT_SOURCE_TYPES = {"file", "note", "kb"}
_CONTEXT_SOP_SOURCE_TYPE = "sop"
_STREAMING_FLUSH_INTERVAL_S = 0.15
_TOOL_PROGRESS_RENDER_INTERVAL_S = 0.15
_TOOL_HEARTBEAT_RENDER_MIN_STEP_S = 0.5
_TOOL_OUTPUT_RETENTION_MAX_CHARS = 4096
_TOOL_START_COALESCE_INTERVAL_S = 0.1
_TOOL_FAST_COMPLETE_SUPPRESS_START_S = 0.5
_THINKING_DISPLAY_DEBOUNCE_S = 0.2
_THINKING_ANIMATION_INTERVAL_S = 0.5
_THINKING_EXPORT_MAX_CHARS = 5000
_TRANSIENT_TOAST_TIMEOUT_S = 5.0
_FATAL_TOAST_TIMEOUT_S = 3600.0
_SOP_SOURCE_LOCAL = "local"
_SOP_SOURCE_STRANDS = "strands-sops"
_SOP_SOURCE_PRIORITY: dict[str, int] = {
    _SOP_SOURCE_LOCAL: 0,
    "pack": 1,
    _SOP_SOURCE_STRANDS: 2,
}
_COPY_COMMAND_MAP: dict[str, str] = {
    "/copy": "transcript",
    ":copy": "transcript",
    "/copy plan": "plan",
    ":copy plan": "plan",
    "/copy issues": "issues",
    ":copy issues": "issues",
    "/copy artifacts": "artifacts",
    ":copy artifacts": "artifacts",
    "/copy last": "last",
    ":copy last": "last",
    "/copy all": "all",
    ":copy all": "all",
}


def _sanitize_profile_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


def _default_profile_id(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"profile-{ts.strftime('%Y%m%d-%H%M%S')}"


def _default_profile_name(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"Profile {ts.strftime('%Y-%m-%d %H:%M')}"


def _default_team_preset_id(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"team-{ts.strftime('%Y%m%d-%H%M%S')}"


def _default_team_preset_name(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"Team Preset {ts.strftime('%Y-%m-%d %H:%M')}"


def _normalize_team_preset_spec(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    try:
        encoded = _json.dumps(raw, ensure_ascii=False, sort_keys=True)
        decoded = _json.loads(encoded)
    except Exception:
        return None
    return decoded if isinstance(decoded, dict) else None


def normalize_team_preset(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    raw_name = str(raw.get("name", "")).strip()
    if not raw_name:
        return None

    raw_id = str(raw.get("id", "")).strip()
    preset_id = _sanitize_profile_token(raw_id or raw_name)
    if not preset_id:
        return None

    spec = _normalize_team_preset_spec(raw.get("spec"))
    if spec is None:
        return None

    return {
        "id": preset_id,
        "name": raw_name,
        "description": str(raw.get("description", "")).strip(),
        "spec": spec,
    }


def normalize_team_presets(raw_presets: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_presets, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_preset in raw_presets:
        preset = normalize_team_preset(raw_preset)
        if preset is None:
            continue
        preset_id = str(preset.get("id", "")).strip()
        if not preset_id or preset_id in seen_ids:
            continue
        seen_ids.add(preset_id)
        normalized.append(preset)
    return normalized


def build_team_preset_run_prompt(preset: dict[str, Any]) -> str:
    normalized = normalize_team_preset(preset)
    if normalized is None:
        return ""

    spec_json = _json.dumps(normalized["spec"], ensure_ascii=False, indent=2, sort_keys=True)
    return (
        f"Run team preset '{normalized['name']}' (id: {normalized['id']}).\n"
        "Call the `swarm` tool exactly once with the JSON `spec` object below.\n"
        "After the tool returns, summarize results and next actions.\n\n"
        "spec:\n"
        "```json\n"
        f"{spec_json}\n"
        "```"
    )


@dataclass(frozen=True)
class ParsedEvent:
    kind: str
    text: str
    meta: dict[str, str] | None = None


class _DaemonTransport:
    @property
    def pid(self) -> int:
        raise NotImplementedError

    def poll(self) -> int | None:
        raise NotImplementedError

    def wait(self, timeout: float | None = None) -> int:
        raise NotImplementedError

    def read_line(self) -> str:
        raise NotImplementedError

    def send_command(self, cmd_dict: dict[str, Any]) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _SubprocessTransport(_DaemonTransport):
    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def poll(self) -> int | None:
        return self._proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        return int(self._proc.wait(timeout=timeout))

    def read_line(self) -> str:
        stdout = self._proc.stdout
        if stdout is None:
            return ""
        return stdout.readline()

    def send_command(self, cmd_dict: dict[str, Any]) -> bool:
        stdin = self._proc.stdin
        if stdin is None:
            return False
        try:
            payload = _json.dumps(cmd_dict, ensure_ascii=False) + "\n"
            stdin.write(payload)
            stdin.flush()
        except Exception:
            return False
        return True

    def close(self) -> None:
        stop_process(self._proc)


class _SocketTransport(_DaemonTransport):
    def __init__(
        self,
        *,
        client: RuntimeServiceClient,
        session_id: str,
        broker_pid: int | None = None,
        pending_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._broker_pid = int(broker_pid) if isinstance(broker_pid, int) else None
        self._pending_events = list(pending_events or [])
        self._closed = False
        self._poll_code: int | None = None

    @classmethod
    def connect(
        cls,
        *,
        session_id: str,
        cwd: Path,
        client_name: str,
        surface: str,
    ) -> _SocketTransport:
        discovery = runtime_discovery_path(cwd=cwd)
        if not discovery.exists():
            raise FileNotFoundError(f"Runtime discovery file not found: {discovery}")

        client = RuntimeServiceClient.from_discovery_file(discovery)
        client.connect()
        hello = client.hello(client_name=client_name, surface=surface) or {}
        if str(hello.get("event", "")).strip().lower() == "error":
            message = str(hello.get("message", hello)).strip() or "hello failed"
            client.close()
            raise RuntimeError(message)

        attach = client.attach(session_id=session_id, cwd=str(cwd)) or {}
        if str(attach.get("event", "")).strip().lower() == "error":
            message = str(attach.get("message", attach)).strip() or "attach failed"
            client.close()
            raise RuntimeError(message)

        broker_pid = hello.get("pid")
        return cls(
            client=client,
            session_id=session_id,
            broker_pid=(int(broker_pid) if isinstance(broker_pid, int) else None),
            pending_events=[attach] if isinstance(attach, dict) else [],
        )

    @property
    def pid(self) -> int:
        return int(self._broker_pid) if self._broker_pid is not None else -1

    def poll(self) -> int | None:
        return self._poll_code if self._closed else None

    def wait(self, timeout: float | None = None) -> int:
        start = time.monotonic()
        while not self._closed:
            if timeout is not None and (time.monotonic() - start) >= timeout:
                raise subprocess.TimeoutExpired(["runtime-socket"], timeout)
            time.sleep(0.01)
        return int(self._poll_code or 0)

    def read_line(self) -> str:
        if self._pending_events:
            event = self._pending_events.pop(0)
            return _json.dumps(event, ensure_ascii=False) + "\n"
        if self._closed:
            return ""
        event = self._client.read_event()
        if event is None:
            self._closed = True
            self._poll_code = 0
            return ""
        return _json.dumps(event, ensure_ascii=False) + "\n"

    def send_command(self, cmd_dict: dict[str, Any]) -> bool:
        if self._closed:
            return False
        payload = dict(cmd_dict)
        cmd = str(payload.get("cmd", "")).strip().lower()
        if cmd == "shutdown":
            payload = {"cmd": "shutdown_session"}
        try:
            self._client.send_command(payload)
        except Exception:
            self._closed = True
            self._poll_code = 1
            return False
        if str(payload.get("cmd", "")).strip().lower() == "shutdown_session":
            self.close()
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._poll_code = 0
        self._client.close()


def _sanitize_context_source_id(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


def _normalize_context_source(source: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(source, dict):
        return None
    source_type = str(source.get("type", "")).strip().lower()
    if source_type not in _CONTEXT_SOURCE_TYPES:
        return None

    normalized: dict[str, str] = {"type": source_type}
    source_id = str(source.get("id", "")).strip()
    if source_id:
        normalized["id"] = _sanitize_context_source_id(source_id)
    else:
        source_seed = (
            source.get("path", "")
            or source.get("text", "")
            or source.get("name", "")
            or source.get("kb_id", "")
        )
        seed = (
            str(source_seed).strip()
            or uuid.uuid4().hex
        )
        normalized["id"] = _sanitize_context_source_id(f"{source_type}-{seed}")

    if source_type == "file":
        path = str(source.get("path", "")).strip()
        if not path:
            return None
        normalized["path"] = path
        return normalized
    if source_type == "note":
        text = str(source.get("text", "")).strip()
        if not text:
            return None
        normalized["text"] = text
        return normalized
    if source_type == "sop":
        name = str(source.get("name", "")).strip()
        if not name:
            return None
        normalized["name"] = name
        return normalized
    if source_type == "kb":
        kb_id = str(source.get("id", source.get("kb_id", ""))).strip()
        if not kb_id:
            return None
        normalized["id"] = kb_id
        return normalized
    if source_type == "url":
        url = str(source.get("url", source.get("path", ""))).strip()
        if not url:
            return None
        normalized["url"] = url
        return normalized
    return None


def _normalize_context_sources(raw_sources: Any) -> list[dict[str, str]]:
    if not isinstance(raw_sources, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_sources:
        source = _normalize_context_source(item if isinstance(item, dict) else {})
        if source is None:
            continue
        source_type = source.get("type", "")
        value_key = (
            source.get("path")
            or source.get("text")
            or source.get("name")
            or source.get("id")
            or source.get("url")
            or ""
        ).strip()
        dedupe_key = (source_type, value_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(source)
    return normalized


def _strip_sop_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    text = (markdown or "").lstrip("\ufeff")
    if not text.startswith("---\n"):
        return {}, text.strip()
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text.strip()
    header = text[4:end].strip()
    body = text[end + len("\n---\n") :].strip()
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        k = key.strip().lower()
        v = value.strip()
        if k and v:
            meta[k] = v
    return meta, body


def _first_sop_paragraph(markdown: str) -> str:
    _meta, body = _strip_sop_frontmatter(markdown)
    lines = [line.rstrip() for line in body.splitlines()]
    paragraph_lines: list[str] = []
    started = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if started:
                break
            continue
        if line.startswith("#") and not started:
            continue
        paragraph_lines.append(line)
        started = True
    preview = " ".join(paragraph_lines).strip()
    if not preview:
        return "(no preview available)"
    if len(preview) > 220:
        return preview[:219].rstrip() + "…"
    return preview


def _load_sop_file(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, body = _strip_sop_frontmatter(raw)
    file_name = path.name
    derived = file_name[: -len(_SOP_FILE_SUFFIX)] if file_name.endswith(_SOP_FILE_SUFFIX) else path.stem
    name = str(meta.get("name", derived)).strip() or derived
    return name, body.strip()


def discover_available_sops() -> list[dict[str, str]]:
    records: dict[str, dict[str, str]] = {}

    def _record_source_priority(source: str) -> int:
        if source == _SOP_SOURCE_LOCAL:
            return _SOP_SOURCE_PRIORITY[_SOP_SOURCE_LOCAL]
        if source.startswith("pack:"):
            return _SOP_SOURCE_PRIORITY["pack"]
        return _SOP_SOURCE_PRIORITY[_SOP_SOURCE_STRANDS]

    def _add_record(*, name: str, path: str, source: str, content: str) -> None:
        sop_name = name.strip()
        if not sop_name:
            return
        existing = records.get(sop_name)
        priority = _record_source_priority(source)
        if existing is not None:
            existing_priority = _record_source_priority(existing.get("source", ""))
            if existing_priority <= priority:
                return
        body = content.strip()
        preview = _first_sop_paragraph(body)
        records[sop_name] = {
            "name": sop_name,
            "path": path.strip(),
            "source": source.strip(),
            "first_paragraph_preview": preview,
            "content": body,
        }

    local_dirs: list[Path] = []
    for candidate in (Path.cwd() / "sops", Path(__file__).resolve().parents[1] / "sops"):
        if candidate.exists() and candidate.is_dir():
            local_dirs.append(candidate.resolve())
    seen_local: set[str] = set()
    for directory in local_dirs:
        key = str(directory)
        if key in seen_local:
            continue
        seen_local.add(key)
        for file_path in sorted(directory.glob(f"*{_SOP_FILE_SUFFIX}")):
            if not file_path.is_file():
                continue
            with contextlib.suppress(Exception):
                name, content = _load_sop_file(file_path)
                _add_record(name=name, path=str(file_path.resolve()), source=_SOP_SOURCE_LOCAL, content=content)

    try:
        from swarmee_river.packs import iter_packs
        from swarmee_river.settings import load_settings

        settings = load_settings()
        for pack in iter_packs(settings):
            if not pack.enabled:
                continue
            sop_dir = pack.sops_dir
            if not sop_dir.exists() or not sop_dir.is_dir():
                continue
            source_label = f"pack:{pack.name}"
            for file_path in sorted(sop_dir.glob(f"*{_SOP_FILE_SUFFIX}")):
                if not file_path.is_file():
                    continue
                with contextlib.suppress(Exception):
                    name, content = _load_sop_file(file_path)
                    _add_record(name=name, path=str(file_path.resolve()), source=source_label, content=content)
    except Exception:
        pass

    try:
        import strands_agents_sops as strands_sops

        module_path = Path(getattr(strands_sops, "__file__", "")).resolve().parent
        candidate_dirs = [module_path / "sops", module_path]
        saw_file = False
        for directory in candidate_dirs:
            if not directory.exists() or not directory.is_dir():
                continue
            for file_path in sorted(directory.glob(f"*{_SOP_FILE_SUFFIX}")):
                if not file_path.is_file():
                    continue
                saw_file = True
                with contextlib.suppress(Exception):
                    name, content = _load_sop_file(file_path)
                    _add_record(name=name, path=str(file_path.resolve()), source=_SOP_SOURCE_STRANDS, content=content)
        if not saw_file:
            for attr_name in dir(strands_sops):
                if attr_name.startswith("_"):
                    continue
                value = getattr(strands_sops, attr_name, None)
                if not isinstance(value, str) or len(value.strip()) < 40:
                    continue
                meta, body = _strip_sop_frontmatter(value)
                name = str(meta.get("name", attr_name)).strip() or attr_name
                _add_record(
                    name=name,
                    path=f"{strands_sops.__name__}.{attr_name}",
                    source=_SOP_SOURCE_STRANDS,
                    content=body.strip(),
                )
    except Exception:
        pass

    return [records[name] for name in sorted(records.keys())]


def discover_available_sop_names() -> list[str]:
    return [record["name"] for record in discover_available_sops()]


def build_swarmee_cmd(prompt: str, *, auto_approve: bool) -> list[str]:
    """Build a subprocess command for a one-shot Swarmee run."""
    command = [sys.executable, "-u", "-m", "swarmee_river.swarmee"]
    if auto_approve:
        command.append("--yes")
    command.append(prompt)
    return command


def build_swarmee_daemon_cmd() -> list[str]:
    """Build a subprocess command for a long-running Swarmee daemon."""
    return [sys.executable, "-u", "-m", "swarmee_river.swarmee", "--tui-daemon"]


def extract_plan_section(output: str) -> str | None:
    """Extract the one-shot plan block beginning at 'Proposed plan:' if present."""
    marker = "Proposed plan:"
    marker_index = output.find(marker)
    if marker_index < 0:
        return None

    trailing_hint_prefix = "Plan generated. Re-run with --yes"
    candidate = output[marker_index:]
    extracted_lines: list[str] = []
    for line in candidate.splitlines():
        if not extracted_lines:
            local_index = line.find(marker)
            if local_index >= 0:
                line = line[local_index:]
        if line.strip().startswith(trailing_hint_prefix):
            break
        extracted_lines.append(line.rstrip())

    while extracted_lines and not extracted_lines[-1].strip():
        extracted_lines.pop()

    if not extracted_lines:
        return None

    extracted = "\n".join(extracted_lines).strip()
    return extracted or None


def looks_like_plan_output(text: str) -> bool:
    """Detect whether one-shot output likely contains a generated plan."""
    return extract_plan_section(text) is not None


def render_tui_hint_after_plan() -> str:
    """Hint shown when a plan-only run is detected."""
    return "Plan detected. Type /approve to execute, /replan to regenerate, /clearplan to clear."


def is_multiline_newline_key(event: Any) -> bool:
    """Detect Shift+Enter, Alt+Enter, or Ctrl+J — NOT plain Enter."""
    key = str(getattr(event, "key", "")).lower()
    aliases = [str(a).lower() for a in getattr(event, "aliases", [])]
    event_name = str(getattr(event, "name", "")).lower()

    # Explicit modifier+enter combinations only.
    # Plain Enter must NOT match — it submits the prompt.
    modifier_enter_keys = {
        "shift+enter", "shift+return", "shift+ctrl+m",
        "alt+enter", "alt+return",
        "ctrl+j",
    }
    if key in modifier_enter_keys:
        return True
    if event_name in {"shift_enter", "shift_return"}:
        return True
    if any(alias in modifier_enter_keys for alias in aliases):
        return True
    return False


def _extract_paths_from_text(text: str) -> list[str]:
    return [match.strip() for match in _PATH_TOKEN_RE.findall(text) if match.strip()]


def sanitize_output_text(text: str) -> str:
    """Remove common control sequences that render poorly in a TUI transcript."""
    cleaned = text.replace("\r", "")
    cleaned = _OSC_ESCAPE_RE.sub("", cleaned)
    cleaned = _ANSI_ESCAPE_RE.sub("", cleaned)
    # Remove any stray ESC bytes left by malformed or partial sequences.
    return cleaned.replace("\x1b", "")


def resolve_model_config_summary(*, provider_override: str | None = None, tier_override: str | None = None) -> str:
    """
    Best-effort summary of the configured model selection (provider/tier/model_id) for display in the TUI.

    This is intentionally approximate: final provider/model can still vary at runtime based on CLI args,
    environment variables, and credential availability.
    """
    try:
        from swarmee_river.settings import load_settings
        from swarmee_river.utils.provider_utils import resolve_model_provider
    except Exception:
        return "Model: (unavailable)"

    try:
        settings = load_settings()
    except Exception:
        return "Model: (unavailable)"

    selected_provider, notice = resolve_model_provider(
        cli_provider=None,
        env_provider=provider_override if provider_override is not None else os.getenv("SWARMEE_MODEL_PROVIDER"),
        settings_provider=settings.models.provider,
    )
    tier = (
        tier_override
        if tier_override is not None
        else (os.getenv("SWARMEE_MODEL_TIER") or settings.models.default_tier)
    )
    tier = (tier or "balanced").strip().lower()

    model_id: str | None = None
    try:
        # Provider-specific tiers are the primary source.
        provider_cfg = settings.models.providers.get(selected_provider)
        if provider_cfg:
            tier_cfg = provider_cfg.tiers.get(tier)
            if tier_cfg and tier_cfg.model_id:
                model_id = tier_cfg.model_id
        # Global tier overrides.
        if model_id is None:
            global_tier_cfg = settings.models.tiers.get(tier)
            if global_tier_cfg and global_tier_cfg.model_id:
                model_id = global_tier_cfg.model_id
    except Exception:
        model_id = None

    suffix = f" ({model_id})" if model_id else ""
    fallback = " (fallback)" if notice else ""
    return f"Model: {selected_provider}/{tier}{suffix}{fallback}"


def _model_option_model_id(
    *,
    settings: Any,
    provider_name: str,
    tier_name: str,
) -> str | None:
    model_id: str | None = None
    provider_cfg = settings.models.providers.get(provider_name)
    if provider_cfg is not None:
        provider_tier_cfg = provider_cfg.tiers.get(tier_name)
        if provider_tier_cfg is not None and provider_tier_cfg.model_id:
            model_id = provider_tier_cfg.model_id
    if model_id is None:
        global_tier_cfg = settings.models.tiers.get(tier_name)
        if global_tier_cfg is not None and global_tier_cfg.model_id:
            model_id = global_tier_cfg.model_id
    return model_id


def model_select_options(
    *,
    provider_override: str | None = None,
    tier_override: str | None = None,
) -> tuple[list[tuple[str, str]], str]:
    """
    Build model selector dropdown options and the currently selected option value.

    Returns:
        (options, selected_value)
    """
    auto_summary = resolve_model_config_summary().removeprefix("Model: ").strip()
    options: list[tuple[str, str]] = [(f"Auto ({auto_summary})", _MODEL_AUTO_VALUE)]
    selected_value = _MODEL_AUTO_VALUE

    try:
        from swarmee_river.settings import load_settings
        from swarmee_river.utils.provider_utils import resolve_model_provider
    except Exception:
        return options, selected_value

    try:
        settings = load_settings()
    except Exception:
        return options, selected_value

    selected_provider, _ = resolve_model_provider(
        cli_provider=None,
        env_provider=provider_override if provider_override is not None else os.getenv("SWARMEE_MODEL_PROVIDER"),
        settings_provider=settings.models.provider,
    )
    selected_tier = (
        tier_override
        if tier_override is not None
        else (os.getenv("SWARMEE_MODEL_TIER") or settings.models.default_tier)
    )
    selected_provider = (selected_provider or "").strip().lower()
    selected_tier = (selected_tier or "").strip().lower()

    provider_cfg = settings.models.providers.get(selected_provider)
    tier_names = sorted(provider_cfg.tiers.keys()) if provider_cfg and provider_cfg.tiers else []

    for tier_name in tier_names:
        value = f"{selected_provider}|{tier_name}"
        model_id = _model_option_model_id(settings=settings, provider_name=selected_provider, tier_name=tier_name)
        suffix = f" ({model_id})" if model_id else ""
        options.append((f"{selected_provider}/{tier_name}{suffix}", value))
        if tier_name == selected_tier:
            selected_value = value

    if provider_override is None and tier_override is None:
        selected_value = _MODEL_AUTO_VALUE

    available_values = {value for _, value in options}
    if selected_value not in available_values:
        selected_value = _MODEL_AUTO_VALUE
    return options, selected_value


def parse_model_select_value(value: str | None) -> tuple[str, str] | None:
    """Parse a model selector value like ``provider|tier``."""
    selected = (value or "").strip().lower()
    if not selected or selected in {_MODEL_AUTO_VALUE, _MODEL_LOADING_VALUE}:
        return None
    if "|" not in selected:
        return None
    provider, tier = selected.split("|", 1)
    provider_name = provider.strip().lower()
    tier_name = tier.strip().lower()
    if not provider_name or not tier_name:
        return None
    return provider_name, tier_name


def should_skip_active_run_tier_warning(
    *,
    requested_provider: str,
    requested_tier: str,
    pending_value: str | None,
) -> bool:
    """Return True when a select change is a stale echo of an already-pending pre-run tier switch."""
    requested = f"{requested_provider.strip().lower()}|{requested_tier.strip().lower()}"
    pending = (pending_value or "").strip().lower()
    return bool(requested) and requested == pending


def choose_daemon_model_select_value(
    *,
    provider: str,
    tier: str,
    option_values: list[str],
    pending_value: str | None = None,
    override_provider: str | None = None,
    override_tier: str | None = None,
) -> str | None:
    """Choose which model-select value should be shown for daemon-backed options."""
    available = [str(v).strip().lower() for v in option_values if isinstance(v, str) and str(v).strip()]
    if not available:
        return None

    provider_name = (provider or "").strip().lower()
    tier_name = (tier or "").strip().lower()

    pending = (pending_value or "").strip().lower()
    if pending and pending in available:
        return pending

    override_provider_name = (override_provider or "").strip().lower()
    override_tier_name = (override_tier or "").strip().lower()
    if provider_name and override_provider_name == provider_name and override_tier_name:
        override_value = f"{provider_name}|{override_tier_name}"
        if override_value in available:
            return override_value

    daemon_value = f"{provider_name}|{tier_name}" if provider_name and tier_name else ""
    if daemon_value and daemon_value in available:
        return daemon_value
    return available[0]


def daemon_model_select_options(
    *,
    provider: str,
    tier: str,
    tiers: list[dict[str, Any]],
    pending_value: str | None = None,
    override_provider: str | None = None,
    override_tier: str | None = None,
) -> tuple[list[tuple[str, str]], str]:
    """Build model selector options for daemon-backed provider/tier metadata."""
    provider_name = (provider or "").strip().lower()

    options: list[tuple[str, str]] = []
    for item in tiers:
        item_provider = str(item.get("provider", "")).strip().lower()
        item_tier = str(item.get("name", "")).strip().lower()
        if not item_tier or item_provider != provider_name:
            continue
        if not bool(item.get("available", False)):
            continue
        model_id = str(item.get("model_id", "")).strip()
        suffix = f" ({model_id})" if model_id else ""
        value = f"{item_provider}|{item_tier}"
        options.append((f"{item_provider}/{item_tier}{suffix}", value))

    if not options:
        return [("No available tiers", _MODEL_LOADING_VALUE)], _MODEL_LOADING_VALUE

    selected_value = choose_daemon_model_select_value(
        provider=provider_name,
        tier=tier,
        option_values=[value for _label, value in options],
        pending_value=pending_value,
        override_provider=override_provider,
        override_tier=override_tier,
    )
    if selected_value is None:
        selected_value = options[0][1]
    return options, selected_value


def choose_model_summary_parts(
    *,
    daemon_provider: str | None,
    daemon_tier: str | None,
    daemon_model_id: str | None,
    daemon_tiers: list[dict[str, Any]] | None = None,
    pending_value: str | None = None,
    override_provider: str | None = None,
    override_tier: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Choose provider/tier/model_id for top-level model summary display."""
    def _lookup_model_id(provider_name: str, tier_name: str) -> str | None:
        tiers = daemon_tiers if isinstance(daemon_tiers, list) else []
        for item in tiers:
            item_provider = str(item.get("provider", "")).strip().lower()
            item_tier = str(item.get("name", "")).strip().lower()
            if item_provider != provider_name or item_tier != tier_name:
                continue
            model_id_value = str(item.get("model_id", "")).strip()
            if model_id_value:
                return model_id_value
        return None

    pending = (pending_value or "").strip().lower()
    if pending and "|" in pending:
        pending_provider, pending_tier = pending.split("|", 1)
        pending_provider = pending_provider.strip().lower()
        pending_tier = pending_tier.strip().lower()
        if pending_provider and pending_tier:
            # Keep selected tier stable in header while daemon confirmation is in-flight.
            if (
                daemon_provider
                and daemon_tier
                and daemon_model_id
                and pending_provider == daemon_provider.strip().lower()
                and pending_tier == daemon_tier.strip().lower()
            ):
                return pending_provider, pending_tier, daemon_model_id
            return pending_provider, pending_tier, _lookup_model_id(pending_provider, pending_tier)

    override_provider_name = (override_provider or "").strip().lower()
    override_tier_name = (override_tier or "").strip().lower()
    if override_provider_name and override_tier_name:
        if (
            daemon_provider
            and daemon_tier
            and daemon_model_id
            and override_provider_name == daemon_provider.strip().lower()
            and override_tier_name == daemon_tier.strip().lower()
        ):
            return override_provider_name, override_tier_name, daemon_model_id
        return override_provider_name, override_tier_name, _lookup_model_id(override_provider_name, override_tier_name)

    provider_name = (daemon_provider or "").strip().lower()
    tier_name = (daemon_tier or "").strip().lower()
    if provider_name and tier_name:
        model_id = str(daemon_model_id).strip() if daemon_model_id is not None else None
        if not model_id:
            model_id = _lookup_model_id(provider_name, tier_name)
        return provider_name, tier_name, (model_id or None)
    return None, None, None


def classify_copy_command(normalized: str) -> str | None:
    """Classify copy command variants into action keys."""
    return _COPY_COMMAND_MAP.get(normalized)


def classify_model_command(normalized: str) -> tuple[str, str | None] | None:
    """Classify /model commands into action + optional argument."""
    if normalized == "/model":
        return "help", None
    if normalized == "/model show":
        return "show", None
    if normalized == "/model list":
        return "list", None
    if normalized == "/model reset":
        return "reset", None
    if normalized.startswith("/model provider "):
        return "provider", normalized.split(maxsplit=2)[2].strip()
    if normalized.startswith("/model tier "):
        return "tier", normalized.split(maxsplit=2)[2].strip()
    return None


def classify_pre_run_command(text: str) -> tuple[str, str | None] | None:
    """Classify commands handled before active-run gating."""
    normalized = text.lower()
    if normalized == "/restore":
        return "restore", None
    if normalized == "/new":
        return "new", None
    if normalized == "/context":
        return "context_usage", None
    if normalized.startswith("/context "):
        return "context", text[len("/context "):]
    if normalized == "/sop":
        return "sop_usage", None
    if normalized.startswith("/sop "):
        return "sop", text[len("/sop "):]
    if normalized.startswith("/open "):
        return "open", text[len("/open "):]
    if normalized == "/open":
        return "open_usage", None
    if normalized.startswith("/expand "):
        return "expand", text[len("/expand "):]
    if normalized == "/expand":
        return "expand_usage", None
    if normalized.startswith("/search "):
        return "search", text[len("/search "):]
    if normalized == "/search":
        return "search_usage", None
    if normalized == "/text":
        return "text", None
    if normalized.startswith("/text "):
        return "text_usage", None
    if normalized == "/thinking":
        return "thinking", None
    if normalized.startswith("/thinking "):
        return "thinking_usage", None
    if normalized == "/compact":
        return "compact", None
    if normalized.startswith("/compact "):
        return "compact_usage", None
    if normalized in {"/stop", ":stop"}:
        return "stop", None
    if normalized in {"/exit", ":exit"}:
        return "exit", None
    if normalized in {"/daemon restart", "/restart-daemon"}:
        return "daemon_restart", None
    if normalized == "/consent":
        return "consent_usage", None
    if normalized.startswith("/consent "):
        return "consent", normalized.split(maxsplit=1)[1].strip()
    if normalized == "/connect":
        return "connect", "github_copilot"
    if normalized.startswith("/connect "):
        return "connect", normalized.split(maxsplit=1)[1].strip()
    if normalized == "/auth":
        return "auth_usage", None
    if normalized.startswith("/auth "):
        return "auth", text[len("/auth "):].strip()
    model = classify_model_command(normalized)
    if model is not None:
        action, argument = model
        return f"model:{action}", argument
    return None


def classify_post_run_command(text: str) -> tuple[str, str | None] | None:
    """Classify commands handled after active-run gating."""
    normalized = text.lower()
    if normalized == "/approve":
        return "approve", None
    if normalized == "/replan":
        return "replan", None
    if normalized == "/clearplan":
        return "clearplan", None
    if normalized == "/plan":
        return "plan_mode", None
    if text.startswith("/plan "):
        return "plan_prompt", text[len("/plan "):].strip()
    if normalized == "/run":
        return "run_mode", None
    if text.startswith("/run "):
        return "run_prompt", text[len("/run "):].strip()
    return None


def classify_tui_error_event(event: dict[str, Any]) -> dict[str, Any]:
    message = str(event.get("message", event.get("text", ""))).strip()
    category_hint = normalize_error_category(event.get("category"))
    tool_use_id = str(event.get("tool_use_id", "")).strip() or None
    classified = classify_error_message(message, category_hint=category_hint, tool_use_id=tool_use_id)
    retry_after_raw = event.get("retry_after_s")
    retry_after_s: int | None = None
    if isinstance(retry_after_raw, (int, float)):
        retry_after_s = int(retry_after_raw)
    elif isinstance(retry_after_raw, str) and retry_after_raw.strip().isdigit():
        retry_after_s = int(retry_after_raw.strip())
    next_tier = str(event.get("next_tier", "")).strip() or None
    return {
        "message": message,
        "category": str(classified.get("category", ERROR_CATEGORY_FATAL)),
        "retryable": bool(event.get("retryable", classified.get("retryable", False))),
        "tool_use_id": str(classified.get("tool_use_id", "")).strip() or None,
        "retry_after_s": retry_after_s if isinstance(retry_after_s, int) and retry_after_s > 0 else None,
        "next_tier": next_tier,
    }


def summarize_error_for_toast(error_info: dict[str, Any]) -> tuple[str, str, float | None]:
    category = str(error_info.get("category", ERROR_CATEGORY_FATAL))
    message = str(error_info.get("message", "")).strip()
    retry_after_s = error_info.get("retry_after_s")

    if category == ERROR_CATEGORY_TRANSIENT:
        delay = int(retry_after_s) if isinstance(retry_after_s, int) and retry_after_s > 0 else 1
        return f"Rate limited - retrying in {delay}s", "warning", _TRANSIENT_TOAST_TIMEOUT_S

    if category == ERROR_CATEGORY_TOOL_ERROR:
        tool_use_id = str(error_info.get("tool_use_id", "")).strip()
        if tool_use_id:
            return f"Tool failed ({tool_use_id})", "error", 6.0
        return "Tool execution failed", "error", 6.0

    if category == ERROR_CATEGORY_ESCALATABLE:
        return "Model/context limit hit - escalation available", "warning", 8.0

    if category == ERROR_CATEGORY_AUTH_ERROR:
        return "Auth/permissions error - check credentials", "error", 10.0

    if message:
        first = message.splitlines()[0].strip()
        if first:
            return first[:140], "error", _FATAL_TOAST_TIMEOUT_S
    return "Fatal error", "error", _FATAL_TOAST_TIMEOUT_S


def parse_output_line(line: str) -> ParsedEvent | None:
    """Best-effort parser for notable subprocess output events."""
    text = line.rstrip("\n")
    stripped = text.strip()
    lower = stripped.lower()

    if stripped.startswith(_PROVIDER_NOISE_PREFIX) and _PROVIDER_FALLBACK_PHRASE in lower:
        return ParsedEvent(kind="noise", text=text)

    if "~ consent>" in lower:
        return ParsedEvent(kind="consent_prompt", text=text)

    if stripped.startswith("Proposed plan:"):
        return ParsedEvent(kind="plan_header", text=text)

    if "[tool result truncated:" in lower:
        match = _TRUNCATED_ARTIFACT_RE.search(text)
        if match:
            path = match.group("path").strip()
            return ParsedEvent(kind="artifact", text=text, meta={"path": path})
        return ParsedEvent(kind="tool_truncated", text=text)

    if lower.startswith("patch:"):
        path = stripped.split(":", 1)[1].strip()
        if path:
            return ParsedEvent(kind="artifact", text=text, meta={"path": path})
        return ParsedEvent(kind="patch", text=text)

    if lower.startswith("backups:"):
        rest = stripped.split(":", 1)[1].strip()
        paths = _extract_paths_from_text(rest)
        if paths:
            return ParsedEvent(kind="artifact", text=text, meta={"paths": ",".join(paths)})
        return ParsedEvent(kind="backups", text=text)

    if stripped.startswith("Error:") or stripped.startswith("ERROR:"):
        return ParsedEvent(kind="error", text=text)

    if "operation not permitted" in lower and "/bin/ps" in lower:
        return ParsedEvent(kind="warning", text=text)

    if "warning" in lower or "deprecationwarning" in lower or "runtimewarning" in lower or "userwarning" in lower:
        return ParsedEvent(kind="warning", text=text)

    if lower.startswith("[tool ") and any(token in lower for token in {" start", " started", " running"}):
        return ParsedEvent(kind="tool_start", text=text)

    if lower.startswith("[tool ") and any(token in lower for token in {" done", " end", " finished", " stopped"}):
        return ParsedEvent(kind="tool_stop", text=text)

    return None


def parse_tui_event(line: str) -> dict[str, Any] | None:
    """Parse a JSONL event line emitted by TuiCallbackHandler. Returns None for non-JSON lines."""
    stripped = sanitize_output_text(line).strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = _json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, _json.JSONDecodeError):
        return None


def extract_tui_text_chunk(event: dict[str, Any]) -> str:
    """Extract a text chunk from a structured TUI event payload."""
    for key in ("data", "text", "delta", "content", "output_text", "outputText", "textDelta"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return ""


def extract_plan_section_from_output(output: str) -> str | None:
    """Extract plan text from mixed output by ignoring structured JSONL event lines."""
    plain_lines = [line for line in output.splitlines() if parse_tui_event(line) is None]
    if not plain_lines:
        return None
    return extract_plan_section("\n".join(plain_lines))


def artifact_paths_from_event(event: ParsedEvent) -> list[str]:
    """Extract artifact paths from a parsed event."""
    if event.meta is None:
        return []
    result: list[str] = []
    path = event.meta.get("path")
    if path:
        result.append(path)
    paths = event.meta.get("paths")
    if paths:
        for item in paths.split(","):
            token = item.strip()
            if token:
                result.append(token)
    return result


def add_recent_artifacts(existing: list[str], new_paths: list[str], *, max_items: int = 20) -> list[str]:
    """Dedupe and cap artifact paths while preserving recency."""
    updated = list(existing)
    for item in new_paths:
        path = item.strip()
        if not path:
            continue
        if path in updated:
            updated.remove(path)
        updated.append(path)
    if len(updated) > max_items:
        updated = updated[-max_items:]
    return updated


def normalize_artifact_index_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize an artifact index record for TUI list/detail rendering."""
    if not isinstance(entry, dict):
        return None
    raw_path = str(entry.get("path", "")).strip()
    if not raw_path:
        return None

    meta_raw = entry.get("meta")
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    artifact_id = str(entry.get("id", "")).strip() or Path(raw_path).name
    name = str(meta.get("name", meta.get("title", ""))).strip()
    if not name:
        name = artifact_id

    kind = str(entry.get("kind", "")).strip() or "unknown"
    created_at = str(entry.get("created_at", "")).strip()
    bytes_value = entry.get("bytes")
    chars_value = entry.get("chars")

    return {
        "item_id": raw_path,
        "id": artifact_id,
        "name": name,
        "kind": kind,
        "created_at": created_at,
        "path": raw_path,
        "bytes": int(bytes_value) if isinstance(bytes_value, int) else None,
        "chars": int(chars_value) if isinstance(chars_value, int) else None,
        "meta": meta,
    }


def build_artifact_sidebar_items(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build SidebarList item payloads from normalized artifact entries."""
    items: list[dict[str, str]] = []
    for entry in entries:
        normalized = normalize_artifact_index_entry(entry)
        if normalized is None:
            continue
        kind = str(normalized.get("kind", "unknown")).strip() or "unknown"
        artifact_id = str(normalized.get("id", "")).strip() or "(no-id)"
        name = str(normalized.get("name", normalized.get("id", ""))).strip() or "(unnamed)"
        label = f"{name} ({artifact_id})" if name != artifact_id else artifact_id
        created_at = str(normalized.get("created_at", "")).strip() or "unknown time"
        path = str(normalized.get("path", "")).strip()
        items.append(
            {
                "id": str(normalized.get("item_id", path)).strip() or path,
                "title": f"{kind} · {label}",
                "subtitle": f"{created_at} · {path}",
                "state": "default",
            }
        )
    return items


def artifact_context_source_payload(path: str, *, source_id: str | None = None) -> dict[str, str]:
    """Build a file context-source payload for an artifact path."""
    normalized_path = str(path or "").strip()
    if not normalized_path:
        raise ValueError("artifact path is required")
    return {
        "type": "file",
        "path": normalized_path,
        "id": _sanitize_context_source_id(source_id or uuid.uuid4().hex),
    }


def build_session_issue_sidebar_items(issues: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build SidebarList payloads from structured session issues."""
    def _truncate_text(value: str, *, max_chars: int = 88) -> str:
        text = value.strip().replace("\n", " ")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"

    items: list[dict[str, str]] = []
    for issue in issues:
        issue_id = str(issue.get("id", "")).strip()
        if not issue_id:
            continue
        severity = str(issue.get("severity", "warning")).strip().lower()
        if severity not in {"warning", "error"}:
            severity = "warning"
        title = str(issue.get("title", "")).strip() or "Issue"
        created_at = str(issue.get("created_at", "")).strip()
        text = str(issue.get("text", "")).strip()
        subtitle_parts = []
        if created_at:
            subtitle_parts.append(created_at)
        if text:
            subtitle_parts.append(_truncate_text(text, max_chars=88))
        subtitle = " | ".join(subtitle_parts)
        items.append(
            {
                "id": issue_id,
                "title": title,
                "subtitle": subtitle,
                "state": "error" if severity == "error" else "warning",
            }
        )
    return items


def render_session_issue_detail_text(issue: dict[str, Any] | None) -> str:
    """Render a detail panel body for a selected session issue."""
    if not isinstance(issue, dict):
        return "(no issue selected)"
    lines = [
        f"Severity: {str(issue.get('severity', 'warning')).strip() or 'warning'}",
        f"Title: {str(issue.get('title', 'Issue')).strip() or 'Issue'}",
        f"When: {str(issue.get('created_at', '')).strip() or '(unknown)'}",
        "",
        str(issue.get("text", "")).strip() or "(no details)",
    ]
    tool_use_id = str(issue.get("tool_use_id", "")).strip()
    if tool_use_id:
        lines.append("")
        lines.append(f"Tool Use ID: {tool_use_id}")
    tool_name = str(issue.get("tool_name", "")).strip()
    if tool_name:
        lines.append(f"Tool: {tool_name}")
    next_tier = str(issue.get("next_tier", "")).strip()
    if next_tier:
        lines.append(f"Suggested tier: {next_tier}")
    return "\n".join(lines)


def session_issue_actions(issue: dict[str, Any] | None) -> list[dict[str, str]]:
    """Return available action buttons for a selected session issue."""
    if not isinstance(issue, dict):
        return []
    category = str(issue.get("category", "")).strip().lower()
    tool_use_id = str(issue.get("tool_use_id", "")).strip()
    actions: list[dict[str, str]] = []
    if category == "tool_failure" and tool_use_id:
        actions.append({"id": "session_issue_retry_tool", "label": "Retry", "variant": "default"})
        actions.append({"id": "session_issue_skip_tool", "label": "Skip", "variant": "default"})
        actions.append({"id": "session_issue_escalate_tier", "label": "Escalate", "variant": "default"})
        actions.append({"id": "session_issue_interrupt", "label": "Interrupt", "variant": "default"})
    return actions


def normalize_session_view_mode(mode: str | None) -> str:
    """Normalize session panel mode for Timeline/Issues toggle."""
    normalized = str(mode or "").strip().lower()
    if normalized == "issues":
        return "issues"
    return "timeline"


def classify_session_timeline_event_kind(event: dict[str, Any] | None) -> str:
    """Classify timeline event kind used for badges/icons."""
    if not isinstance(event, dict):
        return "event"
    name = str(event.get("event", "")).strip().lower()
    has_error = bool(str(event.get("error", "")).strip())
    if name == "after_tool_call":
        if has_error or event.get("success") is False:
            return "error"
        return "tool"
    if name == "after_model_call":
        return "model"
    if name == "after_invocation":
        return "invocation"
    if has_error:
        return "error"
    return "event"


def summarize_session_timeline_event(event: dict[str, Any] | None) -> str:
    """Render compact one-line timeline summary."""
    if not isinstance(event, dict):
        return "event"
    kind = classify_session_timeline_event_kind(event)
    duration = event.get("duration_s")
    duration_text = ""
    if isinstance(duration, (int, float)):
        duration_text = f" ({float(duration):.1f}s)"
    if kind in {"tool", "error"}:
        tool = str(event.get("tool", "")).strip() or "unknown"
        label = f"tool: {tool}{duration_text}"
        if kind == "error":
            return f"{label} error"
        return label
    if kind == "model":
        return f"model call{duration_text}"
    if kind == "invocation":
        return f"invocation{duration_text}"
    return (str(event.get("event", "")).strip() or "event") + duration_text


def build_session_timeline_sidebar_items(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build SidebarList payload for session timeline events."""
    icon_map = {
        "tool": "⚙",
        "model": "◉",
        "invocation": "▶",
        "error": "✖",
        "event": "•",
    }
    state_map = {
        "tool": "default",
        "model": "default",
        "invocation": "active",
        "error": "error",
        "event": "default",
    }
    items: list[dict[str, str]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", "")).strip() or f"timeline-{index + 1}"
        kind = classify_session_timeline_event_kind(event)
        summary = summarize_session_timeline_event(event)
        ts = str(event.get("ts", "")).strip()
        label = str(event.get("event", "")).strip().lower()
        subtitle = ts if ts else label
        if ts and label:
            subtitle = f"{ts} | {label}"
        items.append(
            {
                "id": event_id,
                "title": f"{icon_map.get(kind, '•')} {summary}",
                "subtitle": subtitle,
                "state": state_map.get(kind, "default"),
            }
        )
    return items


def render_session_timeline_detail_text(event: dict[str, Any] | None) -> str:
    """Render detail body for selected timeline event."""
    if not isinstance(event, dict):
        return "(no timeline event selected)"
    payload = dict(event)
    payload.pop("id", None)
    summary = summarize_session_timeline_event(event)
    try:
        rendered = _json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = str(payload)
    return f"Summary: {summary}\n\nPayload:\n{rendered}"


def session_timeline_actions(event: dict[str, Any] | None) -> list[dict[str, str]]:
    """Actions available for selected timeline event."""
    if not isinstance(event, dict):
        return []
    return [
        {"id": "session_timeline_copy_json", "label": "Copy JSON", "variant": "default"},
        {"id": "session_timeline_copy_summary", "label": "Copy summary", "variant": "default"},
    ]


def normalize_agent_studio_view_mode(mode: str | None) -> str:
    """Normalize Agent Studio sub-view mode."""
    normalized = str(mode or "").strip().lower()
    if normalized in {"profile", "tools", "team"}:
        return normalized
    return "profile"


def _normalized_tool_name_list(raw_values: Any) -> list[str]:
    values = raw_values if isinstance(raw_values, list) else []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        token = str(item).strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(token)
    return normalized


def normalize_session_safety_overrides(raw_overrides: Any) -> dict[str, Any]:
    if not isinstance(raw_overrides, dict):
        return {}
    normalized: dict[str, Any] = {}
    consent = str(raw_overrides.get("tool_consent", "")).strip().lower()
    if consent in _AGENT_TOOL_CONSENT_VALUES:
        normalized["tool_consent"] = consent
    allow = _normalized_tool_name_list(raw_overrides.get("tool_allowlist"))
    if allow:
        normalized["tool_allowlist"] = allow
    block = _normalized_tool_name_list(raw_overrides.get("tool_blocklist"))
    if block:
        normalized["tool_blocklist"] = block
    return normalized


def _env_tool_list(var_name: str) -> list[str]:
    raw = os.getenv(var_name, "")
    if not isinstance(raw, str) or not raw.strip():
        return []
    return _normalized_tool_name_list([token for token in raw.split(",")])


def _policy_tier_profile(tier_name: str | None) -> tuple[list[str], list[str], str]:
    tier = str(tier_name or "").strip().lower()
    try:
        settings = load_settings()
    except Exception:
        return [], [], "ask"
    profile = settings.harness.tier_profiles.get(tier)
    allow = list(profile.tool_allowlist) if profile is not None else []
    block = list(profile.tool_blocklist) if profile is not None else []
    default_consent = str(settings.safety.tool_consent or "ask").strip().lower()
    if default_consent not in _AGENT_TOOL_CONSENT_VALUES:
        default_consent = "ask"
    return _normalized_tool_name_list(allow), _normalized_tool_name_list(block), default_consent


def build_agent_policy_lens(*, tier_name: str | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    tier_allow, tier_block, default_consent = _policy_tier_profile(tier_name)
    normalized_overrides = normalize_session_safety_overrides(overrides)

    effective_allow = (
        _normalized_tool_name_list(normalized_overrides.get("tool_allowlist"))
        if "tool_allowlist" in normalized_overrides
        else list(tier_allow)
    )
    effective_block = (
        _normalized_tool_name_list(normalized_overrides.get("tool_blocklist"))
        if "tool_blocklist" in normalized_overrides
        else list(tier_block)
    )
    effective_consent = str(normalized_overrides.get("tool_consent", default_consent)).strip().lower()
    if effective_consent not in _AGENT_TOOL_CONSENT_VALUES:
        effective_consent = default_consent

    return {
        "tier": str(tier_name or "").strip().lower() or None,
        "default": {
            "tool_consent": default_consent,
            "tool_allowlist": list(tier_allow),
            "tool_blocklist": list(tier_block),
        },
        "session_overrides": dict(normalized_overrides),
        "effective": {
            "tool_consent": effective_consent,
            "tool_allowlist": list(effective_allow),
            "tool_blocklist": list(effective_block),
        },
        "env": {
            "enable_tools": _env_tool_list("SWARMEE_ENABLE_TOOLS"),
            "disable_tools": _env_tool_list("SWARMEE_DISABLE_TOOLS"),
        },
    }


def build_agent_tools_safety_sidebar_items(policy_lens: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Return sidebar items for Tools & Safety Agent Studio view."""
    lens = policy_lens if isinstance(policy_lens, dict) else {}
    effective = lens.get("effective", {}) if isinstance(lens.get("effective"), dict) else {}
    overrides = lens.get("session_overrides", {}) if isinstance(lens.get("session_overrides"), dict) else {}
    consent = str(effective.get("tool_consent", "ask")).strip().lower() or "ask"
    effective_allow = _normalized_tool_name_list(effective.get("tool_allowlist"))
    effective_block = _normalized_tool_name_list(effective.get("tool_blocklist"))
    override_count = len(overrides)
    return [
        {
            "id": "policy_lens",
            "title": "Policy Lens",
            "subtitle": f"consent={consent} | allow={len(effective_allow)} | block={len(effective_block)}",
            "state": "active",
        },
        {
            "id": "session_overrides",
            "title": "Session Overrides",
            "subtitle": f"active fields={override_count}",
            "state": "warning" if override_count else "default",
        },
    ]


def render_agent_tools_safety_detail_text(
    item: dict[str, Any] | None,
    policy_lens: dict[str, Any] | None = None,
) -> str:
    """Render detail text for Tools & Safety records."""
    if not isinstance(item, dict):
        return "(no tools/safety item selected)"
    lens = policy_lens if isinstance(policy_lens, dict) else {}
    item_id = str(item.get("id", "")).strip()
    if item_id == "policy_lens":
        rendered = _json.dumps(lens, ensure_ascii=False, indent=2, sort_keys=True) if lens else "{}"
        return (
            "Tools & Safety: Policy Lens\n\n"
            "Effective tool/safety posture across tier defaults, session overrides, and env controls.\n\n"
            f"{rendered}"
        )
    if item_id == "session_overrides":
        overrides = lens.get("session_overrides", {}) if isinstance(lens.get("session_overrides"), dict) else {}
        rendered = _json.dumps(overrides, ensure_ascii=False, indent=2, sort_keys=True)
        return (
            "Tools & Safety: Session Overrides\n\n"
            "Session-only overrides are layered above tier defaults.\n"
            "Use the form below to apply or reset tool_consent/tool_allowlist/tool_blocklist.\n\n"
            f"{rendered}"
        )
    return str(item.get("title", "Tools & Safety")).strip() or "(no details)"


def build_agent_team_sidebar_items(team_presets: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return Team Agent Studio sidebar items from profile team presets."""
    normalized = normalize_team_presets(team_presets or [])
    if not normalized:
        return [
            {
                "id": "team_preset_none",
                "title": "No Team Presets",
                "subtitle": "Create a preset to compose a multi-agent run.",
                "state": "default",
            }
        ]

    items: list[dict[str, Any]] = []
    for preset in normalized:
        description = str(preset.get("description", "")).strip()
        spec = preset.get("spec", {})
        key_count = len(spec) if isinstance(spec, dict) else 0
        subtitle = description or f"spec keys: {key_count}"
        items.append(
            {
                "id": str(preset.get("id", "")).strip(),
                "title": str(preset.get("name", "")).strip() or "Unnamed Team Preset",
                "subtitle": subtitle,
                "state": "active" if key_count else "default",
                "preset": dict(preset),
            }
        )
    return items


def render_agent_team_detail_text(item: dict[str, Any] | None) -> str:
    """Render detail text for Team preset records."""
    if not isinstance(item, dict):
        return "(no team item selected)"
    item_id = str(item.get("id", "")).strip()
    if item_id == "team_preset_none":
        return (
            "Team Presets\n\n"
            "Create and save a preset to compose multi-agent execution via `swarm`.\n"
            "Use Save Profile after editing to persist the preset catalog."
        )

    preset = normalize_team_preset(item.get("preset"))
    if preset is None:
        return str(item.get("title", "Team")).strip() or "(no details)"
    spec_json = _json.dumps(preset.get("spec", {}), ensure_ascii=False, indent=2, sort_keys=True)
    description = str(preset.get("description", "")).strip() or "(none)"
    return (
        "Team Preset\n\n"
        f"ID: {preset['id']}\n"
        f"Name: {preset['name']}\n"
        f"Description: {description}\n\n"
        "Spec:\n"
        f"{spec_json}"
    )


def detect_consent_prompt(line: str) -> str | None:
    """Detect consent-related subprocess output lines."""
    normalized = line.strip().lower()
    if "~ consent>" in normalized:
        return "prompt"
    if "allow tool '" in normalized:
        return "header"
    return None


def update_consent_capture(
    consent_active: bool,
    consent_buffer: list[str],
    line: str,
    *,
    max_lines: int = 20,
) -> tuple[bool, list[str]]:
    """Update consent capture state from a single output line."""
    kind = detect_consent_prompt(line)
    if kind is None and not consent_active:
        return consent_active, consent_buffer

    updated = list(consent_buffer)
    updated.append(line.rstrip("\n"))
    if len(updated) > max_lines:
        updated = updated[-max_lines:]
    return True, updated


def write_to_proc(proc: subprocess.Popen[str], text: str) -> bool:
    """Write a response line to a subprocess stdin."""
    if proc.stdin is None:
        return False

    payload = text if text.endswith("\n") else f"{text}\n"
    try:
        proc.stdin.write(payload)
        proc.stdin.flush()
    except Exception:
        return False
    return True


def send_daemon_command(proc: Any, cmd_dict: dict[str, Any]) -> bool:
    """Serialize and send a daemon command as JSONL."""
    sender = getattr(proc, "send_command", None)
    if callable(sender):
        try:
            return bool(sender(cmd_dict))
        except Exception:
            return False

    stdin = getattr(proc, "stdin", None)
    if stdin is None:
        return False
    try:
        payload = _json.dumps(cmd_dict, ensure_ascii=False) + "\n"
        stdin.write(payload)
        stdin.flush()
    except Exception:
        return False
    return True


def _build_swarmee_subprocess_env(
    *,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # The CLI callback handler prints spinners using ANSI + carriage returns; disable for TUI subprocess capture.
    env["SWARMEE_SPINNERS"] = "0"
    # Enable structured JSONL event output for TUI consumption.
    env["SWARMEE_TUI_EVENTS"] = "1"
    existing_warning_filters = env.get("PYTHONWARNINGS", "").strip()
    tui_warning_filters = [
        # `PYTHONWARNINGS` is parsed via `warnings._setoption`, which `re.escape`s message+module.
        # Use exact (literal) values, not regex patterns.
        'ignore:Field name "json" in "Http_requestTool" shadows an attribute in parent "BaseModel"'
        ":UserWarning:pydantic.main",
    ]
    env["PYTHONWARNINGS"] = ",".join(
        [item for item in [*tui_warning_filters, existing_warning_filters] if isinstance(item, str) and item.strip()]
    )
    if env_overrides:
        env.update(env_overrides)
    if session_id:
        env["SWARMEE_SESSION_ID"] = session_id
    return env


def _spawn_swarmee_process(
    command: list[str],
    *,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    env = _build_swarmee_subprocess_env(session_id=session_id, env_overrides=env_overrides)
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        # Isolate child subprocesses from the interactive terminal session to
        # prevent terminal-title churn while tools (git/rg/etc.) execute.
        start_new_session=True,
    )


def spawn_swarmee(
    prompt: str,
    *,
    auto_approve: bool,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Spawn Swarmee as a subprocess with line-buffered merged output."""
    return _spawn_swarmee_process(
        build_swarmee_cmd(prompt, auto_approve=auto_approve),
        session_id=session_id,
        env_overrides=env_overrides,
    )


def spawn_swarmee_daemon(
    *,
    session_id: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Spawn Swarmee daemon with line-buffered merged output."""
    return _spawn_swarmee_process(
        build_swarmee_daemon_cmd(),
        session_id=session_id,
        env_overrides=env_overrides,
    )


def stop_process(proc: subprocess.Popen[str], *, timeout_s: float = 2.0) -> None:
    """Stop a running subprocess, escalating from interrupt to terminate to kill."""
    if proc.poll() is not None:
        return

    if os.name == "posix" and hasattr(signal, "SIGINT"):
        signaled = False
        if hasattr(os, "killpg"):
            with contextlib.suppress(Exception):
                os.killpg(proc.pid, signal.SIGINT)
                signaled = True
        if not signaled:
            with contextlib.suppress(Exception):
                proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass

    terminated = False
    if os.name == "posix" and hasattr(os, "killpg") and hasattr(signal, "SIGTERM"):
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGTERM)
            terminated = True
    if not terminated:
        with contextlib.suppress(Exception):
            proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return

    killed = False
    if os.name == "posix" and hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
            killed = True
    if not killed:
        with contextlib.suppress(Exception):
            proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout_s)


def run_tui() -> int:
    """Run the full-screen TUI if Textual is installed."""
    try:
        textual_app = importlib.import_module("textual.app")
        textual_binding = importlib.import_module("textual.binding")
        textual_containers = importlib.import_module("textual.containers")
        textual_widgets = importlib.import_module("textual.widgets")
    except ImportError:
        print(
            "Textual is required for `swarmee tui`.\n"
            'Install with: pip install "swarmee-river[tui]"\n'
            'For editable installs in this repo: pip install -e ".[tui]"',
            file=sys.stderr,
        )
        return 1

    AppBase = textual_app.App
    Binding = textual_binding.Binding
    Horizontal = textual_containers.Horizontal
    Vertical = textual_containers.Vertical
    VerticalScroll = textual_containers.VerticalScroll
    Button = textual_widgets.Button
    Checkbox = textual_widgets.Checkbox
    Header = textual_widgets.Header
    Footer = textual_widgets.Footer
    Input = textual_widgets.Input
    RichLog = textual_widgets.RichLog
    Select = textual_widgets.Select
    Static = textual_widgets.Static
    TabbedContent = textual_widgets.TabbedContent
    TabPane = textual_widgets.TabPane
    TextArea = textual_widgets.TextArea

    from swarmee_river.tui.widgets import (
        ActionSheet,
        AgentProfileActions,
        CommandPalette,
        ConsentPrompt,
        ContextBudgetBar,
        ErrorActionPrompt,
        PlanActions,
        SidebarDetail,
        SidebarHeader,
        SidebarList,
        StatusBar,
        ThinkingBar,
        extract_consent_tool_name,
        format_tool_input_oneliner,
        render_agent_profile_summary_text,
        render_assistant_message,
        render_plan_panel,
        render_system_message,
        render_tool_details_panel,
        render_tool_heartbeat_line,
        render_tool_progress_chunk,
        render_tool_result_line,
        render_tool_start_line_with_input,
        render_user_message,
    )

    class PromptTextArea(TextArea):
        """Prompt editor that submits on Enter and inserts newline on Shift+Enter/Ctrl+J."""

        _ENTER_KEYS = {"enter", "return", "ctrl+m"}

        def _insert_newline(self) -> None:
            for method_name, args in (
                ("insert", ("\n",)),
                ("insert_text_at_cursor", ("\n",)),
                ("action_newline", ()),
                ("action_insert_newline", ()),
            ):
                method = getattr(self, method_name, None)
                if not callable(method):
                    continue
                with contextlib.suppress(Exception):
                    method(*args)
                    return

        def _adjust_height(self) -> None:
            line_count = self.text.count("\n") + 1
            # Height controls the input area only. The prompt container provides the
            # border and a one-line footer row (model selector).
            target = max(4, min(11, line_count))
            if getattr(self, "_last_height", None) != target:
                self._last_height = target
                self.styles.height = target

        def on_text_area_changed(self, event: Any) -> None:
            self._adjust_height()
            app = getattr(self, "app", None)
            if app is not None and hasattr(app, "_update_command_palette"):
                app._update_command_palette(self.text)
            if app is not None and hasattr(app, "_on_prompt_text_changed"):
                app._on_prompt_text_changed(self.text)

        async def _on_key(self, event: Any) -> None:
            """Override TextArea._on_key to control Enter behaviour and route
            special keys (palette, history) before falling back to TextArea's
            default character-insertion handler.

            Textual 8's dispatch uses ``cls.__dict__.get("_on_key") or
            cls.__dict__.get("on_key")`` per MRO class, so defining *both*
            ``_on_key`` and ``on_key`` on the same class causes ``on_key`` to
            be silently skipped.  Therefore ALL key logic lives here.
            """
            key = str(getattr(event, "key", "")).lower()
            app = getattr(self, "app", None)

            # ── Shift+Enter / Alt+Enter / Ctrl+J → insert newline ──
            if is_multiline_newline_key(event):
                event.stop()
                event.prevent_default()
                self._insert_newline()
                return

            # ── Arrow keys: palette navigation (when visible) ──
            if key in {"up", "down"} and app is not None and hasattr(app, "_command_palette"):
                palette = app._command_palette
                if palette is not None and palette.is_visible:
                    event.stop()
                    event.prevent_default()
                    palette.move_selection(-1 if key == "up" else 1)
                    return

            # ── Ctrl+K / Ctrl+Space: action sheet ──
            if (
                key in {"ctrl+k", "ctrl+space", "ctrl+@"}
                and app is not None
                and hasattr(app, "action_open_action_sheet")
            ):
                event.stop()
                event.prevent_default()
                app.action_open_action_sheet()
                return

            # ── Ctrl+T: transcript mode toggle ──
            if key == "ctrl+t" and app is not None and hasattr(app, "action_toggle_transcript_mode"):
                event.stop()
                event.prevent_default()
                app.action_toggle_transcript_mode()
                return

            # ── Arrow keys: prompt history (when palette hidden) ──
            if key == "up" and app is not None and hasattr(app, "_prompt_history"):
                history = app._prompt_history
                if history and app._history_index < len(history) - 1:
                    event.stop()
                    event.prevent_default()
                    app._history_index += 1
                    self.clear()
                    entry = history[-(app._history_index + 1)]
                    for method_name in ("insert", "insert_text_at_cursor"):
                        method = getattr(self, method_name, None)
                        if callable(method):
                            with contextlib.suppress(Exception):
                                method(entry)
                                break
                    return
            if key == "down" and app is not None and hasattr(app, "_prompt_history"):
                if app._history_index > 0:
                    event.stop()
                    event.prevent_default()
                    app._history_index -= 1
                    self.clear()
                    entry = app._prompt_history[-(app._history_index + 1)]
                    for method_name in ("insert", "insert_text_at_cursor"):
                        method = getattr(self, method_name, None)
                        if callable(method):
                            with contextlib.suppress(Exception):
                                method(entry)
                                break
                    return
                elif app._history_index == 0:
                    event.stop()
                    event.prevent_default()
                    app._history_index = -1
                    self.clear()
                    return

            # ── Enter with palette visible → submit ──
            # The palette is suggestions/autocomplete; Enter should still execute the
            # current prompt text (otherwise slash commands never run because the
            # palette reopens on every change).
            if key in self._ENTER_KEYS and app is not None and hasattr(app, "_command_palette"):
                palette = app._command_palette
                if palette is not None and palette.is_visible:
                    event.stop()
                    event.prevent_default()
                    palette.hide()
                    app.action_submit_prompt()
                    return

            # ── Tab with palette visible → select command ──
            if key == "tab" and app is not None and hasattr(app, "_command_palette"):
                palette = app._command_palette
                if palette is not None and palette.is_visible:
                    event.stop()
                    event.prevent_default()
                    selected = palette.get_selected()
                    if selected:
                        self.clear()
                        for method_name in ("insert", "insert_text_at_cursor"):
                            method = getattr(self, method_name, None)
                            if callable(method):
                                with contextlib.suppress(Exception):
                                    method(selected + " ")
                                    break
                    palette.hide()
                    return

            # ── Escape → dismiss palette ──
            if key == "escape" and app is not None and hasattr(app, "_command_palette"):
                palette = app._command_palette
                if palette is not None and palette.is_visible:
                    event.stop()
                    event.prevent_default()
                    palette.hide()
                    return

            # ── Plain Enter → submit prompt ──
            # Do NOT call super() — TextArea would insert a newline.
            if key in self._ENTER_KEYS:
                event.stop()
                event.prevent_default()
                if app is not None:
                    app.action_submit_prompt()
                return

            # ── Space: explicit insert ──
            # Some terminal / Textual 8.0.0 combinations appear to drop space insertion
            # when overriding `_on_key`. Handle it directly to keep typing reliable.
            if key == "space":
                event.stop()
                event.prevent_default()
                with contextlib.suppress(Exception):
                    self.insert(" ")
                    return
                with contextlib.suppress(Exception):
                    self.insert_text_at_cursor(" ")
                    return

            # ── Everything else: delegate to TextArea (space, printable, etc.) ──
            # Textual has changed `TextArea._on_key` between releases (sync vs async),
            # and some versions route default behavior via `on_key` instead.
            # Call what exists, and only await if it returns an awaitable.
            handler = getattr(super(), "_on_key", None) or getattr(super(), "on_key", None)
            if handler is None:
                return
            result = handler(event)
            if inspect.isawaitable(result):
                await result

    class SwarmeeTUI(AppBase):
        CSS = """
        $accent: #6a9955;
        $primary: #6a9955;
        $footer-key-foreground: #6a9955;
        $footer-key-background: #2f2f2f;
        $footer-description-background: #1f1f1f;
        $footer-item-background: #1f1f1f;
        $footer-background: #1f1f1f;

        Screen {
            layout: vertical;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        #panes {
            layout: horizontal;
            height: 1fr;
            width: 100%;
        }

        #transcript, #transcript_text {
            width: 2fr;
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            overflow-y: auto;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }
        #transcript_text {
            display: none;
        }

        #side {
            width: 1fr;
            height: 1fr;
            layout: vertical;
        }

        #side_tabs {
            height: 1fr;
        }

        #plan, #agent_summary {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        #context_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #sops_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #artifacts_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #artifacts_list {
            margin: 0 0 1 0;
            min-height: 10;
        }

        #artifacts_detail {
            height: 1fr;
        }

        #session_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #session_view_switch {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #session_view_switch Button {
            width: 1fr;
            min-width: 12;
            margin: 0 1 0 0;
        }

        #session_timeline_view, #session_issues_view {
            height: 1fr;
            layout: vertical;
        }

        #session_timeline_list, #session_issue_list {
            margin: 0 0 1 0;
            min-height: 10;
        }

        #session_timeline_detail, #session_issue_detail {
            height: 1fr;
        }

        #session_issue_list {
            margin: 0 0 1 0;
            min-height: 10;
        }

        #session_issue_detail {
            height: 1fr;
        }

        #agent_panel {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
            layout: vertical;
        }

        #agent_view_switch {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_view_switch Button {
            width: 1fr;
            min-width: 12;
            margin: 0 1 0 0;
        }

        #agent_profile_view, #agent_tools_view, #agent_team_view {
            height: 1fr;
            layout: vertical;
        }

        #agent_summary_header, #agent_profiles_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #agent_profile_list {
            width: 1fr;
            margin: 0 0 1 0;
            min-height: 8;
        }

        #agent_profile_meta_row {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_profile_id, #agent_profile_name {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_profile_name {
            margin: 0;
        }

        #agent_profile_actions {
            height: auto;
            margin: 0 0 1 0;
        }

        #agent_profile_status {
            height: auto;
            color: $text-muted;
        }

        #agent_tools_list, #agent_team_list {
            width: 1fr;
            margin: 0 0 1 0;
            min-height: 8;
        }

        #agent_tools_detail, #agent_team_detail {
            height: 1fr;
        }

        #agent_tools_overrides_header {
            height: auto;
            color: $text-muted;
            padding: 1 0 0 0;
        }

        #agent_tools_override_consent, #agent_tools_override_allowlist, #agent_tools_override_blocklist {
            width: 1fr;
            margin: 0 0 1 0;
        }

        #agent_tools_override_actions {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_tools_override_actions Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_tools_override_status {
            height: auto;
            color: $text-muted;
        }

        #agent_team_editor_header {
            height: auto;
            color: $text-muted;
            padding: 1 0 0 0;
        }

        #agent_team_meta_row {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_team_preset_id, #agent_team_preset_name {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_team_preset_name {
            margin: 0;
        }

        #agent_team_preset_description {
            width: 1fr;
            margin: 0 0 1 0;
        }

        #agent_team_preset_spec {
            width: 1fr;
            height: 8;
            margin: 0 0 1 0;
        }

        #agent_team_actions {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #agent_team_actions Button {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #agent_team_status {
            height: auto;
            color: $text-muted;
        }

        #context_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #sops_header {
            height: auto;
            color: $text-muted;
            padding: 0 0 1 0;
        }

        #context_sources_list {
            height: 1fr;
            border: round #3b3b3b;
            padding: 0 1;
            margin: 0 0 1 0;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        #sop_list {
            height: 1fr;
            border: round #3b3b3b;
            padding: 0 1;
            margin: 0 0 1 0;
            scrollbar-background: #2f2f2f;
            scrollbar-background-hover: #3a3a3a;
            scrollbar-background-active: #454545;
            scrollbar-color: #7f7f7f;
            scrollbar-color-hover: #999999;
            scrollbar-color-active: #b3b3b3;
        }

        .context-source-row {
            layout: horizontal;
            height: auto;
            margin: 0;
            padding: 0;
        }

        .sop-row {
            height: auto;
            margin: 0 0 1 0;
            padding: 0;
            layout: vertical;
            border-bottom: solid #333333;
        }

        .sop-row-header {
            height: auto;
            layout: horizontal;
            margin: 0;
            padding: 0;
        }

        .sop-source-label {
            width: auto;
            min-width: 14;
            color: $text-muted;
            padding: 0 0 0 1;
        }

        .sop-preview {
            color: $text-muted;
            padding: 0 0 0 3;
        }

        .context-source-label {
            width: 1fr;
            color: $text;
            padding: 0 1 0 0;
        }

        .context-remove-btn {
            width: 3;
            min-width: 3;
            margin: 0;
            padding: 0;
        }

        #context_add_row {
            height: auto;
            layout: horizontal;
            margin: 0 0 1 0;
        }

        #context_add_row Button {
            width: 1fr;
            min-width: 7;
            margin: 0 1 0 0;
        }

        #context_input_row, #context_sop_row {
            height: auto;
            layout: horizontal;
            margin: 0;
        }

        #context_input, #context_sop_select {
            width: 1fr;
            margin: 0 1 0 0;
        }

        #plan_actions {
            height: auto;
        }

        #consent_prompt {
            display: none;
            height: auto;
            min-height: 5;
            margin: 0 0 0 0;
        }

        #error_action_prompt {
            display: none;
            height: auto;
            min-height: 0;
            margin: 0 0 0 0;
        }

        #prompt_box {
            min-height: 5;
            max-height: 12;
            height: auto;
            width: 1fr;
            border: round $accent;
            padding: 0 1;
        }

        #prompt {
            border: none;
            padding: 0;
            height: auto;
        }

        #prompt_bottom {
            height: 1;
            layout: horizontal;
            align: left middle;
        }

        #prompt_metrics {
            width: 1fr;
        }

        #model_select {
            width: 34;
            min-width: 24;
        }

        Footer {
            color: $text-muted;
            background: #1f1f1f;
        }

        Footer FooterKey {
            background: #1f1f1f;
        }

        Footer FooterKey .footer-key--key {
            color: $accent;
            background: #2f2f2f;
        }

        Footer FooterKey .footer-key--description {
            color: $text-muted;
            background: #1f1f1f;
        }
        """
        BINDINGS = [
            ("f5", "submit_prompt", "Send prompt"),
            ("escape", "interrupt_run", "Interrupt run"),
            ("ctrl+t", "toggle_transcript_mode", "Toggle transcript mode"),
            Binding("ctrl+k", "open_action_sheet", "Actions", priority=True),
            Binding("ctrl+space", "open_action_sheet", "Actions", priority=True),
            ("ctrl+c", "copy_selection", "Copy selection"),
            ("meta+c", "copy_selection", "Copy selection"),
            ("super+c", "copy_selection", "Copy selection"),
            ("ctrl+d", "quit", "Quit"),
            ("tab", "focus_prompt", "Focus prompt"),
            Binding("ctrl+left", "widen_side", "Widen side", priority=True),
            Binding("ctrl+right", "widen_transcript", "Widen transcript", priority=True),
            Binding("ctrl+shift+left", "widen_side", "Widen side", priority=True),
            Binding("ctrl+shift+right", "widen_transcript", "Widen transcript", priority=True),
            ("f6", "widen_side", "Widen side"),
            ("f7", "widen_transcript", "Widen transcript"),
            ("ctrl+f", "search_transcript", "Search"),
        ]

        _proc: _DaemonTransport | None = None
        _runner_thread: threading.Thread | None = None
        _last_prompt: str | None = None
        _pending_plan_prompt: str | None = None
        _last_run_auto_approve: bool = False
        _default_auto_approve: bool = False
        _consent_active: bool = False
        _consent_buffer: list[str] = []
        _consent_history_lines: list[str] = []
        _consent_tool_name: str = "tool"
        _consent_prompt_nonce: int = 0
        _consent_hide_timer: Any = None
        _context_sources: list[dict[str, str]] = []
        _context_sop_names: list[str] = []
        _context_add_mode: str | None = None
        _context_ready_for_sync: bool = False
        _sop_catalog: list[dict[str, str]] = []
        _active_sop_names: set[str] = set()
        _sop_toggle_id_to_name: dict[str, str] = {}
        _sops_ready_for_sync: bool = False
        _artifacts: list[str] = []
        _artifact_entries: list[dict[str, Any]] = []
        _artifact_selected_item_id: str | None = None
        _plan_text: str = ""
        _issues_lines: list[str] = []
        _session_issues: list[dict[str, Any]] = []
        _session_selected_issue_id: str | None = None
        _session_view_mode: str = "timeline"
        _session_timeline_index: dict[str, Any] | None = None
        _session_timeline_events: list[dict[str, Any]] = []
        _session_timeline_selected_event_id: str | None = None
        _session_timeline_refresh_timer: Any = None
        _session_timeline_refresh_inflight: bool = False
        _session_timeline_refresh_pending: bool = False
        _issues_repeat_line: str | None = None
        _issues_repeat_count: int = 0
        _warning_count: int = 0
        _error_count: int = 0
        _model_provider_override: str | None = None
        _model_tier_override: str | None = None
        _model_select_syncing: bool = False
        _pending_model_select_value: str | None = None
        # Conversation view state
        _current_assistant_chunks: list[str] = []
        _streaming_buffer: list[str] = []
        _streaming_flush_timer: Any = None
        _tool_progress_pending_ids: set[str] = set()
        _tool_progress_flush_timer: Any = None
        _current_assistant_model: str | None = None
        _current_assistant_timestamp: str | None = None
        _assistant_placeholder_written: bool = False
        _current_thinking: bool = False
        _thinking_buffer: list[str] = []
        _thinking_char_count: int = 0
        _thinking_display_timer: Any = None
        _thinking_animation_timer: Any = None
        _thinking_started_mono: float | None = None
        _thinking_frame_index: int = 0
        _last_thinking_text: str = ""
        _tool_blocks: dict[str, dict[str, Any]] = {}
        _tool_pending_start: dict[str, float] = {}
        _tool_pending_start_timers: dict[str, Any] = {}
        _transcript_mode: str = "rich"
        _current_plan_steps_total: int = 0
        _current_plan_summary: str = ""
        _current_plan_steps: list[str] = []
        _current_plan_step_statuses: list[str] = []
        _current_plan_active_step: int | None = None
        _plan_updates_seen: bool = False
        _transcript_fallback_lines: list[str] = []
        _consent_prompt_widget: Any = None  # ConsentPrompt | None
        _error_action_prompt_widget: Any = None  # ErrorActionPrompt | None
        _pending_error_action: dict[str, Any] | None = None
        _context_sources_list: Any = None  # VerticalScroll | None
        _sop_list: Any = None  # VerticalScroll | None
        _context_input: Any = None  # Input | None
        _context_sop_select: Any = None  # Select | None
        _session_header: Any = None  # SidebarHeader | None
        _session_view_timeline_button: Any = None  # Button | None
        _session_view_issues_button: Any = None  # Button | None
        _session_timeline_view: Any = None  # Vertical | None
        _session_issues_view: Any = None  # Vertical | None
        _session_timeline_header: Any = None  # SidebarHeader | None
        _session_timeline_list: Any = None  # SidebarList | None
        _session_timeline_detail: Any = None  # SidebarDetail | None
        _session_issue_list: Any = None  # SidebarList | None
        _session_issue_detail: Any = None  # SidebarDetail | None
        _artifacts_header: Any = None  # SidebarHeader | None
        _artifacts_list: Any = None  # SidebarList | None
        _artifacts_detail: Any = None  # SidebarDetail | None
        _command_palette: Any = None  # CommandPalette | None
        _action_sheet: Any = None  # ActionSheet | None
        _action_sheet_mode: str = "root"
        _action_sheet_previous_focus: Any = None
        _status_bar: Any = None  # StatusBar | None
        _thinking_bar: Any = None  # ThinkingBar | None
        _prompt_metrics: Any = None  # ContextBudgetBar | None
        _prompt_input_tokens_est: int | None = None
        _prompt_estimate_timer: Any = None
        _pending_prompt_estimate_text: str = ""
        _run_tool_count: int = 0
        _run_start_time: float | None = None
        _status_timer: Any = None
        _last_assistant_text: str = ""
        _prompt_history: list[str] = []
        _history_index: int = -1
        _MAX_PROMPT_HISTORY: int = 50
        _TRANSCRIPT_MAX_LINES: int = 5000
        _split_ratio: int = 2
        _search_active: bool = False
        _plan_step_counter: int = 0
        _plan_completion_announced: bool = False
        _received_structured_plan: bool = False
        _daemon_ready: bool = False
        _query_active: bool = False
        _daemon_tiers: list[dict[str, Any]] = []
        _daemon_provider: str | None = None
        _daemon_tier: str | None = None
        _daemon_model_id: str | None = None
        _current_daemon_model: str | None = None
        _turn_output_chunks: list[str] = []
        _daemon_session_id: str | None = None
        _available_restore_session_id: str | None = None
        _available_restore_turn_count: int = 0
        _last_restored_turn_count: int = 0
        _is_shutting_down: bool = False
        _last_usage: dict[str, Any] | None = None
        _last_cost_usd: float | None = None
        _last_prompt_tokens_est: int | None = None
        _last_budget_tokens: int | None = None
        _saved_profiles: list[AgentProfile] = []
        _effective_profile: AgentProfile | None = None
        _agent_draft_dirty: bool = False
        _agent_form_syncing: bool = False
        _agent_studio_view_mode: str = "profile"
        _agent_tools_items: list[dict[str, Any]] = []
        _agent_team_presets: list[dict[str, Any]] = []
        _agent_team_items: list[dict[str, Any]] = []
        _agent_tools_selected_item_id: str | None = None
        _agent_team_selected_item_id: str | None = None
        _session_safety_overrides: dict[str, Any] = {}
        _agent_tools_policy_lens: dict[str, Any] = {}
        _agent_tools_form_syncing: bool = False
        _agent_team_form_syncing: bool = False
        _agent_view_profile_button: Any = None  # Button | None
        _agent_view_tools_button: Any = None  # Button | None
        _agent_view_team_button: Any = None  # Button | None
        _agent_profile_view: Any = None  # Vertical | None
        _agent_tools_view: Any = None  # Vertical | None
        _agent_team_view: Any = None  # Vertical | None
        _agent_summary: Any = None  # TextArea | None
        _agent_profile_list: Any = None  # SidebarList | None
        _agent_tools_header: Any = None  # SidebarHeader | None
        _agent_tools_list: Any = None  # SidebarList | None
        _agent_tools_detail: Any = None  # SidebarDetail | None
        _agent_tools_override_consent_input: Any = None  # Input | None
        _agent_tools_override_allowlist_input: Any = None  # Input | None
        _agent_tools_override_blocklist_input: Any = None  # Input | None
        _agent_tools_override_status: Any = None  # Static | None
        _agent_team_header: Any = None  # SidebarHeader | None
        _agent_team_list: Any = None  # SidebarList | None
        _agent_team_detail: Any = None  # SidebarDetail | None
        _agent_team_preset_id_input: Any = None  # Input | None
        _agent_team_preset_name_input: Any = None  # Input | None
        _agent_team_preset_description_input: Any = None  # Input | None
        _agent_team_preset_spec_input: Any = None  # TextArea | None
        _agent_team_status: Any = None  # Static | None
        _agent_profile_id_input: Any = None  # Input | None
        _agent_profile_name_input: Any = None  # Input | None
        _agent_profile_status: Any = None  # Static | None
        _run_active_tier_warning_emitted: bool = False

        def compose(self) -> Any:
            yield Header()
            with Horizontal(id="panes"):
                yield RichLog(id="transcript")
                yield TextArea(
                    text="",
                    read_only=True,
                    show_line_numbers=False,
                    id="transcript_text",
                    soft_wrap=True,
                )
                with Vertical(id="side"):
                    with TabbedContent(id="side_tabs"):
                        with TabPane("Plan", id="tab_plan"):
                            yield TextArea(
                                text="",
                                language="markdown",
                                read_only=True,
                                show_cursor=False,
                                id="plan",
                                soft_wrap=True,
                            )
                            yield PlanActions(id="plan_actions")
                        with TabPane("Context", id="tab_context"):
                            with Vertical(id="context_panel"):
                                yield Static("Active Context Sources", id="context_header")
                                yield VerticalScroll(id="context_sources_list")
                                with Horizontal(id="context_add_row"):
                                    yield Button("File", id="context_add_file", compact=True, variant="default")
                                    yield Button("Note", id="context_add_note", compact=True, variant="default")
                                    yield Button("SOP", id="context_add_sop", compact=True, variant="default")
                                    yield Button("KB", id="context_add_kb", compact=True, variant="default")
                                with Horizontal(id="context_input_row"):
                                    yield Input(placeholder="Enter context value", id="context_input")
                                    yield Button("Add", id="context_add_commit", compact=True, variant="success")
                                    yield Button("Cancel", id="context_add_cancel", compact=True, variant="default")
                                with Horizontal(id="context_sop_row"):
                                    yield Select(
                                        options=[("Select SOP...", _CONTEXT_SELECT_PLACEHOLDER)],
                                        allow_blank=False,
                                        id="context_sop_select",
                                        compact=True,
                                    )
                                    yield Button("Add", id="context_sop_commit", compact=True, variant="success")
                                    yield Button("Cancel", id="context_sop_cancel", compact=True, variant="default")
                        with TabPane("SOPs", id="tab_sops"):
                            with Vertical(id="sops_panel"):
                                yield Static("Available SOPs", id="sops_header")
                                yield VerticalScroll(id="sop_list")
                        with TabPane("Artifacts", id="tab_artifacts"):
                            with Vertical(id="artifacts_panel"):
                                yield SidebarHeader("Artifacts", id="artifacts_header")
                                yield SidebarList(id="artifacts_list")
                                yield SidebarDetail(id="artifacts_detail")
                        with TabPane("Session", id="tab_session"):
                            with Vertical(id="session_panel"):
                                with Horizontal(id="session_view_switch"):
                                    yield Button(
                                        "Timeline",
                                        id="session_view_timeline",
                                        compact=True,
                                        variant="primary",
                                    )
                                    yield Button("Issues", id="session_view_issues", compact=True, variant="default")
                                with Vertical(id="session_timeline_view"):
                                    yield SidebarHeader("Timeline", id="session_timeline_header")
                                    yield SidebarList(id="session_timeline_list")
                                    yield SidebarDetail(id="session_timeline_detail")
                                with Vertical(id="session_issues_view"):
                                    yield SidebarHeader("Issues", id="session_issues_header")
                                    yield SidebarList(id="session_issue_list")
                                    yield SidebarDetail(id="session_issue_detail")
                        with TabPane("Agent", id="tab_agent"):
                            with Vertical(id="agent_panel"):
                                with Horizontal(id="agent_view_switch"):
                                    yield Button(
                                        "Profile",
                                        id="agent_view_profile",
                                        compact=True,
                                        variant="primary",
                                    )
                                    yield Button(
                                        "Tools & Safety",
                                        id="agent_view_tools",
                                        compact=True,
                                        variant="default",
                                    )
                                    yield Button("Team", id="agent_view_team", compact=True, variant="default")
                                with Vertical(id="agent_profile_view"):
                                    yield Static("Effective Session Profile", id="agent_summary_header")
                                    yield TextArea(
                                        text="",
                                        read_only=True,
                                        show_cursor=False,
                                        id="agent_summary",
                                        soft_wrap=True,
                                    )
                                    yield Static("Saved Profiles", id="agent_profiles_header")
                                    yield SidebarList(id="agent_profile_list")
                                    with Horizontal(id="agent_profile_meta_row"):
                                        yield Input(
                                            placeholder="Profile id",
                                            id="agent_profile_id",
                                        )
                                        yield Input(
                                            placeholder="Profile name",
                                            id="agent_profile_name",
                                        )
                                    yield AgentProfileActions(id="agent_profile_actions")
                                    yield Static("", id="agent_profile_status")
                                with Vertical(id="agent_tools_view"):
                                    yield SidebarHeader("Tools & Safety", id="agent_tools_header")
                                    yield SidebarList(id="agent_tools_list")
                                    yield SidebarDetail(id="agent_tools_detail")
                                    yield Static("Session Overrides", id="agent_tools_overrides_header")
                                    yield Input(
                                        placeholder="tool_consent: ask|allow|deny (blank = inherit)",
                                        id="agent_tools_override_consent",
                                    )
                                    yield Input(
                                        placeholder="tool_allowlist: comma-separated tools (blank = inherit)",
                                        id="agent_tools_override_allowlist",
                                    )
                                    yield Input(
                                        placeholder="tool_blocklist: comma-separated tools (blank = inherit)",
                                        id="agent_tools_override_blocklist",
                                    )
                                    with Horizontal(id="agent_tools_override_actions"):
                                        yield Button(
                                            "Apply",
                                            id="agent_tools_overrides_apply",
                                            compact=True,
                                            variant="success",
                                        )
                                        yield Button(
                                            "Reset",
                                            id="agent_tools_overrides_reset",
                                            compact=True,
                                            variant="default",
                                        )
                                    yield Static("", id="agent_tools_override_status")
                                with Vertical(id="agent_team_view"):
                                    yield SidebarHeader("Team Presets", id="agent_team_header")
                                    yield SidebarList(id="agent_team_list")
                                    yield SidebarDetail(id="agent_team_detail")
                                    yield Static("Preset Editor", id="agent_team_editor_header")
                                    with Horizontal(id="agent_team_meta_row"):
                                        yield Input(
                                            placeholder="Preset id",
                                            id="agent_team_preset_id",
                                        )
                                        yield Input(
                                            placeholder="Preset name",
                                            id="agent_team_preset_name",
                                        )
                                    yield Input(
                                        placeholder="Description (optional)",
                                        id="agent_team_preset_description",
                                    )
                                    yield TextArea(
                                        text="{}",
                                        language="json",
                                        id="agent_team_preset_spec",
                                        soft_wrap=True,
                                    )
                                    with Horizontal(id="agent_team_actions"):
                                        yield Button("New", id="agent_team_new", compact=True, variant="default")
                                        yield Button("Save", id="agent_team_save", compact=True, variant="success")
                                        yield Button("Delete", id="agent_team_delete", compact=True, variant="warning")
                                        yield Button(
                                            "Insert Run Prompt",
                                            id="agent_team_insert_prompt",
                                            compact=True,
                                            variant="primary",
                                        )
                                        yield Button(
                                            "Run Now",
                                            id="agent_team_run_now",
                                            compact=True,
                                            variant="default",
                                        )
                                    yield Static("", id="agent_team_status")
            yield CommandPalette(id="command_palette")
            yield ActionSheet(id="action_sheet")
            yield ThinkingBar(id="thinking_bar")
            yield StatusBar(id="status_bar")
            yield ErrorActionPrompt(id="error_action_prompt")
            yield ConsentPrompt(id="consent_prompt")
            with Vertical(id="prompt_box"):
                yield PromptTextArea(
                    text="",
                    placeholder="Type prompt. Enter submits, Shift+Enter inserts newline.",
                    id="prompt",
                    soft_wrap=True,
                )
                with Horizontal(id="prompt_bottom"):
                    yield ContextBudgetBar(id="prompt_metrics")
                    yield Select(
                        options=[("Loading model info...", _MODEL_LOADING_VALUE)],
                        allow_blank=False,
                        id="model_select",
                        compact=True,
                    )
            yield Footer()

        def on_mount(self) -> None:
            self._command_palette = self.query_one("#command_palette", CommandPalette)
            self._action_sheet = self.query_one("#action_sheet", ActionSheet)
            self._thinking_bar = self.query_one("#thinking_bar", ThinkingBar)
            self._status_bar = self.query_one("#status_bar", StatusBar)
            self._consent_prompt_widget = self.query_one("#consent_prompt", ConsentPrompt)
            self._error_action_prompt_widget = self.query_one("#error_action_prompt", ErrorActionPrompt)
            self._context_sources_list = self.query_one("#context_sources_list", VerticalScroll)
            self._sop_list = self.query_one("#sop_list", VerticalScroll)
            self._context_input = self.query_one("#context_input", Input)
            self._context_sop_select = self.query_one("#context_sop_select", Select)
            self._session_header = self.query_one("#session_issues_header", SidebarHeader)
            self._session_view_timeline_button = self.query_one("#session_view_timeline", Button)
            self._session_view_issues_button = self.query_one("#session_view_issues", Button)
            self._session_timeline_view = self.query_one("#session_timeline_view", Vertical)
            self._session_issues_view = self.query_one("#session_issues_view", Vertical)
            self._session_timeline_header = self.query_one("#session_timeline_header", SidebarHeader)
            self._session_timeline_list = self.query_one("#session_timeline_list", SidebarList)
            self._session_timeline_detail = self.query_one("#session_timeline_detail", SidebarDetail)
            self._session_issue_list = self.query_one("#session_issue_list", SidebarList)
            self._session_issue_detail = self.query_one("#session_issue_detail", SidebarDetail)
            self._artifacts_header = self.query_one("#artifacts_header", SidebarHeader)
            self._artifacts_list = self.query_one("#artifacts_list", SidebarList)
            self._artifacts_detail = self.query_one("#artifacts_detail", SidebarDetail)
            self._agent_view_profile_button = self.query_one("#agent_view_profile", Button)
            self._agent_view_tools_button = self.query_one("#agent_view_tools", Button)
            self._agent_view_team_button = self.query_one("#agent_view_team", Button)
            self._agent_profile_view = self.query_one("#agent_profile_view", Vertical)
            self._agent_tools_view = self.query_one("#agent_tools_view", Vertical)
            self._agent_team_view = self.query_one("#agent_team_view", Vertical)
            self._agent_summary = self.query_one("#agent_summary", TextArea)
            self._agent_profile_list = self.query_one("#agent_profile_list", SidebarList)
            self._agent_tools_header = self.query_one("#agent_tools_header", SidebarHeader)
            self._agent_tools_list = self.query_one("#agent_tools_list", SidebarList)
            self._agent_tools_detail = self.query_one("#agent_tools_detail", SidebarDetail)
            self._agent_tools_override_consent_input = self.query_one("#agent_tools_override_consent", Input)
            self._agent_tools_override_allowlist_input = self.query_one("#agent_tools_override_allowlist", Input)
            self._agent_tools_override_blocklist_input = self.query_one("#agent_tools_override_blocklist", Input)
            self._agent_tools_override_status = self.query_one("#agent_tools_override_status", Static)
            self._agent_team_header = self.query_one("#agent_team_header", SidebarHeader)
            self._agent_team_list = self.query_one("#agent_team_list", SidebarList)
            self._agent_team_detail = self.query_one("#agent_team_detail", SidebarDetail)
            self._agent_team_preset_id_input = self.query_one("#agent_team_preset_id", Input)
            self._agent_team_preset_name_input = self.query_one("#agent_team_preset_name", Input)
            self._agent_team_preset_description_input = self.query_one("#agent_team_preset_description", Input)
            self._agent_team_preset_spec_input = self.query_one("#agent_team_preset_spec", TextArea)
            self._agent_team_status = self.query_one("#agent_team_status", Static)
            self._agent_profile_id_input = self.query_one("#agent_profile_id", Input)
            self._agent_profile_name_input = self.query_one("#agent_profile_name", Input)
            self._agent_profile_status = self.query_one("#agent_profile_status", Static)
            self._prompt_metrics = self.query_one("#prompt_metrics", ContextBudgetBar)
            self._status_bar.set_model(self._current_model_summary())
            self.query_one("#prompt", PromptTextArea).focus()
            self._reset_plan_panel()
            self._reset_issues_panel()
            self._reset_session_timeline_panel()
            self._reset_artifacts_panel()
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            self._set_session_view_mode("timeline")
            self._set_context_add_mode(None)
            self._refresh_context_sop_options()
            self._render_context_sources_panel()
            self._refresh_sop_catalog()
            self._render_sop_panel()
            self._reload_saved_profiles()
            self._render_agent_tools_panel()
            self._render_agent_team_panel()
            self._set_agent_studio_view_mode("profile")
            if self._saved_profiles:
                self._load_profile_into_draft(self._saved_profiles[0])
            else:
                self._new_agent_profile_draft(announce=False)
            self._refresh_agent_summary()
            self._refresh_model_select()
            self.title = "Swarmee"
            self.sub_title = self._current_model_summary()
            self._update_prompt_placeholder()
            # Show ASCII art banner at the top of the transcript.
            # Write plain lines so selection/export keeps exact banner text.
            from swarmee_river.utils.welcome_utils import SWARMEE_BANNER
            for banner_line in SWARMEE_BANNER.strip().splitlines():
                self._mount_transcript_widget(banner_line, plain_text=banner_line)
            self._write_transcript("Starting Swarmee daemon...")
            self._write_transcript(self.sub_title)
            self._write_transcript("Tips: use /commands in the prompt and the Agent tab for profile actions.")
            transcript = self.query_one("#transcript", RichLog)
            with contextlib.suppress(Exception):
                transcript.auto_scroll = True
            with contextlib.suppress(Exception):
                transcript.max_lines = self._TRANSCRIPT_MAX_LINES
            self._set_transcript_mode("rich", notify=False)
            self._load_session()
            if self._daemon_session_id:
                self._schedule_session_timeline_refresh(delay=0.1)
            self._refresh_agent_summary()
            self._spawn_daemon()

        def _record_transcript_fallback(self, text: str) -> None:
            clean = sanitize_output_text(text).rstrip("\n")
            if not clean:
                return
            self._transcript_fallback_lines.extend(clean.splitlines())
            if len(self._transcript_fallback_lines) > self._TRANSCRIPT_MAX_LINES:
                self._transcript_fallback_lines = self._transcript_fallback_lines[-self._TRANSCRIPT_MAX_LINES :]

        def _sync_transcript_text_widget(self, *, scroll_to_end: bool = True) -> None:
            text_widget = self.query_one("#transcript_text", TextArea)
            text = "\n".join(self._transcript_fallback_lines).rstrip()
            if text:
                text += "\n"
            text_widget.load_text(text)
            if scroll_to_end:
                self._scroll_transcript_text_to_end()

        def _scroll_transcript_text_to_end(self) -> None:
            text_widget = self.query_one("#transcript_text", TextArea)
            with contextlib.suppress(Exception):
                text_widget.scroll_end(animate=False)
            for method_name in ("action_cursor_document_end", "action_end"):
                method = getattr(text_widget, method_name, None)
                if callable(method):
                    with contextlib.suppress(Exception):
                        method()
                        break

        def _get_scroll_proportion(self, widget: Any) -> float:
            """Get 0.0-1.0 proportion of current scroll position."""
            try:
                scroll_y = float(getattr(getattr(widget, "scroll_offset", None), "y", 0.0) or 0.0)
                virtual_h = float(getattr(getattr(widget, "virtual_size", None), "height", 0.0) or 0.0)
                viewport_h = float(getattr(getattr(widget, "size", None), "height", 0.0) or 0.0)
                max_scroll = virtual_h - viewport_h
                if max_scroll <= 0:
                    return 1.0
                return min(1.0, max(0.0, scroll_y / max_scroll))
            except Exception:
                return 1.0

        def _set_scroll_proportion(self, widget: Any, proportion: float) -> None:
            """Set scroll position from 0.0-1.0 proportion."""
            try:
                normalized = min(1.0, max(0.0, float(proportion)))
                virtual_h = float(getattr(getattr(widget, "virtual_size", None), "height", 0.0) or 0.0)
                viewport_h = float(getattr(getattr(widget, "size", None), "height", 0.0) or 0.0)
                max_scroll = virtual_h - viewport_h
                if max_scroll <= 0:
                    return
                target = int(normalized * max_scroll)
                scroll_to = getattr(widget, "scroll_to", None)
                if callable(scroll_to):
                    scroll_to(0, target, animate=False)
                    return
                scroll_relative = getattr(widget, "scroll_relative", None)
                if callable(scroll_relative):
                    current_y = float(getattr(getattr(widget, "scroll_offset", None), "y", 0.0) or 0.0)
                    scroll_relative(y=target - current_y, animate=False)
            except Exception:
                pass

        def _set_transcript_mode(self, mode: str, *, notify: bool = True) -> None:
            normalized = mode.strip().lower()
            if normalized not in {"rich", "text"}:
                return
            rich_widget = self.query_one("#transcript", RichLog)
            text_widget = self.query_one("#transcript_text", TextArea)
            if normalized == "text":
                proportion = self._get_scroll_proportion(rich_widget)
                at_bottom = proportion > 0.95
                self._sync_transcript_text_widget(scroll_to_end=at_bottom)
                rich_widget.styles.display = "none"
                text_widget.styles.display = "block"
                if at_bottom:
                    self._scroll_transcript_text_to_end()
                else:
                    self.set_timer(0.05, lambda p=proportion: self._set_scroll_proportion(text_widget, p))
                self._transcript_mode = "text"
                if notify:
                    self._notify("Text mode: select text with mouse. /text to return.", severity="information")
                return

            proportion = self._get_scroll_proportion(text_widget)
            at_bottom = proportion > 0.95
            text_widget.styles.display = "none"
            rich_widget.styles.display = "block"
            if at_bottom:
                with contextlib.suppress(Exception):
                    rich_widget.scroll_end(animate=False)
            else:
                self.set_timer(0.05, lambda p=proportion: self._set_scroll_proportion(rich_widget, p))
            self._transcript_mode = "rich"
            if notify:
                self._notify("Rich mode restored.", severity="information")

        def _toggle_transcript_mode(self) -> None:
            target = "text" if self._transcript_mode != "text" else "rich"
            self._set_transcript_mode(target, notify=True)

        def _mount_transcript_widget(self, renderable: Any, *, plain_text: str | None = None) -> None:
            """Write a renderable into the transcript RichLog."""
            transcript = self.query_one("#transcript", RichLog)
            transcript.write(renderable)
            if isinstance(plain_text, str):
                self._record_transcript_fallback(plain_text)
            elif isinstance(renderable, str):
                self._record_transcript_fallback(renderable)
            if self._transcript_mode == "text":
                self._sync_transcript_text_widget()
            with contextlib.suppress(Exception):
                transcript.scroll_end(animate=False)

        def _write_transcript(self, line: str) -> None:
            """Write a system/info message to the transcript."""
            self._mount_transcript_widget(render_system_message(line), plain_text=line)

        def _cancel_thinking_display_timer(self) -> None:
            timer = self._thinking_display_timer
            self._thinking_display_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _cancel_thinking_animation_timer(self) -> None:
            timer = self._thinking_animation_timer
            self._thinking_animation_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _thinking_elapsed_s(self) -> float:
            started = self._thinking_started_mono
            if not isinstance(started, float):
                return 0.0
            return max(0.0, time.monotonic() - started)

        def _thinking_preview(self) -> str:
            for chunk in reversed(self._thinking_buffer):
                text = sanitize_output_text(str(chunk or "")).strip()
                if text:
                    return text
            return ""

        def _render_thinking_bar(self) -> None:
            bar = self._thinking_bar
            if bar is None:
                return
            bar.show_thinking(
                char_count=max(0, int(self._thinking_char_count)),
                elapsed_s=self._thinking_elapsed_s(),
                preview=self._thinking_preview(),
                frame_index=self._thinking_frame_index,
            )

        def _on_thinking_display_timer(self) -> None:
            self._thinking_display_timer = None
            if self._current_thinking:
                self._render_thinking_bar()

        def _schedule_thinking_display_update(self) -> None:
            self._cancel_thinking_display_timer()
            self._thinking_display_timer = self.set_timer(_THINKING_DISPLAY_DEBOUNCE_S, self._on_thinking_display_timer)

        def _on_thinking_animation_tick(self) -> None:
            if not self._current_thinking:
                return
            self._thinking_frame_index = (self._thinking_frame_index + 1) % 3
            self._render_thinking_bar()

        def _ensure_thinking_animation_timer(self) -> None:
            if self._thinking_animation_timer is not None:
                return
            self._thinking_animation_timer = self.set_interval(
                _THINKING_ANIMATION_INTERVAL_S,
                self._on_thinking_animation_tick,
            )

        def _reset_thinking_state(self) -> None:
            self._cancel_thinking_display_timer()
            self._cancel_thinking_animation_timer()
            self._current_thinking = False
            self._thinking_buffer = []
            self._thinking_char_count = 0
            self._thinking_started_mono = None
            self._thinking_frame_index = 0
            bar = self._thinking_bar
            if bar is not None:
                with contextlib.suppress(Exception):
                    bar.hide_thinking()

        def _record_thinking_event(self, thinking_text: str) -> None:
            chunk = sanitize_output_text(str(thinking_text or ""))
            first_event = not self._current_thinking and not self._thinking_buffer and self._thinking_char_count == 0
            if first_event:
                self._current_thinking = True
                self._thinking_started_mono = time.monotonic()
                self._thinking_frame_index = 0
                self._ensure_thinking_animation_timer()
            else:
                self._current_thinking = True
            if chunk:
                self._thinking_buffer.append(chunk)
                self._thinking_char_count += len(chunk)
            self._render_thinking_bar()
            self._schedule_thinking_display_update()

        def _dismiss_thinking(self, *, emit_summary: bool = False) -> None:
            """Hide thinking indicator and optionally persist a transcript summary."""
            had_thinking = bool(
                self._current_thinking
                or self._thinking_started_mono is not None
                or self._thinking_char_count > 0
                or self._thinking_buffer
            )
            if not had_thinking:
                bar = self._thinking_bar
                if bar is not None:
                    with contextlib.suppress(Exception):
                        bar.hide_thinking()
                return

            elapsed_s = self._thinking_elapsed_s()
            full_thinking = "".join(self._thinking_buffer)
            self._last_thinking_text = full_thinking
            char_count = self._thinking_char_count

            if emit_summary:
                elapsed_label = max(0, int(round(elapsed_s)))
                summary_line = f"💭 Thought for {elapsed_label}s"
                if char_count > 0:
                    summary_line += f" ({char_count:,} chars)"
                self._mount_transcript_widget(render_system_message(summary_line), plain_text=summary_line)

            self._reset_thinking_state()

        def _cancel_streaming_flush_timer(self) -> None:
            timer = self._streaming_flush_timer
            self._streaming_flush_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _schedule_streaming_flush(self) -> None:
            if self._streaming_flush_timer is None:
                self._streaming_flush_timer = self.set_timer(
                    _STREAMING_FLUSH_INTERVAL_S,
                    self._on_streaming_flush_timer,
                )

        def _on_streaming_flush_timer(self) -> None:
            self._streaming_flush_timer = None
            self._flush_streaming_buffer()

        def _flush_streaming_buffer(self) -> None:
            if not self._streaming_buffer:
                return
            text = "".join(self._streaming_buffer)
            self._streaming_buffer = []
            if not text:
                return
            self._current_assistant_chunks.append(text)

        def _cancel_tool_progress_flush_timer(self) -> None:
            timer = self._tool_progress_flush_timer
            self._tool_progress_flush_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _schedule_tool_progress_flush(self, tool_use_id: str | None = None) -> None:
            if isinstance(tool_use_id, str) and tool_use_id.strip():
                self._tool_progress_pending_ids.add(tool_use_id.strip())
            if self._tool_progress_flush_timer is None:
                self._tool_progress_flush_timer = self.set_timer(
                    _STREAMING_FLUSH_INTERVAL_S,
                    self._on_tool_progress_flush_timer,
                )

        def _on_tool_progress_flush_timer(self) -> None:
            self._tool_progress_flush_timer = None
            pending_ids = list(self._tool_progress_pending_ids)
            self._tool_progress_pending_ids.clear()
            for tool_use_id in pending_ids:
                rendered = self._flush_tool_progress_render(tool_use_id)
                record = self._tool_blocks.get(tool_use_id)
                has_pending = bool(record and str(record.get("pending_output", "")))
                if has_pending and not rendered:
                    self._tool_progress_pending_ids.add(tool_use_id)
            if self._tool_progress_pending_ids:
                self._schedule_tool_progress_flush()

        def _flush_all_streaming_buffers(self) -> None:
            self._cancel_streaming_flush_timer()
            self._flush_streaming_buffer()
            self._cancel_tool_progress_flush_timer()
            self._tool_progress_pending_ids.clear()
            for tool_use_id in list(self._tool_blocks.keys()):
                self._flush_tool_progress_render(tool_use_id, force=True)

        def _write_transcript_line(self, line: str) -> None:
            """Write a plain text line to the transcript (used for TUI-internal messages)."""
            if self._query_active:
                self._turn_output_chunks.append(sanitize_output_text(f"[tui] {line}\n"))
            self._write_transcript(line)

        def _tool_input_summary(self, tool_name: str, tool_input: Any) -> str:
            if not isinstance(tool_input, dict):
                return ""
            return format_tool_input_oneliner(tool_name, tool_input)

        def _tool_start_plain_text(self, tool_name: str, tool_input: Any) -> str:
            summary = self._tool_input_summary(tool_name, tool_input)
            if summary:
                return f"⚙ {tool_name} — {summary} ..."
            return f"⚙ {tool_name} ..."

        def _tool_result_plain_text(self, tool_name: str, status: str, duration_s: float, tool_input: Any) -> str:
            succeeded = status == "success"
            glyph = "✓" if succeeded else "✗"
            summary = self._tool_input_summary(tool_name, tool_input)
            base = f"{glyph} {tool_name} ({duration_s:.1f}s)"
            if summary:
                base = f"{base} — {summary}"
            if not succeeded:
                label = (status or "error").strip()
                base = f"{base} ({label})"
            return base

        def _cancel_tool_start_timer(self, tool_use_id: str) -> None:
            timer = self._tool_pending_start_timers.pop(tool_use_id, None)
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _clear_pending_tool_starts(self) -> None:
            for tool_use_id in list(self._tool_pending_start_timers.keys()):
                self._cancel_tool_start_timer(tool_use_id)
            self._tool_pending_start_timers = {}
            self._tool_pending_start = {}

        def _emit_tool_start_line(self, tool_use_id: str) -> bool:
            record = self._tool_blocks.get(tool_use_id)
            if record is None:
                self._tool_pending_start.pop(tool_use_id, None)
                self._cancel_tool_start_timer(tool_use_id)
                return False
            if bool(record.get("start_rendered")):
                self._tool_pending_start.pop(tool_use_id, None)
                self._cancel_tool_start_timer(tool_use_id)
                return False
            tool_name = str(record.get("tool", "unknown"))
            tool_input = record.get("input")
            self._mount_transcript_widget(
                render_tool_start_line_with_input(tool_name, tool_input=tool_input, tool_use_id=tool_use_id),
                plain_text=self._tool_start_plain_text(tool_name, tool_input),
            )
            record["start_rendered"] = True
            self._tool_pending_start.pop(tool_use_id, None)
            self._cancel_tool_start_timer(tool_use_id)
            return True

        def _on_tool_start_coalesce_timer(self, tool_use_id: str) -> None:
            self._tool_pending_start_timers.pop(tool_use_id, None)
            self._emit_tool_start_line(tool_use_id)

        def _schedule_tool_start_line(self, tool_use_id: str) -> None:
            if not tool_use_id:
                return
            self._tool_pending_start[tool_use_id] = time.monotonic()
            self._cancel_tool_start_timer(tool_use_id)
            self._tool_pending_start_timers[tool_use_id] = self.set_timer(
                _TOOL_START_COALESCE_INTERVAL_S,
                lambda tid=tool_use_id: self._on_tool_start_coalesce_timer(tid),
            )

        def _append_tool_output(self, record: dict[str, Any], chunk: str) -> None:
            text = sanitize_output_text(str(chunk or ""))
            if not text:
                return
            output = str(record.get("output", "")) + text
            if len(output) > _TOOL_OUTPUT_RETENTION_MAX_CHARS:
                output = output[-_TOOL_OUTPUT_RETENTION_MAX_CHARS:]
            record["output"] = output

        def _queue_tool_progress_content(self, record: dict[str, Any], *, content: str, stream: str) -> None:
            text = sanitize_output_text(str(content or ""))
            if not text:
                return
            self._append_tool_output(record, text)
            pending = str(record.get("pending_output", ""))
            pending_stream = str(record.get("pending_stream", "stdout") or "stdout")
            normalized_stream = stream if stream in {"stdout", "stderr", "mixed"} else "stdout"
            chunk = text
            if pending:
                if pending_stream != normalized_stream:
                    pending_stream = "mixed"
                    chunk = f"[{normalized_stream}] {text}"
            else:
                pending_stream = normalized_stream
            pending += chunk
            if len(pending) > _TOOL_OUTPUT_RETENTION_MAX_CHARS:
                pending = pending[-_TOOL_OUTPUT_RETENTION_MAX_CHARS:]
            record["pending_output"] = pending
            record["pending_stream"] = pending_stream

        def _flush_tool_progress_render(self, tool_use_id: str, *, force: bool = False) -> bool:
            record = self._tool_blocks.get(tool_use_id)
            if record is None:
                return False
            now = time.monotonic()
            last = float(record.get("last_progress_render_mono", 0.0))
            pending = str(record.get("pending_output", ""))
            if pending:
                if force or (now - last) >= _TOOL_PROGRESS_RENDER_INTERVAL_S:
                    stream = str(record.get("pending_stream", "stdout") or "stdout")
                    self._mount_transcript_widget(
                        render_tool_progress_chunk(pending, stream=stream),
                        plain_text=pending,
                    )
                    record["pending_output"] = ""
                    record["pending_stream"] = "stdout"
                    record["last_progress_render_mono"] = now
                    return True
                return False

            elapsed = record.get("elapsed_s")
            if force or not isinstance(elapsed, (int, float)):
                return False
            elapsed_s = float(elapsed)
            previous = float(record.get("last_heartbeat_rendered_s", 0.0))
            if (elapsed_s - previous) < _TOOL_HEARTBEAT_RENDER_MIN_STEP_S:
                return False
            if (now - last) < _TOOL_PROGRESS_RENDER_INTERVAL_S:
                return False
            tool_name = str(record.get("tool", "unknown"))
            self._mount_transcript_widget(
                render_tool_heartbeat_line(tool_name, elapsed_s=elapsed_s, tool_use_id=tool_use_id),
                plain_text=f"⚙ {tool_name} running... ({elapsed_s:.1f}s)",
            )
            record["last_progress_render_mono"] = now
            record["last_heartbeat_rendered_s"] = elapsed_s
            return True

        def _call_from_thread_safe(self, callback: Any, *args: Any, **kwargs: Any) -> None:
            if self._is_shutting_down:
                return
            with contextlib.suppress(Exception):
                self.call_from_thread(callback, *args, **kwargs)

        def _warn_run_active_tier_change_once(self) -> None:
            if self._run_active_tier_warning_emitted:
                return
            self._run_active_tier_warning_emitted = True
            self._write_transcript_line(_RUN_ACTIVE_TIER_WARNING)

        def _write_user_input(self, text: str) -> None:
            timestamp = self._turn_timestamp()
            plain = f"YOU> {text}\n{timestamp}"
            self._mount_transcript_widget(render_user_message(text, timestamp=timestamp), plain_text=plain)

        def _write_user_message(self, text: str, *, timestamp: str | None = None) -> None:
            resolved_timestamp = (timestamp or "").strip() or self._turn_timestamp()
            plain = f"YOU> {text}\n{resolved_timestamp}"
            self._mount_transcript_widget(render_user_message(text, timestamp=resolved_timestamp), plain_text=plain)

        def _write_assistant_message(
            self,
            text: str,
            *,
            model: str | None = None,
            timestamp: str | None = None,
        ) -> None:
            resolved_timestamp = (timestamp or "").strip() or self._turn_timestamp()
            self._last_assistant_text = text
            plain_lines = [text]
            meta_parts = [part for part in [model, resolved_timestamp] if isinstance(part, str) and part.strip()]
            if meta_parts:
                plain_lines.append(" · ".join(meta_parts))
            self._mount_transcript_widget(
                render_assistant_message(text, model=model, timestamp=resolved_timestamp),
                plain_text="\n".join(plain_lines),
            )

        def _turn_timestamp(self) -> str:
            return datetime.now().strftime("%I:%M %p").lstrip("0")

        def _append_plain_text(self, text: str) -> None:
            """Write a plain text line for non-event fallback output."""
            if not text.strip():
                return
            self._mount_transcript_widget(text, plain_text=text)

        def _set_plan_panel(self, content: str) -> None:
            self._plan_text = content
            plan_panel = self.query_one("#plan", TextArea)
            text = content if content.strip() else "(no plan)"
            plan_panel.load_text(text)
            plan_panel.scroll_end(animate=False)

        def _extract_plan_step_descriptions(self, plan_json: dict[str, Any]) -> list[str]:
            steps_raw = plan_json.get("steps", [])
            if not isinstance(steps_raw, list):
                return []
            steps: list[str] = []
            for step in steps_raw:
                if isinstance(step, str):
                    desc = step.strip()
                elif isinstance(step, dict):
                    desc = str(step.get("description", step.get("title", step))).strip()
                else:
                    desc = str(step).strip()
                if desc:
                    steps.append(desc)
            return steps

        def _refresh_plan_status_bar(self) -> None:
            if self._status_bar is None:
                return
            if not self._query_active:
                self._status_bar.set_plan_step(current=None, total=None)
                return
            total = self._current_plan_steps_total
            if total <= 0:
                self._status_bar.set_plan_step(current=None, total=None)
                return
            current: int | None = None
            if isinstance(self._current_plan_active_step, int) and self._current_plan_active_step >= 0:
                current = self._current_plan_active_step + 1
            else:
                completed = sum(1 for item in self._current_plan_step_statuses if item == "completed")
                if completed >= total:
                    current = total
                elif completed > 0:
                    current = completed
            self._status_bar.set_plan_step(current=current, total=total)

        def _render_plan_panel_from_status(self) -> None:
            if self._current_plan_steps_total <= 0 or not self._current_plan_steps:
                return
            text_lines: list[str] = []
            if self._current_plan_summary:
                text_lines.append(self._current_plan_summary)
                text_lines.append("")
            for index, desc in enumerate(self._current_plan_steps, start=1):
                status = (
                    self._current_plan_step_statuses[index - 1]
                    if index - 1 < len(self._current_plan_step_statuses)
                    else "pending"
                )
                marker = "☐"
                if status == "in_progress":
                    marker = "▶"
                elif status == "completed":
                    marker = "☑"
                text_lines.append(f"{marker} {index}. {desc}")
            text_lines.append("")
            text_lines.append("/approve  /replan  /clearplan")
            self._set_plan_panel("\n".join(text_lines))
            self._refresh_plan_status_bar()

        def _selected_agent_profile_id(self) -> str | None:
            sidebar_list = self._agent_profile_list
            if sidebar_list is None:
                return None
            getter = getattr(sidebar_list, "selected_id", None)
            if not callable(getter):
                return None
            value = str(getter() or "").strip()
            if not value or value == _AGENT_PROFILE_SELECT_NONE:
                return None
            return value

        def _lookup_saved_profile(self, profile_id: str | None) -> AgentProfile | None:
            target = str(profile_id or "").strip()
            if not target:
                return None
            for profile in self._saved_profiles:
                if profile.id == target:
                    return profile
            return None

        def _set_agent_form_values(self, *, profile_id: str, profile_name: str) -> None:
            self._agent_form_syncing = True
            try:
                if self._agent_profile_id_input is not None:
                    self._agent_profile_id_input.value = profile_id
                if self._agent_profile_name_input is not None:
                    self._agent_profile_name_input.value = profile_name
            finally:
                self._agent_form_syncing = False

        def _agent_tools_item_by_id(self, item_id: str | None) -> dict[str, Any] | None:
            target = str(item_id or "").strip()
            if not target:
                return None
            for item in self._agent_tools_items:
                if str(item.get("id", "")).strip() == target:
                    return item
            return None

        def _set_agent_tools_status(self, message: str) -> None:
            widget = self._agent_tools_override_status
            if widget is None:
                return
            text = message.strip() if isinstance(message, str) else ""
            widget.update(text)

        def _set_agent_tools_override_form_values(self, overrides: dict[str, Any] | None) -> None:
            normalized = normalize_session_safety_overrides(overrides)
            consent = str(normalized.get("tool_consent", "")).strip().lower()
            allow = _normalized_tool_name_list(normalized.get("tool_allowlist"))
            block = _normalized_tool_name_list(normalized.get("tool_blocklist"))
            self._agent_tools_form_syncing = True
            try:
                if self._agent_tools_override_consent_input is not None:
                    self._agent_tools_override_consent_input.value = consent
                if self._agent_tools_override_allowlist_input is not None:
                    self._agent_tools_override_allowlist_input.value = ", ".join(allow)
                if self._agent_tools_override_blocklist_input is not None:
                    self._agent_tools_override_blocklist_input.value = ", ".join(block)
            finally:
                self._agent_tools_form_syncing = False

        def _parse_agent_tools_csv_list(self, value: str) -> list[str]:
            if not value.strip():
                return []
            tokens = [token.strip() for token in value.split(",")]
            return _normalized_tool_name_list(tokens)

        def _agent_tools_form_update_payload(self) -> dict[str, Any] | None:
            raw_consent = str(getattr(self._agent_tools_override_consent_input, "value", "")).strip().lower()
            if raw_consent and raw_consent not in _AGENT_TOOL_CONSENT_VALUES:
                self._notify("tool_consent must be ask|allow|deny.", severity="warning")
                return None
            allow_csv = str(getattr(self._agent_tools_override_allowlist_input, "value", ""))
            block_csv = str(getattr(self._agent_tools_override_blocklist_input, "value", ""))
            allow_list = self._parse_agent_tools_csv_list(allow_csv)
            block_list = self._parse_agent_tools_csv_list(block_csv)
            return {
                "tool_consent": raw_consent or None,
                "tool_allowlist": allow_list or None,
                "tool_blocklist": block_list or None,
            }

        def _current_agent_policy_tier_name(self) -> str | None:
            return (
                str(self._daemon_tier or "").strip().lower()
                or str(self._model_tier_override or "").strip().lower()
                or None
            )

        def _refresh_agent_tools_policy_lens(self) -> None:
            self._agent_tools_policy_lens = build_agent_policy_lens(
                tier_name=self._current_agent_policy_tier_name(),
                overrides=self._session_safety_overrides,
            )

        def _apply_agent_tools_safety_overrides(self, *, reset: bool = False) -> None:
            proc = self._proc
            if self._query_active:
                self._set_agent_tools_status("Cannot update overrides while a run is active.")
                self._notify("Cannot update overrides while a run is active.", severity="warning")
                return
            if not self._daemon_ready or proc is None or proc.poll() is not None:
                self._set_agent_tools_status("Daemon is not ready.")
                self._notify("Daemon is not ready.", severity="warning")
                return
            payload = (
                {"tool_consent": None, "tool_allowlist": None, "tool_blocklist": None}
                if reset
                else self._agent_tools_form_update_payload()
            )
            if payload is None:
                return
            command: dict[str, Any] = {"cmd": "set_safety_overrides"}
            command.update(payload)
            if not send_daemon_command(proc, command):
                self._set_agent_tools_status("Failed to send override update.")
                self._notify("Failed to send safety override update.", severity="warning")
                return
            if reset:
                self._set_agent_tools_status("Resetting overrides...")
            else:
                self._set_agent_tools_status("Applying overrides...")

        def _set_agent_tools_selection(self, item: dict[str, Any] | None) -> None:
            detail = self._agent_tools_detail
            if detail is None:
                return
            if item is None:
                self._agent_tools_selected_item_id = None
                detail.set_preview("(no tools/safety items)")
                detail.set_actions([])
                return
            self._agent_tools_selected_item_id = str(item.get("id", "")).strip() or None
            detail.set_preview(render_agent_tools_safety_detail_text(item, self._agent_tools_policy_lens))
            detail.set_actions([])

        def _agent_team_item_by_id(self, item_id: str | None) -> dict[str, Any] | None:
            target = str(item_id or "").strip()
            if not target:
                return None
            for item in self._agent_team_items:
                if str(item.get("id", "")).strip() == target:
                    return item
            return None

        def _set_agent_team_status(self, message: str) -> None:
            widget = self._agent_team_status
            if widget is None:
                return
            text = message.strip() if isinstance(message, str) else ""
            widget.update(text)

        def _set_agent_team_form_values(self, preset: dict[str, Any] | None) -> None:
            normalized = normalize_team_preset(preset) if isinstance(preset, dict) else None
            preset_id = str(normalized.get("id", "")).strip() if normalized else ""
            name = str(normalized.get("name", "")).strip() if normalized else ""
            description = str(normalized.get("description", "")).strip() if normalized else ""
            spec_text = (
                _json.dumps(normalized.get("spec", {}), ensure_ascii=False, indent=2, sort_keys=True)
                if normalized
                else "{}"
            )
            self._agent_team_form_syncing = True
            try:
                if self._agent_team_preset_id_input is not None:
                    self._agent_team_preset_id_input.value = preset_id
                if self._agent_team_preset_name_input is not None:
                    self._agent_team_preset_name_input.value = name
                if self._agent_team_preset_description_input is not None:
                    self._agent_team_preset_description_input.value = description
                if self._agent_team_preset_spec_input is not None:
                    self._agent_team_preset_spec_input.load_text(spec_text)
            finally:
                self._agent_team_form_syncing = False

        def _agent_team_form_payload(self) -> dict[str, Any] | None:
            raw_name = str(getattr(self._agent_team_preset_name_input, "value", "")).strip()
            if not raw_name:
                self._notify("Preset name is required.", severity="warning")
                return None
            raw_id = str(getattr(self._agent_team_preset_id_input, "value", "")).strip()
            preset_id = _sanitize_profile_token(raw_id or raw_name)
            description = str(getattr(self._agent_team_preset_description_input, "value", "")).strip()
            spec_text = str(getattr(self._agent_team_preset_spec_input, "text", "")).strip() or "{}"
            try:
                spec = _json.loads(spec_text)
            except Exception:
                self._notify("Preset spec must be valid JSON.", severity="warning")
                return None
            if not isinstance(spec, dict):
                self._notify("Preset spec must be a JSON object.", severity="warning")
                return None
            normalized = normalize_team_preset(
                {"id": preset_id, "name": raw_name, "description": description, "spec": spec}
            )
            if normalized is None:
                self._notify("Preset is invalid.", severity="warning")
                return None
            return normalized

        def _selected_team_preset(self) -> dict[str, Any] | None:
            selected_item = self._agent_team_item_by_id(self._agent_team_selected_item_id)
            if selected_item is None:
                return None
            return normalize_team_preset(selected_item.get("preset"))

        def _set_prompt_editor_text(self, text: str) -> bool:
            content = str(text or "")
            if not content.strip():
                return False
            with contextlib.suppress(Exception):
                prompt_widget = self.query_one("#prompt", PromptTextArea)
                prompt_widget.clear()
                loader = getattr(prompt_widget, "load_text", None)
                if callable(loader):
                    loader(content)
                else:
                    prompt_widget.insert(content)
                prompt_widget.focus()
                return True
            return False

        def _new_agent_team_preset_draft(self) -> None:
            seed = normalize_team_preset(
                {
                    "id": _default_team_preset_id(),
                    "name": _default_team_preset_name(),
                    "description": "",
                    "spec": {},
                }
            )
            if seed is None:
                return
            self._agent_team_selected_item_id = None
            self._set_agent_team_form_values(seed)
            self._set_agent_team_status("New team preset draft.")
            self._set_agent_draft_dirty(True, note="Team preset draft updated.")

        def _save_agent_team_preset_draft(self) -> None:
            payload = self._agent_team_form_payload()
            if payload is None:
                return
            selected_id = str(self._agent_team_selected_item_id or "").strip()
            saved_id = str(payload.get("id", "")).strip()
            next_presets = [dict(item) for item in normalize_team_presets(self._agent_team_presets)]
            if selected_id and selected_id != saved_id:
                next_presets = [item for item in next_presets if str(item.get("id", "")).strip() != selected_id]
            replaced = False
            for idx, existing in enumerate(next_presets):
                if str(existing.get("id", "")).strip() == saved_id:
                    next_presets[idx] = payload
                    replaced = True
                    break
            if not replaced:
                next_presets.append(payload)
            self._agent_team_presets = normalize_team_presets(next_presets)
            self._agent_team_selected_item_id = saved_id
            self._render_agent_team_panel()
            self._set_agent_team_status(f"Saved preset '{payload['name']}' in draft.")
            self._set_agent_draft_dirty(True, note=f"Team preset '{payload['name']}' saved in draft.")

        def _delete_selected_agent_team_preset(self) -> None:
            selected = self._selected_team_preset()
            if selected is None:
                self._notify("Select a team preset first.", severity="warning")
                return
            selected_id = str(selected.get("id", "")).strip()
            next_presets = [
                item
                for item in self._agent_team_presets
                if str(item.get("id", "")).strip() != selected_id
            ]
            self._agent_team_presets = normalize_team_presets(next_presets)
            self._agent_team_selected_item_id = None
            self._render_agent_team_panel()
            self._set_agent_team_status(f"Deleted preset '{selected.get('name', selected_id)}' from draft.")
            self._set_agent_draft_dirty(True, note=f"Team preset '{selected_id}' removed from draft.")

        def _insert_agent_team_preset_run_prompt(self, *, run_now: bool = False) -> None:
            preset = self._selected_team_preset()
            if preset is None:
                preset = self._agent_team_form_payload()
            if preset is None:
                return
            prompt = build_team_preset_run_prompt(preset)
            if not prompt:
                self._notify("Failed to build team preset prompt.", severity="warning")
                return
            if not self._set_prompt_editor_text(prompt):
                self._notify("Prompt editor is unavailable.", severity="warning")
                return
            self._set_agent_team_status(f"Inserted run prompt for '{preset['name']}'.")
            if run_now:
                self.action_submit_prompt()

        def _set_agent_team_selection(self, item: dict[str, Any] | None) -> None:
            detail = self._agent_team_detail
            if detail is None:
                return
            if item is None:
                self._agent_team_selected_item_id = None
                detail.set_preview("(no team items)")
                detail.set_actions([])
                self._set_agent_team_form_values(None)
                return
            self._agent_team_selected_item_id = str(item.get("id", "")).strip() or None
            detail.set_preview(render_agent_team_detail_text(item))
            detail.set_actions([])
            preset = normalize_team_preset(item.get("preset"))
            self._set_agent_team_form_values(preset)

        def _refresh_agent_tools_header(self) -> None:
            header = self._agent_tools_header
            if header is None:
                return
            effective = (
                self._agent_tools_policy_lens.get("effective", {})
                if isinstance(self._agent_tools_policy_lens.get("effective"), dict)
                else {}
            )
            overrides = (
                self._agent_tools_policy_lens.get("session_overrides", {})
                if isinstance(self._agent_tools_policy_lens.get("session_overrides"), dict)
                else {}
            )
            consent = str(effective.get("tool_consent", "ask")).strip().lower() or "ask"
            header.set_badges([f"consent {consent}", f"overrides {len(overrides)}"])

        def _refresh_agent_team_header(self) -> None:
            header = self._agent_team_header
            if header is None:
                return
            header.set_badges([f"presets {len(self._agent_team_presets)}"])

        def _render_agent_tools_panel(self) -> None:
            self._refresh_agent_tools_policy_lens()
            self._agent_tools_items = [
                dict(item) for item in build_agent_tools_safety_sidebar_items(self._agent_tools_policy_lens)
            ]
            list_widget = self._agent_tools_list
            if list_widget is not None:
                selected_id = self._agent_tools_selected_item_id
                if not selected_id and self._agent_tools_items:
                    selected_id = str(self._agent_tools_items[0].get("id", "")).strip()
                list_widget.set_items(self._agent_tools_items, selected_id=selected_id, emit=False)
                selected_id = list_widget.selected_id()
                selected_item = self._agent_tools_item_by_id(selected_id)
                if selected_item is None and self._agent_tools_items:
                    selected_item = self._agent_tools_items[0]
                    with contextlib.suppress(Exception):
                        list_widget.select_by_id(str(selected_item.get("id", "")), emit=False)
                self._set_agent_tools_selection(selected_item)
            else:
                self._set_agent_tools_selection(self._agent_tools_items[0] if self._agent_tools_items else None)
            self._refresh_agent_tools_header()
            self._set_agent_tools_override_form_values(self._session_safety_overrides)

        def _render_agent_team_panel(self) -> None:
            self._agent_team_presets = normalize_team_presets(self._agent_team_presets)
            self._agent_team_items = [dict(item) for item in build_agent_team_sidebar_items(self._agent_team_presets)]
            list_widget = self._agent_team_list
            if list_widget is not None:
                selected_id = self._agent_team_selected_item_id
                if not selected_id and self._agent_team_items:
                    selected_id = str(self._agent_team_items[0].get("id", "")).strip()
                list_widget.set_items(self._agent_team_items, selected_id=selected_id, emit=False)
                selected_id = list_widget.selected_id()
                selected_item = self._agent_team_item_by_id(selected_id)
                if selected_item is None and self._agent_team_items:
                    selected_item = self._agent_team_items[0]
                    with contextlib.suppress(Exception):
                        list_widget.select_by_id(str(selected_item.get("id", "")), emit=False)
                self._set_agent_team_selection(selected_item)
            else:
                self._set_agent_team_selection(self._agent_team_items[0] if self._agent_team_items else None)
            self._refresh_agent_team_header()

        def _set_agent_studio_view_mode(self, mode: str) -> None:
            normalized = normalize_agent_studio_view_mode(mode)
            self._agent_studio_view_mode = normalized

            profile_view = self._agent_profile_view
            tools_view = self._agent_tools_view
            team_view = self._agent_team_view
            if profile_view is not None:
                profile_view.styles.display = "block" if normalized == "profile" else "none"
            if tools_view is not None:
                tools_view.styles.display = "block" if normalized == "tools" else "none"
            if team_view is not None:
                team_view.styles.display = "block" if normalized == "team" else "none"

            profile_button = self._agent_view_profile_button
            tools_button = self._agent_view_tools_button
            team_button = self._agent_view_team_button
            if profile_button is not None:
                profile_button.variant = "primary" if normalized == "profile" else "default"
            if tools_button is not None:
                tools_button.variant = "primary" if normalized == "tools" else "default"
            if team_button is not None:
                team_button.variant = "primary" if normalized == "team" else "default"

        def _kb_id_from_context_sources(self) -> str | None:
            for source in self._context_sources:
                source_type = str(source.get("type", "")).strip().lower()
                if source_type != "kb":
                    continue
                kb_id = str(source.get("id", "")).strip()
                if kb_id:
                    return kb_id
            return None

        def _session_effective_profile(self) -> AgentProfile:
            provider_name, tier_name, _model_id = choose_model_summary_parts(
                daemon_provider=self._daemon_provider,
                daemon_tier=self._daemon_tier,
                daemon_model_id=self._daemon_model_id,
                daemon_tiers=self._daemon_tiers,
                pending_value=self._pending_model_select_value,
                override_provider=self._model_provider_override,
                override_tier=self._model_tier_override,
            )
            current = self._effective_profile
            return AgentProfile(
                id=(current.id if current is not None else "session-effective"),
                name=(current.name if current is not None else "Session Effective"),
                provider=provider_name,
                tier=tier_name,
                system_prompt_snippets=(list(current.system_prompt_snippets) if current is not None else []),
                context_sources=self._context_sources_payload(),
                active_sops=sorted(self._active_sop_names),
                knowledge_base_id=(
                    self._kb_id_from_context_sources() or (current.knowledge_base_id if current else None)
                ),
                team_presets=(normalize_team_presets(current.team_presets) if current is not None else []),
            )

        def _set_agent_status(self, message: str) -> None:
            widget = self._agent_profile_status
            if widget is None:
                return
            text = message.strip() if isinstance(message, str) else ""
            widget.update(text)

        def _set_agent_draft_dirty(self, dirty: bool, *, note: str | None = None) -> None:
            self._agent_draft_dirty = bool(dirty)
            if self._agent_draft_dirty:
                base = "Draft changes pending."
            else:
                base = "Draft synced."
            if isinstance(note, str) and note.strip():
                self._set_agent_status(f"{base} {note.strip()}")
            else:
                self._set_agent_status(base)
            if self._agent_profile_list is not None:
                self._reload_saved_profiles()

        def _reload_saved_profiles(self, *, selected_id: str | None = None) -> None:
            self._saved_profiles = sorted(
                list_profiles(),
                key=lambda item: (item.name.lower(), item.id.lower()),
            )
            sidebar_list = self._agent_profile_list
            if sidebar_list is None:
                return

            items: list[dict[str, str]] = [
                {
                    "id": _AGENT_PROFILE_SELECT_NONE,
                    "title": "Draft / Session",
                    "subtitle": "Unsaved local draft",
                    "state": "syncing" if self._agent_draft_dirty else "default",
                }
            ]
            for profile in self._saved_profiles:
                profile_subtitle_parts = [profile.id]
                model_summary = "/".join(
                    part for part in [str(profile.provider or "").strip(), str(profile.tier or "").strip()] if part
                )
                if model_summary:
                    profile_subtitle_parts.append(model_summary)
                items.append(
                    {
                        "id": profile.id,
                        "title": profile.name,
                        "subtitle": " | ".join(profile_subtitle_parts),
                        "state": (
                            "active"
                            if self._effective_profile and self._effective_profile.id == profile.id
                            else "default"
                        ),
                    }
                )

            getter = getattr(sidebar_list, "selected_id", None)
            current_value = str(getter() or "").strip() if callable(getter) else ""
            candidate = selected_id if selected_id else current_value
            saved_ids = {profile.id for profile in self._saved_profiles}
            if candidate == _AGENT_PROFILE_SELECT_NONE:
                pass
            elif candidate not in saved_ids:
                candidate = self._saved_profiles[0].id if self._saved_profiles else _AGENT_PROFILE_SELECT_NONE

            self._agent_form_syncing = True
            try:
                setter = getattr(sidebar_list, "set_items", None)
                if callable(setter):
                    setter(items, selected_id=candidate, emit=False)
            finally:
                self._agent_form_syncing = False

        def _refresh_agent_summary(self) -> None:
            summary = self._agent_summary
            if summary is None:
                return
            effective = self._session_effective_profile()
            self._effective_profile = effective
            summary.load_text(render_agent_profile_summary_text(effective.to_dict()))
            summary.scroll_home(animate=False)

        def _new_agent_profile_draft(self, *, announce: bool = True) -> None:
            snapshot = self._session_effective_profile()
            draft = AgentProfile(
                id=_default_profile_id(),
                name=_default_profile_name(),
                provider=snapshot.provider,
                tier=snapshot.tier,
                system_prompt_snippets=list(snapshot.system_prompt_snippets),
                context_sources=[dict(source) for source in snapshot.context_sources],
                active_sops=list(snapshot.active_sops),
                knowledge_base_id=snapshot.knowledge_base_id,
                team_presets=normalize_team_presets(snapshot.team_presets),
            )
            sidebar_list = self._agent_profile_list
            if sidebar_list is not None:
                self._agent_form_syncing = True
                try:
                    select_by_id = getattr(sidebar_list, "select_by_id", None)
                    if callable(select_by_id):
                        select_by_id(_AGENT_PROFILE_SELECT_NONE, emit=False)
                finally:
                    self._agent_form_syncing = False
            self._agent_team_presets = normalize_team_presets(draft.team_presets)
            self._agent_team_selected_item_id = None
            self._render_agent_team_panel()
            self._set_agent_form_values(profile_id=draft.id, profile_name=draft.name)
            self._set_agent_draft_dirty(True, note=("New profile draft." if announce else None))

        def _load_profile_into_draft(self, profile: AgentProfile) -> None:
            sidebar_list = self._agent_profile_list
            if sidebar_list is not None:
                self._agent_form_syncing = True
                try:
                    select_by_id = getattr(sidebar_list, "select_by_id", None)
                    if callable(select_by_id):
                        select_by_id(profile.id, emit=False)
                finally:
                    self._agent_form_syncing = False
            self._agent_team_presets = normalize_team_presets(profile.team_presets)
            self._agent_team_selected_item_id = None
            self._render_agent_team_panel()
            self._set_agent_form_values(profile_id=profile.id, profile_name=profile.name)
            self._set_agent_draft_dirty(False, note=f"Loaded profile '{profile.name}'.")

        def _profile_from_draft(self) -> AgentProfile:
            selected_profile = self._lookup_saved_profile(self._selected_agent_profile_id())
            seed = selected_profile if selected_profile is not None else self._session_effective_profile()

            raw_id = str(getattr(self._agent_profile_id_input, "value", "")).strip()
            raw_name = str(getattr(self._agent_profile_name_input, "value", "")).strip()
            profile_id = _sanitize_profile_token(raw_id or seed.id or _default_profile_id())
            profile_name = raw_name or seed.name or _default_profile_name()

            return AgentProfile.from_dict(
                {
                    "id": profile_id,
                    "name": profile_name,
                    "provider": seed.provider,
                    "tier": seed.tier,
                    "system_prompt_snippets": list(seed.system_prompt_snippets),
                    "context_sources": [dict(source) for source in seed.context_sources],
                    "active_sops": list(seed.active_sops),
                    "knowledge_base_id": seed.knowledge_base_id,
                    "team_presets": normalize_team_presets(self._agent_team_presets),
                }
            )

        def _save_agent_profile_draft(self) -> None:
            profile = self._profile_from_draft()
            saved = save_profile(profile)
            self._reload_saved_profiles(selected_id=saved.id)
            self._set_agent_form_values(profile_id=saved.id, profile_name=saved.name)
            self._set_agent_draft_dirty(False, note=f"Saved profile '{saved.name}'.")

        def _delete_selected_agent_profile(self) -> None:
            selected_id = self._selected_agent_profile_id()
            if not selected_id:
                self._notify("Select a saved profile first.", severity="warning")
                return
            removed = delete_profile(selected_id)
            if not removed:
                self._notify("Profile not found.", severity="warning")
                return
            self._reload_saved_profiles()
            if self._saved_profiles:
                self._load_profile_into_draft(self._saved_profiles[0])
                self._set_agent_draft_dirty(False, note=f"Deleted profile '{selected_id}'.")
            else:
                self._new_agent_profile_draft(announce=False)
                self._set_agent_draft_dirty(True, note=f"Deleted profile '{selected_id}'.")

        def _apply_agent_profile_draft(self) -> None:
            if self._query_active:
                self._write_transcript_line("[agent] cannot apply profile while a run is active.")
                return
            if not self._daemon_ready:
                self._write_transcript_line("[agent] daemon is not ready.")
                return
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._write_transcript_line("[agent] daemon is not running.")
                self._daemon_ready = False
                return
            profile = self._profile_from_draft()
            payload = {"cmd": "set_profile", "profile": profile.to_dict()}
            if not send_daemon_command(proc, payload):
                self._write_transcript_line("[agent] failed to send set_profile.")
                return
            self._set_agent_status(f"Applying profile '{profile.name}'...")

        def _reset_plan_panel(self) -> None:
            self._set_plan_panel("(no plan)")
            self._current_plan_steps_total = 0
            self._current_plan_summary = ""
            self._current_plan_steps = []
            self._current_plan_step_statuses = []
            self._current_plan_active_step = None
            self._plan_updates_seen = False
            self._plan_step_counter = 0
            self._plan_completion_announced = False
            self._refresh_plan_status_bar()

        def _reset_issues_panel(self) -> None:
            self._issues_lines = []
            self._session_issues = []
            self._session_selected_issue_id = None
            self._issues_repeat_line = None
            self._issues_repeat_count = 0
            self._warning_count = 0
            self._error_count = 0
            self._render_session_panel()
            self._update_header_status()

        def _reset_session_timeline_panel(self) -> None:
            self._session_timeline_index = None
            self._session_timeline_events = []
            self._session_timeline_selected_event_id = None
            self._render_session_timeline_panel()

        def _session_issue_by_id(self, issue_id: str | None) -> dict[str, Any] | None:
            target = str(issue_id or "").strip()
            if not target:
                return None
            for issue in self._session_issues:
                if str(issue.get("id", "")).strip() == target:
                    return issue
            return None

        def _append_session_issue(
            self,
            *,
            severity: str,
            title: str,
            text: str,
            category: str = "issue",
            tool_use_id: str | None = None,
            tool_name: str | None = None,
            next_tier: str | None = None,
        ) -> None:
            normalized_severity = severity.strip().lower()
            if normalized_severity not in {"warning", "error"}:
                normalized_severity = "warning"
            issue = {
                "id": uuid.uuid4().hex[:12],
                "severity": normalized_severity,
                "title": title.strip() or "Issue",
                "text": text.strip(),
                "category": category.strip().lower() or "issue",
                "tool_use_id": (tool_use_id or "").strip(),
                "tool_name": (tool_name or "").strip(),
                "next_tier": (next_tier or "").strip().lower(),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._session_issues.append(issue)
            if len(self._session_issues) > 500:
                self._session_issues = self._session_issues[-500:]
            self._render_session_panel()

        def _session_timeline_event_by_id(self, event_id: str | None) -> dict[str, Any] | None:
            target = str(event_id or "").strip()
            if not target:
                return None
            for event in self._session_timeline_events:
                if str(event.get("id", "")).strip() == target:
                    return event
            return None

        def _session_issue_from_line(self, line: str) -> dict[str, Any]:
            text = line.strip()
            lowered = text.lower()
            severity = "error" if lowered.startswith("error:") else "warning"
            title = "Error" if severity == "error" else "Warning"
            category = "issue"
            tool_use_id = ""
            tool_name = ""
            next_tier = ""

            match = re.search(
                r"^error:\s*tool (?P<tool>.+?) failed \((?P<status>.+?)\)\s*\[(?P<tool_use_id>[^\]]+)\]",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                category = "tool_failure"
                title = f"Tool Failed: {match.group('tool').strip()}"
                tool_use_id = match.group("tool_use_id").strip()
                tool_name = match.group("tool").strip()
                next_tier = self._next_available_tier_name() or ""
            return {
                "severity": severity,
                "title": title,
                "text": text,
                "category": category,
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "next_tier": next_tier,
            }

        def _set_session_issue_selection(self, issue: dict[str, Any] | None) -> None:
            detail = self._session_issue_detail
            if detail is None:
                return
            if issue is None:
                self._session_selected_issue_id = None
                detail.set_preview("(no issues yet)")
                detail.set_actions([])
                return
            self._session_selected_issue_id = str(issue.get("id", "")).strip() or None
            detail.set_preview(render_session_issue_detail_text(issue))
            detail.set_actions(session_issue_actions(issue))

        def _render_session_panel(self) -> None:
            issues = list(self._session_issues)
            items = build_session_issue_sidebar_items(issues)
            list_widget = self._session_issue_list
            if list_widget is not None:
                selected_id = self._session_selected_issue_id
                if not selected_id and issues:
                    selected_id = str(issues[-1].get("id", "")).strip()
                list_widget.set_items(items, selected_id=selected_id, emit=False)
                selected_id = list_widget.selected_id()
                selected_issue = self._session_issue_by_id(selected_id)
                if selected_issue is None and issues:
                    selected_issue = issues[-1]
                    with contextlib.suppress(Exception):
                        list_widget.select_by_id(str(selected_issue.get("id", "")), emit=False)
                self._set_session_issue_selection(selected_issue)
            else:
                self._set_session_issue_selection(issues[-1] if issues else None)
            self._refresh_session_header()

        def _set_session_timeline_selection(self, event: dict[str, Any] | None) -> None:
            detail = self._session_timeline_detail
            if detail is None:
                return
            if event is None:
                self._session_timeline_selected_event_id = None
                detail.set_preview("(no timeline events yet)")
                detail.set_actions([])
                return
            self._session_timeline_selected_event_id = str(event.get("id", "")).strip() or None
            detail.set_preview(render_session_timeline_detail_text(event))
            detail.set_actions(session_timeline_actions(event))

        def _render_session_timeline_panel(self) -> None:
            events = [item for item in self._session_timeline_events if isinstance(item, dict)]
            items = build_session_timeline_sidebar_items(events)
            list_widget = self._session_timeline_list
            if list_widget is not None:
                selected_id = self._session_timeline_selected_event_id
                if not selected_id and events:
                    selected_id = str(events[-1].get("id", "")).strip()
                list_widget.set_items(items, selected_id=selected_id, emit=False)
                selected_id = list_widget.selected_id()
                selected_event = self._session_timeline_event_by_id(selected_id)
                if selected_event is None and events:
                    selected_event = events[-1]
                    with contextlib.suppress(Exception):
                        list_widget.select_by_id(str(selected_event.get("id", "")), emit=False)
                self._set_session_timeline_selection(selected_event)
            else:
                self._set_session_timeline_selection(events[-1] if events else None)
            self._refresh_session_timeline_header()

        def _refresh_session_header(self) -> None:
            header = self._session_header
            if header is None:
                return
            badges = [
                f"warn {self._warning_count}",
                f"err {self._error_count}",
                f"issues {len(self._session_issues)}",
            ]
            header.set_badges(badges)
            self._refresh_session_timeline_header()

        def _refresh_session_timeline_header(self) -> None:
            header = self._session_timeline_header
            if header is None:
                return
            events = list(self._session_timeline_events)
            error_count = 0
            for event in events:
                if classify_session_timeline_event_kind(event) == "error":
                    error_count += 1
            badges = [f"events {len(events)}", f"errors {error_count}"]
            header.set_badges(badges)

        def _set_session_view_mode(self, mode: str) -> None:
            normalized = normalize_session_view_mode(mode)
            self._session_view_mode = normalized

            timeline_view = self._session_timeline_view
            issues_view = self._session_issues_view
            if timeline_view is not None:
                timeline_view.styles.display = "block" if normalized == "timeline" else "none"
            if issues_view is not None:
                issues_view.styles.display = "block" if normalized == "issues" else "none"

            timeline_button = self._session_view_timeline_button
            issues_button = self._session_view_issues_button
            if timeline_button is not None:
                timeline_button.variant = "primary" if normalized == "timeline" else "default"
            if issues_button is not None:
                issues_button.variant = "primary" if normalized == "issues" else "default"

        def _schedule_session_timeline_refresh(self, *, delay: float = 0.35) -> None:
            timer = self._session_timeline_refresh_timer
            self._session_timeline_refresh_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()
            self._session_timeline_refresh_timer = self.set_timer(delay, self._launch_session_timeline_refresh)

        def _launch_session_timeline_refresh(self) -> None:
            self._session_timeline_refresh_timer = None
            with contextlib.suppress(RuntimeError):
                asyncio.create_task(self._refresh_session_timeline_async())

        async def _refresh_session_timeline_async(self) -> None:
            session_id = str(self._daemon_session_id or "").strip()
            if not session_id:
                self._reset_session_timeline_panel()
                return
            if self._session_timeline_refresh_inflight:
                self._session_timeline_refresh_pending = True
                return
            self._session_timeline_refresh_inflight = True
            next_pending = False
            try:
                existing_index: dict[str, Any] | None = None
                try:
                    loaded = await asyncio.to_thread(load_session_graph_index, session_id)
                    existing_index = loaded if isinstance(loaded, dict) else None
                except Exception:
                    existing_index = None
                try:
                    built_index = await asyncio.to_thread(build_session_graph_index, session_id, cwd=Path.cwd())
                    await asyncio.to_thread(write_session_graph_index, session_id, built_index)
                except Exception:
                    built_index = None
                index = built_index
                if isinstance(index, dict):
                    built_events = index.get("events")
                    existing_events = existing_index.get("events") if isinstance(existing_index, dict) else None
                    if (
                        isinstance(existing_events, list)
                        and existing_events
                        and (not isinstance(built_events, list) or not built_events)
                    ):
                        index = existing_index
                elif isinstance(existing_index, dict):
                    index = existing_index
                if isinstance(index, dict):
                    events_raw = index.get("events")
                    normalized_events: list[dict[str, Any]] = []
                    if isinstance(events_raw, list):
                        for offset, raw in enumerate(events_raw, start=1):
                            if not isinstance(raw, dict):
                                continue
                            event = dict(raw)
                            event.setdefault("id", f"timeline-{offset}")
                            normalized_events.append(event)
                    self._session_timeline_index = index
                    self._session_timeline_events = normalized_events
                    self._render_session_timeline_panel()
            finally:
                self._session_timeline_refresh_inflight = False
                next_pending = self._session_timeline_refresh_pending
                self._session_timeline_refresh_pending = False
            if next_pending:
                self._schedule_session_timeline_refresh(delay=0.1)

        def _write_issue(self, line: str) -> None:
            if self._issues_repeat_line == line:
                self._issues_repeat_count += 1
                return
            if self._issues_repeat_line is not None and self._issues_repeat_count > 0:
                repeated = f"… repeated {self._issues_repeat_count} more time(s): {self._issues_repeat_line}"
                self._issues_lines.append(repeated)
            self._issues_repeat_line = line
            self._issues_repeat_count = 0
            self._issues_lines.append(line)
            if len(self._issues_lines) > 2000:
                self._issues_lines = self._issues_lines[-2000:]
            issue_meta = self._session_issue_from_line(line)
            self._append_session_issue(
                severity=str(issue_meta.get("severity", "warning")),
                title=str(issue_meta.get("title", "Issue")),
                text=str(issue_meta.get("text", line)),
                category=str(issue_meta.get("category", "issue")),
                tool_use_id=str(issue_meta.get("tool_use_id", "")) or None,
                tool_name=str(issue_meta.get("tool_name", "")) or None,
                next_tier=str(issue_meta.get("next_tier", "")) or None,
            )

        def _flush_issue_repeats(self) -> None:
            if self._issues_repeat_line is None or self._issues_repeat_count <= 0:
                self._issues_repeat_line = None
                self._issues_repeat_count = 0
                return
            repeated = f"… repeated {self._issues_repeat_count} more time(s): {self._issues_repeat_line}"
            self._issues_lines.append(repeated)
            self._issues_repeat_line = None
            self._issues_repeat_count = 0
            self._append_session_issue(
                severity="warning",
                title="Repeated Issue",
                text=repeated,
                category="issue",
            )

        def _update_header_status(self) -> None:
            counts = []
            if self._warning_count:
                counts.append(f"warn={self._warning_count}")
            if self._error_count:
                counts.append(f"err={self._error_count}")
            suffix = (" | " + " ".join(counts)) if counts else ""
            self.sub_title = f"{self._current_model_summary()}{suffix}"
            if self._status_bar is not None:
                self._status_bar.set_counts(warnings=self._warning_count, errors=self._error_count)
            self._refresh_session_header()

        def _current_model_summary(self) -> str:
            provider_name, tier_name, model_id = choose_model_summary_parts(
                daemon_provider=self._daemon_provider,
                daemon_tier=self._daemon_tier,
                daemon_model_id=self._daemon_model_id,
                daemon_tiers=self._daemon_tiers,
                pending_value=self._pending_model_select_value,
                override_provider=self._model_provider_override,
                override_tier=self._model_tier_override,
            )
            if provider_name and tier_name:
                suffix = f" ({model_id})" if model_id else ""
                return f"Model: {provider_name}/{tier_name}{suffix}"
            return resolve_model_config_summary(
                provider_override=self._model_provider_override,
                tier_override=self._model_tier_override,
            )

        def _model_env_overrides(self) -> dict[str, str]:
            overrides: dict[str, str] = {}
            if self._model_provider_override:
                overrides["SWARMEE_MODEL_PROVIDER"] = self._model_provider_override
            if self._model_tier_override:
                overrides["SWARMEE_MODEL_TIER"] = self._model_tier_override
            return overrides

        def _refresh_model_select(self) -> None:
            if self._daemon_provider and self._daemon_tier and self._daemon_tiers:
                self._refresh_model_select_from_daemon(
                    provider=self._daemon_provider,
                    tier=self._daemon_tier,
                    tiers=self._daemon_tiers,
                )
                return

            options, selected_value = self._model_select_options()
            self._apply_model_select_options(options, selected_value)

        def _apply_model_select_options(self, options: list[tuple[str, str]], selected_value: str) -> None:
            selector = self.query_one("#model_select", Select)
            self._model_select_syncing = True
            try:
                selector.set_options(options)
                with contextlib.suppress(Exception):
                    selector.value = selected_value
            finally:
                self._model_select_syncing = False

        def _refresh_model_select_from_daemon(
            self,
            *,
            provider: str,
            tier: str,
            tiers: list[dict[str, Any]],
        ) -> None:
            options, selected_value = daemon_model_select_options(
                provider=provider,
                tier=tier,
                tiers=tiers,
                pending_value=self._pending_model_select_value,
                override_provider=self._model_provider_override,
                override_tier=self._model_tier_override,
            )
            self._apply_model_select_options(options, selected_value)

        def _model_select_options(self) -> tuple[list[tuple[str, str]], str]:
            if self._daemon_tiers and self._daemon_provider:
                return daemon_model_select_options(
                    provider=self._daemon_provider,
                    tier=(self._daemon_tier or ""),
                    tiers=self._daemon_tiers,
                    pending_value=self._pending_model_select_value,
                    override_provider=self._model_provider_override,
                    override_tier=self._model_tier_override,
                )
            return model_select_options(
                provider_override=self._model_provider_override,
                tier_override=self._model_tier_override,
            )

        def _handle_model_info(self, event: dict[str, Any]) -> None:
            provider = str(event.get("provider", "")).strip().lower()
            tier = str(event.get("tier", "")).strip().lower()
            model_id = event.get("model_id")
            tiers = event.get("tiers")

            self._daemon_provider = provider or None
            self._daemon_tier = tier or None
            self._daemon_model_id = str(model_id).strip() if model_id is not None and str(model_id).strip() else None
            self._daemon_tiers = tiers if isinstance(tiers, list) else []
            self._current_daemon_model = self._daemon_model_id or (
                f"{self._daemon_provider}/{self._daemon_tier}" if self._daemon_provider and self._daemon_tier else None
            )
            pending_value = (self._pending_model_select_value or "").strip().lower()
            if pending_value and "|" in pending_value:
                pending_provider, pending_tier = pending_value.split("|", 1)
                if pending_provider == provider and pending_tier == tier:
                    self._pending_model_select_value = None
                    pending_value = ""

            if not pending_value and self._daemon_provider and self._daemon_tier:
                self._model_provider_override = self._daemon_provider
                self._model_tier_override = self._daemon_tier

            if self._daemon_provider and self._daemon_tier:
                self._refresh_model_select_from_daemon(
                    provider=self._daemon_provider,
                    tier=self._daemon_tier,
                    tiers=self._daemon_tiers,
                )
            else:
                self._refresh_model_select()

            self._update_header_status()
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())
            self._refresh_agent_summary()
            self._render_agent_tools_panel()

        def _update_prompt_placeholder(self) -> None:
            input_widget = self.query_one("#prompt", PromptTextArea)
            approval = "on" if self._default_auto_approve else "off"
            input_widget.placeholder = (
                f"Auto-approve: {approval}. Enter submits. Shift+Enter/Ctrl+J adds newline."
            )

        def _update_command_palette(self, text: str) -> None:
            if self._command_palette is None:
                return
            stripped = text.strip()
            if stripped.startswith("/") and "\n" not in stripped:
                self._command_palette.filter(stripped)
            else:
                self._command_palette.hide()

        def _switch_side_tab(self, tab_id: str) -> None:
            with contextlib.suppress(Exception):
                tabs = self.query_one("#side_tabs", TabbedContent)
                tabs.active = tab_id

        def _seed_prompt_with_command(self, command: str) -> None:
            prompt_widget = self.query_one("#prompt", PromptTextArea)
            existing = (prompt_widget.text or "").strip()
            prompt_widget.clear()
            command_text = command.strip()
            if existing and not existing.startswith("/"):
                seeded = f"{command_text} {existing}".strip() + " "
            else:
                seeded = f"{command_text} "
            for method_name in ("insert", "insert_text_at_cursor"):
                method = getattr(prompt_widget, method_name, None)
                if callable(method):
                    with contextlib.suppress(Exception):
                        method(seeded)
                        break
            prompt_widget.focus()

        def _set_model_tier_from_value(self, value: str) -> None:
            parsed = parse_model_select_value(value)
            if parsed is None:
                return
            requested_provider, requested_tier = parsed
            self._pending_model_select_value = None
            self._model_provider_override = requested_provider or None
            self._model_tier_override = requested_tier or None
            self._refresh_model_select()
            self._update_header_status()
            self._update_prompt_placeholder()
            if (
                self._daemon_ready
                and self._proc is not None
                and self._proc.poll() is None
                and not self._query_active
            ):
                if not send_daemon_command(self._proc, {"cmd": "set_tier", "tier": requested_tier}):
                    self._write_transcript_line("[model] failed to send tier change to daemon.")
                else:
                    self._pending_model_select_value = f"{requested_provider}|{requested_tier}"
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())

        def _build_action_sheet_actions(self) -> tuple[str, list[dict[str, str]]]:
            if self._action_sheet_mode == "tier_menu":
                options, _selected = self._model_select_options()
                tier_actions: list[dict[str, str]] = [
                    {"id": "tiers:back", "icon": "←", "label": "Back", "shortcut": "Esc"},
                ]
                for _label, value in options:
                    parsed = parse_model_select_value(value)
                    if parsed is None:
                        continue
                    provider_name, tier_name = parsed
                    tier_actions.append(
                        {
                            "id": f"tier:{value}",
                            "icon": "◌",
                            "label": f"{provider_name}/{tier_name}",
                            "shortcut": "Enter",
                        }
                    )
                return "Switch Model Tier", tier_actions

            if self._consent_active:
                return (
                    "Consent Pending",
                    [
                        {"id": "consent:y", "icon": "✓", "label": "Allow", "shortcut": "y"},
                        {"id": "consent:n", "icon": "✗", "label": "Deny", "shortcut": "n"},
                        {"id": "consent:a", "icon": "★", "label": "Always allow", "shortcut": "a"},
                        {"id": "consent:v", "icon": "🚫", "label": "Never allow", "shortcut": "v"},
                    ],
                )

            if self._query_active:
                return (
                    "Run Actions",
                    [
                        {"id": "run:stop", "icon": "■", "label": "Stop run", "shortcut": "Esc"},
                        {"id": "view:plan", "icon": "▶", "label": "View plan progress", "shortcut": "P"},
                        {"id": "view:issues", "icon": "⚠", "label": "View session issues", "shortcut": "I"},
                    ],
                )

            if self._pending_plan_prompt:
                return (
                    "Plan Review",
                    [
                        {"id": "plan:approve", "icon": "✓", "label": "Approve plan", "shortcut": "/approve"},
                        {"id": "plan:replan", "icon": "↻", "label": "Replan", "shortcut": "/replan"},
                        {"id": "plan:clear", "icon": "⌫", "label": "Clear plan", "shortcut": "/clearplan"},
                        {"id": "plan:edit", "icon": "✎", "label": "Edit plan", "shortcut": "Future"},
                    ],
                )

            actions: list[dict[str, str]] = [
                {"id": "idle:new_query", "icon": "✍", "label": "New query", "shortcut": "Tab"},
                {"id": "idle:plan_mode", "icon": "🧭", "label": "Plan mode", "shortcut": "/plan"},
                {"id": "idle:run_mode", "icon": "▶", "label": "Run mode", "shortcut": "/run"},
            ]
            if self._available_restore_session_id:
                actions.append({"id": "idle:restore", "icon": "↺", "label": "Restore session", "shortcut": "/restore"})
            actions.extend(
                [
                    {"id": "idle:compact", "icon": "⇢", "label": "Compact context", "shortcut": "/compact"},
                    {"id": "idle:tiers", "icon": "⚙", "label": "Switch model tier", "shortcut": "Enter"},
                ]
            )
            return "Actions", actions

        def _show_action_sheet(self) -> None:
            sheet = self._action_sheet
            if sheet is None:
                return
            title, actions = self._build_action_sheet_actions()
            sheet.set_actions(title=title, actions=actions)
            sheet.show_sheet(focus=True)

        def _dismiss_action_sheet(self, *, restore_focus: bool = True) -> None:
            sheet = self._action_sheet
            if sheet is not None:
                sheet.hide_sheet()
            previous = self._action_sheet_previous_focus
            self._action_sheet_previous_focus = None
            self._action_sheet_mode = "root"
            if restore_focus and previous is not None:
                with contextlib.suppress(Exception):
                    previous.focus()

        def action_open_action_sheet(self) -> None:
            sheet = self._action_sheet
            if sheet is None:
                return
            if sheet.is_visible:
                self._dismiss_action_sheet(restore_focus=True)
                return
            if self._command_palette is not None:
                self._command_palette.hide()
            self._action_sheet_previous_focus = getattr(self, "focused", None)
            self._action_sheet_mode = "root"
            self._show_action_sheet()

        def _execute_action_sheet_action(self, action_id: str) -> None:
            action = action_id.strip().lower()
            if not action:
                return
            if action == "tiers:back":
                self._action_sheet_mode = "root"
                self._show_action_sheet()
                return
            if action.startswith("tier:"):
                value = action_id.split(":", 1)[1].strip()
                self._set_model_tier_from_value(value)
                return

            if action == "idle:new_query":
                self.action_focus_prompt()
                return
            if action == "idle:plan_mode":
                self._seed_prompt_with_command("/plan")
                return
            if action == "idle:run_mode":
                self._seed_prompt_with_command("/run")
                return
            if action == "idle:restore":
                self._restore_available_session()
                return
            if action == "idle:compact":
                self._request_context_compact()
                return
            if action == "idle:tiers":
                self._action_sheet_mode = "tier_menu"
                self._show_action_sheet()
                return

            if action == "run:stop":
                self._stop_run()
                return
            if action == "view:plan":
                self._switch_side_tab("tab_plan")
                return
            if action == "view:issues":
                self._switch_side_tab("tab_session")
                return

            if action.startswith("consent:"):
                choice = action.split(":", 1)[1].strip().lower()
                self._submit_consent_choice(choice)
                return

            if action == "plan:approve":
                self._dispatch_plan_action("approve")
                return
            if action == "plan:replan":
                self._dispatch_plan_action("replan")
                return
            if action == "plan:clear":
                self._dispatch_plan_action("clearplan")
                return
            if action == "plan:edit":
                self._seed_prompt_with_command("/plan")
                self._write_transcript_line("[plan] inline editing is not available yet; adjust prompt and submit.")
                return

        def _estimate_prompt_tokens(self, text: str) -> int:
            # Lightweight heuristic for pre-send token estimate.
            return max(0, (len(text or "") + 3) // 4)

        def _apply_prompt_estimate(self) -> None:
            text = self._pending_prompt_estimate_text
            self._prompt_input_tokens_est = self._estimate_prompt_tokens(text)
            self._refresh_prompt_metrics()

        def _schedule_prompt_estimate_update(self, text: str) -> None:
            self._pending_prompt_estimate_text = text or ""
            timer = self._prompt_estimate_timer
            self._prompt_estimate_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()
            self._prompt_estimate_timer = self.set_timer(0.2, self._apply_prompt_estimate)

        def _on_prompt_text_changed(self, text: str) -> None:
            self._schedule_prompt_estimate_update(text)

        def _notify(self, message: str, *, severity: str = "information", timeout: float | None = 2.5) -> None:
            with contextlib.suppress(Exception):
                self.notify(message, severity=severity, timeout=timeout)

        def _copy_text(self, text: str, *, label: str) -> None:
            payload = text or ""
            if not payload.strip():
                self._notify(f"{label}: nothing to copy.", severity="warning")
                return

            # Prefer native clipboard commands in terminal contexts (more reliable than Textual clipboard bridges).
            try:
                if sys.platform == "darwin" and shutil.which("pbcopy"):
                    subprocess.run(["pbcopy"], input=payload, text=True, encoding="utf-8", check=True)
                    self._notify(f"{label}: copied to clipboard.")
                    return
                if os.name == "nt" and shutil.which("clip"):
                    subprocess.run(["clip"], input=payload, text=True, encoding="utf-8", check=True)
                    self._notify(f"{label}: copied to clipboard.")
                    return
                if shutil.which("wl-copy"):
                    subprocess.run(["wl-copy"], input=payload, text=True, encoding="utf-8", check=True)
                    self._notify(f"{label}: copied to clipboard.")
                    return
                if shutil.which("xclip"):
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=payload,
                        text=True,
                        encoding="utf-8",
                        check=True,
                    )
                    self._notify(f"{label}: copied to clipboard.")
                    return
            except Exception:
                pass

            try:
                self.copy_to_clipboard(payload)
                self._notify(f"{label}: copied to clipboard.")
                return
            except Exception:
                pass

            artifact_path = self._persist_run_transcript(
                pid=(self._proc.pid if self._proc is not None else None),
                session_id=self._daemon_session_id,
                prompt=f"(copy) {label}",
                auto_approve=False,
                exit_code=0,
                output_text=payload,
            )
            if artifact_path:
                self._add_artifact_paths([artifact_path])
                self._write_transcript_line(
                    f"[copy] {label}: clipboard unavailable. Saved to artifact: {artifact_path}"
                )
            else:
                self._write_transcript_line(f"[copy] {label}: clipboard unavailable.")

        def _get_transcript_text(self) -> str:
            if self._transcript_fallback_lines:
                return "\n".join(self._transcript_fallback_lines).rstrip() + "\n"
            return ""

        def _get_richlog_selection_text(self, transcript: Any) -> str:
            if isinstance(transcript, TextArea):
                selected = transcript.selected_text or ""
                return selected if isinstance(selected, str) else ""
            return ""

        def _get_all_text(self) -> str:
            parts = [
                "# Transcript",
                self._get_transcript_text().rstrip(),
                "",
                "# Plan",
                (self._plan_text or "").rstrip() or "(no plan)",
                "",
                "# Session Issues",
                "\n".join(self._issues_lines).rstrip() or "(no issues)",
                "",
                "# Artifacts",
                self._get_artifacts_text().rstrip() or "(no artifacts)",
                "",
                "# Agent Profile",
                render_agent_profile_summary_text(self._session_effective_profile().to_dict()),
                "",
                "# Context Sources",
                "\n".join(self._context_list_lines()).rstrip() or "(no context sources)",
                "",
                "# Consent History",
                "\n".join(self._consent_history_lines).rstrip() or "(no consent decisions)",
                "",
            ]
            return "\n".join(parts).rstrip() + "\n"

        def _load_indexed_artifact_entries(self, *, limit: int = 200) -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            seen_paths: set[str] = set()
            try:
                store = ArtifactStore()
                indexed = store.list(limit=limit)
            except Exception:
                indexed = []

            for raw in indexed:
                normalized = normalize_artifact_index_entry(raw if isinstance(raw, dict) else {})
                if normalized is None:
                    continue
                path = str(normalized.get("path", "")).strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                entries.append(normalized)

            # Keep compatibility with legacy in-memory artifact paths that may not
            # have an index record (e.g., session logs).
            for raw_path in self._artifacts:
                path = str(raw_path or "").strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                entries.append(
                    {
                        "item_id": path,
                        "id": Path(path).name,
                        "name": Path(path).name,
                        "kind": "path",
                        "created_at": "",
                        "path": path,
                        "bytes": None,
                        "chars": None,
                        "meta": {},
                    }
                )
            return entries

        def _artifact_entry_by_item_id(self, item_id: str | None) -> dict[str, Any] | None:
            target = str(item_id or "").strip()
            if not target:
                return None
            for entry in self._artifact_entries:
                if str(entry.get("item_id", "")).strip() == target:
                    return entry
            return None

        def _artifact_looks_textual(self, entry: dict[str, Any]) -> bool:
            path = str(entry.get("path", "")).strip()
            kind = str(entry.get("kind", "")).strip().lower()
            if kind in {"tui_transcript", "tool_result", "diagnostic", "project_map"}:
                return True
            suffix = Path(path).suffix.lower()
            if suffix in {".txt", ".md", ".json", ".jsonl", ".log", ".yaml", ".yml", ".csv", ".patch", ".diff"}:
                return True
            if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".tar", ".bz2"}:
                return False
            return True

        def _artifact_metadata_preview(self, entry: dict[str, Any]) -> str:
            lines = [
                f"Kind: {entry.get('kind', 'unknown')}",
                f"ID: {entry.get('id', '(none)')}",
                f"Name: {entry.get('name', '(none)')}",
                f"Created: {entry.get('created_at', '(unknown)')}",
                f"Path: {entry.get('path', '(none)')}",
            ]
            bytes_value = entry.get("bytes")
            chars_value = entry.get("chars")
            if isinstance(bytes_value, int):
                lines.append(f"Bytes: {bytes_value}")
            if isinstance(chars_value, int):
                lines.append(f"Chars: {chars_value}")
            meta = entry.get("meta")
            if isinstance(meta, dict) and meta:
                try:
                    lines.append("")
                    lines.append("Meta:")
                    lines.append(_json.dumps(meta, indent=2, ensure_ascii=False))
                except Exception:
                    pass
            return "\n".join(lines)

        def _artifact_preview_text(self, entry: dict[str, Any]) -> str:
            path = str(entry.get("path", "")).strip()
            if not path:
                return self._artifact_metadata_preview(entry)
            artifact_path = Path(path).expanduser()
            if not artifact_path.exists() or not artifact_path.is_file():
                return self._artifact_metadata_preview(entry) + "\n\nFile not found."
            if not self._artifact_looks_textual(entry):
                return self._artifact_metadata_preview(entry)
            try:
                store = ArtifactStore()
                body = store.read_text(artifact_path, max_chars=5000)
            except Exception as exc:
                return self._artifact_metadata_preview(entry) + f"\n\nFailed to read artifact: {exc}"
            header = self._artifact_metadata_preview(entry)
            return f"{header}\n\nPreview:\n{body}"

        def _set_artifact_selection(self, entry: dict[str, Any] | None) -> None:
            detail = self._artifacts_detail
            if detail is None:
                return
            if entry is None:
                self._artifact_selected_item_id = None
                detail.set_preview("(no artifacts yet)")
                detail.set_actions([])
                return
            self._artifact_selected_item_id = str(entry.get("item_id", "")).strip() or None
            detail.set_preview(self._artifact_preview_text(entry))
            detail.set_actions(
                [
                    {"id": "artifact_action_open", "label": "Open", "variant": "default"},
                    {"id": "artifact_action_copy_path", "label": "Copy path", "variant": "default"},
                    {"id": "artifact_action_add_context", "label": "Add context", "variant": "default"},
                ]
            )

        def _render_artifacts_panel(self) -> None:
            self._artifact_entries = self._load_indexed_artifact_entries(limit=200)
            if self._artifacts_header is not None:
                badge_count = len(self._artifact_entries)
                self._artifacts_header.set_badges([f"{badge_count} item{'s' if badge_count != 1 else ''}"])
            list_widget = self._artifacts_list
            if list_widget is None:
                return
            items = build_artifact_sidebar_items(self._artifact_entries)
            selected_id = self._artifact_selected_item_id
            if not selected_id and self._artifact_entries:
                selected_id = str(self._artifact_entries[0].get("item_id", "")).strip()
            list_widget.set_items(items, selected_id=selected_id, emit=False)
            selected_item_id = list_widget.selected_id()
            selected_entry = self._artifact_entry_by_item_id(selected_item_id)
            if selected_entry is None and self._artifact_entries:
                selected_entry = self._artifact_entries[0]
                list_widget.select_by_id(str(selected_entry.get("item_id", "")), emit=False)
            self._set_artifact_selection(selected_entry)

        def _get_artifacts_text(self) -> str:
            entries = self._artifact_entries or self._load_indexed_artifact_entries(limit=200)
            if not entries:
                return "(no artifacts)\n"
            lines: list[str] = []
            for index, entry in enumerate(entries, start=1):
                kind = str(entry.get("kind", "unknown")).strip() or "unknown"
                artifact_id = str(entry.get("id", "")).strip() or "(no-id)"
                name = str(entry.get("name", "")).strip() or artifact_id
                created_at = str(entry.get("created_at", "")).strip() or "unknown time"
                path = str(entry.get("path", "")).strip() or "(no-path)"
                lines.append(f"{index}. {kind} | {name} | {created_at} | {path}")
            return "\n".join(lines).rstrip() + "\n"

        def _reset_artifacts_panel(self) -> None:
            self._artifacts = []
            self._artifact_entries = []
            self._artifact_selected_item_id = None
            self._render_artifacts_panel()

        def _add_artifact_paths(self, paths: list[str]) -> None:
            updated = add_recent_artifacts(self._artifacts, paths, max_items=20)
            if updated != self._artifacts:
                self._artifacts = updated
            self._render_artifacts_panel()

        def _context_source_label(self, source: dict[str, str]) -> str:
            source_type = source.get("type", "")
            if source_type == "file":
                return source.get("path", "")
            if source_type == "note":
                return source.get("text", "")
            if source_type == "sop":
                return source.get("name", "")
            if source_type == "kb":
                return source.get("id", "")
            if source_type == "url":
                return source.get("url", source.get("path", ""))
            return str(source)

        def _truncate_context_label(self, value: str, *, max_chars: int = _CONTEXT_SOURCE_MAX_LABEL) -> str:
            text = value.strip().replace("\n", " ")
            if len(text) <= max_chars:
                return text
            if max_chars <= 1:
                return text[:max_chars]
            return text[: max_chars - 1].rstrip() + "…"

        def _render_context_sources_panel(self) -> None:
            container = self._context_sources_list
            if container is None:
                with contextlib.suppress(Exception):
                    container = self.query_one("#context_sources_list", VerticalScroll)
                    self._context_sources_list = container
            if container is None:
                return

            for child in list(container.children):
                with contextlib.suppress(Exception):
                    child.remove()

            if not self._context_sources:
                container.mount(Static("[dim](no context sources attached)[/dim]"))
                return

            for index, source in enumerate(self._context_sources):
                source_type = source.get("type", "").strip().lower()
                icon = _CONTEXT_SOURCE_ICONS.get(source_type, "•")
                label = self._truncate_context_label(self._context_source_label(source))
                row = Horizontal(classes="context-source-row")
                container.mount(row)
                row.mount(Static(f"{icon} {label}", classes="context-source-label"))
                row.mount(
                    Button(
                        "✕",
                        id=f"context_remove_{index}",
                        classes="context-remove-btn",
                        compact=True,
                        variant="error",
                    )
                )

        def _refresh_context_sop_options(self) -> None:
            self._context_sop_names = discover_available_sop_names()
            selector = self._context_sop_select
            if selector is None:
                with contextlib.suppress(Exception):
                    selector = self.query_one("#context_sop_select", Select)
                    self._context_sop_select = selector
            if selector is None:
                return
            options: list[tuple[str, str]] = [("Select SOP...", _CONTEXT_SELECT_PLACEHOLDER)]
            options.extend((name, name) for name in self._context_sop_names)
            with contextlib.suppress(Exception):
                selector.set_options(options)
                selector.value = _CONTEXT_SELECT_PLACEHOLDER

        def _sop_checkbox_id(self, name: str) -> str:
            token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip().lower())
            token = token.strip("_") or "sop"
            return f"sop_toggle_{token}"

        def _refresh_sop_catalog(self) -> None:
            self._sop_catalog = discover_available_sops()
            available_names = {item.get("name", "").strip() for item in self._sop_catalog}
            self._active_sop_names = {name for name in self._active_sop_names if name in available_names}
            self._sops_ready_for_sync = bool(self._active_sop_names)

        def _lookup_sop(self, name: str) -> dict[str, str] | None:
            target = name.strip().lower()
            for item in self._sop_catalog:
                sop_name = str(item.get("name", "")).strip()
                if sop_name.lower() == target:
                    return item
            return None

        def _render_sop_panel(self) -> None:
            container = self._sop_list
            if container is None:
                with contextlib.suppress(Exception):
                    container = self.query_one("#sop_list", VerticalScroll)
                    self._sop_list = container
            if container is None:
                return

            self._sop_toggle_id_to_name = {}
            for child in list(container.children):
                with contextlib.suppress(Exception):
                    child.remove()

            if not self._sop_catalog:
                container.mount(Static("[dim](no SOPs found)[/dim]"))
                return

            for sop in self._sop_catalog:
                name = str(sop.get("name", "")).strip()
                if not name:
                    continue
                source = str(sop.get("source", "")).strip() or "unknown"
                preview = str(sop.get("first_paragraph_preview", "")).strip() or "(no preview available)"
                checkbox_id = self._sop_checkbox_id(name)
                self._sop_toggle_id_to_name[checkbox_id] = name
                row = Vertical(classes="sop-row")
                container.mount(row)
                header = Horizontal(classes="sop-row-header")
                row.mount(header)
                header.mount(Checkbox(name, id=checkbox_id, value=(name in self._active_sop_names)))
                header.mount(Static(f"[dim]{source}[/dim]", classes="sop-source-label"))
                row.mount(Static(preview, classes="sop-preview"))

        def _sync_active_sops_with_daemon(self, *, notify_on_failure: bool = False) -> bool:
            proc = self._proc
            if not self._daemon_ready or proc is None or proc.poll() is not None:
                self._sops_ready_for_sync = bool(self._active_sop_names)
                return False
            for name in sorted(self._active_sop_names):
                record = self._lookup_sop(name)
                if record is None:
                    continue
                content = str(record.get("content", "")).strip()
                if not content:
                    continue
                payload = {"cmd": "set_sop", "name": name, "content": content}
                if not send_daemon_command(proc, payload):
                    self._sops_ready_for_sync = True
                    if notify_on_failure:
                        self._write_transcript_line("[sop] failed to sync active SOPs with daemon.")
                    return False
            self._sops_ready_for_sync = False
            return True

        def _set_sop_active(self, name: str, active: bool, *, sync: bool = True, announce: bool = True) -> bool:
            record = self._lookup_sop(name)
            if record is None:
                self._write_transcript_line(f"[sop] unknown SOP: {name}")
                return False
            sop_name = str(record.get("name", "")).strip()
            if not sop_name:
                return False

            changed = False
            if active and sop_name not in self._active_sop_names:
                self._active_sop_names.add(sop_name)
                changed = True
            elif (not active) and sop_name in self._active_sop_names:
                self._active_sop_names.remove(sop_name)
                changed = True

            if not changed:
                return True

            self._render_sop_panel()
            self._save_session()
            self._refresh_agent_summary()

            if sync:
                proc = self._proc
                if not self._daemon_ready or proc is None or proc.poll() is not None:
                    self._sops_ready_for_sync = bool(self._active_sop_names)
                else:
                    payload = {
                        "cmd": "set_sop",
                        "name": sop_name,
                        "content": (str(record.get("content", "")).strip() if active else None),
                    }
                    if not send_daemon_command(proc, payload):
                        self._sops_ready_for_sync = True
                        self._write_transcript_line("[sop] failed to sync with daemon.")
                    else:
                        self._sops_ready_for_sync = False

            if announce:
                status = "activated" if active else "deactivated"
                self._write_transcript_line(f"[sop] {status}: {sop_name}")
            return True

        def _sop_list_lines(self) -> list[str]:
            if not self._sop_catalog:
                return ["[sop] no SOPs found."]
            lines: list[str] = ["[sop] available:"]
            for index, sop in enumerate(self._sop_catalog, start=1):
                name = str(sop.get("name", "")).strip()
                source = str(sop.get("source", "")).strip() or "unknown"
                marker = "✓" if name in self._active_sop_names else " "
                lines.append(f"{index}. [{marker}] {name} ({source})")
            return lines

        def _handle_sop_command(self, argument: str) -> bool:
            raw = (argument or "").strip()
            if not raw:
                self._write_transcript_line(_SOP_USAGE_TEXT)
                return True
            lowered = raw.lower()

            if lowered == "list":
                for line in self._sop_list_lines():
                    self._write_transcript_line(line)
                return True

            if lowered.startswith("activate "):
                target = raw.split(maxsplit=1)[1].strip()
                if not target:
                    self._write_transcript_line(_SOP_USAGE_TEXT)
                    return True
                self._set_sop_active(target, True, sync=True, announce=True)
                return True

            if lowered.startswith("deactivate "):
                target = raw.split(maxsplit=1)[1].strip()
                if not target:
                    self._write_transcript_line(_SOP_USAGE_TEXT)
                    return True
                self._set_sop_active(target, False, sync=True, announce=True)
                return True

            if lowered.startswith("preview "):
                target = raw.split(maxsplit=1)[1].strip()
                if not target:
                    self._write_transcript_line(_SOP_USAGE_TEXT)
                    return True
                record = self._lookup_sop(target)
                if record is None:
                    self._write_transcript_line(f"[sop] unknown SOP: {target}")
                    return True
                name = str(record.get("name", target)).strip()
                source = str(record.get("source", "")).strip() or "unknown"
                content = str(record.get("content", "")).strip()
                if not content:
                    self._write_transcript_line(f"[sop] no content available for {name}")
                    return True
                markdown = f"# SOP: {name}\n\n[dim]Source: {source}[/dim]\n\n{content}"
                self._mount_transcript_widget(
                    render_assistant_message(markdown),
                    plain_text=f"SOP: {name}\n\n{content}",
                )
                return True

            self._write_transcript_line(_SOP_USAGE_TEXT)
            return True

        def _set_context_add_mode(self, mode: str | None) -> None:
            normalized = (mode or "").strip().lower() or None
            self._context_add_mode = normalized
            input_row = self.query_one("#context_input_row", Horizontal)
            sop_row = self.query_one("#context_sop_row", Horizontal)
            input_widget = self._context_input
            if input_widget is None:
                with contextlib.suppress(Exception):
                    input_widget = self.query_one("#context_input", Input)
                    self._context_input = input_widget

            if normalized in _CONTEXT_INPUT_SOURCE_TYPES:
                input_row.styles.display = "block"
                sop_row.styles.display = "none"
                if input_widget is not None:
                    if normalized == "file":
                        input_widget.placeholder = "Enter file path"
                    elif normalized == "kb":
                        input_widget.placeholder = "Enter knowledge base ID"
                    else:
                        input_widget.placeholder = "Enter context note"
                    with contextlib.suppress(Exception):
                        input_widget.value = ""
                        input_widget.focus()
                return

            if normalized == _CONTEXT_SOP_SOURCE_TYPE:
                input_row.styles.display = "none"
                sop_row.styles.display = "block"
                self._refresh_context_sop_options()
                if self._context_sop_select is not None:
                    with contextlib.suppress(Exception):
                        self._context_sop_select.focus()
                return

            input_row.styles.display = "none"
            sop_row.styles.display = "none"

        def _context_sources_payload(self) -> list[dict[str, str]]:
            payload: list[dict[str, str]] = []
            for source in self._context_sources:
                source_type = source.get("type", "")
                if source_type == "file":
                    path = source.get("path", "").strip()
                    if path:
                        payload.append({"type": "file", "path": path, "id": source.get("id", "")})
                elif source_type == "note":
                    text = source.get("text", "").strip()
                    if text:
                        payload.append({"type": "note", "text": text, "id": source.get("id", "")})
                elif source_type == "sop":
                    name = source.get("name", "").strip()
                    if name:
                        payload.append({"type": "sop", "name": name, "id": source.get("id", "")})
                elif source_type == "kb":
                    kb_id = source.get("id", "").strip()
                    if kb_id:
                        payload.append({"type": "kb", "id": kb_id})
                elif source_type == "url":
                    url = source.get("url", "").strip()
                    if url:
                        payload.append({"type": "url", "url": url, "id": source.get("id", "")})
            return payload

        def _sync_context_sources_with_daemon(self, *, notify_on_failure: bool = False) -> bool:
            proc = self._proc
            if not self._daemon_ready or proc is None or proc.poll() is not None:
                self._context_ready_for_sync = True
                return False
            payload = {"cmd": "set_context_sources", "sources": self._context_sources_payload()}
            if send_daemon_command(proc, payload):
                self._context_ready_for_sync = False
                return True
            self._context_ready_for_sync = True
            if notify_on_failure:
                self._write_transcript_line("[context] failed to sync sources with daemon.")
            return False

        def _set_context_sources(self, sources: list[dict[str, str]], *, sync: bool = True) -> None:
            self._context_sources = _normalize_context_sources(sources)
            self._render_context_sources_panel()
            self._save_session()
            self._refresh_agent_summary()
            if sync:
                self._sync_context_sources_with_daemon(notify_on_failure=True)

        def _add_context_source(self, source: dict[str, str]) -> None:
            normalized = _normalize_context_source(source)
            if normalized is None:
                return
            updated = [*self._context_sources, normalized]
            self._set_context_sources(updated, sync=True)

        def _remove_context_source(self, index: int) -> None:
            if index < 0 or index >= len(self._context_sources):
                self._write_transcript_line(f"[context] invalid index: {index + 1}.")
                return
            updated = [item for i, item in enumerate(self._context_sources) if i != index]
            self._set_context_sources(updated, sync=True)

        def _clear_context_sources(self) -> None:
            self._set_context_sources([], sync=True)

        def _context_list_lines(self) -> list[str]:
            if not self._context_sources:
                return ["[context] no sources attached."]
            lines: list[str] = ["[context] active sources:"]
            for index, source in enumerate(self._context_sources, start=1):
                source_type = source.get("type", "unknown")
                label = self._truncate_context_label(self._context_source_label(source), max_chars=96)
                lines.append(f"{index}. {source_type}: {label}")
            return lines

        def _commit_context_add_from_ui(self) -> None:
            mode = (self._context_add_mode or "").strip().lower()
            if mode in _CONTEXT_INPUT_SOURCE_TYPES:
                if self._context_input is None:
                    return
                raw_value = str(getattr(self._context_input, "value", "")).strip()
                if not raw_value:
                    self._notify("Context value is required.", severity="warning")
                    return
                source: dict[str, str]
                if mode == "file":
                    source = {"type": "file", "path": raw_value, "id": _sanitize_context_source_id(uuid.uuid4().hex)}
                elif mode == "kb":
                    source = {"type": "kb", "id": raw_value}
                else:
                    source = {"type": "note", "text": raw_value, "id": _sanitize_context_source_id(uuid.uuid4().hex)}
                self._add_context_source(source)
                self._set_context_add_mode(None)
                self._write_transcript_line(f"[context] added {mode} source.")
                return

            if mode == _CONTEXT_SOP_SOURCE_TYPE:
                selector = self._context_sop_select
                selected = str(getattr(selector, "value", "")).strip() if selector is not None else ""
                if not selected or selected == _CONTEXT_SELECT_PLACEHOLDER:
                    self._notify("Select an SOP first.", severity="warning")
                    return
                self._add_context_source(
                    {
                        "type": "sop",
                        "name": selected,
                        "id": _sanitize_context_source_id(f"sop-{selected}"),
                    }
                )
                self._set_context_add_mode(None)
                self._write_transcript_line(f"[context] added sop source: {selected}")

        def _handle_context_command(self, argument: str) -> bool:
            raw = (argument or "").strip()
            if not raw:
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True

            lowered = raw.lower()
            if lowered == "list":
                for line in self._context_list_lines():
                    self._write_transcript_line(line)
                return True

            if lowered == "clear":
                self._clear_context_sources()
                self._write_transcript_line("[context] cleared all sources.")
                return True

            if lowered.startswith("remove "):
                token = raw.split(maxsplit=1)[1].strip()
                try:
                    index = int(token) - 1
                except ValueError:
                    self._write_transcript_line("Usage: /context remove <index>")
                    return True
                self._remove_context_source(index)
                return True

            add_match = re.match(r"^add\s+(file|note|sop|kb)\s+(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
            if add_match is None:
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True
            source_type = add_match.group(1).strip().lower()
            value = add_match.group(2).strip()
            if not value:
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True

            if source_type == "file":
                self._add_context_source(
                    {
                        "type": "file",
                        "path": value,
                        "id": _sanitize_context_source_id(uuid.uuid4().hex),
                    }
                )
            elif source_type == "note":
                self._add_context_source(
                    {
                        "type": "note",
                        "text": value,
                        "id": _sanitize_context_source_id(uuid.uuid4().hex),
                    }
                )
            elif source_type == "sop":
                self._add_context_source(
                    {"type": "sop", "name": value, "id": _sanitize_context_source_id(f"sop-{value}")}
                )
            elif source_type == "kb":
                self._add_context_source({"type": "kb", "id": value})
            else:
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True

            self._write_transcript_line(f"[context] added {source_type} source.")
            return True

        def _record_consent_history(self, line: str) -> None:
            entry = line.strip()
            if not entry:
                return
            self._consent_history_lines.append(entry)
            if len(self._consent_history_lines) > 200:
                self._consent_history_lines = self._consent_history_lines[-200:]

        def _show_thinking_text(self) -> None:
            current_text = "".join(self._thinking_buffer).strip()
            text = current_text or (self._last_thinking_text or "").strip()
            if not text:
                self._write_transcript_line("No thinking content from this turn.")
                return

            total_chars = len(text)
            if total_chars > _THINKING_EXPORT_MAX_CHARS:
                shown = text[-_THINKING_EXPORT_MAX_CHARS :]
                self._write_transcript_line(
                    f"[thinking] showing last {_THINKING_EXPORT_MAX_CHARS:,} of {total_chars:,} chars."
                )
                text = shown
            else:
                self._write_transcript_line(f"[thinking] showing {total_chars:,} chars.")

            self._mount_transcript_widget(render_system_message(text), plain_text=text)

        def _cancel_consent_hide_timer(self) -> None:
            timer = self._consent_hide_timer
            self._consent_hide_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _complete_consent_prompt_hide(self, expected_nonce: int) -> None:
            self._consent_hide_timer = None
            if expected_nonce != self._consent_prompt_nonce:
                return
            widget = self._consent_prompt_widget
            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.hide_prompt()
            with contextlib.suppress(Exception):
                self.query_one("#prompt", TextArea).focus()

        def _schedule_consent_prompt_hide(self, *, delay: float = 1.0) -> None:
            self._cancel_consent_hide_timer()
            nonce = self._consent_prompt_nonce
            self._consent_hide_timer = self.set_timer(
                delay,
                lambda: self._complete_consent_prompt_hide(nonce),
            )

        def _show_consent_prompt(self, *, context: str, options: list[str] | None = None, alert: bool = True) -> None:
            widget = self._consent_prompt_widget
            if widget is None:
                with contextlib.suppress(Exception):
                    widget = self.query_one("#consent_prompt", ConsentPrompt)
                    self._consent_prompt_widget = widget
            if widget is None:
                return
            self._cancel_consent_hide_timer()
            self._consent_prompt_nonce += 1
            self._consent_active = True
            self._consent_tool_name = extract_consent_tool_name(context)
            normalized_options = options or ["y", "n", "a", "v"]
            widget.set_prompt(context=context, options=normalized_options, alert=alert)

        def _reset_consent_panel(self) -> None:
            self._cancel_consent_hide_timer()
            self._consent_prompt_nonce += 1
            self._consent_active = False
            self._consent_buffer = []
            self._consent_tool_name = "tool"
            widget = self._consent_prompt_widget
            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.hide_prompt()

        def _reset_error_action_prompt(self) -> None:
            self._pending_error_action = None
            widget = self._error_action_prompt_widget
            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.hide_prompt()

        def _next_available_tier_name(self) -> str | None:
            current_tier = (self._daemon_tier or "").strip().lower()
            available = [
                str(item.get("name", "")).strip().lower()
                for item in self._daemon_tiers
                if isinstance(item, dict) and bool(item.get("available"))
            ]
            if not available:
                return None
            if current_tier in available:
                idx = available.index(current_tier)
                for candidate in available[idx + 1 :]:
                    if candidate:
                        return candidate
            for candidate in available:
                if candidate and candidate != current_tier:
                    return candidate
            return None

        def _show_tool_error_actions(self, *, tool_use_id: str, tool_name: str) -> None:
            widget = self._error_action_prompt_widget
            if widget is None:
                return
            self._pending_error_action = {
                "kind": "tool",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
            }
            with contextlib.suppress(Exception):
                widget.show_tool_error(tool_name=tool_name, tool_use_id=tool_use_id)

        def _show_escalation_actions(self, *, next_tier: str | None = None) -> None:
            widget = self._error_action_prompt_widget
            if widget is None:
                return
            resolved_next = (next_tier or "").strip().lower() or self._next_available_tier_name()
            self._pending_error_action = {
                "kind": "escalation",
                "next_tier": resolved_next or None,
            }
            with contextlib.suppress(Exception):
                widget.show_escalation(next_tier=resolved_next or None)

        def _resume_after_error(self, *, escalate: bool) -> None:
            if self._query_active:
                self._write_transcript_line("[run] already running; use /stop.")
                return
            prompt = (self._last_prompt or "").strip()
            if not prompt:
                self._write_transcript_line("[run] no previous prompt to continue.")
                self._reset_error_action_prompt()
                return
            proc = self._proc
            if proc is None or proc.poll() is not None or not self._daemon_ready:
                self._write_transcript_line("[run] daemon is not ready. Use /daemon restart.")
                self._reset_error_action_prompt()
                return
            pending = self._pending_error_action or {}
            if escalate:
                next_tier = str(pending.get("next_tier", "")).strip().lower()
                if next_tier:
                    if not send_daemon_command(proc, {"cmd": "set_tier", "tier": next_tier}):
                        self._write_transcript_line("[model] failed to request tier escalation.")
                        return
                    self._write_transcript_line(f"[model] escalated to {next_tier}.")
                else:
                    self._write_transcript_line("[model] no higher tier available; continuing on current tier.")
            self._reset_error_action_prompt()
            self._start_run(prompt, auto_approve=self._last_run_auto_approve, mode="execute")

        def _retry_failed_tool(self) -> None:
            action = self._pending_error_action or {}
            tool_use_id = str(action.get("tool_use_id", "")).strip()
            if not tool_use_id:
                self._write_transcript_line("[recovery] no failed tool selected.")
                self._reset_error_action_prompt()
                return
            proc = self._proc
            if proc is None or proc.poll() is not None or not self._daemon_ready:
                self._write_transcript_line("[recovery] daemon is not ready.")
                self._reset_error_action_prompt()
                return
            if send_daemon_command(proc, {"cmd": "retry_tool", "tool_use_id": tool_use_id}):
                self._write_transcript_line(f"[recovery] retry requested for tool {tool_use_id}.")
                self._reset_error_action_prompt()
            else:
                self._write_transcript_line("[recovery] failed to send retry request.")

        def _skip_failed_tool(self) -> None:
            action = self._pending_error_action or {}
            tool_use_id = str(action.get("tool_use_id", "")).strip()
            if not tool_use_id:
                self._write_transcript_line("[recovery] no failed tool selected.")
                self._reset_error_action_prompt()
                return
            proc = self._proc
            if proc is None or proc.poll() is not None or not self._daemon_ready:
                self._write_transcript_line("[recovery] daemon is not ready.")
                self._reset_error_action_prompt()
                return
            if send_daemon_command(proc, {"cmd": "skip_tool", "tool_use_id": tool_use_id}):
                self._write_transcript_line(f"[recovery] skip requested for tool {tool_use_id}.")
                self._reset_error_action_prompt()
            else:
                self._write_transcript_line("[recovery] failed to send skip request.")

        def _apply_consent_capture(self, line: str) -> None:
            previously_active = self._consent_active
            next_active, next_buffer = update_consent_capture(
                self._consent_active,
                self._consent_buffer,
                line,
                max_lines=20,
            )
            if next_active != self._consent_active or next_buffer != self._consent_buffer:
                self._consent_active = next_active
                self._consent_buffer = next_buffer
                if self._consent_active:
                    context = "\n".join(self._consent_buffer[-4:])
                    self._show_consent_prompt(
                        context=context,
                        options=["y", "n", "a", "v"],
                        alert=not previously_active and next_active,
                    )

        def _consent_decision_line(self, choice: str) -> str:
            tool_name = self._consent_tool_name or "tool"
            if choice == "y":
                return f"✓ {tool_name} allowed"
            if choice == "n":
                return f"✗ {tool_name} denied"
            if choice == "a":
                return f"✓ {tool_name} always allowed (session)"
            if choice == "v":
                return f"✗ {tool_name} never allowed (session)"
            return f"[consent] response: {choice}"

        def _submit_consent_choice(self, choice: str) -> None:
            normalized_choice = choice.strip().lower()
            if normalized_choice not in _CONSENT_CHOICES:
                self._write_transcript_line("Usage: /consent <y|n|a|v>")
                return
            if not self._consent_active:
                self._write_transcript_line("[consent] no active prompt.")
                return
            if self._proc is None or self._proc.poll() is not None:
                self._write_transcript_line("[consent] daemon is not running.")
                self._reset_consent_panel()
                return
            decision_line = self._consent_decision_line(normalized_choice)
            self._write_transcript(decision_line)
            self._record_consent_history(decision_line)
            self._consent_active = False
            self._consent_buffer = []
            approved = normalized_choice in {"y", "a"}
            widget = self._consent_prompt_widget
            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.show_confirmation(decision_line, approved=approved)
            self._schedule_consent_prompt_hide(delay=1.0)
            if not send_daemon_command(self._proc, {"cmd": "consent_response", "choice": normalized_choice}):
                self._write_transcript_line("[consent] failed to send response (stdin unavailable).")
                return

        def _finalize_assistant_message(self) -> None:
            self._cancel_streaming_flush_timer()
            self._flush_streaming_buffer()
            if not self._current_assistant_chunks:
                self._current_assistant_model = None
                self._current_assistant_timestamp = None
                self._assistant_placeholder_written = False
                return

            full_text = "".join(self._current_assistant_chunks)
            self._last_assistant_text = full_text
            model = self._current_assistant_model
            timestamp = self._current_assistant_timestamp or self._turn_timestamp()
            plain_lines = [full_text]
            meta_parts = [part for part in [model, timestamp] if isinstance(part, str) and part.strip()]
            if meta_parts:
                plain_lines.append(" · ".join(meta_parts))
            self._mount_transcript_widget(
                render_assistant_message(full_text, model=model, timestamp=timestamp),
                plain_text="\n".join(plain_lines),
            )

            self._current_assistant_chunks = []
            self._streaming_buffer = []
            self._current_assistant_model = None
            self._current_assistant_timestamp = None
            self._assistant_placeholder_written = False

        def _handle_output_line(self, line: str, raw_line: str | None = None) -> None:
            if self._query_active:
                chunk = raw_line if raw_line is not None else (line + "\n")
                self._turn_output_chunks.append(sanitize_output_text(chunk))
            sanitized = sanitize_output_text(line)
            # Try structured JSONL first (emitted by TuiCallbackHandler).
            tui_event = parse_tui_event(sanitized)
            if tui_event is not None:
                self._handle_tui_event(tui_event)
                return

            # Legacy fallback for non-JSON lines (stderr leakage, library warnings, etc.).
            event = parse_output_line(sanitized)
            if event is None:
                if sanitized.strip() == "return meta(":
                    return
                self._append_plain_text(sanitized)
                self._apply_consent_capture(sanitized)
                return

            if event.kind == "noise":
                return

            if event.kind == "error":
                text = event.text
                if not text.startswith("ERROR:"):
                    text = f"ERROR: {text}"
                self._error_count += 1
                self._write_issue(text)
                self._update_header_status()
            elif event.kind == "warning":
                text = event.text
                if not text.startswith("WARN:"):
                    text = f"WARN: {text}"
                self._warning_count += 1
                self._write_issue(text)
                self._update_header_status()
            else:
                self._append_plain_text(event.text)

            artifact_paths = artifact_paths_from_event(event)
            if artifact_paths:
                self._add_artifact_paths(artifact_paths)

            self._apply_consent_capture(sanitized)

        def _handle_tui_event(self, event: dict[str, Any]) -> None:
            """Process a structured JSONL event from the subprocess."""
            etype = str(event.get("event", "")).strip().lower()

            if etype in {"ready", "attached"}:
                self._daemon_ready = True
                session_id = str(event.get("session_id", "")).strip()
                if session_id:
                    self._daemon_session_id = session_id
                    self._save_session()
                if etype == "attached":
                    clients_raw = event.get("clients")
                    clients = int(clients_raw) if isinstance(clients_raw, int) else None
                    if clients is not None and clients > 1:
                        self._write_transcript_line(
                            f"[daemon] attached to shared runtime session ({clients} clients connected)."
                        )
                    else:
                        self._write_transcript_line("[daemon] attached to shared runtime session.")
                else:
                    self._write_transcript("Swarmee daemon ready. Enter a prompt to run Swarmee.")
                if self._context_sources or self._context_ready_for_sync:
                    self._sync_context_sources_with_daemon(notify_on_failure=True)
                if self._active_sop_names or self._sops_ready_for_sync:
                    self._sync_active_sops_with_daemon(notify_on_failure=True)
                self._refresh_agent_summary()
                self._schedule_session_timeline_refresh()

            elif etype == "session_available":
                session_id = str(event.get("session_id", "")).strip()
                turn_count_raw = event.get("turn_count", 0)
                try:
                    turn_count = int(turn_count_raw or 0)
                except (TypeError, ValueError):
                    turn_count = 0
                self._available_restore_session_id = session_id or None
                self._available_restore_turn_count = max(0, turn_count)
                if session_id:
                    self._write_transcript_line(
                        f"Previous session found ({self._available_restore_turn_count} turns). "
                        "Type /restore to resume or /new to start fresh."
                    )

            elif etype == "session_restored":
                session_id = str(event.get("session_id", "")).strip()
                if session_id:
                    self._daemon_session_id = session_id
                turn_count_raw = event.get("turn_count", 0)
                try:
                    self._last_restored_turn_count = max(0, int(turn_count_raw or 0))
                except (TypeError, ValueError):
                    self._last_restored_turn_count = 0
                self._available_restore_session_id = None
                self._available_restore_turn_count = 0
                self._save_session()
                self._schedule_session_timeline_refresh()

            elif etype == "replay_turn":
                role = str(event.get("role", "")).strip().lower()
                text = sanitize_output_text(str(event.get("text", "")))
                if not text.strip():
                    return
                timestamp = str(event.get("timestamp", "")).strip() or None
                if role == "user":
                    self._write_user_message(text, timestamp=timestamp)
                elif role == "assistant":
                    model = str(event.get("model", "")).strip() or None
                    self._write_assistant_message(text, model=model, timestamp=timestamp)

            elif etype == "replay_complete":
                turn_count_raw = event.get("turn_count", self._last_restored_turn_count)
                try:
                    turns = max(0, int(turn_count_raw or 0))
                except (TypeError, ValueError):
                    turns = max(0, self._last_restored_turn_count)
                self._write_transcript_line(f"Session restored ({turns} turns).")

            elif etype == "turn_complete":
                exit_status = str(event.get("exit_status", "ok"))
                self._finalize_turn(exit_status=exit_status)
                if exit_status in {"ok", "interrupted"}:
                    self._reset_error_action_prompt()
                self._schedule_session_timeline_refresh()

            elif etype == "model_info":
                self._handle_model_info(event)

            elif etype == "profile_applied":
                raw_profile = event.get("profile")
                try:
                    applied_profile = AgentProfile.from_dict(raw_profile)
                except Exception:
                    self._write_transcript_line("[agent] received invalid profile_applied payload.")
                    return
                self._effective_profile = applied_profile
                self._refresh_agent_summary()
                self._reload_saved_profiles(selected_id=applied_profile.id)
                self._agent_team_presets = normalize_team_presets(applied_profile.team_presets)
                self._agent_team_selected_item_id = None
                self._render_agent_team_panel()
                self._set_agent_form_values(profile_id=applied_profile.id, profile_name=applied_profile.name)
                self._set_agent_draft_dirty(False, note=f"Applied profile '{applied_profile.name}'.")

            elif etype == "safety_overrides":
                self._session_safety_overrides = normalize_session_safety_overrides(event.get("overrides"))
                self._render_agent_tools_panel()
                if self._session_safety_overrides:
                    self._set_agent_tools_status("Session overrides active.")
                else:
                    self._set_agent_tools_status("Session overrides cleared.")

            elif etype == "context":
                prompt_tokens_est = event.get("prompt_tokens_est")
                budget_tokens = event.get("budget_tokens")
                self._last_prompt_tokens_est = int(prompt_tokens_est) if isinstance(prompt_tokens_est, int) else None
                self._last_budget_tokens = int(budget_tokens) if isinstance(budget_tokens, int) else None
                if self._status_bar is not None:
                    self._status_bar.set_context(
                        prompt_tokens_est=self._last_prompt_tokens_est,
                        budget_tokens=self._last_budget_tokens,
                    )
                self._refresh_prompt_metrics()

            elif etype == "usage":
                usage = event.get("usage")
                self._last_usage = usage if isinstance(usage, dict) else None
                cost = event.get("cost_usd")
                self._last_cost_usd = float(cost) if isinstance(cost, (int, float)) else None
                if self._status_bar is not None:
                    self._status_bar.set_usage(self._last_usage, cost_usd=self._last_cost_usd)
                self._refresh_prompt_metrics()

            elif etype == "compact_complete":
                compacted = bool(event.get("compacted", False))
                warning_text = str(event.get("warning", "")).strip()
                before_tokens = event.get("before_tokens_est")
                after_tokens = event.get("after_tokens_est")
                if isinstance(before_tokens, int) and isinstance(after_tokens, int):
                    self._last_prompt_tokens_est = after_tokens
                    if self._status_bar is not None:
                        self._status_bar.set_context(
                            prompt_tokens_est=self._last_prompt_tokens_est,
                            budget_tokens=self._last_budget_tokens,
                        )
                    self._refresh_prompt_metrics()
                if compacted:
                    self._notify("Context compacted.", severity="information", timeout=4.0)
                elif warning_text:
                    self._notify(warning_text, severity="warning", timeout=6.0)
                else:
                    self._notify("Context compaction made no changes.", severity="information", timeout=4.0)

            elif etype in {"text_delta", "message_delta", "output_text_delta", "delta"}:
                chunk = sanitize_output_text(extract_tui_text_chunk(event))
                if not chunk:
                    return
                self._dismiss_thinking(emit_summary=True)
                if not self._current_assistant_chunks and not self._streaming_buffer:
                    self._current_assistant_model = self._current_daemon_model
                    self._current_assistant_timestamp = self._turn_timestamp()
                self._streaming_buffer.append(chunk)
                self._schedule_streaming_flush()

            elif etype in {"text_complete", "message_complete", "output_text_complete", "complete"}:
                self._cancel_streaming_flush_timer()
                self._flush_streaming_buffer()
                self._finalize_assistant_message()

            elif etype == "thinking":
                self._record_thinking_event(str(event.get("text", "")))

            elif etype == "tool_start":
                self._dismiss_thinking(emit_summary=True)
                tid = str(event.get("tool_use_id", "")).strip() or f"tool-{self._run_tool_count + 1}"
                tool_name = str(event.get("tool", "unknown"))
                self._tool_blocks[tid] = {
                    "tool_use_id": tid,
                    "tool": tool_name,
                    "status": "running",
                    "duration_s": 0.0,
                    "input": None,
                    "output": "",
                    "pending_output": "",
                    "pending_stream": "stdout",
                    "elapsed_s": 0.0,
                    "last_progress_render_mono": 0.0,
                    "last_heartbeat_rendered_s": 0.0,
                    "start_rendered": False,
                }
                self._schedule_tool_start_line(tid)
                self._run_tool_count += 1
                if self._status_bar is not None:
                    self._status_bar.set_tool_count(self._run_tool_count)

            elif etype == "tool_progress":
                tid = str(event.get("tool_use_id", "")).strip()
                record = self._tool_blocks.get(tid)
                if record is None and tid:
                    fallback_tool_name = str(event.get("tool", "unknown"))
                    record = {
                        "tool_use_id": tid,
                        "tool": fallback_tool_name,
                        "status": "running",
                        "duration_s": 0.0,
                        "input": None,
                        "output": "",
                        "pending_output": "",
                        "pending_stream": "stdout",
                        "elapsed_s": 0.0,
                        "last_progress_render_mono": 0.0,
                        "last_heartbeat_rendered_s": 0.0,
                        "start_rendered": False,
                    }
                    self._tool_blocks[tid] = record
                    self._schedule_tool_start_line(tid)
                if record is not None:
                    chars = event.get("chars")
                    if isinstance(chars, int):
                        record["chars"] = chars
                    elapsed_raw = event.get("elapsed_s")
                    if isinstance(elapsed_raw, (int, float)):
                        record["elapsed_s"] = float(elapsed_raw)
                    content = event.get("content")
                    if isinstance(content, str) and content:
                        stream = str(event.get("stream", "stdout")).strip().lower() or "stdout"
                        self._queue_tool_progress_content(record, content=content, stream=stream)
                    self._schedule_tool_progress_flush(tid)

            elif etype == "tool_input":
                tid = str(event.get("tool_use_id", "")).strip()
                record = self._tool_blocks.get(tid)
                if record is not None:
                    record["input"] = event.get("input", {})
                    if tid in self._tool_pending_start:
                        self._emit_tool_start_line(tid)

            elif etype == "tool_result":
                tid = str(event.get("tool_use_id", "")).strip()
                status = str(event.get("status", "unknown"))
                duration_raw = event.get("duration_s", 0.0)
                try:
                    duration_s = float(duration_raw or 0.0)
                except (TypeError, ValueError):
                    duration_s = 0.0
                record = self._tool_blocks.get(tid)
                tool_name = str(event.get("tool", "unknown"))
                if record is not None:
                    record["status"] = status
                    record["duration_s"] = duration_s
                    record["elapsed_s"] = duration_s
                    tool_name = str(record.get("tool", tool_name))
                    pending_since = self._tool_pending_start.get(tid)
                    if pending_since is not None:
                        self._tool_pending_start.pop(tid, None)
                        self._cancel_tool_start_timer(tid)
                        if duration_s >= _TOOL_FAST_COMPLETE_SUPPRESS_START_S:
                            self._emit_tool_start_line(tid)
                    self._tool_progress_pending_ids.discard(tid)
                    self._flush_tool_progress_render(tid, force=True)
                tool_input = record.get("input") if isinstance(record, dict) else None
                plain = self._tool_result_plain_text(tool_name, status, duration_s, tool_input)
                self._mount_transcript_widget(
                    render_tool_result_line(
                        tool_name,
                        status=status,
                        duration_s=duration_s,
                        tool_input=tool_input if isinstance(tool_input, dict) else None,
                        tool_use_id=tid,
                    ),
                    plain_text=plain,
                )
                if status != "success":
                    self._error_count += 1
                    self._write_issue(f"ERROR: tool {tool_name} failed ({status}) [{tid}]")
                    self._update_header_status()
                    self._notify(f"{tool_name} tool failed", severity="error", timeout=6.0)
                    if tid:
                        self._mount_transcript_widget(
                            render_system_message("Tool failed. Retry or skip using buttons above the prompt."),
                            plain_text="Tool failed. Retry or skip using buttons above the prompt.",
                        )
                        self._show_tool_error_actions(tool_use_id=tid, tool_name=tool_name)

            elif etype == "consent_prompt":
                context = str(event.get("context", ""))
                raw_options = event.get("options", ["y", "n", "a", "v"])
                options = (
                    [str(item).strip() for item in raw_options if str(item).strip()]
                    if isinstance(raw_options, (list, tuple))
                    else ["y", "n", "a", "v"]
                )
                if not options:
                    options = ["y", "n", "a", "v"]
                self._consent_buffer = [context]
                self._show_consent_prompt(context=context, options=options, alert=True)

            elif etype == "plan":
                rendered = event.get("rendered", "")
                plan_json = event.get("plan_json")
                if plan_json and not rendered:
                    rendered = _json.dumps(plan_json, indent=2)
                self._set_plan_panel(rendered)
                self._received_structured_plan = True
                self._plan_completion_announced = False
                self._plan_step_counter = 0
                self._current_plan_steps_total = 0
                self._current_plan_summary = ""
                self._current_plan_steps = []
                self._current_plan_step_statuses = []
                self._current_plan_active_step = None
                self._plan_updates_seen = False
                if not self._last_run_auto_approve and self._last_prompt:
                    self._pending_plan_prompt = self._last_prompt
                if plan_json and isinstance(plan_json, dict):
                    self._current_plan_summary = str(plan_json.get("summary", plan_json.get("title", ""))).strip()
                    self._current_plan_steps = self._extract_plan_step_descriptions(plan_json)
                    self._current_plan_steps_total = len(self._current_plan_steps)
                    self._current_plan_step_statuses = ["pending"] * self._current_plan_steps_total
                    self._render_plan_panel_from_status()
                    self._mount_transcript_widget(
                        render_plan_panel(plan_json),
                        plain_text=rendered if isinstance(rendered, str) else _json.dumps(plan_json, indent=2),
                    )
                else:
                    self._refresh_plan_status_bar()

            elif etype == "plan_step_update":
                step_index_raw = event.get("step_index")
                status = str(event.get("status", "")).strip().lower()
                if not isinstance(step_index_raw, int):
                    with contextlib.suppress(Exception):
                        step_index_raw = int(step_index_raw)
                if not isinstance(step_index_raw, int):
                    return
                step_index = step_index_raw
                if step_index < 0:
                    return
                if not self._current_plan_step_statuses:
                    return
                if step_index >= len(self._current_plan_step_statuses):
                    self._write_transcript_line(f"[plan] ignoring out-of-range step index: {step_index + 1}")
                    return
                if status not in {"in_progress", "completed"}:
                    return
                self._plan_updates_seen = True
                if status == "in_progress":
                    self._current_plan_active_step = step_index
                    if self._current_plan_step_statuses[step_index] != "completed":
                        self._current_plan_step_statuses[step_index] = "in_progress"
                elif status == "completed":
                    self._current_plan_step_statuses[step_index] = "completed"
                    if self._current_plan_active_step == step_index:
                        self._current_plan_active_step = None
                self._plan_step_counter = sum(1 for item in self._current_plan_step_statuses if item == "completed")
                self._render_plan_panel_from_status()
                if (
                    self._current_plan_steps_total > 0
                    and self._plan_step_counter >= self._current_plan_steps_total
                    and not self._plan_completion_announced
                ):
                    self._plan_completion_announced = True
                    self._write_transcript_line("Plan complete. Clear?")

            elif etype == "plan_complete":
                self._plan_step_counter = self._current_plan_steps_total
                if self._current_plan_step_statuses:
                    self._current_plan_step_statuses = ["completed"] * len(self._current_plan_step_statuses)
                self._current_plan_active_step = None
                self._render_plan_panel_from_status()
                if not self._plan_completion_announced:
                    self._plan_completion_announced = True
                    self._write_transcript_line("Plan complete. Clear?")

            elif etype == "artifact":
                paths = event.get("paths", [])
                if paths:
                    self._add_artifact_paths(paths)

            elif etype == "error":
                error_info = classify_tui_error_event(event)
                error_message = str(error_info.get("message", "")).strip()
                error_text = error_message
                if not error_text.startswith("ERROR:"):
                    error_text = f"ERROR: {error_text}"
                normalized_error = error_text.lower()
                if self._pending_model_select_value and (
                    "set_tier" in normalized_error
                    or "cannot set tier" in normalized_error
                    or "tier" in normalized_error
                ):
                    self._pending_model_select_value = None
                    self._refresh_model_select()
                self._error_count += 1
                self._write_issue(error_text)
                self._update_header_status()
                toast_message, severity, timeout = summarize_error_for_toast(error_info)
                self._notify(toast_message, severity=severity, timeout=timeout)

                category = str(error_info.get("category", ERROR_CATEGORY_FATAL))
                if category == ERROR_CATEGORY_TRANSIENT:
                    self._reset_error_action_prompt()
                elif category == ERROR_CATEGORY_TOOL_ERROR:
                    tool_use_id = str(error_info.get("tool_use_id", "")).strip()
                    if tool_use_id:
                        tool_record = self._tool_blocks.get(tool_use_id)
                        tool_name = str(tool_record.get("tool", "tool")) if isinstance(tool_record, dict) else "tool"
                        self._show_tool_error_actions(tool_use_id=tool_use_id, tool_name=tool_name)
                elif category == ERROR_CATEGORY_ESCALATABLE:
                    next_tier = str(error_info.get("next_tier", "")).strip() or self._next_available_tier_name()
                    self._show_escalation_actions(next_tier=next_tier or None)
                elif category == ERROR_CATEGORY_AUTH_ERROR:
                    self._mount_transcript_widget(
                        render_system_message(
                            "Authentication failed. Verify credentials/permissions for the active provider."
                        ),
                        plain_text="Authentication failed. Verify credentials/permissions for the active provider.",
                    )
                    self._reset_error_action_prompt()
                elif category == ERROR_CATEGORY_FATAL:
                    self._reset_error_action_prompt()

            elif etype == "warning":
                warn_text = event.get("text", "")
                if not warn_text.startswith("WARN:"):
                    warn_text = f"WARN: {warn_text}"
                self._warning_count += 1
                self._write_issue(warn_text)
                self._update_header_status()

        def _discover_session_log_path(self, session_id: str | None) -> str | None:
            if not session_id:
                return None
            try:
                matches = list(logs_dir().glob(f"*_{session_id}.jsonl"))
            except Exception:
                return None
            if not matches:
                return None
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(matches[0])

        def _persist_run_transcript(
            self,
            *,
            pid: int | None,
            session_id: str | None,
            prompt: str,
            auto_approve: bool,
            exit_code: int,
            output_text: str,
        ) -> str | None:
            if not output_text:
                return None
            try:
                store = ArtifactStore()
                ref = store.write_text(
                    kind="tui_transcript",
                    text=output_text,
                    metadata={
                        "pid": pid,
                        "session_id": session_id,
                        "prompt": prompt,
                        "auto_approve": auto_approve,
                        "exit_code": exit_code,
                    },
                )
                return str(ref.path)
            except Exception:
                return None

        def _finalize_turn(self, *, exit_status: str) -> None:
            self._run_active_tier_warning_emitted = False
            if self._status_timer is not None:
                self._status_timer.stop()
                self._status_timer = None
            elapsed = time.time() - self._run_start_time if self._run_start_time is not None else 0.0
            if self._status_bar is not None:
                self._status_bar.set_state("idle")
                self._status_bar.set_elapsed(elapsed)
                self._status_bar.set_plan_step(current=None, total=None)
            self._run_start_time = None
            self._query_active = False
            self._clear_pending_tool_starts()
            self._cancel_tool_progress_flush_timer()
            self._tool_progress_pending_ids = set()
            for tool_use_id in list(self._tool_blocks.keys()):
                self._flush_tool_progress_render(tool_use_id, force=True)

            self._write_transcript(
                f"[run] completed in {elapsed:.1f}s ({self._run_tool_count} tool calls, status={exit_status})"
            )

            self._finalize_assistant_message()
            self._dismiss_thinking(emit_summary=True)

            output_text = "".join(self._turn_output_chunks)
            self._turn_output_chunks = []
            transcript_path = self._persist_run_transcript(
                pid=(self._proc.pid if self._proc is not None else None),
                session_id=self._daemon_session_id,
                prompt=self._last_prompt or "",
                auto_approve=self._last_run_auto_approve,
                exit_code=0 if exit_status == "ok" else 1,
                output_text=output_text,
            )
            if transcript_path:
                self._add_artifact_paths([transcript_path])

            log_path = self._discover_session_log_path(self._daemon_session_id)
            if log_path:
                self._add_artifact_paths([log_path])

            if not self._received_structured_plan:
                extracted_plan = extract_plan_section_from_output(sanitize_output_text(output_text))
                if extracted_plan:
                    self._pending_plan_prompt = self._last_prompt
                    self._set_plan_panel(extracted_plan)
                    self._write_transcript_line(render_tui_hint_after_plan())

            self._reset_consent_panel()
            self._received_structured_plan = False
            self._save_session()

        def _handle_daemon_exit(self, proc: _DaemonTransport, *, return_code: int) -> None:
            if self._proc is not proc:
                return
            was_query_active = self._query_active
            self._daemon_ready = False
            self._pending_model_select_value = None
            self._query_active = False
            self._context_ready_for_sync = bool(self._context_sources)
            self._sops_ready_for_sync = bool(self._active_sop_names)
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            self._clear_pending_tool_starts()
            self._proc = None
            self._runner_thread = None

            if self._status_timer is not None:
                self._status_timer.stop()
                self._status_timer = None
            if self._status_bar is not None:
                self._status_bar.set_state("idle")

            if was_query_active:
                self._finalize_turn(exit_status="error")
            else:
                self._reset_thinking_state()
            if self._is_shutting_down:
                return
            self._write_transcript_line(f"[daemon] exited unexpectedly (code {return_code}).")
            self._write_transcript_line("[daemon] run /daemon restart to restart the background agent.")

        def _stream_daemon_output(self, proc: _DaemonTransport) -> None:
            try:
                while True:
                    raw_line = proc.read_line()
                    if raw_line == "":
                        break
                    self._call_from_thread_safe(self._handle_output_line, raw_line.rstrip("\n"), raw_line)
            except Exception as exc:
                self._call_from_thread_safe(self._write_transcript_line, f"[daemon] output stream error: {exc}")
            finally:
                return_code = 0
                with contextlib.suppress(Exception):
                    return_code = proc.wait()
                self._call_from_thread_safe(self._handle_daemon_exit, proc, return_code=return_code)

        def _shutdown_transport(self, proc: _DaemonTransport) -> None:
            send_daemon_command(proc, {"cmd": "shutdown"})
            with contextlib.suppress(Exception):
                proc.wait(timeout=3.0)
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.close()

        def _spawn_daemon(self, *, restart: bool = False) -> None:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                if restart:
                    self._pending_model_select_value = None
                    self._shutdown_transport(proc)
                    self._proc = None
                else:
                    return

            requested_session_id = (self._daemon_session_id or "").strip() or uuid.uuid4().hex
            self._daemon_session_id = requested_session_id
            daemon: _DaemonTransport | None = None
            broker_error: Exception | None = None

            try:
                daemon = _SocketTransport.connect(
                    session_id=requested_session_id,
                    cwd=Path.cwd(),
                    client_name="swarmee-tui",
                    surface="tui",
                )
            except Exception as exc:
                broker_error = exc

            if daemon is None:
                try:
                    daemon_proc = spawn_swarmee_daemon(
                        session_id=requested_session_id,
                        env_overrides=self._model_env_overrides(),
                    )
                    daemon = _SubprocessTransport(daemon_proc)
                except Exception as exc:
                    self._daemon_ready = False
                    self._write_transcript_line(f"[daemon] failed to start: {exc}")
                    return

            self._proc = daemon
            self._daemon_ready = False
            self._context_ready_for_sync = bool(self._context_sources)
            self._sops_ready_for_sync = bool(self._active_sop_names)
            self._runner_thread = threading.Thread(
                target=self._stream_daemon_output,
                args=(daemon,),
                daemon=True,
                name="swarmee-tui-daemon-stream",
            )
            self._runner_thread.start()
            if isinstance(daemon, _SocketTransport):
                self._write_transcript_line("[daemon] connected to runtime broker, waiting for ready event.")
            else:
                if broker_error is not None and not isinstance(broker_error, FileNotFoundError):
                    self._write_transcript_line(
                        f"[daemon] runtime broker unavailable ({broker_error}); using local daemon."
                    )
                self._write_transcript_line("[daemon] started, waiting for ready event.")
            self._save_session()

        def _tick_status(self) -> None:
            if self._run_start_time is not None and self._status_bar is not None:
                self._status_bar.set_elapsed(time.time() - self._run_start_time)

        def _start_run(self, prompt: str, *, auto_approve: bool, mode: str | None = None) -> None:
            if not self._daemon_ready:
                self._write_transcript_line("[run] daemon is not ready. Use /daemon restart.")
                return
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._write_transcript_line("[run] daemon is not running. Use /daemon restart.")
                self._daemon_ready = False
                return
            if self._query_active:
                self._write_transcript_line("[run] already running; use /stop.")
                return
            self._dismiss_action_sheet(restore_focus=False)
            self._sync_selected_model_before_run()

            self._pending_plan_prompt = None
            self._reset_artifacts_panel()
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            self._reset_issues_panel()
            self._current_assistant_chunks = []
            self._streaming_buffer = []
            self._cancel_streaming_flush_timer()
            self._current_assistant_model = None
            self._current_assistant_timestamp = None
            self._assistant_placeholder_written = False
            self._reset_thinking_state()
            self._last_thinking_text = ""
            self._tool_blocks = {}
            self._clear_pending_tool_starts()
            self._tool_progress_pending_ids = set()
            self._cancel_tool_progress_flush_timer()
            self._run_tool_count = 0
            self._run_start_time = time.time()
            self._run_active_tier_warning_emitted = False
            self._plan_step_counter = 0
            self._plan_completion_announced = False
            mode_normalized = (mode or "").strip().lower()
            if mode_normalized == "execute" and self._current_plan_steps_total > 0:
                self._current_plan_step_statuses = ["pending"] * self._current_plan_steps_total
                self._current_plan_active_step = None
                self._plan_updates_seen = False
                self._render_plan_panel_from_status()
            else:
                self._current_plan_steps_total = 0
                self._current_plan_summary = ""
                self._current_plan_steps = []
                self._current_plan_step_statuses = []
                self._current_plan_active_step = None
                self._plan_updates_seen = False
            self._received_structured_plan = False
            self._turn_output_chunks = []
            self._last_usage = None
            self._last_cost_usd = None
            if self._status_bar is not None:
                self._status_bar.set_state("running")
                self._status_bar.set_tool_count(0)
                self._status_bar.set_elapsed(0.0)
                self._status_bar.set_model(self._current_model_summary())
                self._status_bar.set_usage(None, cost_usd=None)
                self._status_bar.set_context(
                    prompt_tokens_est=self._last_prompt_tokens_est,
                    budget_tokens=self._last_budget_tokens,
                )
                if mode_normalized == "execute" and self._current_plan_steps_total > 0:
                    self._refresh_plan_status_bar()
                else:
                    self._status_bar.set_plan_step(current=None, total=None)
            self._refresh_prompt_metrics()
            if self._status_timer is not None:
                self._status_timer.stop()
            self._status_timer = self.set_interval(1.0, self._tick_status)
            self._last_prompt = prompt
            self._last_run_auto_approve = auto_approve
            self._query_active = True
            desired_tier = ""
            pending_value = (self._pending_model_select_value or "").strip().lower()
            if "|" in pending_value:
                _pending_provider, pending_tier = pending_value.split("|", 1)
                desired_tier = pending_tier.strip().lower()
            if not desired_tier:
                desired_tier = (self._model_tier_override or "").strip().lower()
            if not desired_tier:
                with contextlib.suppress(Exception):
                    selector = self.query_one("#model_select", Select)
                    selected_value = str(getattr(selector, "value", "")).strip()
                    parsed = parse_model_select_value(selected_value)
                    if parsed is not None:
                        _provider_name, parsed_tier = parsed
                        desired_tier = parsed_tier.strip().lower()
            command: dict[str, Any] = {
                "cmd": "query",
                "text": prompt,
                "auto_approve": bool(auto_approve),
            }
            if desired_tier:
                command["tier"] = desired_tier
            if mode:
                command["mode"] = mode
            if not send_daemon_command(proc, command):
                self._query_active = False
                if self._status_timer is not None:
                    self._status_timer.stop()
                    self._status_timer = None
                if self._status_bar is not None:
                    self._status_bar.set_state("idle")
                self._write_transcript_line("[run] failed to send query to daemon.")

        def _stop_run(self) -> None:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._write_transcript_line("[run] no active run.")
                self._daemon_ready = False
                self._reset_consent_panel()
                self._reset_error_action_prompt()
                return
            if not self._query_active:
                self._write_transcript_line("[run] no active run.")
                self._reset_consent_panel()
                self._reset_error_action_prompt()
                return
            self._flush_all_streaming_buffers()
            self._clear_pending_tool_starts()
            self._dismiss_thinking(emit_summary=True)
            if send_daemon_command(proc, {"cmd": "interrupt"}):
                self._write_transcript_line("[run] interrupt requested.")
            else:
                self._write_transcript_line("[run] failed to send interrupt.")
            self._reset_consent_panel()
            self._reset_error_action_prompt()

        def _refresh_prompt_metrics(self) -> None:
            if self._prompt_metrics is None:
                return
            set_context = getattr(self._prompt_metrics, "set_context", None)
            if callable(set_context):
                set_context(
                    prompt_tokens_est=self._last_prompt_tokens_est,
                    budget_tokens=self._last_budget_tokens,
                    animate=True,
                )
            set_prompt_estimate = getattr(self._prompt_metrics, "set_prompt_input_estimate", None)
            if callable(set_prompt_estimate):
                set_prompt_estimate(self._prompt_input_tokens_est)

        def action_quit(self) -> None:
            self._is_shutting_down = True
            timer = self._prompt_estimate_timer
            self._prompt_estimate_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()
            timeline_timer = self._session_timeline_refresh_timer
            self._session_timeline_refresh_timer = None
            if timeline_timer is not None:
                with contextlib.suppress(Exception):
                    timeline_timer.stop()
            self._cancel_streaming_flush_timer()
            self._cancel_tool_progress_flush_timer()
            self._clear_pending_tool_starts()
            self._reset_thinking_state()
            if self._proc is not None and self._proc.poll() is None:
                self._shutdown_transport(self._proc)
            if self._runner_thread is not None and self._runner_thread.is_alive():
                with contextlib.suppress(Exception):
                    self._runner_thread.join(timeout=1.0)
            self._save_session()
            self.exit(return_code=0)

        def action_copy_transcript(self) -> None:
            self._copy_text(self._get_transcript_text(), label="transcript")

        def action_copy_plan(self) -> None:
            self._copy_text((self._plan_text or "").rstrip() + "\n", label="plan")

        def action_copy_issues(self) -> None:
            self._flush_issue_repeats()
            payload = ("\n".join(self._issues_lines).rstrip() + "\n") if self._issues_lines else ""
            self._copy_text(payload, label="issues")

        def action_copy_artifacts(self) -> None:
            self._copy_text(self._get_artifacts_text(), label="artifacts")

        def action_focus_prompt(self) -> None:
            self.query_one("#prompt", PromptTextArea).focus()

        def action_submit_prompt(self) -> None:
            prompt_widget = self.query_one("#prompt", PromptTextArea)
            text = (prompt_widget.text or "").strip()
            prompt_widget.clear()
            if text:
                self._prompt_history.append(text)
                if len(self._prompt_history) > self._MAX_PROMPT_HISTORY:
                    self._prompt_history = self._prompt_history[-self._MAX_PROMPT_HISTORY:]
                self._history_index = -1
                self._handle_user_input(text)

        def action_interrupt_run(self) -> None:
            proc = self._proc
            if proc is None or proc.poll() is not None or not self._query_active:
                self._reset_consent_panel()
                self._reset_error_action_prompt()
                return
            self._flush_all_streaming_buffers()
            self._clear_pending_tool_starts()
            self._dismiss_thinking(emit_summary=True)
            send_daemon_command(proc, {"cmd": "interrupt"})
            self._write_transcript_line("[run] interrupted.")
            self._reset_consent_panel()
            self._reset_error_action_prompt()

        def action_copy_selection(self) -> None:
            focused = getattr(self, "focused", None)
            if isinstance(focused, TextArea):
                selected_text = focused.selected_text or ""
                if selected_text.strip():
                    self._copy_text(selected_text, label="selection")
                    return
                if focused.id == "transcript_text":
                    transcript_text = self._get_transcript_text()
                    if transcript_text.strip():
                        self._copy_text(transcript_text, label="transcript")
                        return
                    self._notify("transcript: nothing to copy.", severity="warning")
                    return
                focused_text = (getattr(focused, "text", "") or "").strip()
                if focused.id in {"issues", "plan", "artifacts", "agent_summary"} and focused_text:
                    self._copy_text(focused_text + "\n", label=f"{focused.id} pane")
                    return
                self._notify("Select text first.", severity="warning")
                return

            transcript_widget: Any
            if self._transcript_mode == "text":
                transcript_widget = self.query_one("#transcript_text", TextArea)
            else:
                transcript_widget = self.query_one("#transcript", RichLog)

            if isinstance(transcript_widget, TextArea):
                selected_text = self._get_richlog_selection_text(transcript_widget)
                if selected_text.strip():
                    self._copy_text(selected_text, label="selection")
                    return
                transcript_text = self._get_transcript_text()
                if transcript_text.strip():
                    self._copy_text(transcript_text, label="transcript")
                    return
                self._notify("Select text first.", severity="warning")
                return

            node = focused
            while node is not None:
                if node is transcript_widget:
                    self._copy_text(self._get_transcript_text(), label="transcript")
                    return
                node = getattr(node, "parent", None)

            self._notify("No text area focused.", severity="warning")

        def action_widen_side(self) -> None:
            if self._split_ratio > 1:
                self._split_ratio -= 1
                self._apply_split_ratio()

        def action_widen_transcript(self) -> None:
            if self._split_ratio < 4:
                self._split_ratio += 1
                self._apply_split_ratio()

        def _apply_split_ratio(self) -> None:
            transcript = self.query_one("#transcript", RichLog)
            transcript_text = self.query_one("#transcript_text", TextArea)
            side = self.query_one("#side", Vertical)
            transcript.styles.width = f"{self._split_ratio}fr"
            transcript_text.styles.width = f"{self._split_ratio}fr"
            side.styles.width = "1fr"
            self.refresh(layout=True)

        def action_toggle_transcript_mode(self) -> None:
            self._toggle_transcript_mode()

        def action_search_transcript(self) -> None:
            prompt_widget = self.query_one("#prompt", PromptTextArea)
            prompt_widget.clear()
            # Hide palette first to avoid rendering issues
            if self._command_palette is not None:
                self._command_palette.hide()
            for method_name in ("insert", "insert_text_at_cursor"):
                method = getattr(prompt_widget, method_name, None)
                if callable(method):
                    with contextlib.suppress(Exception):
                        method("/search ")
                        break
            # Hide palette again after insert (on_text_area_changed may re-show it)
            if self._command_palette is not None:
                self._command_palette.hide()
            prompt_widget.focus()

        def _search_transcript(self, term: str) -> None:
            if not term.strip():
                self._write_transcript_line("Usage: /search <term>")
                return
            term_lower = term.lower()
            transcript_text = self._get_transcript_text()
            if term_lower in transcript_text.lower():
                with contextlib.suppress(Exception):
                    self.query_one("#transcript", RichLog).scroll_end(animate=True)
                self._write_transcript_line("[search] found match in transcript.")
                return
            self._write_transcript_line(f"[search] no match for '{term}'.")

        def _request_context_compact(self) -> None:
            if self._query_active:
                self._write_transcript_line("[compact] unavailable while a run is active.")
                return
            if not self._daemon_ready:
                self._write_transcript_line("[compact] daemon is not ready. Use /daemon restart.")
                return
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._daemon_ready = False
                self._write_transcript_line("[compact] daemon is not running. Use /daemon restart.")
                return
            self._notify("Compacting context...", severity="information", timeout=5.0)
            if send_daemon_command(proc, {"cmd": "compact"}):
                self._write_transcript_line("[compact] requested context compaction.")
            else:
                self._write_transcript_line("[compact] failed to send compact command.")

        def _expand_tool_call(self, tool_use_id: str) -> None:
            tid = tool_use_id.strip()
            if not tid:
                self._write_transcript_line(_EXPAND_USAGE_TEXT)
                return
            record = self._tool_blocks.get(tid)
            if record is None:
                self._write_transcript_line(f"[expand] unknown tool id: {tid}")
                return
            self._mount_transcript_widget(
                render_tool_details_panel(record),
                plain_text=_json.dumps(record, indent=2, ensure_ascii=False),
            )

        def _open_artifact_path(self, path: str) -> None:
            resolved = str(path or "").strip()
            if not resolved:
                self._write_transcript_line("[open] invalid artifact path.")
                return
            editor = os.environ.get("EDITOR", "")
            try:
                if editor:
                    subprocess.Popen([editor, resolved])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", resolved])
                elif shutil.which("xdg-open"):
                    subprocess.Popen(["xdg-open", resolved])
                else:
                    self._write_transcript_line(f"[open] set $EDITOR. Path: {resolved}")
                    return
                self._write_transcript_line(f"[open] opened: {resolved}")
            except Exception as exc:
                self._write_transcript_line(f"[open] failed: {exc}")

        def _open_artifact(self, index_str: str) -> None:
            try:
                index = int(index_str.strip()) - 1
            except ValueError:
                self._write_transcript_line("Usage: /open <number>")
                return
            self._render_artifacts_panel()
            entries = self._artifact_entries
            if index < 0 or index >= len(entries):
                self._write_transcript_line(f"[open] invalid index. {len(entries)} artifacts available.")
                return
            path = str(entries[index].get("path", "")).strip()
            self._open_artifact_path(path)

        def _copy_selected_artifact_path(self) -> None:
            selected = self._artifact_entry_by_item_id(self._artifact_selected_item_id)
            if selected is None:
                self._notify("Select an artifact first.", severity="warning")
                return
            path = str(selected.get("path", "")).strip()
            if not path:
                self._notify("Selected artifact has no path.", severity="warning")
                return
            self._copy_text(path, label="artifact path")

        def _add_selected_artifact_as_context(self) -> None:
            selected = self._artifact_entry_by_item_id(self._artifact_selected_item_id)
            if selected is None:
                self._notify("Select an artifact first.", severity="warning")
                return
            path = str(selected.get("path", "")).strip()
            if not path:
                self._notify("Selected artifact has no path.", severity="warning")
                return
            payload = artifact_context_source_payload(path)
            self._add_context_source(payload)
            self._write_transcript_line(f"[context] added file source from artifact: {path}")

        def _session_retry_tool(self, tool_use_id: str) -> None:
            tool_id = str(tool_use_id or "").strip()
            if not tool_id:
                self._notify("Issue has no tool_use_id.", severity="warning")
                return
            proc = self._proc
            if not self._daemon_ready or proc is None or proc.poll() is not None:
                self._write_transcript_line("[session] daemon is not ready.")
                return
            if send_daemon_command(proc, {"cmd": "retry_tool", "tool_use_id": tool_id}):
                self._write_transcript_line(f"[session] retry requested for tool {tool_id}.")
            else:
                self._write_transcript_line("[session] failed to send retry request.")

        def _session_skip_tool(self, tool_use_id: str) -> None:
            tool_id = str(tool_use_id or "").strip()
            if not tool_id:
                self._notify("Issue has no tool_use_id.", severity="warning")
                return
            proc = self._proc
            if not self._daemon_ready or proc is None or proc.poll() is not None:
                self._write_transcript_line("[session] daemon is not ready.")
                return
            if send_daemon_command(proc, {"cmd": "skip_tool", "tool_use_id": tool_id}):
                self._write_transcript_line(f"[session] skip requested for tool {tool_id}.")
            else:
                self._write_transcript_line("[session] failed to send skip request.")

        def _session_escalate_tier(self, tier_name: str | None = None) -> None:
            next_tier = str(tier_name or "").strip().lower() or (self._next_available_tier_name() or "")
            if not next_tier:
                self._write_transcript_line("[session] no higher tier available.")
                return
            proc = self._proc
            if not self._daemon_ready or proc is None or proc.poll() is not None:
                self._write_transcript_line("[session] daemon is not ready.")
                return
            if send_daemon_command(proc, {"cmd": "set_tier", "tier": next_tier}):
                self._write_transcript_line(f"[session] tier change requested: {next_tier}")
            else:
                self._write_transcript_line("[session] failed to send tier change request.")

        def _session_interrupt(self) -> None:
            proc = self._proc
            if not self._daemon_ready or proc is None or proc.poll() is not None:
                self._write_transcript_line("[session] daemon is not ready.")
                return
            if send_daemon_command(proc, {"cmd": "interrupt"}):
                self._write_transcript_line("[session] interrupt requested.")
            else:
                self._write_transcript_line("[session] failed to send interrupt.")

        def _save_session(self) -> None:
            try:
                session_path = sessions_dir() / "tui_session.json"
                session_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "prompt_history": self._prompt_history[-self._MAX_PROMPT_HISTORY:],
                    "last_prompt": self._last_prompt,
                    "plan_text": self._plan_text,
                    "artifacts": self._artifacts,
                    "context_sources": self._context_sources,
                    "active_sop_names": sorted(self._active_sop_names),
                    "daemon_session_id": self._daemon_session_id,
                    "available_restore_session_id": self._available_restore_session_id,
                    "available_restore_turn_count": self._available_restore_turn_count,
                    "model_provider_override": self._model_provider_override,
                    "model_tier_override": self._model_tier_override,
                    "default_auto_approve": self._default_auto_approve,
                    "split_ratio": self._split_ratio,
                    "session_view_mode": self._session_view_mode,
                    "agent_studio_view_mode": self._agent_studio_view_mode,
                }
                session_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass

        def _load_session(self) -> None:
            try:
                session_path = sessions_dir() / "tui_session.json"
                if not session_path.exists():
                    return
                data = _json.loads(session_path.read_text(encoding="utf-8", errors="replace"))
                self._prompt_history = data.get("prompt_history", [])[-self._MAX_PROMPT_HISTORY:]
                self._last_prompt = data.get("last_prompt")
                plan_text = data.get("plan_text", "")
                if plan_text and plan_text != "(no plan)":
                    self._set_plan_panel(plan_text)
                artifacts = data.get("artifacts", [])
                if artifacts:
                    self._artifacts = artifacts
                    self._render_artifacts_panel()
                self._context_sources = _normalize_context_sources(data.get("context_sources", []))
                self._render_context_sources_panel()
                self._context_ready_for_sync = bool(self._context_sources)
                loaded_active_sops = data.get("active_sop_names", [])
                if isinstance(loaded_active_sops, list):
                    self._active_sop_names = {str(item).strip() for item in loaded_active_sops if str(item).strip()}
                self._refresh_sop_catalog()
                self._render_sop_panel()
                self._sops_ready_for_sync = bool(self._active_sop_names)
                self._daemon_session_id = str(data.get("daemon_session_id", "")).strip() or None
                self._available_restore_session_id = str(data.get("available_restore_session_id", "")).strip() or None
                restore_turn_count_raw = data.get("available_restore_turn_count", 0)
                try:
                    self._available_restore_turn_count = max(0, int(restore_turn_count_raw or 0))
                except (TypeError, ValueError):
                    self._available_restore_turn_count = 0
                # Do not restore model overrides from prior sessions.
                # The daemon-reported model_info is the source of truth for startup model state.
                self._model_provider_override = None
                self._model_tier_override = None
                self._default_auto_approve = data.get("default_auto_approve", False)
                self._split_ratio = data.get("split_ratio", 2)
                self._session_view_mode = normalize_session_view_mode(data.get("session_view_mode"))
                self._agent_studio_view_mode = normalize_agent_studio_view_mode(data.get("agent_studio_view_mode"))
                self._apply_split_ratio()
                self._set_session_view_mode(self._session_view_mode)
                self._set_agent_studio_view_mode(self._agent_studio_view_mode)
                self._refresh_model_select()
                self._update_header_status()
                self._update_prompt_placeholder()
                if self._status_bar is not None:
                    self._status_bar.set_model(self._current_model_summary())
                if self._prompt_history:
                    self._write_transcript(f"[session] restored ({len(self._prompt_history)} history entries).")
            except Exception:
                pass

        def on_key(self, event: Any) -> None:
            key = str(getattr(event, "key", "")).lower()

            if key in {"ctrl+k", "ctrl+space", "ctrl+@"}:
                event.stop()
                event.prevent_default()
                self.action_open_action_sheet()
                return

            if self._consent_active and key in _CONSENT_CHOICES:
                event.stop()
                event.prevent_default()
                self._submit_consent_choice(key)
                return

            if key in {"ctrl+c", "meta+c", "super+c", "cmd+c", "command+c"}:
                event.stop()
                event.prevent_default()
                self.action_copy_selection()
                return

        def on_checkbox_changed(self, event: Any) -> None:
            checkbox = getattr(event, "checkbox", None)
            checkbox_id = str(getattr(checkbox, "id", "")).strip()
            if not checkbox_id:
                return
            sop_name = self._sop_toggle_id_to_name.get(checkbox_id)
            if not sop_name:
                return
            value = bool(getattr(event, "value", False))
            self._set_sop_active(sop_name, value, sync=True, announce=True)

        def on_select_changed(self, event: Any) -> None:
            select_widget = getattr(event, "select", None)
            select_id = str(getattr(select_widget, "id", "")).strip().lower()
            if select_id != "model_select":
                return
            if self._model_select_syncing:
                return

            value = str(getattr(event, "value", "")).strip()
            if not value:
                return
            if value == _MODEL_AUTO_VALUE:
                self._pending_model_select_value = None
                self._model_provider_override = None
                self._model_tier_override = None
            elif value == _MODEL_LOADING_VALUE:
                return
            elif "|" in value:
                provider, tier = value.split("|", 1)
                requested_provider = provider.strip().lower()
                requested_tier = tier.strip().lower()
                if not requested_provider or not requested_tier:
                    return
                if self._daemon_ready and self._proc is not None and self._proc.poll() is None:
                    current_tier = (self._daemon_tier or "").strip().lower()
                    current_provider = (self._daemon_provider or "").strip().lower()
                    if requested_provider == current_provider and requested_tier == current_tier:
                        self._pending_model_select_value = None
                        self._model_provider_override = requested_provider or None
                        self._model_tier_override = requested_tier or None
                        self._update_header_status()
                        self._update_prompt_placeholder()
                        if self._status_bar is not None:
                            self._status_bar.set_model(self._current_model_summary())
                        return
                    if self._query_active:
                        if should_skip_active_run_tier_warning(
                            requested_provider=requested_provider,
                            requested_tier=requested_tier,
                            pending_value=self._pending_model_select_value,
                        ):
                            self._model_provider_override = requested_provider or None
                            self._model_tier_override = requested_tier or None
                            self._update_header_status()
                            self._update_prompt_placeholder()
                            if self._status_bar is not None:
                                self._status_bar.set_model(self._current_model_summary())
                            return
                        self._pending_model_select_value = f"{requested_provider}|{requested_tier}"
                        self._model_provider_override = requested_provider or None
                        self._model_tier_override = requested_tier or None
                        self._update_header_status()
                        self._update_prompt_placeholder()
                        if self._status_bar is not None:
                            self._status_bar.set_model(self._current_model_summary())
                        return
                    else:
                        # Persist desired selection locally; the next query command carries `tier` and applies
                        # atomically in daemon before invocation.
                        self._pending_model_select_value = f"{requested_provider}|{requested_tier}"
                else:
                    self._pending_model_select_value = None
                self._model_provider_override = requested_provider or None
                self._model_tier_override = requested_tier or None
            self._update_header_status()
            self._update_prompt_placeholder()
            if self._status_bar is not None:
                self._status_bar.set_model(self._current_model_summary())
            self._refresh_agent_summary()
            # The model selector is always visible; avoid transient notifications.

        def on_sidebar_list_selection_changed(self, event: Any) -> None:
            sidebar_list = getattr(event, "sidebar_list", None)
            if sidebar_list is None:
                return

            if sidebar_list is self._session_issue_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_issue = self._session_issue_by_id(selected_id)
                self._set_session_issue_selection(selected_issue)
                return

            if sidebar_list is self._session_timeline_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_event = self._session_timeline_event_by_id(selected_id)
                self._set_session_timeline_selection(selected_event)
                return

            if sidebar_list is self._artifacts_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_entry = self._artifact_entry_by_item_id(selected_id)
                self._set_artifact_selection(selected_entry)
                return

            if sidebar_list is self._agent_profile_list:
                if self._agent_form_syncing:
                    return
                selected_id = str(getattr(event, "item_id", "")).strip()
                if not selected_id or selected_id == _AGENT_PROFILE_SELECT_NONE:
                    return
                profile = self._lookup_saved_profile(selected_id)
                if profile is None:
                    return
                self._load_profile_into_draft(profile)
                return

            if sidebar_list is self._agent_tools_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_item = self._agent_tools_item_by_id(selected_id)
                self._set_agent_tools_selection(selected_item)
                return

            if sidebar_list is self._agent_team_list:
                selected_id = str(getattr(event, "item_id", "")).strip()
                selected_item = self._agent_team_item_by_id(selected_id)
                self._set_agent_team_selection(selected_item)
                return

        def on_sidebar_detail_action_selected(self, event: Any) -> None:
            detail = getattr(event, "detail", None)
            action_id = str(getattr(event, "action_id", "")).strip().lower()
            if detail is None or not action_id:
                return

            if detail is self._session_issue_detail:
                issue = self._session_issue_by_id(self._session_selected_issue_id)
                if issue is None:
                    self._notify("Select an issue first.", severity="warning")
                    return
                tool_use_id = str(issue.get("tool_use_id", "")).strip()
                if action_id == "session_issue_retry_tool":
                    self._session_retry_tool(tool_use_id)
                    return
                if action_id == "session_issue_skip_tool":
                    self._session_skip_tool(tool_use_id)
                    return
                if action_id == "session_issue_escalate_tier":
                    self._session_escalate_tier(str(issue.get("next_tier", "")).strip())
                    return
                if action_id == "session_issue_interrupt":
                    self._session_interrupt()
                    return

            if detail is self._session_timeline_detail:
                selected_event = self._session_timeline_event_by_id(self._session_timeline_selected_event_id)
                if selected_event is None:
                    self._notify("Select a timeline event first.", severity="warning")
                    return
                if action_id == "session_timeline_copy_json":
                    payload = _json.dumps(selected_event, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                    self._copy_text(payload, label="timeline event json")
                    return
                if action_id == "session_timeline_copy_summary":
                    summary = summarize_session_timeline_event(selected_event).rstrip() + "\n"
                    self._copy_text(summary, label="timeline event summary")
                    return

            if detail is self._artifacts_detail:
                selected = self._artifact_entry_by_item_id(self._artifact_selected_item_id)
                if selected is None:
                    self._notify("Select an artifact first.", severity="warning")
                    return
                path = str(selected.get("path", "")).strip()
                if action_id == "artifact_action_open":
                    self._open_artifact_path(path)
                    return
                if action_id == "artifact_action_copy_path":
                    self._copy_selected_artifact_path()
                    return
                if action_id == "artifact_action_add_context":
                    self._add_selected_artifact_as_context()
                    return

        def on_input_changed(self, event: Any) -> None:
            input_widget = getattr(event, "input", None)
            input_id = str(getattr(input_widget, "id", "")).strip().lower()
            if input_id in {"agent_profile_id", "agent_profile_name"}:
                if self._agent_form_syncing:
                    return
                self._set_agent_draft_dirty(True)
                return
            if input_id in {
                "agent_tools_override_consent",
                "agent_tools_override_allowlist",
                "agent_tools_override_blocklist",
            }:
                if self._agent_tools_form_syncing:
                    return
                self._set_agent_tools_status("Override draft changes pending.")
                return
            if input_id in {"agent_team_preset_id", "agent_team_preset_name", "agent_team_preset_description"}:
                if self._agent_team_form_syncing:
                    return
                self._set_agent_team_status("Team preset draft changes pending.")
                self._set_agent_draft_dirty(True)
                return

        def on_text_area_changed(self, event: Any) -> None:
            text_area = getattr(event, "text_area", None)
            text_area_id = str(getattr(text_area, "id", "")).strip().lower()
            if text_area_id != "agent_team_preset_spec":
                return
            if self._agent_team_form_syncing:
                return
            self._set_agent_team_status("Team preset draft changes pending.")
            self._set_agent_draft_dirty(True)

        def _sync_selected_model_before_run(self) -> None:
            selected_value: str | None = None
            with contextlib.suppress(Exception):
                selector = self.query_one("#model_select", Select)
                selected_value = str(getattr(selector, "value", "")).strip()
            parsed = parse_model_select_value(selected_value)
            if parsed is None:
                return

            requested_provider, requested_tier = parsed
            self._model_provider_override = requested_provider or None
            self._model_tier_override = requested_tier or None

            current_provider = (self._daemon_provider or "").strip().lower()
            current_tier = (self._daemon_tier or "").strip().lower()
            if (
                current_provider
                and current_tier
                and requested_provider == current_provider
                and requested_tier == current_tier
            ):
                self._pending_model_select_value = None
                return
            self._pending_model_select_value = f"{requested_provider}|{requested_tier}"

        def _dispatch_plan_action(self, action: str) -> None:
            normalized = action.strip().lower()
            if normalized == "approve":
                if not self._pending_plan_prompt:
                    self._write_transcript_line("[run] no pending plan.")
                    return
                self._start_run(self._pending_plan_prompt, auto_approve=True, mode="execute")
                return
            if normalized == "replan":
                if not self._last_prompt:
                    self._write_transcript_line("[run] no previous prompt to replan.")
                    return
                self._start_run(self._last_prompt, auto_approve=False, mode="plan")
                return
            if normalized == "clearplan":
                self._pending_plan_prompt = None
                self._reset_plan_panel()
                self._write_transcript_line("[run] plan cleared.")
                return

        def _restore_available_session(self) -> None:
            if self._query_active:
                self._write_transcript_line("[restore] cannot restore while a run is active.")
                return
            session_id = (self._available_restore_session_id or "").strip()
            if not session_id:
                self._write_transcript_line("[restore] no previous session available.")
                return
            proc = self._proc
            if not self._daemon_ready or proc is None or proc.poll() is not None:
                self._write_transcript_line("[restore] daemon is not ready.")
                return
            if send_daemon_command(proc, {"cmd": "restore_session", "session_id": session_id}):
                self._write_transcript_line(f"[restore] requesting session restore: {session_id}")
            else:
                self._write_transcript_line("[restore] failed to send restore command.")

        def _start_fresh_session(self) -> None:
            self._available_restore_session_id = None
            self._available_restore_turn_count = 0
            self._last_restored_turn_count = 0
            self._write_transcript_line("[session] starting fresh.")
            self._save_session()

        def _handle_copy_command(self, normalized: str) -> bool:
            command = classify_copy_command(normalized)
            if command == "transcript":
                self.action_copy_transcript()
                return True

            if command == "plan":
                self.action_copy_plan()
                return True

            if command == "issues":
                self.action_copy_issues()
                return True

            if command == "artifacts":
                self.action_copy_artifacts()
                return True

            if command == "last":
                self._copy_text(self._last_assistant_text, label="last response")
                return True

            if command == "all":
                self._copy_text(self._get_all_text(), label="all")
                return True

            return False

        def _handle_model_command(self, normalized: str) -> bool:
            command = classify_model_command(normalized)
            if command is None:
                return False
            action, argument = command

            if action == "help":
                self._write_transcript_line(self._current_model_summary())
                self._write_transcript_line(_MODEL_USAGE_TEXT)
                return True

            if action == "show":
                self._write_transcript_line(self._current_model_summary())
                return True

            if action == "list":
                options, _ = self._model_select_options()
                for label, _value in options:
                    self._write_transcript_line(f"- {label}")
                return True

            if action == "reset":
                self._pending_model_select_value = None
                self._model_provider_override = None
                self._model_tier_override = None
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] reset. {self._current_model_summary()}")
                return True

            if action == "provider":
                provider = (argument or "").strip()
                if not provider:
                    self._write_transcript_line("Usage: /model provider <name>")
                    return True
                self._pending_model_select_value = None
                self._model_provider_override = provider
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] provider set to {provider}.")
                if self._daemon_ready:
                    self._write_transcript_line("[model] restart daemon to apply provider changes.")
                self._write_transcript_line(self._current_model_summary())
                return True

            if action == "tier":
                tier = (argument or "").strip()
                if not tier:
                    self._write_transcript_line("Usage: /model tier <name>")
                    return True
                if self._daemon_ready and self._proc is not None and self._proc.poll() is None and self._query_active:
                    self._warn_run_active_tier_change_once()
                    return True
                self._pending_model_select_value = None
                self._model_tier_override = tier
                self._refresh_model_select()
                self._update_header_status()
                self._write_transcript_line(f"[model] tier set to {tier}.")
                if (
                    self._daemon_ready
                    and self._proc is not None
                    and self._proc.poll() is None
                    and not self._query_active
                ):
                    requested_provider = (self._model_provider_override or self._daemon_provider or "").strip().lower()
                    requested_tier = tier.strip().lower()
                    if not send_daemon_command(self._proc, {"cmd": "set_tier", "tier": tier}):
                        self._pending_model_select_value = None
                        self._write_transcript_line("[model] failed to send tier change to daemon.")
                    elif requested_provider and requested_tier:
                        self._pending_model_select_value = f"{requested_provider}|{requested_tier}"
                self._write_transcript_line(self._current_model_summary())
                return True

            return False

        def _handle_pre_run_command(self, text: str) -> bool:
            classified = classify_pre_run_command(text)
            if classified is None:
                return False

            action, argument = classified
            if action == "open":
                self._open_artifact(argument or "")
                return True
            if action == "open_usage":
                self._write_transcript_line(_OPEN_USAGE_TEXT)
                return True
            if action == "restore":
                self._restore_available_session()
                return True
            if action == "new":
                self._start_fresh_session()
                return True
            if action == "context":
                return self._handle_context_command(argument or "")
            if action == "context_usage":
                self._write_transcript_line(_CONTEXT_USAGE_TEXT)
                return True
            if action == "sop":
                return self._handle_sop_command(argument or "")
            if action == "sop_usage":
                self._write_transcript_line(_SOP_USAGE_TEXT)
                return True
            if action == "expand":
                self._expand_tool_call(argument or "")
                return True
            if action == "expand_usage":
                self._write_transcript_line(_EXPAND_USAGE_TEXT)
                return True
            if action == "search":
                self._search_transcript(argument or "")
                return True
            if action == "search_usage":
                self._write_transcript_line(_SEARCH_USAGE_TEXT)
                return True
            if action == "text":
                self._toggle_transcript_mode()
                return True
            if action == "text_usage":
                self._write_transcript_line(_TEXT_USAGE_TEXT)
                return True
            if action == "thinking":
                self._show_thinking_text()
                return True
            if action == "thinking_usage":
                self._write_transcript_line(_THINKING_USAGE_TEXT)
                return True
            if action == "compact":
                self._request_context_compact()
                return True
            if action == "compact_usage":
                self._write_transcript_line(_COMPACT_USAGE_TEXT)
                return True
            if action == "stop":
                self._stop_run()
                return True
            if action == "exit":
                self.action_quit()
                return True
            if action == "daemon_restart":
                self._spawn_daemon(restart=True)
                return True
            if action == "consent_usage":
                self._write_transcript_line(_CONSENT_USAGE_TEXT)
                return True
            if action == "consent":
                self._submit_consent_choice((argument or "").strip())
                return True
            if action == "connect":
                provider = (argument or "").strip() or "github_copilot"
                proc = self._proc
                if not self._daemon_ready or proc is None or proc.poll() is not None:
                    self._write_transcript_line("[connect] daemon is not ready.")
                    return True
                if self._query_active:
                    self._write_transcript_line("[connect] cannot connect while a run is active.")
                    return True
                self._write_transcript_line(f"[connect] starting provider auth for {provider}...")
                if not send_daemon_command(
                    proc,
                    {"cmd": "connect", "provider": provider, "method": "device", "open_browser": True},
                ):
                    self._write_transcript_line("[connect] failed to send command.")
                return True
            if action == "auth_usage":
                self._write_transcript_line(_AUTH_USAGE_TEXT)
                return True
            if action == "auth":
                raw = (argument or "").strip()
                normalized = raw.lower()
                proc = self._proc
                if not self._daemon_ready or proc is None or proc.poll() is not None:
                    self._write_transcript_line("[auth] daemon is not ready.")
                    return True
                if self._query_active:
                    self._write_transcript_line("[auth] cannot run while a run is active.")
                    return True
                if not raw:
                    self._write_transcript_line(_AUTH_USAGE_TEXT)
                    return True
                if normalized in {"list", "ls"}:
                    send_daemon_command(proc, {"cmd": "auth", "action": "list"})
                    return True
                if normalized.startswith("logout"):
                    parts = raw.split()
                    provider = parts[1].strip() if len(parts) >= 2 else "github_copilot"
                    send_daemon_command(proc, {"cmd": "auth", "action": "logout", "provider": provider})
                    return True
                self._write_transcript_line(_AUTH_USAGE_TEXT)
                return True
            if action.startswith("model:"):
                normalized = text.lower()
                return self._handle_model_command(normalized)
            return False

        def _handle_post_run_command(self, text: str) -> bool:
            classified = classify_post_run_command(text)
            if classified is None:
                return False

            action, argument = classified

            if action == "approve":
                self._dispatch_plan_action("approve")
                return True

            if action == "replan":
                self._dispatch_plan_action("replan")
                return True

            if action == "clearplan":
                self._dispatch_plan_action("clearplan")
                return True

            if action == "plan_mode":
                self._default_auto_approve = False
                self._update_prompt_placeholder()
                self._write_transcript_line("[mode] auto-approve disabled for default prompts.")
                return True

            if action == "plan_prompt":
                prompt = (argument or "").strip()
                if not prompt:
                    self._write_transcript_line("Usage: /plan <prompt>")
                    return True
                self._start_run(prompt, auto_approve=False, mode="plan")
                return True

            if action == "run_mode":
                self._default_auto_approve = True
                self._update_prompt_placeholder()
                self._write_transcript_line("[mode] auto-approve enabled for default prompts.")
                return True

            if action == "run_prompt":
                prompt = (argument or "").strip()
                if not prompt:
                    self._write_transcript_line("Usage: /run <prompt>")
                    return True
                self._start_run(prompt, auto_approve=True, mode="execute")
                return True

            return False

        def on_action_sheet_action_selected(self, event: Any) -> None:
            action_id = str(getattr(event, "action_id", "")).strip()
            event.stop()
            if not action_id:
                self._dismiss_action_sheet(restore_focus=True)
                return
            # Keep the sheet open when traversing sub-menus; otherwise close first.
            if action_id.startswith("tier:") or action_id == "tiers:back" or action_id == "idle:tiers":
                self._execute_action_sheet_action(action_id)
                return
            self._dismiss_action_sheet(restore_focus=False)
            self._execute_action_sheet_action(action_id)

        def on_action_sheet_dismissed(self, event: Any) -> None:
            event.stop()
            self._dismiss_action_sheet(restore_focus=True)

        def on_button_pressed(self, event: Any) -> None:
            button_id = str(getattr(getattr(event, "button", None), "id", "")).strip().lower()
            if button_id == "agent_view_profile":
                self._set_agent_studio_view_mode("profile")
                return
            if button_id == "agent_view_tools":
                self._set_agent_studio_view_mode("tools")
                return
            if button_id == "agent_view_team":
                self._set_agent_studio_view_mode("team")
                return
            if button_id == "agent_team_new":
                self._new_agent_team_preset_draft()
                return
            if button_id == "agent_team_save":
                self._save_agent_team_preset_draft()
                return
            if button_id == "agent_team_delete":
                self._delete_selected_agent_team_preset()
                return
            if button_id == "agent_team_insert_prompt":
                self._insert_agent_team_preset_run_prompt(run_now=False)
                return
            if button_id == "agent_team_run_now":
                self._insert_agent_team_preset_run_prompt(run_now=True)
                return
            if button_id == "agent_tools_overrides_apply":
                self._apply_agent_tools_safety_overrides(reset=False)
                return
            if button_id == "agent_tools_overrides_reset":
                self._apply_agent_tools_safety_overrides(reset=True)
                return
            if button_id == "session_view_timeline":
                self._set_session_view_mode("timeline")
                return
            if button_id == "session_view_issues":
                self._set_session_view_mode("issues")
                return
            if button_id.startswith("context_remove_"):
                suffix = button_id.removeprefix("context_remove_")
                with contextlib.suppress(ValueError):
                    self._remove_context_source(int(suffix))
                return
            if button_id == "context_add_file":
                self._set_context_add_mode("file")
                return
            if button_id == "context_add_note":
                self._set_context_add_mode("note")
                return
            if button_id == "context_add_sop":
                self._set_context_add_mode("sop")
                return
            if button_id == "context_add_kb":
                self._set_context_add_mode("kb")
                return
            if button_id in {"context_add_commit", "context_sop_commit"}:
                self._commit_context_add_from_ui()
                return
            if button_id in {"context_add_cancel", "context_sop_cancel"}:
                self._set_context_add_mode(None)
                return
            if button_id == "consent_choice_y":
                self._submit_consent_choice("y")
                return
            if button_id == "consent_choice_n":
                self._submit_consent_choice("n")
                return
            if button_id == "consent_choice_a":
                self._submit_consent_choice("a")
                return
            if button_id == "consent_choice_v":
                self._submit_consent_choice("v")
                return
            if button_id == "error_action_retry_tool":
                self._retry_failed_tool()
                return
            if button_id == "error_action_skip_tool":
                self._skip_failed_tool()
                return
            if button_id == "error_action_escalate":
                self._resume_after_error(escalate=True)
                return
            if button_id == "error_action_continue":
                self._resume_after_error(escalate=False)
                return
            if button_id == "plan_action_approve":
                self._dispatch_plan_action("approve")
                return
            if button_id == "plan_action_replan":
                self._dispatch_plan_action("replan")
                return
            if button_id == "plan_action_clear":
                self._dispatch_plan_action("clearplan")
                return
            if button_id == "agent_profile_new":
                self._new_agent_profile_draft()
                return
            if button_id == "agent_profile_save":
                self._save_agent_profile_draft()
                return
            if button_id == "agent_profile_delete":
                self._delete_selected_agent_profile()
                return
            if button_id == "agent_profile_apply":
                self._apply_agent_profile_draft()
                return

        def _handle_user_input(self, text: str) -> None:
            self._write_user_input(text)

            normalized = text.lower()

            if self._handle_copy_command(normalized):
                return

            if self._handle_pre_run_command(text):
                return

            if self._query_active:
                self._write_transcript_line("[run] already running; use /stop.")
                return

            if self._handle_post_run_command(text):
                return

            if text.startswith("/") or text.startswith(":"):
                self._write_transcript_line(f"[run] unknown command: {text}")
                return

            self._start_run(text, auto_approve=self._default_auto_approve)

    try:
        SwarmeeTUI().run()
    except KeyboardInterrupt:
        return 130
    return 0
