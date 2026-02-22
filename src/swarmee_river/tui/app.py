"""Optional Textual app scaffold for `swarmee tui`."""

from __future__ import annotations

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
_SEARCH_USAGE_TEXT = "Usage: /search <term>"
_OPEN_USAGE_TEXT = "Usage: /open <number>"
_EXPAND_USAGE_TEXT = "Usage: /expand <tool_use_id>"
_COMPACT_USAGE_TEXT = "Usage: /compact"
_TEXT_USAGE_TEXT = "Usage: /text"
_CONTEXT_USAGE_TEXT = (
    "Usage: /context add file <path> | /context add note <text> | /context add sop <name> | "
    "/context add kb <id> | /context remove <index> | /context list | /context clear"
)
_SOP_USAGE_TEXT = "Usage: /sop list | /sop activate <name> | /sop deactivate <name> | /sop preview <name>"
_RUN_ACTIVE_TIER_WARNING = "[model] cannot change tier while a run is active."
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


@dataclass(frozen=True)
class ParsedEvent:
    kind: str
    text: str
    meta: dict[str, str] | None = None


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
        seed = (
            str(source.get("path", "") or source.get("text", "") or source.get("name", "") or source.get("kb_id", "")).strip()
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


def send_daemon_command(proc: subprocess.Popen[str], cmd_dict: dict[str, Any]) -> bool:
    """Serialize and send a daemon command as JSONL."""
    if proc.stdin is None:
        return False
    try:
        payload = _json.dumps(cmd_dict, ensure_ascii=False) + "\n"
        proc.stdin.write(payload)
        proc.stdin.flush()
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
        CommandPalette,
        ContextBudgetBar,
        ConsentPrompt,
        ErrorActionPrompt,
        PlanActions,
        StatusBar,
        extract_consent_tool_name,
        render_assistant_message,
        render_plan_panel,
        render_system_message,
        render_thinking_message,
        render_tool_details_panel,
        render_tool_heartbeat_line,
        render_tool_progress_chunk,
        render_tool_result_line,
        render_tool_start_line,
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
            if key in {"ctrl+k", "ctrl+space", "ctrl+@"} and app is not None and hasattr(app, "action_open_action_sheet"):
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

        #plan, #issues, #artifacts {
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

        _proc: subprocess.Popen[str] | None = None
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
        _plan_text: str = ""
        _issues_lines: list[str] = []
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
        _tool_blocks: dict[str, dict[str, Any]] = {}
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
        _command_palette: Any = None  # CommandPalette | None
        _action_sheet: Any = None  # ActionSheet | None
        _action_sheet_mode: str = "root"
        _action_sheet_previous_focus: Any = None
        _status_bar: Any = None  # StatusBar | None
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
        _help_text: str = ""
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
                            yield TextArea(
                                text="",
                                read_only=True,
                                show_cursor=False,
                                id="artifacts",
                                soft_wrap=True,
                            )
                        with TabPane("Issues", id="tab_issues"):
                            yield TextArea(
                                text="",
                                read_only=True,
                                show_cursor=False,
                                id="issues",
                                soft_wrap=True,
                            )
                        with TabPane("Help", id="tab_help"):
                            yield TextArea(
                                text="",
                                read_only=True,
                                show_cursor=False,
                                id="help",
                                soft_wrap=True,
                            )
            yield CommandPalette(id="command_palette")
            yield ActionSheet(id="action_sheet")
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
            self._status_bar = self.query_one("#status_bar", StatusBar)
            self._consent_prompt_widget = self.query_one("#consent_prompt", ConsentPrompt)
            self._error_action_prompt_widget = self.query_one("#error_action_prompt", ErrorActionPrompt)
            self._context_sources_list = self.query_one("#context_sources_list", VerticalScroll)
            self._sop_list = self.query_one("#sop_list", VerticalScroll)
            self._context_input = self.query_one("#context_input", Input)
            self._context_sop_select = self.query_one("#context_sop_select", Select)
            self._prompt_metrics = self.query_one("#prompt_metrics", ContextBudgetBar)
            self._status_bar.set_model(self._current_model_summary())
            self.query_one("#prompt", PromptTextArea).focus()
            self._reset_help_panel()
            self._reset_plan_panel()
            self._reset_issues_panel()
            self._reset_artifacts_panel()
            self._reset_consent_panel()
            self._reset_error_action_prompt()
            self._set_context_add_mode(None)
            self._refresh_context_sop_options()
            self._render_context_sources_panel()
            self._refresh_sop_catalog()
            self._render_sop_panel()
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
            self._write_transcript("Tips: see the Help tab for commands/keys/copy shortcuts.")
            self._set_help_panel(
                "\n".join(
                    [
                        "Commands:",
                        "- /plan <prompt>, /run <prompt>",
                        "- /restore, /new",
                        "- /compact",
                        "- /context add/list/remove/clear",
                        "- /sop list, /sop activate <name>, /sop deactivate <name>, /sop preview <name>",
                        "- /approve, /replan, /clearplan",
                        "- /model, /text, /stop, /daemon restart, /expand <tool_id>, /exit",
                        "",
                        "Consent:",
                        "- /consent <y|n|a|v> (or press y/n/a/v when prompted)",
                        "- Inline consent buttons appear above the prompt box",
                        "",
                        "Keys:",
                        "- Enter submit, Shift+Enter newline (Ctrl+J/Alt+Enter fallback)",
                        "- Ctrl+Left/Right (or F6/F7) resize panes, F5 submit",
                        "- Ctrl+T toggles transcript rich/text mode",
                        "- Ctrl+K or Ctrl+Space opens action sheet",
                        "- Esc interrupt run, Ctrl+C/Cmd+C copy selection",
                        "",
                        "Copy/export:",
                        "- /copy, /copy plan, /copy issues, /copy all",
                        "",
                        "Notes:",
                        "- Model selector is in the prompt box footer dropdown.",
                        "- Transcript supports native mouse/keyboard text selection.",
                        "- Ctrl+C/Cmd+C copies selected text; /copy exports full transcript.",
                    ]
                )
            )
            transcript = self.query_one("#transcript", RichLog)
            with contextlib.suppress(Exception):
                transcript.auto_scroll = True
            with contextlib.suppress(Exception):
                transcript.max_lines = self._TRANSCRIPT_MAX_LINES
            self._set_transcript_mode("rich", notify=False)
            self._load_session()
            self._spawn_daemon()

        def _record_transcript_fallback(self, text: str) -> None:
            clean = sanitize_output_text(text).rstrip("\n")
            if not clean:
                return
            self._transcript_fallback_lines.extend(clean.splitlines())
            if len(self._transcript_fallback_lines) > self._TRANSCRIPT_MAX_LINES:
                self._transcript_fallback_lines = self._transcript_fallback_lines[-self._TRANSCRIPT_MAX_LINES :]

        def _sync_transcript_text_widget(self) -> None:
            text_widget = self.query_one("#transcript_text", TextArea)
            text = "\n".join(self._transcript_fallback_lines).rstrip()
            if text:
                text += "\n"
            text_widget.load_text(text)
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

        def _set_transcript_mode(self, mode: str, *, notify: bool = True) -> None:
            normalized = mode.strip().lower()
            if normalized not in {"rich", "text"}:
                return
            rich_widget = self.query_one("#transcript", RichLog)
            text_widget = self.query_one("#transcript_text", TextArea)
            if normalized == "text":
                self._sync_transcript_text_widget()
                rich_widget.styles.display = "none"
                text_widget.styles.display = "block"
                self._scroll_transcript_text_to_end()
                self._transcript_mode = "text"
                if notify:
                    self._notify("Text mode: select text with mouse. /text to return.", severity="information")
                return

            text_widget.styles.display = "none"
            rich_widget.styles.display = "block"
            with contextlib.suppress(Exception):
                rich_widget.scroll_end(animate=False)
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

        def _dismiss_thinking(self) -> None:
            """Mark the thinking placeholder as dismissed."""
            self._current_thinking = False

        def _cancel_streaming_flush_timer(self) -> None:
            timer = self._streaming_flush_timer
            self._streaming_flush_timer = None
            if timer is not None:
                with contextlib.suppress(Exception):
                    timer.stop()

        def _schedule_streaming_flush(self) -> None:
            if self._streaming_flush_timer is None:
                self._streaming_flush_timer = self.set_timer(_STREAMING_FLUSH_INTERVAL_S, self._on_streaming_flush_timer)

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
                plain_text=f"⚙ {tool_name} [{tool_use_id}] running... ({elapsed_s:.1f}s)",
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

        def _set_help_panel(self, content: str) -> None:
            self._help_text = content
            panel = self.query_one("#help", TextArea)
            text = content if content.strip() else "(no help yet)"
            panel.load_text(text)
            panel.scroll_home(animate=False)

        def _reset_help_panel(self) -> None:
            self._set_help_panel("(help)")

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
            self._issues_repeat_line = None
            self._issues_repeat_count = 0
            self._warning_count = 0
            self._error_count = 0
            panel = self.query_one("#issues", TextArea)
            panel.load_text("(no issues yet)")
            self._update_header_status()

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
            self._render_issues_panel()

        def _render_issues_panel(self) -> None:
            issues_panel = self.query_one("#issues", TextArea)
            text = "\n".join(self._issues_lines) if self._issues_lines else "(no issues yet)"
            issues_panel.load_text(text)
            issues_panel.scroll_end(animate=False)

        def _flush_issue_repeats(self) -> None:
            if self._issues_repeat_line is None or self._issues_repeat_count <= 0:
                self._issues_repeat_line = None
                self._issues_repeat_count = 0
                return
            repeated = f"… repeated {self._issues_repeat_count} more time(s): {self._issues_repeat_line}"
            self._issues_lines.append(repeated)
            self._issues_repeat_line = None
            self._issues_repeat_count = 0
            self._render_issues_panel()

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
                for label, value in options:
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
                        {"id": "view:issues", "icon": "⚠", "label": "View issues", "shortcut": "I"},
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
                self._switch_side_tab("tab_issues")
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
                "# Issues",
                "\n".join(self._issues_lines).rstrip() or "(no issues)",
                "",
                "# Artifacts",
                "\n".join(self._artifacts).rstrip() or "(no artifacts)",
                "",
                "# Context Sources",
                "\n".join(self._context_list_lines()).rstrip() or "(no context sources)",
                "",
                "# Consent History",
                "\n".join(self._consent_history_lines).rstrip() or "(no consent decisions)",
                "",
            ]
            return "\n".join(parts).rstrip() + "\n"

        def _render_artifacts_panel(self) -> None:
            panel = self.query_one("#artifacts", TextArea)
            with contextlib.suppress(Exception):
                panel.soft_wrap = True
            if not self._artifacts:
                panel.load_text("(no artifacts yet)")
            else:
                lines = [f"{i + 1}. {path}" for i, path in enumerate(self._artifacts)]
                panel.load_text("\n".join(lines))
            panel.scroll_end(animate=False)

        def _get_artifacts_text(self) -> str:
            if not self._artifacts:
                return "(no artifacts)\n"
            return "\n".join(self._artifacts).rstrip() + "\n"

        def _reset_artifacts_panel(self) -> None:
            self._artifacts = []
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
                self._mount_transcript_widget(render_assistant_message(markdown), plain_text=f"SOP: {name}\n\n{content}")
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
                self._add_context_source({"type": "file", "path": value, "id": _sanitize_context_source_id(uuid.uuid4().hex)})
            elif source_type == "note":
                self._add_context_source({"type": "note", "text": value, "id": _sanitize_context_source_id(uuid.uuid4().hex)})
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
                self._dismiss_thinking()
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
            self._dismiss_thinking()

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
            etype = event.get("event", "")

            if etype == "ready":
                self._daemon_ready = True
                session_id = str(event.get("session_id", "")).strip()
                if session_id:
                    self._daemon_session_id = session_id
                    self._save_session()
                self._write_transcript("Swarmee daemon ready. Enter a prompt to run Swarmee.")
                if self._context_sources or self._context_ready_for_sync:
                    self._sync_context_sources_with_daemon(notify_on_failure=True)
                if self._active_sop_names or self._sops_ready_for_sync:
                    self._sync_active_sops_with_daemon(notify_on_failure=True)

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

            elif etype == "model_info":
                self._handle_model_info(event)

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
                if not self._assistant_placeholder_written:
                    self._mount_transcript_widget(render_thinking_message(), plain_text="thinking...")
                    self._assistant_placeholder_written = True
                    self._current_thinking = True
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
                self._current_thinking = True

            elif etype == "tool_start":
                self._dismiss_thinking()
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
                }
                self._mount_transcript_widget(
                    render_tool_start_line(tool_name, tool_use_id=tid),
                    plain_text=f"⚙ {tool_name} [{tid}] running...",
                )
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
                    }
                    self._tool_blocks[tid] = record
                    self._mount_transcript_widget(
                        render_tool_start_line(fallback_tool_name, tool_use_id=tid),
                        plain_text=f"⚙ {fallback_tool_name} [{tid}] running...",
                    )
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
                    self._tool_progress_pending_ids.discard(tid)
                    self._flush_tool_progress_render(tid, force=True)
                plain = f"⚙ {tool_name} ({duration_s:.1f}s) {'✓' if status == 'success' else '✗'} [{tid}]"
                self._mount_transcript_widget(
                    render_tool_result_line(tool_name, status=status, duration_s=duration_s, tool_use_id=tid),
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
            self._cancel_tool_progress_flush_timer()
            self._tool_progress_pending_ids = set()
            for tool_use_id in list(self._tool_blocks.keys()):
                self._flush_tool_progress_render(tool_use_id, force=True)

            self._write_transcript(
                f"[run] completed in {elapsed:.1f}s ({self._run_tool_count} tool calls, status={exit_status})"
            )

            self._finalize_assistant_message()
            self._dismiss_thinking()

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

        def _handle_daemon_exit(self, proc: subprocess.Popen[str], *, return_code: int) -> None:
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
            self._proc = None
            self._runner_thread = None

            if self._status_timer is not None:
                self._status_timer.stop()
                self._status_timer = None
            if self._status_bar is not None:
                self._status_bar.set_state("idle")

            if was_query_active:
                self._finalize_turn(exit_status="error")
            if self._is_shutting_down:
                return
            self._write_transcript_line(f"[daemon] exited unexpectedly (code {return_code}).")
            self._write_transcript_line("[daemon] run /daemon restart to restart the background agent.")

        def _stream_daemon_output(self, proc: subprocess.Popen[str]) -> None:
            if proc.stdout is None:
                self._call_from_thread_safe(
                    self._write_transcript_line,
                    "[daemon] error: subprocess stdout unavailable.",
                )
                return_code = proc.poll()
                self._call_from_thread_safe(
                    self._handle_daemon_exit,
                    proc,
                    return_code=(return_code if return_code is not None else 1),
                )
                return

            try:
                for raw_line in proc.stdout:
                    self._call_from_thread_safe(self._handle_output_line, raw_line.rstrip("\n"), raw_line)
            except Exception as exc:
                self._call_from_thread_safe(self._write_transcript_line, f"[daemon] output stream error: {exc}")
            finally:
                with contextlib.suppress(Exception):
                    proc.stdout.close()
                return_code = proc.wait()
                self._call_from_thread_safe(self._handle_daemon_exit, proc, return_code=return_code)

        def _spawn_daemon(self, *, restart: bool = False) -> None:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                if restart:
                    self._pending_model_select_value = None
                    send_daemon_command(proc, {"cmd": "shutdown"})
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=3.0)
                    if proc.poll() is None:
                        stop_process(proc)
                    self._proc = None
                else:
                    return

            try:
                requested_session_id = (self._daemon_session_id or "").strip() or uuid.uuid4().hex
                self._daemon_session_id = requested_session_id
                daemon = spawn_swarmee_daemon(
                    session_id=requested_session_id,
                    env_overrides=self._model_env_overrides(),
                )
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
            self._current_thinking = False
            self._tool_blocks = {}
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
            self._cancel_streaming_flush_timer()
            self._cancel_tool_progress_flush_timer()
            if self._proc is not None and self._proc.poll() is None:
                send_daemon_command(self._proc, {"cmd": "shutdown"})
                with contextlib.suppress(Exception):
                    self._proc.wait(timeout=3.0)
                if self._proc.poll() is None:
                    stop_process(self._proc)
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
                if focused.id in {"issues", "plan", "artifacts", "help"} and focused_text:
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

        def _open_artifact(self, index_str: str) -> None:
            try:
                index = int(index_str.strip()) - 1
            except ValueError:
                self._write_transcript_line("Usage: /open <number>")
                return
            if index < 0 or index >= len(self._artifacts):
                self._write_transcript_line(f"[open] invalid index. {len(self._artifacts)} artifacts available.")
                return
            path = self._artifacts[index]
            editor = os.environ.get("EDITOR", "")
            try:
                if editor:
                    subprocess.Popen([editor, path])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", path])
                elif shutil.which("xdg-open"):
                    subprocess.Popen(["xdg-open", path])
                else:
                    self._write_transcript_line(f"[open] set $EDITOR. Path: {path}")
                    return
                self._write_transcript_line(f"[open] opened: {path}")
            except Exception as exc:
                self._write_transcript_line(f"[open] failed: {exc}")

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
                self._apply_split_ratio()
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
            if self._model_select_syncing:
                return
            select_widget = getattr(event, "select", None)
            if getattr(select_widget, "id", None) != "model_select":
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
            # The model selector is always visible; avoid transient notifications.

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
            if current_provider and current_tier and requested_provider == current_provider and requested_tier == current_tier:
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
