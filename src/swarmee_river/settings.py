from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swarmee_river.config.env_policy import filter_project_env_overrides
from swarmee_river.utils.provider_utils import normalize_provider_name


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        out_value = out.get(key)
        if isinstance(value, dict) and isinstance(out_value, dict):
            out[key] = deep_merge_dict(out_value, value)
        else:
            out[key] = value
    return out


def _default_settings_path() -> Path:
    return Path.cwd() / ".swarmee" / "settings.json"


def _load_settings_payload(path: Path | None = None) -> dict[str, Any]:
    settings_path = path or _default_settings_path()
    if not settings_path.exists() or not settings_path.is_file():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def normalize_project_env_overrides(raw_env: Any) -> dict[str, str]:
    # Migration-only: restrict env overrides to internal wiring keys.
    return filter_project_env_overrides(raw_env)


def load_project_env_overrides(path: Path | None = None) -> dict[str, str]:
    payload = _load_settings_payload(path)
    return normalize_project_env_overrides(payload.get("env"))


def apply_project_env_overrides(path: Path | None = None, *, overwrite: bool = True) -> dict[str, str]:
    """
    Apply internal-only environment overrides from `.swarmee/settings.json`.

    Note: This function intentionally ignores arbitrary env overrides. End-user
    configuration must be expressed via structured settings fields instead.
    """
    applied: dict[str, str] = {}
    for key, value in load_project_env_overrides(path).items():
        if not overwrite and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def _as_int(value: Any, *, default: int | None = None, min_value: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        out = int(str(value).strip())
    except Exception:
        return default
    if min_value is not None and out < min_value:
        return default
    return out


def _as_float(value: Any, *, default: float | None = None, min_value: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(str(value).strip())
    except Exception:
        return default
    if min_value is not None and out < min_value:
        return default
    return out


def _as_bool(value: Any, *, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return _truthy(str(value))


def _normalized_choice(value: Any, *, allowed: set[str], default: str) -> str:
    token = str(value or "").strip().lower()
    return token if token in allowed else default


@dataclass(frozen=True)
class ModelReasoningConfig:
    effort: str = "medium"  # low|medium|high

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelReasoningConfig":
        return cls(
            effort=_normalized_choice(
                raw.get("effort"),
                allowed={"low", "medium", "high"},
                default="medium",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"effort": self.effort}


@dataclass(frozen=True)
class ModelToolingConfig:
    mode: str = "standard"  # minimal|standard|tool-heavy
    discovery: str = "off"  # off|search

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelToolingConfig":
        return cls(
            mode=_normalized_choice(
                raw.get("mode"),
                allowed={"minimal", "standard", "tool-heavy"},
                default="standard",
            ),
            discovery=_normalized_choice(
                raw.get("discovery"),
                allowed={"off", "search"},
                default="off",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "discovery": self.discovery,
        }


@dataclass(frozen=True)
class ModelContextBehavior:
    strategy: str = "balanced"  # balanced|cache_safe|long_running
    compaction: str = "auto"  # auto|manual

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelContextBehavior":
        return cls(
            strategy=_normalized_choice(
                raw.get("strategy"),
                allowed={"balanced", "cache_safe", "long_running"},
                default="balanced",
            ),
            compaction=_normalized_choice(
                raw.get("compaction"),
                allowed={"auto", "manual"},
                default="auto",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "compaction": self.compaction,
        }


@dataclass(frozen=True)
class ContextConfig:
    manager: str = "summarize"  # summarize|sliding|none
    max_prompt_tokens: int = 20000
    window_size: int = 20
    per_turn: int = 1
    truncate_results: bool = True
    preserve_recent_messages: int = 10
    summary_ratio: float = 0.3
    cache_safe_summary: bool = False
    strategy: str = "balanced"  # balanced|cache_safe|long_running
    compaction: str = "auto"  # auto|manual

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ContextConfig":
        manager = str(raw.get("manager") or "summarize").strip().lower()
        if manager in {"summary", "summarizing"}:
            manager = "summarize"
        if manager in {"null", "off", "disabled"}:
            manager = "none"
        if manager not in {"summarize", "sliding", "none"}:
            manager = "summarize"
        return cls(
            manager=manager,
            max_prompt_tokens=_as_int(raw.get("max_prompt_tokens"), default=20000, min_value=1) or 20000,
            window_size=_as_int(raw.get("window_size"), default=20, min_value=1) or 20,
            per_turn=_as_int(raw.get("per_turn"), default=1, min_value=1) or 1,
            truncate_results=bool(_as_bool(raw.get("truncate_results"), default=True)),
            preserve_recent_messages=_as_int(raw.get("preserve_recent_messages"), default=10, min_value=0) or 10,
            summary_ratio=_as_float(raw.get("summary_ratio"), default=0.3, min_value=0.0) or 0.3,
            cache_safe_summary=bool(_as_bool(raw.get("cache_safe_summary"), default=False)),
            strategy=_normalized_choice(
                raw.get("strategy"),
                allowed={"balanced", "cache_safe", "long_running"},
                default="balanced",
            ),
            compaction=_normalized_choice(
                raw.get("compaction"),
                allowed={"auto", "manual"},
                default="auto",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manager": self.manager,
            "max_prompt_tokens": self.max_prompt_tokens,
            "window_size": self.window_size,
            "per_turn": self.per_turn,
            "truncate_results": self.truncate_results,
            "preserve_recent_messages": self.preserve_recent_messages,
            "summary_ratio": self.summary_ratio,
            "cache_safe_summary": self.cache_safe_summary,
            "strategy": self.strategy,
            "compaction": self.compaction,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    auto_approve: bool = False
    freeze_tools: bool = False
    swarm_enabled: bool = True
    esc_interrupt_enabled: bool = True
    limit_tool_results: bool = True
    knowledge_base_id: str | None = None
    session_s3_bucket: str | None = None
    session_s3_prefix: str = "swarmee/sessions/"
    session_s3_auto_export: bool = False
    session_kb_promote_on_complete: bool = False
    tooling_s3_prefix: str = "swarmee/tooling"
    enabled_tools: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)
    enable_project_context_tool: bool = False
    project_map_enabled: bool = True
    preflight_enabled: bool = True
    preflight_level: str = "summary"  # summary|summary+tree|summary+files
    preflight_max_chars: int = 8000
    preflight_print: bool = False
    interrupt_timeout_sec: float = 2.0
    state_dir: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RuntimeConfig":
        enabled_tools_raw = raw.get("enabled_tools")
        enabled_tools = [str(x).strip() for x in enabled_tools_raw] if isinstance(enabled_tools_raw, list) else []
        enabled_tools = [x for x in enabled_tools if x]
        disabled_tools_raw = raw.get("disabled_tools")
        disabled_tools = [str(x).strip() for x in disabled_tools_raw] if isinstance(disabled_tools_raw, list) else []
        disabled_tools = [x for x in disabled_tools if x]
        preflight_level = str(raw.get("preflight_level") or "summary").strip().lower()
        if preflight_level not in {"summary", "summary+tree", "summary+files"}:
            preflight_level = "summary"
        interrupt_timeout_sec = _as_float(raw.get("interrupt_timeout_sec"), default=2.0, min_value=0.0) or 2.0
        if interrupt_timeout_sec <= 0:
            interrupt_timeout_sec = 2.0
        state_dir = raw.get("state_dir")
        state_dir = str(state_dir).strip() if isinstance(state_dir, str) and str(state_dir).strip() else None
        kb_id = raw.get("knowledge_base_id")
        kb_id = str(kb_id).strip() if isinstance(kb_id, str) and str(kb_id).strip() else None
        s3_bucket = raw.get("session_s3_bucket")
        s3_bucket = str(s3_bucket).strip() if isinstance(s3_bucket, str) and str(s3_bucket).strip() else None
        s3_prefix = str(raw.get("session_s3_prefix") or "swarmee/sessions/").strip() or "swarmee/sessions/"
        tooling_prefix = str(raw.get("tooling_s3_prefix") or "swarmee/tooling").strip() or "swarmee/tooling"
        return cls(
            auto_approve=bool(_as_bool(raw.get("auto_approve"), default=False)),
            freeze_tools=bool(_as_bool(raw.get("freeze_tools"), default=False)),
            swarm_enabled=bool(_as_bool(raw.get("swarm_enabled"), default=True)),
            esc_interrupt_enabled=bool(_as_bool(raw.get("esc_interrupt_enabled"), default=True)),
            limit_tool_results=bool(_as_bool(raw.get("limit_tool_results"), default=True)),
            knowledge_base_id=kb_id,
            session_s3_bucket=s3_bucket,
            session_s3_prefix=s3_prefix,
            session_s3_auto_export=bool(_as_bool(raw.get("session_s3_auto_export"), default=False)),
            session_kb_promote_on_complete=bool(_as_bool(raw.get("session_kb_promote_on_complete"), default=False)),
            tooling_s3_prefix=tooling_prefix,
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            enable_project_context_tool=bool(_as_bool(raw.get("enable_project_context_tool"), default=False)),
            project_map_enabled=bool(_as_bool(raw.get("project_map_enabled"), default=True)),
            preflight_enabled=bool(_as_bool(raw.get("preflight_enabled"), default=True)),
            preflight_level=preflight_level,
            preflight_max_chars=_as_int(raw.get("preflight_max_chars"), default=8000, min_value=0) or 8000,
            preflight_print=bool(_as_bool(raw.get("preflight_print"), default=False)),
            interrupt_timeout_sec=interrupt_timeout_sec,
            state_dir=state_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "auto_approve": self.auto_approve,
            "freeze_tools": self.freeze_tools,
            "swarm_enabled": self.swarm_enabled,
            "esc_interrupt_enabled": self.esc_interrupt_enabled,
            "limit_tool_results": self.limit_tool_results,
            "knowledge_base_id": self.knowledge_base_id,
            "session_s3_bucket": self.session_s3_bucket,
            "session_s3_prefix": self.session_s3_prefix,
            "session_s3_auto_export": self.session_s3_auto_export,
            "session_kb_promote_on_complete": self.session_kb_promote_on_complete,
            "tooling_s3_prefix": self.tooling_s3_prefix,
            "enabled_tools": list(self.enabled_tools),
            "disabled_tools": list(self.disabled_tools),
            "enable_project_context_tool": self.enable_project_context_tool,
            "project_map_enabled": self.project_map_enabled,
            "preflight_enabled": self.preflight_enabled,
            "preflight_level": self.preflight_level,
            "preflight_max_chars": self.preflight_max_chars,
            "preflight_print": self.preflight_print,
            "interrupt_timeout_sec": self.interrupt_timeout_sec,
        }
        if self.state_dir:
            out["state_dir"] = self.state_dir
        return out


@dataclass(frozen=True)
class DiagnosticsConfig:
    level: str = "baseline"  # baseline|verbose
    redact: bool = True
    log_redact: bool = True
    log_events: bool = True
    log_dir: str | None = None
    log_s3_bucket: str | None = None
    log_s3_prefix: str = "swarmee/logs"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DiagnosticsConfig":
        level = str(raw.get("level") or "baseline").strip().lower()
        if level not in {"baseline", "verbose"}:
            level = "baseline"
        log_dir = raw.get("log_dir")
        log_dir = str(log_dir).strip() if isinstance(log_dir, str) and str(log_dir).strip() else None
        s3_bucket = raw.get("log_s3_bucket")
        s3_bucket = str(s3_bucket).strip() if isinstance(s3_bucket, str) and str(s3_bucket).strip() else None
        s3_prefix = str(raw.get("log_s3_prefix") or "swarmee/logs").strip().strip("/")
        s3_prefix = s3_prefix or "swarmee/logs"
        return cls(
            level=level,
            redact=bool(_as_bool(raw.get("redact"), default=True)),
            log_redact=bool(_as_bool(raw.get("log_redact"), default=True)),
            log_events=bool(_as_bool(raw.get("log_events"), default=True)),
            log_dir=log_dir,
            log_s3_bucket=s3_bucket,
            log_s3_prefix=s3_prefix,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "level": self.level,
            "redact": self.redact,
            "log_redact": self.log_redact,
            "log_events": self.log_events,
            "log_s3_bucket": self.log_s3_bucket,
            "log_s3_prefix": self.log_s3_prefix,
        }
        if self.log_dir:
            out["log_dir"] = self.log_dir
        return out


@dataclass(frozen=True)
class PricingOverride:
    input_per_1m: float | None = None
    output_per_1m: float | None = None
    cached_input_per_1m: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PricingOverride":
        return cls(
            input_per_1m=_as_float(raw.get("input_per_1m"), default=None, min_value=0.0),
            output_per_1m=_as_float(raw.get("output_per_1m"), default=None, min_value=0.0),
            cached_input_per_1m=_as_float(raw.get("cached_input_per_1m"), default=None, min_value=0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.input_per_1m is not None:
            out["input_per_1m"] = self.input_per_1m
        if self.output_per_1m is not None:
            out["output_per_1m"] = self.output_per_1m
        if self.cached_input_per_1m is not None:
            out["cached_input_per_1m"] = self.cached_input_per_1m
        return out


@dataclass(frozen=True)
class PricingConfig:
    default: PricingOverride = field(default_factory=PricingOverride)
    providers: dict[str, PricingOverride] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PricingConfig":
        default_raw = raw.get("default")
        default = PricingOverride.from_dict(default_raw) if isinstance(default_raw, dict) else PricingOverride()
        providers_raw = raw.get("providers")
        providers: dict[str, PricingOverride] = {}
        if isinstance(providers_raw, dict):
            for provider_name, provider_raw in providers_raw.items():
                key = str(provider_name or "").strip().lower()
                if not key or not isinstance(provider_raw, dict):
                    continue
                providers[key] = PricingOverride.from_dict(provider_raw)
        return cls(default=default, providers=providers)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "default": self.default.to_dict(),
            "providers": {k: v.to_dict() for k, v in self.providers.items()},
        }
        return out


def _normalize_env_bool(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    token = str(raw).strip().lower()
    if not token:
        return None
    return _truthy(token)


def migrate_legacy_env_overrides(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """
    Translate legacy settings.json `env` overrides into structured settings fields.

    Returns:
        (migrated_payload, migrated_key_to_destination, dropped_keys)
    """
    raw_env = payload.get("env")
    if not isinstance(raw_env, dict) or not raw_env:
        return payload, {}, []

    migrated_payload = dict(payload)
    migrated_payload.pop("env", None)

    migrated: dict[str, str] = {}
    dropped: list[str] = []

    # Ensure target dicts exist.
    context = dict(migrated_payload.get("context") or {}) if isinstance(migrated_payload.get("context"), dict) else {}
    runtime = dict(migrated_payload.get("runtime") or {}) if isinstance(migrated_payload.get("runtime"), dict) else {}
    diagnostics = (
        dict(migrated_payload.get("diagnostics") or {})
        if isinstance(migrated_payload.get("diagnostics"), dict)
        else {}
    )
    safety = dict(migrated_payload.get("safety") or {}) if isinstance(migrated_payload.get("safety"), dict) else {}
    pricing = dict(migrated_payload.get("pricing") or {}) if isinstance(migrated_payload.get("pricing"), dict) else {}
    models = dict(migrated_payload.get("models") or {}) if isinstance(migrated_payload.get("models"), dict) else {}

    providers = dict(models.get("providers") or {}) if isinstance(models.get("providers"), dict) else {}

    def _set_nested(d: dict[str, Any], key: str, value: Any) -> None:
        d[key] = value

    def _ensure_provider(provider_name: str) -> dict[str, Any]:
        token = normalize_provider_name(provider_name) or provider_name
        token = str(token).strip().lower()
        existing = providers.get(token)
        if isinstance(existing, dict):
            return existing
        out: dict[str, Any] = {}
        providers[token] = out
        return out

    def _ensure_provider_tiers(provider_name: str) -> dict[str, Any]:
        provider = _ensure_provider(provider_name)
        tiers = provider.get("tiers")
        if isinstance(tiers, dict):
            return tiers
        out: dict[str, Any] = {}
        provider["tiers"] = out
        return out

    def _set_provider_extra(provider_name: str, extra_key: str, value: Any) -> None:
        provider = _ensure_provider(provider_name)
        provider[extra_key] = value

    default_tier = str(models.get("default_tier") or "balanced").strip().lower() or "balanced"

    for raw_key, raw_value in raw_env.items():
        key = str(raw_key or "").strip()
        if not key:
            continue

        # Secrets must not be persisted into settings.json.
        if key in {"OPENAI_API_KEY", "SWARMEE_GITHUB_COPILOT_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"}:
            dropped.append(key)
            continue

        if key in {"SWARMEE_KNOWLEDGE_BASE_ID", "STRANDS_KNOWLEDGE_BASE_ID"}:
            kb_id = str(raw_value or "").strip()
            if kb_id:
                _set_nested(runtime, "knowledge_base_id", kb_id)
                migrated[key] = "runtime.knowledge_base_id"
                continue
        if key == "SWARMEE_SESSION_S3_BUCKET":
            v = str(raw_value or "").strip()
            if v:
                _set_nested(runtime, "session_s3_bucket", v)
                migrated[key] = "runtime.session_s3_bucket"
                continue
        if key == "SWARMEE_SESSION_S3_PREFIX":
            v = str(raw_value or "").strip()
            if v:
                _set_nested(runtime, "session_s3_prefix", v)
                migrated[key] = "runtime.session_s3_prefix"
                continue
        if key == "SWARMEE_SESSION_S3_AUTO_EXPORT":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(runtime, "session_s3_auto_export", b)
                migrated[key] = "runtime.session_s3_auto_export"
                continue
        if key == "SWARMEE_SESSION_KB_PROMOTE_ON_COMPLETE":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(runtime, "session_kb_promote_on_complete", b)
                migrated[key] = "runtime.session_kb_promote_on_complete"
                continue
        if key == "SWARMEE_TOOLING_S3_PREFIX":
            v = str(raw_value or "").strip()
            if v:
                _set_nested(runtime, "tooling_s3_prefix", v)
                migrated[key] = "runtime.tooling_s3_prefix"
                continue
        if key == "AWS_PROFILE":
            v = str(raw_value or "").strip()
            if v:
                _set_provider_extra("bedrock", "aws_profile", v)
                migrated[key] = "models.providers.bedrock.aws_profile"
                continue
        if key == "BYPASS_TOOL_CONSENT":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(safety, "tool_consent", "allow" if b else "ask")
                migrated[key] = "safety.tool_consent"
                continue

        # Context manager knobs.
        if key == "SWARMEE_CONTEXT_MANAGER":
            manager = str(raw_value or "").strip().lower()
            if manager in {"summary", "summarizing"}:
                manager = "summarize"
            if manager in {"null", "off", "disabled"}:
                manager = "none"
            if manager not in {"summarize", "sliding", "none"}:
                manager = "summarize"
            _set_nested(context, "manager", manager)
            migrated[key] = "context.manager"
            continue
        if key == "SWARMEE_CONTEXT_BUDGET_TOKENS":
            v = _as_int(raw_value, default=None, min_value=1)
            if v is not None:
                _set_nested(context, "max_prompt_tokens", v)
                migrated[key] = "context.max_prompt_tokens"
                continue
        if key == "SWARMEE_WINDOW_SIZE":
            v = _as_int(raw_value, default=None, min_value=1)
            if v is not None:
                _set_nested(context, "window_size", v)
                migrated[key] = "context.window_size"
                continue
        if key == "SWARMEE_CONTEXT_PER_TURN":
            v = _as_int(raw_value, default=None, min_value=1)
            if v is not None:
                _set_nested(context, "per_turn", v)
                migrated[key] = "context.per_turn"
                continue
        if key == "SWARMEE_TRUNCATE_RESULTS":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(context, "truncate_results", b)
                migrated[key] = "context.truncate_results"
                continue
        if key == "SWARMEE_PRESERVE_RECENT_MESSAGES":
            v = _as_int(raw_value, default=None, min_value=0)
            if v is not None:
                _set_nested(context, "preserve_recent_messages", v)
                migrated[key] = "context.preserve_recent_messages"
                continue
        if key == "SWARMEE_SUMMARY_RATIO":
            v = _as_float(raw_value, default=None, min_value=0.0)
            if v is not None:
                _set_nested(context, "summary_ratio", v)
                migrated[key] = "context.summary_ratio"
                continue
        if key == "SWARMEE_CACHE_SAFE_SUMMARY":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(context, "cache_safe_summary", b)
                migrated[key] = "context.cache_safe_summary"
                continue

        # Runtime toggles.
        if key == "SWARMEE_AUTO_APPROVE":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(runtime, "auto_approve", b)
                migrated[key] = "runtime.auto_approve"
                continue
        if key == "SWARMEE_FREEZE_TOOLS":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(runtime, "freeze_tools", b)
                migrated[key] = "runtime.freeze_tools"
                continue
        if key == "SWARMEE_SWARM_ENABLED":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(runtime, "swarm_enabled", b)
                migrated[key] = "runtime.swarm_enabled"
                continue
        if key == "SWARMEE_ESC_INTERRUPT":
            token = str(raw_value or "").strip().lower()
            if token in {"enabled", "disabled"}:
                _set_nested(runtime, "esc_interrupt_enabled", token == "enabled")
                migrated[key] = "runtime.esc_interrupt_enabled"
                continue
        if key == "SWARMEE_LIMIT_TOOL_RESULTS":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(runtime, "limit_tool_results", b)
                migrated[key] = "runtime.limit_tool_results"
                continue
        if key == "SWARMEE_PROJECT_MAP":
            token = str(raw_value or "").strip().lower()
            if token in {"enabled", "disabled"}:
                _set_nested(runtime, "project_map_enabled", token == "enabled")
                migrated[key] = "runtime.project_map_enabled"
                continue
        if key == "SWARMEE_PREFLIGHT":
            token = str(raw_value or "").strip().lower()
            if token in {"enabled", "disabled"}:
                _set_nested(runtime, "preflight_enabled", token == "enabled")
                migrated[key] = "runtime.preflight_enabled"
                continue
        if key == "SWARMEE_PREFLIGHT_LEVEL":
            token = str(raw_value or "").strip().lower()
            if token in {"summary", "summary+tree", "summary+files"}:
                _set_nested(runtime, "preflight_level", token)
                migrated[key] = "runtime.preflight_level"
                continue
        if key == "SWARMEE_PREFLIGHT_MAX_CHARS":
            v = _as_int(raw_value, default=None, min_value=0)
            if v is not None:
                _set_nested(runtime, "preflight_max_chars", v)
                migrated[key] = "runtime.preflight_max_chars"
                continue
        if key == "SWARMEE_PREFLIGHT_PRINT":
            token = str(raw_value or "").strip().lower()
            if token in {"enabled", "disabled"}:
                _set_nested(runtime, "preflight_print", token == "enabled")
                migrated[key] = "runtime.preflight_print"
                continue
        if key == "SWARMEE_INTERRUPT_TIMEOUT_SEC":
            v = _as_float(raw_value, default=None, min_value=0.0)
            if v is not None and v > 0:
                _set_nested(runtime, "interrupt_timeout_sec", v)
                migrated[key] = "runtime.interrupt_timeout_sec"
                continue
        if key == "SWARMEE_STATE_DIR":
            v = str(raw_value or "").strip()
            if v:
                _set_nested(runtime, "state_dir", v)
                migrated[key] = "runtime.state_dir"
                continue

        # Diagnostics.
        if key == "SWARMEE_DIAG_LEVEL":
            token = str(raw_value or "").strip().lower()
            if token in {"baseline", "verbose"}:
                _set_nested(diagnostics, "level", token)
                migrated[key] = "diagnostics.level"
                continue
        if key in {"SWARMEE_DIAG_REDACT", "SWARMEE_LOG_REDACT"}:
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(diagnostics, "redact", b)
                _set_nested(diagnostics, "log_redact", b)
                migrated[key] = "diagnostics.redact"
                continue
        if key == "SWARMEE_LOG_EVENTS":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(diagnostics, "log_events", b)
                migrated[key] = "diagnostics.log_events"
                continue
        if key == "SWARMEE_LOG_DIR":
            v = str(raw_value or "").strip()
            if v:
                _set_nested(diagnostics, "log_dir", v)
                migrated[key] = "diagnostics.log_dir"
                continue
        if key == "SWARMEE_LOG_S3_BUCKET":
            v = str(raw_value or "").strip()
            if v:
                _set_nested(diagnostics, "log_s3_bucket", v)
                migrated[key] = "diagnostics.log_s3_bucket"
                continue
        if key == "SWARMEE_LOG_S3_PREFIX":
            v = str(raw_value or "").strip().strip("/")
            if v:
                _set_nested(diagnostics, "log_s3_prefix", v)
                migrated[key] = "diagnostics.log_s3_prefix"
                continue

        # Tool policy env -> runtime.
        if key in {"SWARMEE_ENABLE_TOOLS", "SWARMEE_DISABLE_TOOLS"}:
            raw_list = str(raw_value or "").strip()
            parts = [p.strip() for p in raw_list.split(",") if p.strip()]
            if key == "SWARMEE_ENABLE_TOOLS":
                _set_nested(runtime, "enabled_tools", parts)
                migrated[key] = "runtime.enabled_tools"
            else:
                _set_nested(runtime, "disabled_tools", parts)
                migrated[key] = "runtime.disabled_tools"
            continue
        if key == "SWARMEE_ENABLE_PROJECT_CONTEXT_TOOL":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                _set_nested(runtime, "enable_project_context_tool", b)
                migrated[key] = "runtime.enable_project_context_tool"
                continue

        # Model selection.
        if key == "SWARMEE_MODEL_PROVIDER":
            provider = normalize_provider_name(raw_value)
            if provider:
                _set_nested(models, "provider", provider)
                migrated[key] = "models.provider"
                continue
        if key == "SWARMEE_MODEL_TIER":
            tier = str(raw_value or "").strip().lower()
            if tier:
                _set_nested(models, "default_tier", tier)
                migrated[key] = "models.default_tier"
                default_tier = tier
                continue
        if key == "SWARMEE_TIER_AUTO":
            b = _normalize_env_bool(raw_value)
            if b is not None:
                auto_raw = models.get("auto_escalation")
                auto = dict(auto_raw or {}) if isinstance(auto_raw, dict) else {}
                auto["enabled"] = b
                models["auto_escalation"] = auto
                migrated[key] = "models.auto_escalation.enabled"
                continue
        if key in {"SWARMEE_MAX_TOKENS", "STRANDS_MAX_TOKENS"}:
            v = _as_int(raw_value, default=None, min_value=1)
            if v is not None:
                _set_nested(models, "max_output_tokens", v)
                migrated[key] = "models.max_output_tokens"
                continue

        # Provider-level tuning.
        if key == "OPENAI_BASE_URL":
            v = str(raw_value or "").strip()
            if v:
                _set_provider_extra("openai", "base_url", v)
                migrated[key] = "models.providers.openai.base_url"
                continue
        if key == "SWARMEE_OPENAI_MAX_RETRIES":
            v = _as_int(raw_value, default=None, min_value=0)
            if v is not None:
                _set_provider_extra("openai", "max_retries", v)
                migrated[key] = "models.providers.openai.max_retries"
                continue
        if key == "SWARMEE_GITHUB_COPILOT_BASE_URL":
            v = str(raw_value or "").strip()
            if v:
                _set_provider_extra("github_copilot", "base_url", v)
                migrated[key] = "models.providers.github_copilot.base_url"
                continue
        if key == "SWARMEE_GITHUB_COPILOT_MAX_RETRIES":
            v = _as_int(raw_value, default=None, min_value=0)
            if v is not None:
                _set_provider_extra("github_copilot", "max_retries", v)
                migrated[key] = "models.providers.github_copilot.max_retries"
                continue
        if key == "SWARMEE_GITHUB_COPILOT_INTEGRATION_ID":
            v = str(raw_value or "").strip()
            if v:
                _set_provider_extra("github_copilot", "integration_id", v)
                migrated[key] = "models.providers.github_copilot.integration_id"
                continue
        if key == "SWARMEE_OLLAMA_HOST":
            v = str(raw_value or "").strip()
            if v:
                _set_provider_extra("ollama", "host", v)
                migrated[key] = "models.providers.ollama.host"
                continue
        if key == "SWARMEE_BEDROCK_READ_TIMEOUT_SEC":
            v = _as_float(raw_value, default=None, min_value=0.0)
            if v is not None and v > 0:
                _set_provider_extra("bedrock", "read_timeout_sec", v)
                migrated[key] = "models.providers.bedrock.read_timeout_sec"
                continue
        if key == "SWARMEE_BEDROCK_CONNECT_TIMEOUT_SEC":
            v = _as_float(raw_value, default=None, min_value=0.0)
            if v is not None and v > 0:
                _set_provider_extra("bedrock", "connect_timeout_sec", v)
                migrated[key] = "models.providers.bedrock.connect_timeout_sec"
                continue
        if key == "SWARMEE_BEDROCK_MAX_RETRIES":
            v = _as_int(raw_value, default=None, min_value=0)
            if v is not None:
                _set_provider_extra("bedrock", "max_retries", v)
                migrated[key] = "models.providers.bedrock.max_retries"
                continue

        # Per-tier model id overrides.
        if key.startswith("SWARMEE_") and key.endswith("_MODEL_ID"):
            # Pattern: SWARMEE_<PROVIDER>_<TIER>_MODEL_ID
            parts = key.split("_")
            if len(parts) >= 5 and parts[0] == "SWARMEE" and parts[-2:] == ["MODEL", "ID"]:
                provider_token = "_".join(parts[1:-3]).strip().lower()
                tier_token = parts[-3].strip().lower()
                model_id = str(raw_value or "").strip()
                if provider_token and tier_token and model_id:
                    tiers = _ensure_provider_tiers(provider_token)
                    tier_cfg = dict(tiers.get(tier_token) or {}) if isinstance(tiers.get(tier_token), dict) else {}
                    tier_cfg["model_id"] = model_id
                    tiers[tier_token] = tier_cfg
                    migrated[key] = f"models.providers.{provider_token}.tiers.{tier_token}.model_id"
                    continue

        # Provider-level model id overrides -> apply to current default tier.
        if key in {
            "SWARMEE_OPENAI_MODEL_ID",
            "SWARMEE_OLLAMA_MODEL_ID",
            "SWARMEE_GITHUB_COPILOT_MODEL_ID",
            "STRANDS_MODEL_ID",
        }:
            model_id = str(raw_value or "").strip()
            if model_id:
                provider_token = {
                    "SWARMEE_OPENAI_MODEL_ID": "openai",
                    "SWARMEE_OLLAMA_MODEL_ID": "ollama",
                    "SWARMEE_GITHUB_COPILOT_MODEL_ID": "github_copilot",
                    "STRANDS_MODEL_ID": "bedrock",
                }[key]
                tiers = _ensure_provider_tiers(provider_token)
                tier_cfg = dict(tiers.get(default_tier) or {}) if isinstance(tiers.get(default_tier), dict) else {}
                tier_cfg["model_id"] = model_id
                tiers[default_tier] = tier_cfg
                migrated[key] = f"models.providers.{provider_token}.tiers.{default_tier}.model_id"
                continue

        # Pricing overrides.
        if key.startswith("SWARMEE_PRICE_"):
            token = str(key[len('SWARMEE_PRICE_'):]).strip()
            # SWARMEE_PRICE_INPUT_PER_1M, SWARMEE_PRICE_<PROVIDER>_INPUT_PER_1M, etc.
            parts = token.split("_")
            rate = _as_float(raw_value, default=None, min_value=0.0)
            if rate is None:
                dropped.append(key)
                continue
            suffix_map = {
                ("INPUT", "PER", "1M"): ("input_per_1m",),
                ("OUTPUT", "PER", "1M"): ("output_per_1m",),
                ("CACHED", "INPUT", "PER", "1M"): ("cached_input_per_1m",),
            }
            dest_field = None
            provider_name = None
            if tuple(parts) in suffix_map:
                dest_field = suffix_map[tuple(parts)][0]
            else:
                # provider-specific: <PROVIDER>_<SUFFIX...>
                for suffix_parts, (field_name,) in suffix_map.items():
                    if len(parts) > len(suffix_parts) and tuple(parts[-len(suffix_parts):]) == suffix_parts:
                        provider_name = "_".join(parts[:-len(suffix_parts)]).strip().lower()
                        dest_field = field_name
                        break
            if not dest_field:
                dropped.append(key)
                continue
            if provider_name:
                providers_value = pricing.get("providers")
                providers_raw = dict(providers_value or {}) if isinstance(providers_value, dict) else {}
                provider_value = providers_raw.get(provider_name)
                provider_cfg = dict(provider_value or {}) if isinstance(provider_value, dict) else {}
                provider_cfg[dest_field] = rate
                providers_raw[provider_name] = provider_cfg
                pricing["providers"] = providers_raw
                migrated[key] = f"pricing.providers.{provider_name}.{dest_field}"
            else:
                default_raw = dict(pricing.get("default") or {}) if isinstance(pricing.get("default"), dict) else {}
                default_raw[dest_field] = rate
                pricing["default"] = default_raw
                migrated[key] = f"pricing.default.{dest_field}"
            continue

        # Everything else is removed.
        dropped.append(key)

    if context:
        migrated_payload["context"] = context
    if runtime:
        migrated_payload["runtime"] = runtime
    if diagnostics:
        migrated_payload["diagnostics"] = diagnostics
    if safety:
        migrated_payload["safety"] = safety
    if pricing:
        migrated_payload["pricing"] = pricing
    if providers:
        models["providers"] = providers
    if models:
        migrated_payload["models"] = models

    return migrated_payload, migrated, dropped


@dataclass(frozen=True)
class ModelTier:
    provider: str = ""
    model_id: str | None = None
    display_name: str | None = None
    description: str | None = None
    transport: str | None = None
    reasoning: ModelReasoningConfig | None = None
    tooling: ModelToolingConfig | None = None
    context: ModelContextBehavior | None = None
    client_args: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_provider: str | None = None) -> "ModelTier":
        provider = normalize_provider_name(raw.get("provider") or default_provider)
        model_id = raw.get("model_id")
        display_name = raw.get("display_name")
        description = raw.get("description")
        transport_raw = raw.get("transport")
        transport = (
            _normalized_choice(transport_raw, allowed={"responses", "chat_completions"}, default="responses")
            if isinstance(transport_raw, str) and transport_raw.strip()
            else None
        )
        reasoning_raw = raw.get("reasoning")
        tooling_raw = raw.get("tooling")
        context_raw = raw.get("context")
        client_args = raw.get("client_args")
        params = raw.get("params")

        extra = {
            k: v
            for k, v in raw.items()
            if k
            not in {
                "provider",
                "model_id",
                "display_name",
                "description",
                "transport",
                "reasoning",
                "tooling",
                "context",
                "client_args",
                "params",
            }
        }
        return cls(
            provider=provider,
            model_id=str(model_id) if isinstance(model_id, str) else None,
            display_name=str(display_name) if isinstance(display_name, str) else None,
            description=str(description) if isinstance(description, str) else None,
            transport=transport,
            reasoning=ModelReasoningConfig.from_dict(reasoning_raw) if isinstance(reasoning_raw, dict) else None,
            tooling=ModelToolingConfig.from_dict(tooling_raw) if isinstance(tooling_raw, dict) else None,
            context=ModelContextBehavior.from_dict(context_raw) if isinstance(context_raw, dict) else None,
            client_args=client_args if isinstance(client_args, dict) else None,
            params=params if isinstance(params, dict) else None,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"provider": self.provider}
        if self.model_id:
            out["model_id"] = self.model_id
        if self.display_name:
            out["display_name"] = self.display_name
        if self.description:
            out["description"] = self.description
        if self.transport:
            out["transport"] = self.transport
        if self.reasoning is not None:
            out["reasoning"] = self.reasoning.to_dict()
        if self.tooling is not None:
            out["tooling"] = self.tooling.to_dict()
        if self.context is not None:
            out["context"] = self.context.to_dict()
        if self.client_args:
            out["client_args"] = self.client_args
        if self.params:
            out["params"] = self.params
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class ProviderModels:
    display_name: str | None = None
    description: str | None = None
    tiers: dict[str, ModelTier] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, provider: str) -> "ProviderModels":
        display_name = raw.get("display_name")
        description = raw.get("description")

        tiers_raw = raw.get("tiers")
        tiers: dict[str, ModelTier] = {}
        if isinstance(tiers_raw, dict):
            for tier_name, tier_value in tiers_raw.items():
                if not isinstance(tier_name, str) or not isinstance(tier_value, dict):
                    continue
                tiers[tier_name.strip().lower()] = ModelTier.from_dict(tier_value, default_provider=provider)

        extra = {k: v for k, v in raw.items() if k not in {"display_name", "description", "tiers"}}
        return cls(
            display_name=str(display_name) if isinstance(display_name, str) else None,
            description=str(description) if isinstance(description, str) else None,
            tiers=tiers,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tiers": {k: v.to_dict() for k, v in self.tiers.items()}}
        if self.display_name:
            out["display_name"] = self.display_name
        if self.description:
            out["description"] = self.description
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class AutoEscalation:
    enabled: bool = False
    max_escalations_per_task: int = 1
    triggers: dict[str, Any] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AutoEscalation":
        enabled = raw.get("enabled")
        max_escalations_per_task = raw.get("max_escalations_per_task")
        triggers = raw.get("triggers")
        order_raw = raw.get("order")
        order: list[str] = []
        seen: set[str] = set()
        if isinstance(order_raw, list):
            for item in order_raw:
                tier_name = str(item or "").strip().lower()
                if not tier_name or tier_name in seen:
                    continue
                seen.add(tier_name)
                order.append(tier_name)
        return cls(
            enabled=bool(enabled)
            if isinstance(enabled, bool)
            else _truthy(str(enabled))
            if enabled is not None
            else False,
            max_escalations_per_task=int(max_escalations_per_task)
            if isinstance(max_escalations_per_task, (int, float, str)) and str(max_escalations_per_task).isdigit()
            else 1,
            triggers=triggers if isinstance(triggers, dict) else {},
            order=order,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_escalations_per_task": self.max_escalations_per_task,
            "triggers": self.triggers,
            "order": list(self.order),
        }


@dataclass(frozen=True)
class DefaultModelSelection:
    provider: str | None = None
    tier: str = "balanced"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DefaultModelSelection":
        provider_raw = raw.get("provider")
        tier_raw = raw.get("tier")
        provider = (
            normalize_provider_name(provider_raw)
            if isinstance(provider_raw, str) and provider_raw.strip()
            else None
        )
        tier = str(tier_raw or "balanced").strip().lower() or "balanced"
        return cls(provider=provider, tier=tier)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tier": self.tier}
        if self.provider:
            out["provider"] = self.provider
        return out


@dataclass(frozen=True)
class ModelsConfig:
    provider: str | None = None
    default_tier: str = "balanced"
    default_selection: DefaultModelSelection = field(default_factory=DefaultModelSelection)
    max_output_tokens: int | None = None
    tiers: dict[str, ModelTier] = field(default_factory=dict)
    providers: dict[str, ProviderModels] = field(default_factory=dict)
    auto_escalation: AutoEscalation = field(default_factory=AutoEscalation)
    availability: dict[str, Any] = field(default_factory=dict)
    hidden_tiers: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelsConfig":
        provider = raw.get("provider")
        default_tier = str(raw.get("default_tier") or "balanced").strip()
        default_selection_raw = raw.get("default_selection")
        max_output_tokens = _as_int(raw.get("max_output_tokens"), default=None, min_value=1)

        tiers_raw = raw.get("tiers")
        tiers: dict[str, ModelTier] = {}
        if isinstance(tiers_raw, dict):
            for tier_name, tier_value in tiers_raw.items():
                if not isinstance(tier_name, str) or not isinstance(tier_value, dict):
                    continue
                tiers[tier_name.strip().lower()] = ModelTier.from_dict(tier_value)

        providers_raw = raw.get("providers")
        providers: dict[str, ProviderModels] = {}
        if isinstance(providers_raw, dict):
            for provider_name, provider_value in providers_raw.items():
                if not isinstance(provider_name, str) or not isinstance(provider_value, dict):
                    continue
                key = normalize_provider_name(provider_name.strip())
                providers[key] = ProviderModels.from_dict(provider_value, provider=key)

        auto = raw.get("auto_escalation")
        availability = raw.get("availability")
        hidden_tiers_raw = raw.get("hidden_tiers")
        hidden_tiers: list[str] = []
        if isinstance(hidden_tiers_raw, list):
            seen_hidden: set[str] = set()
            for item in hidden_tiers_raw:
                token = str(item or "").strip().lower()
                if "|" not in token:
                    continue
                provider_name, tier_name = token.split("|", 1)
                provider_name = normalize_provider_name(provider_name.strip())
                tier_name = tier_name.strip().lower()
                if not provider_name or not tier_name:
                    continue
                key = f"{provider_name}|{tier_name}"
                if key in seen_hidden:
                    continue
                seen_hidden.add(key)
                hidden_tiers.append(key)
        legacy_provider = normalize_provider_name(provider) if isinstance(provider, str) and provider.strip() else None
        legacy_tier = default_tier.strip().lower() or "balanced"
        if isinstance(default_selection_raw, dict):
            parsed_default = DefaultModelSelection.from_dict(default_selection_raw)
            resolved_provider = parsed_default.provider if parsed_default.provider is not None else legacy_provider
            resolved_tier = parsed_default.tier or legacy_tier
            default_selection = DefaultModelSelection(provider=resolved_provider, tier=resolved_tier)
        else:
            default_selection = DefaultModelSelection(provider=legacy_provider, tier=legacy_tier)
        return cls(
            provider=default_selection.provider,
            default_tier=default_selection.tier,
            default_selection=default_selection,
            max_output_tokens=max_output_tokens,
            tiers=tiers,
            providers=providers,
            auto_escalation=AutoEscalation.from_dict(auto) if isinstance(auto, dict) else AutoEscalation(),
            availability=availability if isinstance(availability, dict) else {},
            hidden_tiers=hidden_tiers,
        )

    def to_dict(self) -> dict[str, Any]:
        resolved_provider = self.default_selection.provider if self.default_selection else self.provider
        resolved_tier = self.default_selection.tier if self.default_selection else self.default_tier
        resolved_tier = str(resolved_tier or "balanced").strip().lower() or "balanced"
        out: dict[str, Any] = {
            "default_tier": resolved_tier,
            "default_selection": DefaultModelSelection(provider=resolved_provider, tier=resolved_tier).to_dict(),
            "tiers": {k: v.to_dict() for k, v in self.tiers.items()},
            "providers": {k: v.to_dict() for k, v in self.providers.items()},
            "auto_escalation": self.auto_escalation.to_dict(),
            "availability": self.availability,
            "hidden_tiers": list(self.hidden_tiers),
        }
        if resolved_provider:
            out["provider"] = resolved_provider
        if self.max_output_tokens is not None:
            out["max_output_tokens"] = self.max_output_tokens
        return out


@dataclass(frozen=True)
class ToolRule:
    tool: str
    default: str = "ask"  # ask|allow|deny
    remember: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ToolRule":
        tool_name = str(raw.get("tool") or "").strip()
        default = str(raw.get("default") or "ask").strip().lower()
        remember = raw.get("remember")
        return cls(tool=tool_name, default=default, remember=bool(remember) if isinstance(remember, bool) else True)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "default": self.default, "remember": self.remember}


@dataclass(frozen=True)
class PermissionRule:
    tool: str
    action: str = "ask"  # allow|ask|deny
    remember: bool = True
    when: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PermissionRule":
        tool_name = str(raw.get("tool") or "").strip()
        action = str(raw.get("action") or "ask").strip().lower()
        remember = raw.get("remember")
        when = raw.get("when")
        return cls(
            tool=tool_name,
            action=action,
            remember=bool(remember) if isinstance(remember, bool) else True,
            when=when if isinstance(when, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "action": self.action,
            "remember": self.remember,
            "when": self.when,
        }


@dataclass(frozen=True)
class SafetyConfig:
    tool_consent: str = "ask"  # ask|allow|deny
    tool_rules: list[ToolRule] = field(default_factory=list)
    permission_rules: list[PermissionRule] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SafetyConfig":
        tool_consent = str(raw.get("tool_consent") or "ask").strip().lower()
        rules_raw = raw.get("tool_rules")
        permission_rules_raw = raw.get("permission_rules")
        tool_rules: list[ToolRule] = []
        permission_rules: list[PermissionRule] = []
        if isinstance(rules_raw, list):
            for item in rules_raw:
                if isinstance(item, dict):
                    tool_rules.append(ToolRule.from_dict(item))
        if isinstance(permission_rules_raw, list):
            for item in permission_rules_raw:
                if isinstance(item, dict):
                    permission_rules.append(PermissionRule.from_dict(item))
        return cls(tool_consent=tool_consent, tool_rules=tool_rules, permission_rules=permission_rules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_consent": self.tool_consent,
            "tool_rules": [r.to_dict() for r in self.tool_rules],
            "permission_rules": [r.to_dict() for r in self.permission_rules],
        }


@dataclass(frozen=True)
class PackEntry:
    type: str = "path_pack"
    name: str = ""
    path: str = ""
    enabled: bool = True
    id: str = ""
    provider: str | None = None
    tier: str | None = None
    system_prompt_snippets: list[str] = field(default_factory=list)
    context_sources: list[dict[str, Any]] = field(default_factory=list)
    active_sops: list[str] = field(default_factory=list)
    knowledge_base_id: str | None = None
    agents: list[dict[str, Any]] = field(default_factory=list)
    auto_delegate_assistive: bool = True
    team_presets: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PackEntry":
        entry_type = str(raw.get("type") or "").strip().lower()
        if entry_type not in {"path_pack", "agent_bundle"}:
            entry_type = "path_pack" if str(raw.get("path") or "").strip() else "agent_bundle"
        name = str(raw.get("name") or "").strip()
        path = str(raw.get("path") or "").strip()
        enabled = raw.get("enabled")
        bundle_id = str(raw.get("id") or "").strip()
        provider_raw = str(raw.get("provider") or "").strip().lower()
        tier_raw = str(raw.get("tier") or "").strip().lower()
        snippets_raw = raw.get("system_prompt_snippets")
        context_sources_raw = raw.get("context_sources")
        active_sops_raw = raw.get("active_sops")
        agents_raw = raw.get("agents")
        team_presets_raw = raw.get("team_presets")
        auto_delegate_raw = raw.get("auto_delegate_assistive", True)
        return cls(
            type=entry_type,
            name=name,
            path=path,
            enabled=bool(enabled) if isinstance(enabled, bool) else True,
            id=bundle_id,
            provider=provider_raw or None,
            tier=tier_raw or None,
            system_prompt_snippets=[str(item).strip() for item in snippets_raw if str(item).strip()]
            if isinstance(snippets_raw, list)
            else [],
            context_sources=[dict(item) for item in context_sources_raw if isinstance(item, dict)]
            if isinstance(context_sources_raw, list)
            else [],
            active_sops=[str(item).strip() for item in active_sops_raw if str(item).strip()]
            if isinstance(active_sops_raw, list)
            else [],
            knowledge_base_id=str(raw.get("knowledge_base_id") or "").strip() or None,
            agents=[dict(item) for item in agents_raw if isinstance(item, dict)]
            if isinstance(agents_raw, list)
            else [],
            auto_delegate_assistive=bool(auto_delegate_raw) if isinstance(auto_delegate_raw, bool) else True,
            team_presets=[dict(item) for item in team_presets_raw if isinstance(item, dict)]
            if isinstance(team_presets_raw, list)
            else [],
        )

    def to_dict(self) -> dict[str, Any]:
        if self.type == "agent_bundle":
            return {
                "type": "agent_bundle",
                "id": self.id,
                "name": self.name,
                "provider": self.provider,
                "tier": self.tier,
                "system_prompt_snippets": list(self.system_prompt_snippets),
                "context_sources": [dict(item) for item in self.context_sources],
                "active_sops": list(self.active_sops),
                "knowledge_base_id": self.knowledge_base_id,
                "agents": [dict(item) for item in self.agents],
                "auto_delegate_assistive": bool(self.auto_delegate_assistive),
                "team_presets": [dict(item) for item in self.team_presets],
                "enabled": bool(self.enabled),
            }
        return {"type": "path_pack", "name": self.name, "path": self.path, "enabled": self.enabled}


@dataclass(frozen=True)
class PacksConfig:
    installed: list[PackEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PacksConfig":
        installed_raw = raw.get("installed")
        installed: list[PackEntry] = []
        if isinstance(installed_raw, list):
            for item in installed_raw:
                if isinstance(item, dict):
                    installed.append(PackEntry.from_dict(item))
        return cls(installed=installed)

    def to_dict(self) -> dict[str, Any]:
        return {"installed": [p.to_dict() for p in self.installed]}


@dataclass(frozen=True)
class TierProfile:
    """
    Tier-specific harness defaults.

    These profiles are used to:
    - choose sensible context snapshot depth defaults per tier
    - optionally restrict tool availability per tier (allow/block lists)
    """

    tool_allowlist: list[str] = field(default_factory=list)
    tool_blocklist: list[str] = field(default_factory=list)
    preflight_level: str | None = None  # summary|summary+tree|summary+files

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TierProfile":
        allow = raw.get("tool_allowlist")
        block = raw.get("tool_blocklist")
        preflight_level = raw.get("preflight_level")
        return cls(
            tool_allowlist=[str(x).strip() for x in allow if str(x).strip()] if isinstance(allow, list) else [],
            tool_blocklist=[str(x).strip() for x in block if str(x).strip()] if isinstance(block, list) else [],
            preflight_level=str(preflight_level).strip().lower()
            if isinstance(preflight_level, str) and preflight_level.strip()
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tool_allowlist": list(self.tool_allowlist),
            "tool_blocklist": list(self.tool_blocklist),
        }
        if self.preflight_level:
            out["preflight_level"] = self.preflight_level
        return out


@dataclass(frozen=True)
class HarnessConfig:
    tier_profiles: dict[str, TierProfile] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HarnessConfig":
        profiles_raw = raw.get("tier_profiles")
        tier_profiles: dict[str, TierProfile] = {}
        if isinstance(profiles_raw, dict):
            for tier_name, profile_value in profiles_raw.items():
                if not isinstance(tier_name, str) or not isinstance(profile_value, dict):
                    continue
                tier_profiles[tier_name.strip().lower()] = TierProfile.from_dict(profile_value)
        return cls(tier_profiles=tier_profiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_profiles": {k: v.to_dict() for k, v in self.tier_profiles.items()},
        }


@dataclass(frozen=True)
class SwarmeeSettings:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    packs: PacksConfig = field(default_factory=PacksConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SwarmeeSettings":
        models_raw = raw.get("models")
        context_raw = raw.get("context")
        runtime_raw = raw.get("runtime")
        diagnostics_raw = raw.get("diagnostics")
        pricing_raw = raw.get("pricing")
        safety_raw = raw.get("safety")
        packs_raw = raw.get("packs")
        harness_raw = raw.get("harness")
        return cls(
            models=ModelsConfig.from_dict(models_raw) if isinstance(models_raw, dict) else ModelsConfig(),
            context=ContextConfig.from_dict(context_raw) if isinstance(context_raw, dict) else ContextConfig(),
            runtime=RuntimeConfig.from_dict(runtime_raw) if isinstance(runtime_raw, dict) else RuntimeConfig(),
            diagnostics=DiagnosticsConfig.from_dict(diagnostics_raw)
            if isinstance(diagnostics_raw, dict)
            else DiagnosticsConfig(),
            pricing=PricingConfig.from_dict(pricing_raw) if isinstance(pricing_raw, dict) else PricingConfig(),
            safety=SafetyConfig.from_dict(safety_raw) if isinstance(safety_raw, dict) else SafetyConfig(),
            packs=PacksConfig.from_dict(packs_raw) if isinstance(packs_raw, dict) else PacksConfig(),
            harness=HarnessConfig.from_dict(harness_raw) if isinstance(harness_raw, dict) else HarnessConfig(),
            raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        base = dict(self.raw) if isinstance(self.raw, dict) else {}
        base.update(
            {
                "models": self.models.to_dict(),
                "context": self.context.to_dict(),
                "runtime": self.runtime.to_dict(),
                "diagnostics": self.diagnostics.to_dict(),
                "pricing": self.pricing.to_dict(),
                "safety": self.safety.to_dict(),
                "packs": self.packs.to_dict(),
                "harness": self.harness.to_dict(),
            }
        )
        return base


def load_settings(path: Path | None = None) -> SwarmeeSettings:
    """
    Load Swarmee settings from `.swarmee/settings.json` (project-local).
    """
    settings_path = path or _default_settings_path()
    raw = _load_settings_payload(settings_path)
    if raw and isinstance(raw, dict) and raw.get("env"):
        raw, _migrated, _dropped = migrate_legacy_env_overrides(raw)

    defaults = default_settings_template().to_dict()
    merged = deep_merge_dict(defaults, raw) if raw else defaults
    settings = SwarmeeSettings.from_dict(merged)

    hidden_tiers = list(settings.models.hidden_tiers)
    if hidden_tiers:
        hidden_keys = {str(item).strip().lower() for item in hidden_tiers if str(item).strip()}
        filtered_providers: dict[str, ProviderModels] = {}
        for provider_name, provider in settings.models.providers.items():
            filtered_tiers: dict[str, ModelTier] = {}
            for tier_name, tier in provider.tiers.items():
                if f"{provider_name}|{tier_name}".lower() in hidden_keys:
                    continue
                filtered_tiers[tier_name] = tier
            if filtered_tiers:
                filtered_providers[provider_name] = ProviderModels(
                    display_name=provider.display_name,
                    description=provider.description,
                    tiers=filtered_tiers,
                    extra=provider.extra,
                )
        selected_provider = settings.models.provider
        if selected_provider and selected_provider not in filtered_providers:
            selected_provider = None
        models = ModelsConfig(
            provider=selected_provider,
            default_tier=settings.models.default_tier,
            default_selection=DefaultModelSelection(provider=selected_provider, tier=settings.models.default_tier),
            max_output_tokens=settings.models.max_output_tokens,
            tiers=settings.models.tiers,
            providers=filtered_providers,
            auto_escalation=settings.models.auto_escalation,
            availability=settings.models.availability,
            hidden_tiers=hidden_tiers,
        )

    if hidden_tiers and models is not settings.models:
        # Hidden tiers require key deletion, which `deep_merge_dict` cannot express
        # (it only overlays). Replace the entire `models` object.
        overridden = dict(merged)
        overridden["models"] = models.to_dict()
        return SwarmeeSettings.from_dict(overridden)

    return settings


def save_settings(settings: SwarmeeSettings, path: Path | None = None) -> Path:
    settings_path = path or _default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return settings_path


def default_settings_template() -> SwarmeeSettings:
    # Package defaults: designed to work without users editing `.swarmee/settings.json`.
    #
    # Provider selection precedence (highest -> lowest):
    # - CLI args
    # - `.swarmee/settings.json`
    # - built-ins (this function)
    return SwarmeeSettings(
        models=ModelsConfig(
            provider=None,
            default_tier="balanced",
            default_selection=DefaultModelSelection(provider=None, tier="balanced"),
            tiers={},
            providers={
                "bedrock": ProviderModels(
                    display_name="Amazon Bedrock",
                    description="AWS-managed models (Anthropic Claude, etc).",
                    tiers={
                        "fast": ModelTier(
                            provider="bedrock",
                            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                            display_name="Claude Haiku 4.5 (fast)",
                            description="Fast Bedrock tier for quick low-cost iterations.",
                            reasoning=ModelReasoningConfig(effort="low"),
                            tooling=ModelToolingConfig(mode="minimal", discovery="off"),
                            context=ModelContextBehavior(strategy="balanced", compaction="auto"),
                        ),
                        "balanced": ModelTier(
                            provider="bedrock",
                            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                            display_name="Claude Sonnet 4.5 (balanced)",
                            description="Default Bedrock tier for enterprise analytics and coding work.",
                            reasoning=ModelReasoningConfig(effort="medium"),
                            tooling=ModelToolingConfig(mode="standard", discovery="off"),
                            context=ModelContextBehavior(strategy="balanced", compaction="auto"),
                        ),
                        "deep": ModelTier(
                            provider="bedrock",
                            model_id="us.anthropic.claude-opus-4-6-v1:0",
                            display_name="Claude Opus 4.6 (deep)",
                            description="Adaptive Claude reasoning for harder analytics tasks.",
                            reasoning=ModelReasoningConfig(effort="high"),
                            tooling=ModelToolingConfig(mode="tool-heavy", discovery="search"),
                            context=ModelContextBehavior(strategy="cache_safe", compaction="auto"),
                        ),
                        "long": ModelTier(
                            provider="bedrock",
                            model_id="us.anthropic.claude-opus-4-6-v1:0",
                            display_name="Claude Opus 4.6 (long)",
                            description="Use for long-running Bedrock sessions and larger outputs.",
                            reasoning=ModelReasoningConfig(effort="high"),
                            tooling=ModelToolingConfig(mode="tool-heavy", discovery="search"),
                            context=ModelContextBehavior(strategy="long_running", compaction="auto"),
                        ),
                    },
                ),
                "openai": ProviderModels(
                    display_name="OpenAI",
                    description="OpenAI API models. Requires OPENAI_API_KEY.",
                    tiers={
                        "fast": ModelTier(
                            provider="openai",
                            model_id="gpt-5-nano",
                            display_name="GPT-5 nano",
                            description="Lowest latency/cost; good for quick iterations.",
                            transport="responses",
                            reasoning=ModelReasoningConfig(effort="low"),
                            tooling=ModelToolingConfig(mode="minimal", discovery="off"),
                            context=ModelContextBehavior(strategy="balanced", compaction="manual"),
                        ),
                        "balanced": ModelTier(
                            provider="openai",
                            model_id="gpt-5-mini",
                            display_name="GPT-5 mini",
                            description="Default OpenAI tier for most coding tasks.",
                            transport="responses",
                            reasoning=ModelReasoningConfig(effort="medium"),
                            tooling=ModelToolingConfig(mode="standard", discovery="off"),
                            context=ModelContextBehavior(strategy="balanced", compaction="auto"),
                        ),
                        "deep": ModelTier(
                            provider="openai",
                            model_id="gpt-5.2",
                            display_name="GPT-5.2",
                            description="Deep repo reasoning with tool-heavy workflows.",
                            transport="responses",
                            reasoning=ModelReasoningConfig(effort="high"),
                            tooling=ModelToolingConfig(mode="tool-heavy", discovery="search"),
                            context=ModelContextBehavior(strategy="cache_safe", compaction="auto"),
                        ),
                        "long": ModelTier(
                            provider="openai",
                            model_id="gpt-5.2",
                            display_name="GPT-5.2 (long)",
                            description="Long-running analysis with more aggressive compaction.",
                            transport="responses",
                            reasoning=ModelReasoningConfig(effort="medium"),
                            tooling=ModelToolingConfig(mode="tool-heavy", discovery="search"),
                            context=ModelContextBehavior(strategy="long_running", compaction="auto"),
                        ),
                        "coding": ModelTier(
                            provider="openai",
                            model_id="gpt-5.3-codex",
                            display_name="GPT-5.3 Codex (coding)",
                            description="Optimized tier for coding-heavy workflows and refactors.",
                            transport="responses",
                            reasoning=ModelReasoningConfig(effort="high"),
                            tooling=ModelToolingConfig(mode="tool-heavy", discovery="search"),
                            context=ModelContextBehavior(strategy="cache_safe", compaction="auto"),
                        ),
                    },
                ),
                "ollama": ProviderModels(
                    display_name="Ollama",
                    description="Local models via Ollama.",
                    tiers={
                        "fast": ModelTier(
                            provider="ollama",
                            model_id="llama3.1",
                            display_name="llama3.1",
                            description="Local default for fast iteration.",
                            reasoning=ModelReasoningConfig(effort="low"),
                            tooling=ModelToolingConfig(mode="minimal", discovery="off"),
                            context=ModelContextBehavior(strategy="balanced", compaction="manual"),
                        ),
                        "balanced": ModelTier(
                            provider="ollama",
                            model_id="llama3.1",
                            display_name="llama3.1",
                            description="Local balanced default.",
                            reasoning=ModelReasoningConfig(effort="medium"),
                            tooling=ModelToolingConfig(mode="standard", discovery="off"),
                            context=ModelContextBehavior(strategy="balanced", compaction="auto"),
                        ),
                        "deep": ModelTier(
                            provider="ollama",
                            model_id="llama3.1",
                            display_name="llama3.1",
                            description="Local deep tier.",
                            reasoning=ModelReasoningConfig(effort="high"),
                            tooling=ModelToolingConfig(mode="standard", discovery="off"),
                            context=ModelContextBehavior(strategy="balanced", compaction="auto"),
                        ),
                        "long": ModelTier(
                            provider="ollama",
                            model_id="llama3.1",
                            display_name="llama3.1",
                            description="Local long tier.",
                            reasoning=ModelReasoningConfig(effort="medium"),
                            tooling=ModelToolingConfig(mode="standard", discovery="off"),
                            context=ModelContextBehavior(strategy="long_running", compaction="auto"),
                        ),
                    },
                ),
                "github_copilot": ProviderModels(
                    display_name="GitHub Copilot",
                    description=(
                        "OpenAI-compatible endpoint for GitHub Copilot. "
                        "Uses SWARMEE_GITHUB_COPILOT_API_KEY or GITHUB_TOKEN."
                    ),
                    tiers={
                        "fast": ModelTier(
                            provider="github_copilot",
                            model_id="gpt-4o-mini",
                            display_name="GPT-4o mini (Copilot)",
                            description="Lower latency default for Copilot.",
                            reasoning=ModelReasoningConfig(effort="low"),
                            tooling=ModelToolingConfig(mode="minimal", discovery="off"),
                            context=ModelContextBehavior(strategy="balanced", compaction="manual"),
                        ),
                        "balanced": ModelTier(
                            provider="github_copilot",
                            model_id="gpt-4o",
                            display_name="GPT-4o (Copilot)",
                            description="Default Copilot tier for most coding tasks.",
                            reasoning=ModelReasoningConfig(effort="medium"),
                            tooling=ModelToolingConfig(mode="standard", discovery="off"),
                            context=ModelContextBehavior(strategy="balanced", compaction="auto"),
                        ),
                        "deep": ModelTier(
                            provider="github_copilot",
                            model_id="gpt-5",
                            display_name="GPT-5 (Copilot)",
                            description="Stronger reasoning where available.",
                            reasoning=ModelReasoningConfig(effort="high"),
                            tooling=ModelToolingConfig(mode="tool-heavy", discovery="search"),
                            context=ModelContextBehavior(strategy="cache_safe", compaction="auto"),
                        ),
                        "long": ModelTier(
                            provider="github_copilot",
                            model_id="gpt-5",
                            display_name="GPT-5 (Copilot long)",
                            description="Long-form outputs on Copilot.",
                            reasoning=ModelReasoningConfig(effort="medium"),
                            tooling=ModelToolingConfig(mode="tool-heavy", discovery="search"),
                            context=ModelContextBehavior(strategy="long_running", compaction="auto"),
                        ),
                    },
                ),
            },
            auto_escalation=AutoEscalation(
                enabled=False,
                max_escalations_per_task=1,
                triggers={},
                order=["fast", "balanced", "deep", "long"],
            ),
            availability={},
        ),
        safety=SafetyConfig(
            tool_consent="ask",
            tool_rules=[
                ToolRule(tool="shell", default="ask", remember=True),
                ToolRule(tool="bash", default="ask", remember=True),
                ToolRule(tool="file_write", default="ask", remember=True),
                ToolRule(tool="write", default="ask", remember=True),
                ToolRule(tool="editor", default="ask", remember=True),
                ToolRule(tool="edit", default="ask", remember=True),
                ToolRule(tool="http_request", default="ask", remember=True),
                ToolRule(tool="git", default="ask", remember=True),
                ToolRule(tool="patch_apply", default="ask", remember=True),
                ToolRule(tool="patch", default="ask", remember=True),
                ToolRule(tool="file_search", default="allow", remember=True),
                ToolRule(tool="grep", default="allow", remember=True),
                ToolRule(tool="file_read", default="allow", remember=True),
                ToolRule(tool="read", default="allow", remember=True),
                ToolRule(tool="todoread", default="allow", remember=True),
                ToolRule(tool="todowrite", default="ask", remember=True),
                ToolRule(tool="run_checks", default="ask", remember=True),
            ],
        ),
        packs=PacksConfig(installed=[]),
        harness=HarnessConfig(
            tier_profiles={
                "fast": TierProfile(preflight_level="summary"),
                "balanced": TierProfile(preflight_level="summary+tree"),
                "deep": TierProfile(preflight_level="summary+files"),
                "long": TierProfile(preflight_level="summary+files"),
                "coding": TierProfile(preflight_level="summary+files"),
            }
        ),
        raw={},
    )
