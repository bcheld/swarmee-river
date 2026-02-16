from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _default_settings_path() -> Path:
    return Path.cwd() / ".swarmee" / "settings.json"


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts (override wins). Lists are replaced, not merged."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        out_value = out.get(key)
        if isinstance(value, dict) and isinstance(out_value, dict):
            out[key] = _deep_merge_dict(out_value, value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class ModelTier:
    provider: str = ""
    model_id: str | None = None
    display_name: str | None = None
    description: str | None = None
    client_args: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_provider: str | None = None) -> "ModelTier":
        provider = str(raw.get("provider") or default_provider or "").strip()
        model_id = raw.get("model_id")
        display_name = raw.get("display_name")
        description = raw.get("description")
        client_args = raw.get("client_args")
        params = raw.get("params")

        extra = {
            k: v
            for k, v in raw.items()
            if k not in {"provider", "model_id", "display_name", "description", "client_args", "params"}
        }
        return cls(
            provider=provider,
            model_id=str(model_id) if isinstance(model_id, str) else None,
            display_name=str(display_name) if isinstance(display_name, str) else None,
            description=str(description) if isinstance(description, str) else None,
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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AutoEscalation":
        enabled = raw.get("enabled")
        max_escalations_per_task = raw.get("max_escalations_per_task")
        triggers = raw.get("triggers")
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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_escalations_per_task": self.max_escalations_per_task,
            "triggers": self.triggers,
        }


@dataclass(frozen=True)
class ModelsConfig:
    provider: str | None = None
    default_tier: str = "balanced"
    tiers: dict[str, ModelTier] = field(default_factory=dict)
    providers: dict[str, ProviderModels] = field(default_factory=dict)
    auto_escalation: AutoEscalation = field(default_factory=AutoEscalation)
    availability: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelsConfig":
        provider = raw.get("provider")
        default_tier = str(raw.get("default_tier") or "balanced").strip()

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
                key = provider_name.strip().lower()
                providers[key] = ProviderModels.from_dict(provider_value, provider=key)

        auto = raw.get("auto_escalation")
        availability = raw.get("availability")
        return cls(
            provider=str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else None,
            default_tier=default_tier,
            tiers=tiers,
            providers=providers,
            auto_escalation=AutoEscalation.from_dict(auto) if isinstance(auto, dict) else AutoEscalation(),
            availability=availability if isinstance(availability, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "default_tier": self.default_tier,
            "tiers": {k: v.to_dict() for k, v in self.tiers.items()},
            "providers": {k: v.to_dict() for k, v in self.providers.items()},
            "auto_escalation": self.auto_escalation.to_dict(),
            "availability": self.availability,
        }
        if self.provider:
            out["provider"] = self.provider
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
class SafetyConfig:
    tool_consent: str = "ask"  # ask|allow|deny
    tool_rules: list[ToolRule] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SafetyConfig":
        tool_consent = str(raw.get("tool_consent") or "ask").strip().lower()
        rules_raw = raw.get("tool_rules")
        tool_rules: list[ToolRule] = []
        if isinstance(rules_raw, list):
            for item in rules_raw:
                if isinstance(item, dict):
                    tool_rules.append(ToolRule.from_dict(item))
        return cls(tool_consent=tool_consent, tool_rules=tool_rules)

    def to_dict(self) -> dict[str, Any]:
        return {"tool_consent": self.tool_consent, "tool_rules": [r.to_dict() for r in self.tool_rules]}


@dataclass(frozen=True)
class PackEntry:
    name: str
    path: str
    enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PackEntry":
        name = str(raw.get("name") or "").strip()
        path = str(raw.get("path") or "").strip()
        enabled = raw.get("enabled")
        return cls(name=name, path=path, enabled=bool(enabled) if isinstance(enabled, bool) else True)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "enabled": self.enabled}


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
        return {"tier_profiles": {k: v.to_dict() for k, v in self.tier_profiles.items()}}


@dataclass(frozen=True)
class SwarmeeSettings:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    packs: PacksConfig = field(default_factory=PacksConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SwarmeeSettings":
        models_raw = raw.get("models")
        safety_raw = raw.get("safety")
        packs_raw = raw.get("packs")
        harness_raw = raw.get("harness")
        return cls(
            models=ModelsConfig.from_dict(models_raw) if isinstance(models_raw, dict) else ModelsConfig(),
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
                "safety": self.safety.to_dict(),
                "packs": self.packs.to_dict(),
                "harness": self.harness.to_dict(),
            }
        )
        return base


def load_settings(path: Path | None = None) -> SwarmeeSettings:
    """
    Load Swarmee settings from `.swarmee/settings.json` (project-local).

    Environment overrides:
    - SWARMEE_MODEL_TIER: force the active tier in the CLI
    - SWARMEE_TIER_AUTO: enable/disable auto escalation
    - SWARMEE_MODEL_PROVIDER: choose the default provider (bedrock|openai|ollama)
    """
    settings_path = path or _default_settings_path()
    raw: dict[str, Any] = {}
    if settings_path.exists() and settings_path.is_file():
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}

    defaults = default_settings_template().to_dict()
    merged = _deep_merge_dict(defaults, raw) if raw else defaults
    settings = SwarmeeSettings.from_dict(merged)

    # Apply env overrides into the returned structure (does not persist to disk).
    forced_tier = os.getenv("SWARMEE_MODEL_TIER")
    tier_auto = os.getenv("SWARMEE_TIER_AUTO")
    forced_provider = os.getenv("SWARMEE_MODEL_PROVIDER")

    models = settings.models
    if forced_provider and forced_provider.strip():
        models = ModelsConfig(
            provider=forced_provider.strip().lower(),
            default_tier=models.default_tier,
            tiers=models.tiers,
            providers=models.providers,
            auto_escalation=models.auto_escalation,
            availability=models.availability,
        )

    if forced_tier and forced_tier.strip():
        models = ModelsConfig(
            provider=models.provider,
            default_tier=forced_tier.strip().lower(),
            tiers=models.tiers,
            providers=models.providers,
            auto_escalation=models.auto_escalation,
            availability=models.availability,
        )

    if tier_auto is not None:
        auto = AutoEscalation(
            enabled=_truthy(tier_auto),
            max_escalations_per_task=models.auto_escalation.max_escalations_per_task,
            triggers=models.auto_escalation.triggers,
        )
        models = ModelsConfig(
            provider=models.provider,
            default_tier=models.default_tier,
            tiers=models.tiers,
            providers=models.providers,
            auto_escalation=auto,
            availability=models.availability,
        )

    if models is not settings.models:
        return SwarmeeSettings(
            models=models,
            safety=settings.safety,
            packs=settings.packs,
            harness=settings.harness,
            raw=settings.raw,
        )

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
    # - env (e.g., SWARMEE_MODEL_PROVIDER)
    # - `.swarmee/settings.json`
    # - built-ins (this function)
    return SwarmeeSettings(
        models=ModelsConfig(
            provider=None,
            default_tier="balanced",
            tiers={},
            providers={
                "bedrock": ProviderModels(
                    display_name="Amazon Bedrock",
                    description="AWS-managed models (Anthropic Claude, etc). Uses STRANDS_* env vars by default.",
                    tiers={
                        "fast": ModelTier(
                            provider="bedrock",
                            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                            display_name="Claude Sonnet 4 (fast)",
                            description=(
                                "Lower latency / lower cost default (override via SWARMEE_BEDROCK_FAST_MODEL_ID)."
                            ),
                        ),
                        "balanced": ModelTier(
                            provider="bedrock",
                            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                            display_name="Claude Sonnet 4 (balanced)",
                            description="Default tier for most coding and analysis work.",
                        ),
                        "deep": ModelTier(
                            provider="bedrock",
                            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                            display_name="Claude Sonnet 4 (deep)",
                            description=(
                                "Use when you need stronger reasoning (override via SWARMEE_BEDROCK_DEEP_MODEL_ID)."
                            ),
                            extra={
                                "additional_request_fields": {
                                    "thinking": {
                                        "type": "enabled",
                                        "budget_tokens": 8192,
                                    }
                                }
                            },
                        ),
                        "long": ModelTier(
                            provider="bedrock",
                            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                            display_name="Claude Sonnet 4 (long)",
                            description=(
                                "Use for long outputs and large refactors (override via SWARMEE_BEDROCK_LONG_MODEL_ID)."
                            ),
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
                        ),
                        "balanced": ModelTier(
                            provider="openai",
                            model_id="gpt-5-mini",
                            display_name="GPT-5 mini",
                            description="Default OpenAI tier for most coding tasks.",
                        ),
                        "deep": ModelTier(
                            provider="openai",
                            model_id="gpt-5",
                            display_name="GPT-5",
                            description="Stronger reasoning; slower / more expensive.",
                        ),
                        "long": ModelTier(
                            provider="openai",
                            model_id="gpt-5",
                            display_name="GPT-5 (long)",
                            description="Long-form outputs (override via SWARMEE_OPENAI_LONG_MODEL_ID).",
                        ),
                    },
                ),
                "ollama": ProviderModels(
                    display_name="Ollama",
                    description="Local models via Ollama (SWARMEE_OLLAMA_HOST).",
                    tiers={
                        "fast": ModelTier(
                            provider="ollama",
                            model_id="llama3.1",
                            display_name="llama3.1",
                            description="Local default for fast iteration (override via SWARMEE_OLLAMA_FAST_MODEL_ID).",
                        ),
                        "balanced": ModelTier(
                            provider="ollama",
                            model_id="llama3.1",
                            display_name="llama3.1",
                            description="Local balanced default.",
                        ),
                        "deep": ModelTier(
                            provider="ollama",
                            model_id="llama3.1",
                            display_name="llama3.1",
                            description="Local deep tier (override via SWARMEE_OLLAMA_DEEP_MODEL_ID).",
                        ),
                        "long": ModelTier(
                            provider="ollama",
                            model_id="llama3.1",
                            display_name="llama3.1",
                            description="Local long tier (override via SWARMEE_OLLAMA_LONG_MODEL_ID).",
                        ),
                    },
                ),
            },
            auto_escalation=AutoEscalation(enabled=False, max_escalations_per_task=1, triggers={}),
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
            }
        ),
        raw={},
    )
