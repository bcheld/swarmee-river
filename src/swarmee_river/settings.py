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


@dataclass(frozen=True)
class ModelTier:
    provider: str
    model_id: str | None = None
    client_args: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelTier":
        provider = str(raw.get("provider") or "").strip()
        model_id = raw.get("model_id")
        client_args = raw.get("client_args")
        params = raw.get("params")

        extra = {k: v for k, v in raw.items() if k not in {"provider", "model_id", "client_args", "params"}}
        return cls(
            provider=provider,
            model_id=str(model_id) if isinstance(model_id, str) else None,
            client_args=client_args if isinstance(client_args, dict) else None,
            params=params if isinstance(params, dict) else None,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"provider": self.provider}
        if self.model_id:
            out["model_id"] = self.model_id
        if self.client_args:
            out["client_args"] = self.client_args
        if self.params:
            out["params"] = self.params
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
            enabled=bool(enabled) if isinstance(enabled, bool) else _truthy(str(enabled)) if enabled is not None else False,
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
    default_tier: str = "balanced"
    tiers: dict[str, ModelTier] = field(default_factory=dict)
    auto_escalation: AutoEscalation = field(default_factory=AutoEscalation)
    availability: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelsConfig":
        default_tier = str(raw.get("default_tier") or "balanced").strip()
        tiers_raw = raw.get("tiers")
        tiers: dict[str, ModelTier] = {}
        if isinstance(tiers_raw, dict):
            for tier_name, tier_value in tiers_raw.items():
                if not isinstance(tier_name, str) or not isinstance(tier_value, dict):
                    continue
                tiers[tier_name.strip().lower()] = ModelTier.from_dict(tier_value)
        auto = raw.get("auto_escalation")
        availability = raw.get("availability")
        return cls(
            default_tier=default_tier,
            tiers=tiers,
            auto_escalation=AutoEscalation.from_dict(auto) if isinstance(auto, dict) else AutoEscalation(),
            availability=availability if isinstance(availability, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_tier": self.default_tier,
            "tiers": {k: v.to_dict() for k, v in self.tiers.items()},
            "auto_escalation": self.auto_escalation.to_dict(),
            "availability": self.availability,
        }


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
class SwarmeeSettings:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    packs: PacksConfig = field(default_factory=PacksConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SwarmeeSettings":
        models_raw = raw.get("models")
        safety_raw = raw.get("safety")
        packs_raw = raw.get("packs")
        return cls(
            models=ModelsConfig.from_dict(models_raw) if isinstance(models_raw, dict) else ModelsConfig(),
            safety=SafetyConfig.from_dict(safety_raw) if isinstance(safety_raw, dict) else SafetyConfig(),
            packs=PacksConfig.from_dict(packs_raw) if isinstance(packs_raw, dict) else PacksConfig(),
            raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        base = dict(self.raw) if isinstance(self.raw, dict) else {}
        base.update(
            {
                "models": self.models.to_dict(),
                "safety": self.safety.to_dict(),
                "packs": self.packs.to_dict(),
            }
        )
        return base


def load_settings(path: Path | None = None) -> SwarmeeSettings:
    """
    Load Swarmee settings from `.swarmee/settings.json` (project-local).

    Environment overrides:
    - SWARMEE_MODEL_TIER: force the active tier in the CLI
    - SWARMEE_TIER_AUTO: enable/disable auto escalation
    """
    settings_path = path or _default_settings_path()
    raw: dict[str, Any] = {}
    if settings_path.exists() and settings_path.is_file():
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}

    settings = SwarmeeSettings.from_dict(raw)

    # Apply env overrides into the returned structure (does not persist to disk).
    forced_tier = os.getenv("SWARMEE_MODEL_TIER")
    tier_auto = os.getenv("SWARMEE_TIER_AUTO")

    models = settings.models
    if forced_tier and forced_tier.strip():
        models = ModelsConfig(
            default_tier=forced_tier.strip().lower(),
            tiers=models.tiers,
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
            default_tier=models.default_tier,
            tiers=models.tiers,
            auto_escalation=auto,
            availability=models.availability,
        )

    if models is not settings.models:
        return SwarmeeSettings(models=models, safety=settings.safety, packs=settings.packs, raw=settings.raw)

    return settings


def save_settings(settings: SwarmeeSettings, path: Path | None = None) -> Path:
    settings_path = path or _default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return settings_path


def default_settings_template() -> SwarmeeSettings:
    # Keep defaults conservative; users can edit `.swarmee/settings.json` per environment.
    return SwarmeeSettings(
        models=ModelsConfig(
            default_tier="balanced",
            tiers={
                "fast": ModelTier(provider="bedrock", model_id=""),
                "balanced": ModelTier(provider="bedrock", model_id=""),
                "deep": ModelTier(provider="bedrock", model_id=""),
                "long": ModelTier(provider="bedrock", model_id=""),
            },
            auto_escalation=AutoEscalation(enabled=False, max_escalations_per_task=1, triggers={}),
            availability={},
        ),
        safety=SafetyConfig(
            tool_consent="ask",
            tool_rules=[
                ToolRule(tool="shell", default="ask", remember=True),
                ToolRule(tool="file_write", default="ask", remember=True),
                ToolRule(tool="editor", default="ask", remember=True),
                ToolRule(tool="http_request", default="ask", remember=True),
            ],
        ),
        packs=PacksConfig(installed=[]),
        raw={},
    )

