from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from strands.models import Model

from swarmee_river.settings import ModelTier, ProviderModels, SwarmeeSettings
from swarmee_river.utils import model_utils


_TIER_ORDER: list[str] = ["fast", "balanced", "deep", "long"]


@dataclass
class TierStatus:
    name: str
    provider: str
    model_id: str | None
    display_name: str | None
    description: str | None
    available: bool
    reason: str | None = None


class SessionModelManager:
    """
    Manage model tier selection within a single CLI session.

    Implementation strategy: swap `agent.model` in-place so conversation history, state, hooks,
    and tool registry remain intact.
    """

    def __init__(self, settings: SwarmeeSettings, *, fallback_provider: str = "bedrock") -> None:
        self._settings = settings
        self._fallback_provider = (fallback_provider or "bedrock").strip().lower()
        self._default_provider = (
            (settings.models.provider or self._fallback_provider or "bedrock").strip().lower()
        )
        self._fallback_config: dict[str, Any] | None = None
        self.current_tier: str = settings.models.default_tier.strip().lower() or "balanced"
        self.auto_escalation_enabled: bool = settings.models.auto_escalation.enabled
        self.max_escalations_per_task: int = max(0, settings.models.auto_escalation.max_escalations_per_task)

    def set_fallback_config(self, config: dict[str, Any] | None) -> None:
        self._fallback_config = dict(config) if isinstance(config, dict) else None

    def list_tiers(self) -> list[TierStatus]:
        out: list[TierStatus] = []
        for name in _TIER_ORDER:
            tier = self._resolve_tier(name)
            effective_model_id = self._effective_model_id(tier, tier_name=name)
            available, reason = self._is_tier_available(tier)
            out.append(
                TierStatus(
                    name=name,
                    provider=tier.provider,
                    model_id=effective_model_id,
                    display_name=tier.display_name,
                    description=tier.description,
                    available=available,
                    reason=reason,
                )
            )
        return out

    def set_auto_escalation(self, enabled: bool) -> None:
        self.auto_escalation_enabled = enabled

    def set_tier(self, agent: Any, tier_name: str) -> None:
        tier_name = (tier_name or "").strip().lower()
        if tier_name not in _TIER_ORDER:
            raise ValueError(f"Unknown tier: {tier_name}. Expected one of: {', '.join(_TIER_ORDER)}")

        model = self._build_model_for_tier(tier_name)
        agent.model = model
        self.current_tier = tier_name

    def build_model(self, tier_name: str | None = None) -> Model:
        """Build a model for the given tier (or current tier) without mutating an agent."""
        resolved = (tier_name or self.current_tier or "balanced").strip().lower()
        return self._build_model_for_tier(resolved)

    def maybe_escalate(self, agent: Any, *, attempted: set[str]) -> bool:
        """Escalate to the next tier if possible. Returns True if a switch occurred."""
        if not self.auto_escalation_enabled:
            return False

        try:
            idx = _TIER_ORDER.index(self.current_tier)
        except ValueError:
            idx = 1

        for next_tier in _TIER_ORDER[idx + 1 :]:
            if next_tier in attempted:
                continue
            try:
                self.set_tier(agent, next_tier)
                return True
            except Exception:
                attempted.add(next_tier)
                continue
        return False

    def _build_model_for_tier(self, tier_name: str) -> Model:
        tier = self._resolve_tier(tier_name)

        provider = (tier.provider or "").strip().lower()
        if provider not in {"bedrock", "openai", "ollama"}:
            raise ValueError(f"Unsupported provider {provider!r} for tier {tier_name!r}")

        base = model_utils.default_model_config(provider)
        config = dict(base)

        if self._fallback_config is not None and provider == self._fallback_provider:
            config.update(self._fallback_config)

        # Apply model_id overrides with clear precedence:
        # - per-tier env overrides (SWARMEE_{PROVIDER}_{TIER}_MODEL_ID)
        # - provider-level env override (e.g., STRANDS_MODEL_ID, SWARMEE_OPENAI_MODEL_ID)
        # - settings.json tier config / built-in defaults
        env_tier_model_id = self._env_tier_model_id(provider, tier_name)
        if env_tier_model_id:
            config["model_id"] = env_tier_model_id
        else:
            provider_env_model_id = self._provider_env_model_id(provider)
            if not provider_env_model_id:
                if tier.model_id and tier.model_id.strip():
                    config["model_id"] = tier.model_id.strip()
        if tier.params:
            existing = config.get("params")
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(tier.params)
                config["params"] = merged
            else:
                config["params"] = tier.params
        if tier.client_args:
            existing = config.get("client_args")
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(tier.client_args)
                config["client_args"] = merged
            else:
                config["client_args"] = tier.client_args

        for k, v in (tier.extra or {}).items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k] = self._deep_merge_dict(config[k], v)  # type: ignore[arg-type]
            else:
                config[k] = v

        model_path = model_utils.load_path(provider)
        return model_utils.load_model(model_path, config)

    def _resolve_tier(self, tier_name: str) -> ModelTier:
        tier_name = (tier_name or "").strip().lower()
        global_override = self._settings.models.tiers.get(tier_name)

        provider = self._default_provider
        if global_override and global_override.provider and global_override.provider.strip():
            provider = global_override.provider.strip().lower()

        provider_tier = self._provider_tier(provider, tier_name)
        resolved = provider_tier
        if global_override:
            resolved = self._merge_tiers(resolved, global_override, default_provider=provider)

        # Ensure provider is always set.
        if not resolved.provider:
            resolved = ModelTier.from_dict(resolved.to_dict(), default_provider=provider)
        return resolved

    def _provider_tier(self, provider: str, tier_name: str) -> ModelTier:
        provider = (provider or "").strip().lower()
        tier_name = (tier_name or "").strip().lower()

        pm = self._settings.models.providers.get(provider)
        if isinstance(pm, ProviderModels):
            tier = pm.tiers.get(tier_name)
            if tier is not None:
                return tier
        return ModelTier(provider=provider)

    def _effective_model_id(self, tier: ModelTier, *, tier_name: str) -> str | None:
        provider = (tier.provider or "").strip().lower()
        tier_name = (tier_name or "").strip().lower()
        env_tier_model_id = self._env_tier_model_id(provider, tier_name)
        if env_tier_model_id:
            return env_tier_model_id
        env_provider_model_id = self._provider_env_model_id(provider)
        if env_provider_model_id:
            return env_provider_model_id
        return tier.model_id

    def _env_tier_model_id(self, provider: str, tier_name: str) -> str | None:
        provider = (provider or "").strip().upper()
        tier_name = (tier_name or "").strip().upper()
        if not provider or not tier_name:
            return None
        key = f"SWARMEE_{provider}_{tier_name}_MODEL_ID"
        val = os.getenv(key)
        return val.strip() if isinstance(val, str) and val.strip() else None

    def _provider_env_model_id(self, provider: str) -> str | None:
        provider = (provider or "").strip().lower()
        if provider == "bedrock":
            val = os.getenv("STRANDS_MODEL_ID")
            return val.strip() if isinstance(val, str) and val.strip() else None
        if provider == "openai":
            val = os.getenv("SWARMEE_OPENAI_MODEL_ID")
            return val.strip() if isinstance(val, str) and val.strip() else None
        if provider == "ollama":
            val = os.getenv("SWARMEE_OLLAMA_MODEL_ID") or os.getenv("OLLAMA_MODEL")
            return val.strip() if isinstance(val, str) and val.strip() else None
        return None

    def _merge_tiers(self, base: ModelTier, override: ModelTier, *, default_provider: str) -> ModelTier:
        provider = override.provider.strip().lower() if override.provider and override.provider.strip() else base.provider
        if not provider:
            provider = default_provider

        model_id = base.model_id
        if override.model_id is not None and str(override.model_id).strip():
            model_id = str(override.model_id).strip()

        display_name = base.display_name
        if override.display_name is not None and str(override.display_name).strip():
            display_name = str(override.display_name).strip()

        description = base.description
        if override.description is not None and str(override.description).strip():
            description = str(override.description).strip()

        client_args = base.client_args
        if override.client_args is not None:
            if isinstance(client_args, dict) and isinstance(override.client_args, dict):
                merged = dict(client_args)
                merged.update(override.client_args)
                client_args = merged
            elif isinstance(override.client_args, dict):
                client_args = override.client_args

        params = base.params
        if override.params is not None:
            if isinstance(params, dict) and isinstance(override.params, dict):
                merged = dict(params)
                merged.update(override.params)
                params = merged
            elif isinstance(override.params, dict):
                params = override.params

        extra: dict[str, Any] = dict(base.extra or {})
        extra.update(dict(override.extra or {}))

        return ModelTier(
            provider=provider,
            model_id=model_id,
            display_name=display_name,
            description=description,
            client_args=client_args,
            params=params,
            extra=extra,
        )

    def _deep_merge_dict(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = self._deep_merge_dict(out[k], v)  # type: ignore[arg-type]
            else:
                out[k] = v
        return out

    def _is_tier_available(self, tier: ModelTier) -> tuple[bool, str | None]:
        provider = (tier.provider or "").strip().lower()
        if not provider:
            return False, "provider missing"
        if provider == "openai":
            if not (tier.client_args and tier.client_args.get("api_key")) and not os.getenv("OPENAI_API_KEY"):
                return False, "OPENAI_API_KEY missing"
        return True, None
