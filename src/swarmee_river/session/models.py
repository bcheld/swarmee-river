from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from strands.models import Model

from swarmee_river.settings import ModelTier, SwarmeeSettings
from swarmee_river.utils import model_utils


_TIER_ORDER: list[str] = ["fast", "balanced", "deep", "long"]


@dataclass
class TierStatus:
    name: str
    provider: str
    model_id: str | None
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
        self._fallback_config: dict[str, Any] | None = None
        self.current_tier: str = settings.models.default_tier.strip().lower() or "balanced"
        self.auto_escalation_enabled: bool = settings.models.auto_escalation.enabled
        self.max_escalations_per_task: int = max(0, settings.models.auto_escalation.max_escalations_per_task)

    def set_fallback_config(self, config: dict[str, Any] | None) -> None:
        self._fallback_config = dict(config) if isinstance(config, dict) else None

    def list_tiers(self) -> list[TierStatus]:
        out: list[TierStatus] = []
        for name in _TIER_ORDER:
            tier = self._settings.models.tiers.get(name) or ModelTier(provider=self._fallback_provider)

            available, reason = self._is_tier_available(tier)
            out.append(
                TierStatus(
                    name=name,
                    provider=tier.provider,
                    model_id=tier.model_id,
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
        tier = self._settings.models.tiers.get(tier_name) or ModelTier(provider=self._fallback_provider)

        provider = (tier.provider or "").strip().lower()
        if provider not in {"bedrock", "openai", "ollama"}:
            raise ValueError(f"Unsupported provider {provider!r} for tier {tier_name!r}")

        base = model_utils.default_model_config(provider)
        config = dict(base)

        if self._fallback_config is not None and provider == self._fallback_provider:
            config.update(self._fallback_config)

        # Apply tier overrides.
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
            config[k] = v

        model_path = model_utils.load_path(provider)
        return model_utils.load_model(model_path, config)

    def _is_tier_available(self, tier: ModelTier) -> tuple[bool, str | None]:
        provider = (tier.provider or "").strip().lower()
        if not provider:
            return False, "provider missing"
        if provider == "openai":
            if not (tier.client_args and tier.client_args.get("api_key")) and not os.getenv("OPENAI_API_KEY"):
                return False, "OPENAI_API_KEY missing"
        return True, None
