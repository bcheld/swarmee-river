from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from strands.models import Model

from swarmee_river.settings import (
    ModelContextBehavior,
    ModelReasoningConfig,
    ModelTier,
    ModelToolingConfig,
    ProviderModels,
    SwarmeeSettings,
)
from swarmee_river.utils import model_utils
from swarmee_river.utils.provider_utils import has_aws_credentials, has_github_copilot_token, normalize_provider_name


@dataclass
class TierStatus:
    name: str
    provider: str
    model_id: str | None
    display_name: str | None
    description: str | None
    transport: str | None
    reasoning_effort: str | None
    tooling_mode: str | None
    tooling_discovery: str | None
    context_strategy: str | None
    context_compaction: str | None
    context_max_prompt_tokens: int | None
    reasoning_mode: str | None
    supports_guardrails: bool | None
    supports_cache_tools: bool | None
    supports_forced_tool_with_reasoning: bool | None
    available: bool
    reason: str | None = None


class SessionModelManager:
    """
    Manage model tier selection within a single CLI session.

    Implementation strategy: swap `agent.model` in-place so conversation history, state, hooks,
    and tool registry remain intact.
    """

    def __init__(self, settings: SwarmeeSettings, *, fallback_provider: str | None = None) -> None:
        self._settings = settings
        chosen_provider = normalize_provider_name(fallback_provider or settings.models.provider or "bedrock")
        self._fallback_provider = chosen_provider
        self._default_provider = chosen_provider
        self.current_provider: str = chosen_provider
        self._fallback_config: dict[str, Any] | None = None
        self.current_tier: str = settings.models.default_tier.strip().lower() or "balanced"
        self.auto_escalation_enabled: bool = settings.models.auto_escalation.enabled
        self.max_escalations_per_task: int = max(0, settings.models.auto_escalation.max_escalations_per_task)
        self._auto_escalation_order: list[str] = list(settings.models.auto_escalation.order)

    def _provider_names(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for candidate in (
            self.current_provider,
            self._default_provider,
            *(str(name).strip().lower() for name in self._settings.models.providers.keys() if str(name).strip()),
        ):
            provider = normalize_provider_name(candidate)
            if not provider or provider in seen:
                continue
            seen.add(provider)
            ordered.append(provider)
        return ordered

    def _available_tiers_for_provider(self, provider_name: str) -> list[str]:
        provider_name = normalize_provider_name(provider_name)
        tier_names: set[str] = set()
        provider_cfg = self._settings.models.providers.get(provider_name)
        if isinstance(provider_cfg, ProviderModels):
            for tier_name in provider_cfg.tiers.keys():
                normalized = str(tier_name or "").strip().lower()
                if normalized:
                    tier_names.add(normalized)
        for tier_name in self._settings.models.tiers.keys():
            normalized = str(tier_name or "").strip().lower()
            if normalized:
                tier_names.add(normalized)
        if not tier_names and self.current_tier:
            tier_names.add(self.current_tier.strip().lower())
        ordered: list[str] = []
        seen: set[str] = set()
        for tier_name in self._auto_escalation_order:
            normalized = str(tier_name or "").strip().lower()
            if not normalized or normalized not in tier_names or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        for tier_name in sorted(tier_names):
            if tier_name in seen:
                continue
            ordered.append(tier_name)
        return ordered

    def set_fallback_config(self, config: dict[str, Any] | None) -> None:
        self._fallback_config = dict(config) if isinstance(config, dict) else None

    def list_tiers(self) -> list[TierStatus]:
        out: list[TierStatus] = []
        for provider_name in self._provider_names():
            for name in self._available_tiers_for_provider(provider_name):
                tier = self._resolve_tier(name, provider_name=provider_name)
                effective_model_id = self._effective_model_id(tier, tier_name=name)
                available, reason = self._is_tier_available(tier)
                capabilities = (
                    model_utils.bedrock_model_capabilities(effective_model_id)
                    if tier.provider == "bedrock"
                    else None
                )
                reasoning_effort = tier.reasoning.effort if tier.reasoning is not None else None
                reasoning_mode = capabilities.reasoning_mode if capabilities is not None else None
                if tier.provider == "openai" and not model_utils.openai_model_supports_responses_reasoning(
                    effective_model_id
                ):
                    reasoning_effort = None
                    reasoning_mode = "none"
                out.append(
                    TierStatus(
                        name=name,
                        provider=tier.provider,
                        model_id=effective_model_id,
                        display_name=tier.display_name,
                        description=tier.description,
                        transport=tier.transport,
                        reasoning_effort=reasoning_effort,
                        tooling_mode=tier.tooling.mode if tier.tooling is not None else None,
                        tooling_discovery=tier.tooling.discovery if tier.tooling is not None else None,
                        context_strategy=tier.context.strategy if tier.context is not None else None,
                        context_compaction=tier.context.compaction if tier.context is not None else None,
                        context_max_prompt_tokens=(
                            int(tier.context.max_prompt_tokens)
                            if tier.context is not None and tier.context.max_prompt_tokens is not None
                            else None
                        ),
                        reasoning_mode=reasoning_mode,
                        supports_guardrails=capabilities.supports_guardrails if capabilities is not None else None,
                        supports_cache_tools=capabilities.supports_cache_tools if capabilities is not None else None,
                        supports_forced_tool_with_reasoning=(
                            capabilities.supports_forced_tool_with_reasoning if capabilities is not None else None
                        ),
                        available=available,
                        reason=reason,
                    )
                )
        return out

    def set_auto_escalation(self, enabled: bool) -> None:
        self.auto_escalation_enabled = enabled

    def set_tier(self, agent: Any, tier_name: str) -> None:
        self.set_selection(agent, provider_name=self.current_provider, tier_name=tier_name)

    def set_selection(self, agent: Any, *, provider_name: str | None, tier_name: str) -> None:
        provider = normalize_provider_name(provider_name or self.current_provider or self._default_provider)
        tier_name = (tier_name or "").strip().lower()
        available_tiers = self._available_tiers_for_provider(provider)
        if not tier_name or tier_name not in set(available_tiers):
            raise ValueError(f"Unknown tier: {tier_name}. Available tiers: {', '.join(available_tiers)}")

        model = self._build_model_for_tier(tier_name, provider_name=provider)
        agent.model = model
        self.current_provider = provider
        self.current_tier = tier_name

    def build_model(self, tier_name: str | None = None, *, provider_name: str | None = None) -> Model:
        """Build a model for the given tier (or current tier) without mutating an agent."""
        resolved = (tier_name or self.current_tier or "balanced").strip().lower()
        provider = normalize_provider_name(provider_name or self.current_provider or self._default_provider)
        return self._build_model_for_tier(resolved, provider_name=provider)

    def maybe_escalate(self, agent: Any, *, attempted: set[str]) -> bool:
        """Escalate to the next tier if possible. Returns True if a switch occurred."""
        if not self.auto_escalation_enabled:
            return False
        ordered = [str(item or "").strip().lower() for item in self._auto_escalation_order if str(item or "").strip()]
        if not ordered:
            return False

        try:
            idx = ordered.index(self.current_tier)
        except ValueError:
            return False

        available_tiers = set(self._available_tiers_for_provider(self.current_provider))
        for next_tier in ordered[idx + 1 :]:
            if next_tier not in available_tiers:
                continue
            if next_tier in attempted:
                continue
            try:
                self.set_selection(agent, provider_name=self.current_provider, tier_name=next_tier)
                return True
            except Exception:
                attempted.add(next_tier)
                continue
        return False

    def _build_model_for_tier(self, tier_name: str, *, provider_name: str | None = None) -> Model:
        tier = self._resolve_tier(tier_name, provider_name=provider_name)

        provider = normalize_provider_name(tier.provider)
        if provider not in {"bedrock", "openai", "ollama", "github_copilot"}:
            raise ValueError(f"Unsupported provider {provider!r} for tier {tier_name!r}")

        base = model_utils.default_model_config(provider, self._settings)
        config = dict(base)

        if self._fallback_config is not None and provider == self._fallback_provider:
            config.update(self._fallback_config)

        # Model selection is settings-driven. Per-tier model_id is the source of truth.
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
            config_value = config.get(k)
            if isinstance(v, dict) and isinstance(config_value, dict):
                config[k] = self._deep_merge_dict(config_value, v)
            else:
                config[k] = v

        if provider == "bedrock":
            model_utils.sanitize_bedrock_converse_config(config, tier=tier, settings=self._settings)
        if provider == "openai":
            model_utils.sanitize_openai_responses_config(config, tier=tier, settings=self._settings)
        model_path = model_utils.load_path(provider)
        return model_utils.load_model(model_path, config)

    def current_context_behavior(self, tier_name: str | None = None) -> ModelContextBehavior:
        tier = self._resolve_tier(
            (tier_name or self.current_tier or "balanced").strip().lower(),
            provider_name=self.current_provider,
        )
        return tier.context or ModelContextBehavior()

    def resolve_effective_context_budget(
        self,
        *,
        tier_name: str | None = None,
        provider_name: str | None = None,
        override_tokens: int | None = None,
    ) -> int:
        provider = normalize_provider_name(provider_name or self.current_provider or self._default_provider)
        resolved_tier = (tier_name or self.current_tier or "balanced").strip().lower()
        tier = self._resolve_tier(resolved_tier, provider_name=provider)
        requested = override_tokens
        if requested is None:
            requested = self._settings.context.max_prompt_tokens
        if requested is None and tier.context is not None:
            requested = tier.context.max_prompt_tokens
        cap = self._provider_context_cap(provider=provider, model_id=tier.model_id)
        if requested is None:
            requested = cap if cap is not None else 20000
        resolved = max(1, int(requested))
        if cap is not None:
            resolved = min(resolved, cap)
        return resolved

    def current_reasoning_config(self, tier_name: str | None = None) -> ModelReasoningConfig:
        tier = self._resolve_tier(
            (tier_name or self.current_tier or "balanced").strip().lower(),
            provider_name=self.current_provider,
        )
        return tier.reasoning or ModelReasoningConfig()

    def current_tooling_config(self, tier_name: str | None = None) -> ModelToolingConfig:
        tier = self._resolve_tier(
            (tier_name or self.current_tier or "balanced").strip().lower(),
            provider_name=self.current_provider,
        )
        return tier.tooling or ModelToolingConfig()

    def _resolve_tier(self, tier_name: str, provider_name: str | None = None) -> ModelTier:
        tier_name = (tier_name or "").strip().lower()
        global_override = self._settings.models.tiers.get(tier_name)

        provider = normalize_provider_name(provider_name or self.current_provider or self._default_provider)
        if (
            not provider_name
            and global_override
            and global_override.provider
            and global_override.provider.strip()
        ):
            provider = normalize_provider_name(global_override.provider)

        provider_tier = self._provider_tier(provider, tier_name)
        resolved = provider_tier
        if global_override:
            resolved = self._merge_tiers(resolved, global_override, default_provider=provider)
            if provider_name:
                resolved_dict = resolved.to_dict()
                resolved_dict["provider"] = provider
                resolved = ModelTier.from_dict(resolved_dict, default_provider=provider)

        # Ensure provider is always set.
        if not resolved.provider:
            resolved = ModelTier.from_dict(resolved.to_dict(), default_provider=provider)
        return resolved

    def _provider_tier(self, provider: str, tier_name: str) -> ModelTier:
        provider = normalize_provider_name(provider)
        tier_name = (tier_name or "").strip().lower()

        pm = self._settings.models.providers.get(provider)
        if isinstance(pm, ProviderModels):
            tier = pm.tiers.get(tier_name)
            if tier is not None:
                return tier
        return ModelTier(provider=provider)

    def _effective_model_id(self, tier: ModelTier, *, tier_name: str) -> str | None:
        _ = tier_name
        return tier.model_id

    @staticmethod
    def _provider_context_cap(*, provider: str, model_id: str | None) -> int | None:
        normalized_provider = normalize_provider_name(provider)
        normalized_model_id = str(model_id or "").strip().lower()
        if normalized_provider == "bedrock" and normalized_model_id.startswith("us.anthropic.claude-"):
            return 200000
        if normalized_provider == "openai" and normalized_model_id.startswith("gpt-5"):
            return 400000
        return None

    def _merge_tiers(self, base: ModelTier, override: ModelTier, *, default_provider: str) -> ModelTier:
        provider = normalize_provider_name(
            override.provider if override.provider and override.provider.strip() else base.provider
        )
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

        transport = base.transport
        if override.transport is not None and str(override.transport).strip():
            transport = str(override.transport).strip()

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
            transport=transport,
            reasoning=override.reasoning if override.reasoning is not None else base.reasoning,
            tooling=override.tooling if override.tooling is not None else base.tooling,
            context=override.context if override.context is not None else base.context,
            client_args=client_args,
            params=params,
            extra=extra,
        )

    def _deep_merge_dict(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = dict(base)
        for k, v in override.items():
            out_value = out.get(k)
            if isinstance(v, dict) and isinstance(out_value, dict):
                out[k] = self._deep_merge_dict(out_value, v)
            else:
                out[k] = v
        return out

    def _is_tier_available(self, tier: ModelTier) -> tuple[bool, str | None]:
        provider = normalize_provider_name(tier.provider)
        if not provider:
            return False, "provider missing"
        if provider == "bedrock":
            if not has_aws_credentials():
                return False, "AWS credentials missing/expired"
        if provider == "openai":
            compatibility = model_utils.probe_openai_responses_transport()
            if not compatibility.available:
                return False, compatibility.reason
            if not (tier.client_args and tier.client_args.get("api_key")) and not os.getenv("OPENAI_API_KEY"):
                return False, "OPENAI_API_KEY missing"
        if provider == "github_copilot":
            has_inline_key = isinstance(tier.client_args, dict) and bool(
                str(tier.client_args.get("api_key") or "").strip()
            )
            if not has_inline_key and not has_github_copilot_token():
                return False, "SWARMEE_GITHUB_COPILOT_API_KEY or GITHUB_TOKEN missing"
        return True, None
