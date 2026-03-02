"""Pure model selection helpers for the TUI."""

from __future__ import annotations

import os
from typing import Any

_MODEL_AUTO_VALUE = "__auto__"
_MODEL_LOADING_VALUE = "__loading__"


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
    return f"Model: {selected_provider}/{tier}{suffix}"


def resolve_model_fallback_notice(*, provider_override: str | None = None) -> str | None:
    """Return the provider fallback notice string, if any, for one-time display."""
    try:
        from swarmee_river.settings import load_settings
        from swarmee_river.utils.provider_utils import resolve_model_provider
    except Exception:
        return None
    try:
        settings = load_settings()
    except Exception:
        return None
    _, notice = resolve_model_provider(
        cli_provider=None,
        env_provider=provider_override if provider_override is not None else os.getenv("SWARMEE_MODEL_PROVIDER"),
        settings_provider=settings.models.provider,
    )
    return notice or None


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

    available_values = {value for _label, value in options}

    # Keep the selected tier stable while daemon confirmation catches up:
    # if a pending/override value is for this provider but not yet in the
    # daemon's available list, add an ephemeral option so the dropdown does
    # not bounce back to the previous tier.
    pending = (pending_value or "").strip().lower()
    if pending and "|" in pending and pending not in available_values:
        pending_provider, pending_tier = pending.split("|", 1)
        pending_provider = pending_provider.strip().lower()
        pending_tier = pending_tier.strip().lower()
        if pending_provider == provider_name and pending_tier:
            options.insert(0, (f"{pending_provider}/{pending_tier} (pending)", pending))
            available_values.add(pending)

    override_provider_name = (override_provider or "").strip().lower()
    override_tier_name = (override_tier or "").strip().lower()
    override_value = (
        f"{override_provider_name}|{override_tier_name}" if override_provider_name and override_tier_name else ""
    )
    if override_value and override_value not in available_values and override_provider_name == provider_name:
        options.insert(0, (f"{override_provider_name}/{override_tier_name} (selected)", override_value))
        available_values.add(override_value)

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


__all__ = [
    "_MODEL_AUTO_VALUE",
    "_MODEL_LOADING_VALUE",
    "choose_daemon_model_select_value",
    "choose_model_summary_parts",
    "daemon_model_select_options",
    "model_select_options",
    "parse_model_select_value",
    "resolve_model_config_summary",
    "resolve_model_fallback_notice",
]
