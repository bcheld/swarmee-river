"""Pure Agent Studio domain helpers for the TUI."""

from __future__ import annotations

import json as _json
import os
import re
import uuid
from typing import Any

from swarmee_river.profiles.models import ORCHESTRATOR_AGENT_ID
from swarmee_river.profiles.models import normalize_agent_definition as normalize_profile_agent_definition
from swarmee_river.profiles.models import normalize_agent_definitions as normalize_profile_agent_definitions
from swarmee_river.settings import load_settings

_AGENT_TOOL_CONSENT_VALUES = {"ask", "allow", "deny"}


def _sanitize_profile_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


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


def normalize_agent_studio_view_mode(mode: str | None) -> str:
    """Normalize Agent Studio sub-view mode."""
    normalized = str(mode or "").strip().lower()
    if normalized in {"overview", "builder"}:
        return normalized
    return "overview"


def normalize_agent_definition(raw: Any) -> dict[str, Any] | None:
    return normalize_profile_agent_definition(raw)


def normalize_agent_definitions(raw_agents: Any) -> list[dict[str, Any]]:
    return normalize_profile_agent_definitions(raw_agents)


def build_activated_agent_sidebar_items(
    agents: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_agent_definitions(agents or [])
    activated = [
        agent
        for agent in normalized
        if str(agent.get("id", "")).strip().lower() != ORCHESTRATOR_AGENT_ID and bool(agent.get("activated"))
    ]
    if not activated:
        return [
            {
                "id": "activated_agents_none",
                "title": "No Activated Agents",
                "subtitle": "Use Builder to activate agents for this session.",
                "state": "default",
            }
        ]

    items: list[dict[str, Any]] = []
    for item in activated:
        provider = str(item.get("provider", "")).strip()
        tier = str(item.get("tier", "")).strip()
        model_label = "/".join(token for token in (provider, tier) if token) or "(inherit session model)"
        summary = str(item.get("summary", "")).strip()
        subtitle = summary or model_label
        if summary and model_label:
            subtitle = f"{summary} | {model_label}"
        items.append(
            {
                "id": str(item.get("id", "")).strip(),
                "title": str(item.get("name", "")).strip() or "Unnamed Agent",
                "subtitle": subtitle,
                "state": "active",
                "agent": dict(item),
            }
        )
    return items


def build_activated_agent_table_rows(
    items: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str, str, str, str]]:
    """Build DataTable rows for activated agents.

    Returns tuples of: (id, name, summary, model, activated).
    """
    rows: list[tuple[str, str, str, str, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            continue
        if item_id == "activated_agents_none":
            rows.append((item_id, "No Activated Agents", "", "", "no"))
            continue
        agent = normalize_agent_definition(item.get("agent"))
        if agent is None:
            continue
        provider = str(agent.get("provider", "")).strip()
        tier = str(agent.get("tier", "")).strip()
        model_label = "/".join(token for token in (provider, tier) if token) or "(inherit)"
        rows.append(
            (
                item_id,
                str(agent.get("name", "")).strip() or "Unnamed Agent",
                str(agent.get("summary", "")).strip(),
                model_label,
                "yes" if bool(agent.get("activated")) else "no",
            )
        )
    return rows


def render_activated_agent_detail_text(item: dict[str, Any] | None) -> str:
    if not isinstance(item, dict):
        return "(no activated agent selected)"

    item_id = str(item.get("id", "")).strip()
    if item_id == "activated_agents_none":
        return (
            "Activated Agents\n\n"
            "No agents are currently activated for delegation.\n"
            "Open Builder and enable `Activated` on at least one agent."
        )

    agent = normalize_agent_definition(item.get("agent"))
    if agent is None:
        return "(invalid agent record)"

    provider = str(agent.get("provider", "")).strip()
    tier = str(agent.get("tier", "")).strip()
    model_label = "/".join(token for token in (provider, tier) if token) or "(inherit)"
    tools = _normalized_tool_name_list(agent.get("tool_names"))
    sops = _normalized_tool_name_list(agent.get("sop_names"))
    prompt_refs = _normalized_tool_name_list(agent.get("prompt_refs"))
    kb_id = str(agent.get("knowledge_base_id", "")).strip() or "(none)"
    summary = str(agent.get("summary", "")).strip() or "(none)"
    prompt = str(agent.get("prompt", "")).strip() or "(none)"
    return (
        "Activated Agent\n\n"
        f"ID: {agent['id']}\n"
        f"Name: {agent['name']}\n"
        f"Summary: {summary}\n"
        f"Model: {model_label}\n"
        f"KB: {kb_id}\n"
        f"Tools: {', '.join(tools) if tools else '(inherit/default)'}\n"
        f"SOPs: {', '.join(sops) if sops else '(none)'}\n\n"
        f"Prompt refs: {', '.join(prompt_refs) if prompt_refs else '(none)'}\n\n"
        "Prompt:\n"
        f"{prompt}"
    )


def build_builder_agent_table_rows(items: list[dict[str, Any]] | None = None) -> list[tuple[str, str, str, str, str]]:
    """Build DataTable rows for builder roster entries.

    Returns tuples of: (id, name, summary, model, state).
    """
    rows: list[tuple[str, str, str, str, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            continue
        agent = normalize_agent_definition(item.get("agent"))
        if agent is None:
            continue
        provider = str(agent.get("provider", "")).strip()
        tier = str(agent.get("tier", "")).strip()
        model_label = "/".join(token for token in (provider, tier) if token) or "inherit"
        rows.append(
            (
                item_id,
                str(agent.get("name", "")).strip() or "Unnamed Agent",
                str(agent.get("summary", "")).strip(),
                model_label,
                (
                    "base"
                    if str(agent.get("id", "")).strip().lower() == ORCHESTRATOR_AGENT_ID
                    else ("active" if bool(agent.get("activated")) else "default")
                ),
            )
        )
    return rows


def build_swarm_agent_specs(
    agents: list[dict[str, Any]] | None = None,
    *,
    prompt_assets_by_id: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from swarmee_river.prompt_assets import resolve_agent_prompt_text

    normalized = normalize_agent_definitions(agents or [])
    specs: list[dict[str, Any]] = []
    for agent in normalized:
        if str(agent.get("id", "")).strip().lower() == ORCHESTRATOR_AGENT_ID:
            continue
        if not bool(agent.get("activated")):
            continue
        assets = prompt_assets_by_id if isinstance(prompt_assets_by_id, dict) else {}
        prompt = resolve_agent_prompt_text(agent, assets).strip()
        spec: dict[str, Any] = {
            "name": str(agent.get("name", "")).strip() or str(agent.get("id", "")).strip() or "agent",
            "system_prompt": prompt or "You are a helpful specialist agent.",
        }
        tool_names = _normalized_tool_name_list(agent.get("tool_names"))
        if tool_names:
            spec["tools"] = tool_names
        provider = str(agent.get("provider", "")).strip().lower()
        tier = str(agent.get("tier", "")).strip().lower()
        if provider:
            spec["model_provider"] = provider
        if tier:
            spec["model_settings"] = {"tier": tier}
        specs.append(spec)
    return specs


def build_activated_agents_run_prompt(
    agents: list[dict[str, Any]] | None = None,
    *,
    task: str | None = None,
    prompt_assets_by_id: dict[str, Any] | None = None,
) -> str:
    specs = build_swarm_agent_specs(agents, prompt_assets_by_id=prompt_assets_by_id)
    if not specs:
        return ""
    spec_json = _json.dumps(specs, ensure_ascii=False, indent=2, sort_keys=True)
    task_text = str(task or "Execute the current user task collaboratively.").strip()
    return (
        "Run activated agents with a single `swarm` tool call.\n"
        "Use the `agents` array below exactly as provided.\n\n"
        f"task: {task_text}\n\n"
        "agents:\n"
        "```json\n"
        f"{spec_json}\n"
        "```"
    )


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
        f"Team Preset\n\nID: {preset['id']}\nName: {preset['name']}\nDescription: {description}\n\nSpec:\n{spec_json}"
    )


__all__ = [
    "_normalized_tool_name_list",
    "_policy_tier_profile",
    "build_activated_agent_sidebar_items",
    "build_activated_agent_table_rows",
    "build_activated_agents_run_prompt",
    "build_agent_policy_lens",
    "build_agent_team_sidebar_items",
    "build_agent_tools_safety_sidebar_items",
    "build_swarm_agent_specs",
    "build_builder_agent_table_rows",
    "build_team_preset_run_prompt",
    "normalize_agent_definition",
    "normalize_agent_definitions",
    "normalize_agent_studio_view_mode",
    "normalize_session_safety_overrides",
    "normalize_team_preset",
    "normalize_team_presets",
    "render_activated_agent_detail_text",
    "render_agent_team_detail_text",
    "render_agent_tools_safety_detail_text",
]
