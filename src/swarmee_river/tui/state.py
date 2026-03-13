"""State containers for the TUI app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DaemonState:
    ready: bool = False
    query_active: bool = False
    session_id: str | None = None
    available_restore_session_id: str | None = None
    available_restore_turn_count: int = 0
    last_restored_turn_count: int = 0
    tiers: list[dict[str, Any]] = field(default_factory=list)
    provider: str | None = None
    tier: str | None = None
    model_id: str | None = None
    current_model: str | None = None
    model_provider_override: str | None = None
    model_tier_override: str | None = None
    model_select_syncing: bool = False
    pending_model_select_value: str | None = None
    run_tool_count: int = 0
    run_start_time: float | None = None
    status_timer: Any = None
    run_active_tier_warning_emitted: bool = False
    turn_output_chunks: list[str] = field(default_factory=list)
    last_usage: dict[str, Any] | None = None
    last_cost_usd: float | None = None
    last_prompt_tokens_est: int | None = None
    last_budget_tokens: int | None = None
    last_provider_input_tokens: int | None = None
    last_provider_cached_input_tokens: int | None = None
    last_provider_output_tokens: int | None = None
    proc: Any = None
    runner_thread: Any = None
    is_shutting_down: bool = False


@dataclass
class PlanState:
    text: str = ""
    pending_prompt: str | None = None
    pending_record: dict[str, Any] | None = None
    plan_run_id: str | None = None
    current_steps_total: int = 0
    current_summary: str = ""
    current_steps: list[str] = field(default_factory=list)
    current_step_statuses: list[str] = field(default_factory=list)
    current_active_step: int | None = None
    updates_seen: bool = False
    step_counter: int = 0
    completion_announced: bool = False
    received_structured_plan: bool = False
    plan_json: dict | None = None
    pre_planning_split_ratio: int | None = None


@dataclass
class ArtifactsState:
    session_id: str | None = None
    recent_paths: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    selected_item_id: str | None = None


@dataclass
class SessionState:
    issue_lines: list[str] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    selected_issue_id: str | None = None
    view_mode: str = "timeline"
    timeline_index: dict[str, Any] | None = None
    timeline_events: list[dict[str, Any]] = field(default_factory=list)
    timeline_selected_event_id: str | None = None
    timeline_refresh_timer: Any = None
    timeline_refresh_inflight: bool = False
    timeline_refresh_pending: bool = False
    issues_repeat_line: str | None = None
    issues_repeat_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    bedrock_setup_guidance_shown: bool = False


@dataclass
class ToolingState:
    view_mode: str = "tools"  # "tools", "prompts", "sops", "kbs"
    tool_catalog: list[dict[str, Any]] = field(default_factory=list)
    tool_selected_id: str | None = None
    prompt_assets: list[dict[str, Any]] = field(default_factory=list)
    prompt_selected_id: str | None = None
    sop_catalog: list[dict[str, Any]] = field(default_factory=list)
    sop_selected_id: str | None = None
    kb_entries: list[dict[str, Any]] = field(default_factory=list)
    kb_selected_id: str | None = None


@dataclass
class AgentStudioState:
    view_mode: str = "overview"
    saved_bundles: list[dict[str, Any]] = field(default_factory=list)
    effective_profile: Any = None
    draft_dirty: bool = False
    form_syncing: bool = False
    agents: list[dict[str, Any]] = field(default_factory=list)
    activated_items: list[dict[str, Any]] = field(default_factory=list)
    activated_selected_item_id: str | None = None
    builder_items: list[dict[str, Any]] = field(default_factory=list)
    builder_selected_item_id: str | None = None
    builder_form_syncing: bool = False
    auto_delegate_assistive: bool = True
    tool_catalog: list[str] = field(default_factory=list)
    tools_items: list[dict[str, Any]] = field(default_factory=list)
    team_presets: list[dict[str, Any]] = field(default_factory=list)
    team_items: list[dict[str, Any]] = field(default_factory=list)
    tools_selected_item_id: str | None = None
    team_selected_item_id: str | None = None
    session_safety_overrides: dict[str, Any] = field(default_factory=dict)
    tools_policy_lens: dict[str, Any] = field(default_factory=dict)
    tools_form_syncing: bool = False
    team_form_syncing: bool = False


@dataclass
class BundlesState:
    catalog: list[dict[str, Any]] = field(default_factory=list)
    selected_bundle_id: str | None = None
    form_syncing: bool = False


@dataclass
class AppState:
    daemon: DaemonState = field(default_factory=DaemonState)
    plan: PlanState = field(default_factory=PlanState)
    artifacts: ArtifactsState = field(default_factory=ArtifactsState)
    session: SessionState = field(default_factory=SessionState)
    agent_studio: AgentStudioState = field(default_factory=AgentStudioState)
    bundles: BundlesState = field(default_factory=BundlesState)
    tooling: ToolingState = field(default_factory=ToolingState)
    engage_view_mode: str = "plan"
    tooling_view_mode: str = "tools"
    settings_view_mode: str = "general"
    settings_model_select_syncing: bool = False


__all__ = [
    "AgentStudioState",
    "AppState",
    "ArtifactsState",
    "BundlesState",
    "DaemonState",
    "PlanState",
    "SessionState",
    "ToolingState",
]
