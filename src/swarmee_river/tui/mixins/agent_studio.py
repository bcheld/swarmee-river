from __future__ import annotations

import contextlib
import json as _json
import uuid
from typing import Any

from swarmee_river.profiles import AgentProfile, delete_profile, list_profiles, save_profile
from swarmee_river.profiles.models import normalize_agent_definitions
from swarmee_river.tools import get_tools
from swarmee_river.tui.agent_studio import (
    _normalized_tool_name_list,
    build_activated_agent_sidebar_items,
    build_activated_agents_run_prompt,
    build_agent_policy_lens,
    build_agent_team_sidebar_items,
    build_agent_tools_safety_sidebar_items,
    build_team_preset_run_prompt,
    normalize_agent_definition,
    normalize_agent_studio_view_mode,
    normalize_session_safety_overrides,
    normalize_team_preset,
    normalize_team_presets,
    render_activated_agent_detail_text,
    render_agent_team_detail_text,
    render_agent_tools_safety_detail_text,
)
from swarmee_river.tui.model_select import choose_model_summary_parts
from swarmee_river.tui.transport import send_daemon_command as _transport_send_daemon_command

_AGENT_PROFILE_SELECT_NONE = "__agent_profile_none__"
_AGENT_TOOL_CONSENT_VALUES = {"ask", "allow", "deny"}


def _sanitize_profile_token(value: str) -> str:
    import re
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


def _default_profile_id() -> str:
    from datetime import datetime
    ts = datetime.now()
    return f"profile-{ts.strftime('%Y%m%d-%H%M%S')}"


def _default_profile_name() -> str:
    from datetime import datetime
    ts = datetime.now()
    return f"Profile {ts.strftime('%Y-%m-%d %H:%M')}"


def _default_team_preset_id() -> str:
    from datetime import datetime
    ts = datetime.now()
    return f"team-{ts.strftime('%Y%m%d-%H%M%S')}"


def _default_team_preset_name() -> str:
    from datetime import datetime
    ts = datetime.now()
    return f"Team Preset {ts.strftime('%Y-%m-%d %H:%M')}"


class AgentStudioMixin:
    def _selected_agent_profile_id(self) -> str | None:
        selector = self._agent_profile_select
        if selector is None:
            return None
        value = str(getattr(selector, "value", "")).strip()
        if not value or value == _AGENT_PROFILE_SELECT_NONE:
            return None
        return value

    def _lookup_saved_profile(self, profile_id: str | None) -> AgentProfile | None:
        target = str(profile_id or "").strip()
        if not target:
            return None
        for profile in self.state.agent_studio.saved_profiles:
            if profile.id == target:
                return profile
        return None

    def _set_agent_form_values(self, *, profile_id: str, profile_name: str) -> None:
        self.state.agent_studio.form_syncing = True
        try:
            if self._agent_profile_id_input is not None:
                self._agent_profile_id_input.value = profile_id
            if self._agent_profile_name_input is not None:
                self._agent_profile_name_input.value = profile_name
        finally:
            self.state.agent_studio.form_syncing = False

    def _agent_tools_item_by_id(self, item_id: str | None) -> dict[str, Any] | None:
        target = str(item_id or "").strip()
        if not target:
            return None
        for item in self.state.agent_studio.tools_items:
            if str(item.get("id", "")).strip() == target:
                return item
        return None

    def _set_agent_tools_status(self, message: str) -> None:
        widget = self._agent_tools_override_status
        if widget is None:
            return
        text = message.strip() if isinstance(message, str) else ""
        widget.update(text)

    def _set_agent_tools_override_form_values(self, overrides: dict[str, Any] | None) -> None:
        normalized = normalize_session_safety_overrides(overrides)
        consent = str(normalized.get("tool_consent", "")).strip().lower()
        allow = _normalized_tool_name_list(normalized.get("tool_allowlist"))
        block = _normalized_tool_name_list(normalized.get("tool_blocklist"))
        self.state.agent_studio.tools_form_syncing = True
        try:
            if self._agent_tools_override_consent_input is not None:
                self._agent_tools_override_consent_input.value = consent
            if self._agent_tools_override_allowlist_input is not None:
                self._agent_tools_override_allowlist_input.value = ", ".join(allow)
            if self._agent_tools_override_blocklist_input is not None:
                self._agent_tools_override_blocklist_input.value = ", ".join(block)
        finally:
            self.state.agent_studio.tools_form_syncing = False

    def _parse_agent_tools_csv_list(self, value: str) -> list[str]:
        if not value.strip():
            return []
        tokens = [token.strip() for token in value.split(",")]
        return _normalized_tool_name_list(tokens)

    def _agent_tools_form_update_payload(self) -> dict[str, Any] | None:
        raw_consent = str(getattr(self._agent_tools_override_consent_input, "value", "")).strip().lower()
        if raw_consent and raw_consent not in _AGENT_TOOL_CONSENT_VALUES:
            self._notify("tool_consent must be ask|allow|deny.", severity="warning")
            return None
        allow_csv = str(getattr(self._agent_tools_override_allowlist_input, "value", ""))
        block_csv = str(getattr(self._agent_tools_override_blocklist_input, "value", ""))
        allow_list = self._parse_agent_tools_csv_list(allow_csv)
        block_list = self._parse_agent_tools_csv_list(block_csv)
        return {
            "tool_consent": raw_consent or None,
            "tool_allowlist": allow_list or None,
            "tool_blocklist": block_list or None,
        }

    def _current_agent_policy_tier_name(self) -> str | None:
        return (
            str(self.state.daemon.tier or "").strip().lower()
            or str(self.state.daemon.model_tier_override or "").strip().lower()
            or None
        )

    def _refresh_agent_tools_policy_lens(self) -> None:
        self.state.agent_studio.tools_policy_lens = build_agent_policy_lens(
            tier_name=self._current_agent_policy_tier_name(),
            overrides=self.state.agent_studio.session_safety_overrides,
        )

    def _apply_agent_tools_safety_overrides(self, *, reset: bool = False) -> None:
        proc = self.state.daemon.proc
        if self.state.daemon.query_active:
            self._set_agent_tools_status("Cannot update overrides while a run is active.")
            self._notify("Cannot update overrides while a run is active.", severity="warning")
            return
        if not self.state.daemon.ready or proc is None or proc.poll() is not None:
            self._set_agent_tools_status("Daemon is not ready.")
            self._notify("Daemon is not ready.", severity="warning")
            return
        payload = (
            {"tool_consent": None, "tool_allowlist": None, "tool_blocklist": None}
            if reset
            else self._agent_tools_form_update_payload()
        )
        if payload is None:
            return
        command: dict[str, Any] = {"cmd": "set_safety_overrides"}
        command.update(payload)
        if not _transport_send_daemon_command(proc, command):
            self._set_agent_tools_status("Failed to send override update.")
            self._notify("Failed to send safety override update.", severity="warning")
            return
        if reset:
            self._set_agent_tools_status("Resetting overrides...")
        else:
            self._set_agent_tools_status("Applying overrides...")

    def _set_agent_tools_selection(self, item: dict[str, Any] | None) -> None:
        detail = self._agent_tools_detail
        if detail is None:
            return
        if item is None:
            self.state.agent_studio.tools_selected_item_id = None
            detail.set_preview("(no tools/safety items)")
            detail.set_actions([])
            return
        self.state.agent_studio.tools_selected_item_id = str(item.get("id", "")).strip() or None
        detail.set_preview(render_agent_tools_safety_detail_text(item, self.state.agent_studio.tools_policy_lens))
        detail.set_actions([])

    def _agent_team_item_by_id(self, item_id: str | None) -> dict[str, Any] | None:
        target = str(item_id or "").strip()
        if not target:
            return None
        for item in self.state.agent_studio.team_items:
            if str(item.get("id", "")).strip() == target:
                return item
        return None

    def _set_agent_team_status(self, message: str) -> None:
        widget = self._agent_team_status
        if widget is None:
            return
        text = message.strip() if isinstance(message, str) else ""
        widget.update(text)

    def _set_agent_team_form_values(self, preset: dict[str, Any] | None) -> None:
        normalized = normalize_team_preset(preset) if isinstance(preset, dict) else None
        preset_id = str(normalized.get("id", "")).strip() if normalized else ""
        name = str(normalized.get("name", "")).strip() if normalized else ""
        description = str(normalized.get("description", "")).strip() if normalized else ""
        spec_text = (
            _json.dumps(normalized.get("spec", {}), ensure_ascii=False, indent=2, sort_keys=True)
            if normalized
            else "{}"
        )
        self.state.agent_studio.team_form_syncing = True
        try:
            if self._agent_team_preset_id_input is not None:
                self._agent_team_preset_id_input.value = preset_id
            if self._agent_team_preset_name_input is not None:
                self._agent_team_preset_name_input.value = name
            if self._agent_team_preset_description_input is not None:
                self._agent_team_preset_description_input.value = description
            if self._agent_team_preset_spec_input is not None:
                self._agent_team_preset_spec_input.load_text(spec_text)
        finally:
            self.state.agent_studio.team_form_syncing = False

    def _agent_team_form_payload(self) -> dict[str, Any] | None:
        raw_name = str(getattr(self._agent_team_preset_name_input, "value", "")).strip()
        if not raw_name:
            self._notify("Preset name is required.", severity="warning")
            return None
        raw_id = str(getattr(self._agent_team_preset_id_input, "value", "")).strip()
        preset_id = _sanitize_profile_token(raw_id or raw_name)
        description = str(getattr(self._agent_team_preset_description_input, "value", "")).strip()
        spec_text = str(getattr(self._agent_team_preset_spec_input, "text", "")).strip() or "{}"
        try:
            spec = _json.loads(spec_text)
        except Exception:
            self._notify("Preset spec must be valid JSON.", severity="warning")
            return None
        if not isinstance(spec, dict):
            self._notify("Preset spec must be a JSON object.", severity="warning")
            return None
        normalized = normalize_team_preset(
            {"id": preset_id, "name": raw_name, "description": description, "spec": spec}
        )
        if normalized is None:
            self._notify("Preset is invalid.", severity="warning")
            return None
        return normalized

    def _selected_team_preset(self) -> dict[str, Any] | None:
        selected_item = self._agent_team_item_by_id(self.state.agent_studio.team_selected_item_id)
        if selected_item is None:
            return None
        return normalize_team_preset(selected_item.get("preset"))

    def _set_prompt_editor_text(self, text: str) -> bool:
        content = str(text or "")
        if not content.strip():
            return False
        with contextlib.suppress(Exception):
            # PromptTextArea is defined inside run_tui(); query by ID only.
            prompt_widget = self.query_one("#prompt")
            prompt_widget.clear()
            loader = getattr(prompt_widget, "load_text", None)
            if callable(loader):
                loader(content)
            else:
                prompt_widget.insert(content)
            prompt_widget.focus()
            return True
        return False

    def _new_agent_team_preset_draft(self) -> None:
        seed = normalize_team_preset(
            {
                "id": _default_team_preset_id(),
                "name": _default_team_preset_name(),
                "description": "",
                "spec": {},
            }
        )
        if seed is None:
            return
        self.state.agent_studio.team_selected_item_id = None
        self._set_agent_team_form_values(seed)
        self._set_agent_team_status("New team preset draft.")
        self._set_agent_draft_dirty(True, note="Team preset draft updated.")

    def _save_agent_team_preset_draft(self) -> None:
        payload = self._agent_team_form_payload()
        if payload is None:
            return
        selected_id = str(self.state.agent_studio.team_selected_item_id or "").strip()
        saved_id = str(payload.get("id", "")).strip()
        next_presets = [dict(item) for item in normalize_team_presets(self.state.agent_studio.team_presets)]
        if selected_id and selected_id != saved_id:
            next_presets = [item for item in next_presets if str(item.get("id", "")).strip() != selected_id]
        replaced = False
        for idx, existing in enumerate(next_presets):
            if str(existing.get("id", "")).strip() == saved_id:
                next_presets[idx] = payload
                replaced = True
                break
        if not replaced:
            next_presets.append(payload)
        self.state.agent_studio.team_presets = normalize_team_presets(next_presets)
        self.state.agent_studio.team_selected_item_id = saved_id
        self._render_agent_team_panel()
        self._set_agent_team_status(f"Saved preset '{payload['name']}' in draft.")
        self._set_agent_draft_dirty(True, note=f"Team preset '{payload['name']}' saved in draft.")

    def _delete_selected_agent_team_preset(self) -> None:
        selected = self._selected_team_preset()
        if selected is None:
            self._notify("Select a team preset first.", severity="warning")
            return
        selected_id = str(selected.get("id", "")).strip()
        next_presets = [
            item for item in self.state.agent_studio.team_presets if str(item.get("id", "")).strip() != selected_id
        ]
        self.state.agent_studio.team_presets = normalize_team_presets(next_presets)
        self.state.agent_studio.team_selected_item_id = None
        self._render_agent_team_panel()
        self._set_agent_team_status(f"Deleted preset '{selected.get('name', selected_id)}' from draft.")
        self._set_agent_draft_dirty(True, note=f"Team preset '{selected_id}' removed from draft.")

    def _insert_agent_team_preset_run_prompt(self, *, run_now: bool = False) -> None:
        preset = self._selected_team_preset()
        if preset is None:
            preset = self._agent_team_form_payload()
        if preset is None:
            return
        prompt = build_team_preset_run_prompt(preset)
        if not prompt:
            self._notify("Failed to build team preset prompt.", severity="warning")
            return
        if not self._set_prompt_editor_text(prompt):
            self._notify("Prompt editor is unavailable.", severity="warning")
            return
        self._set_agent_team_status(f"Inserted run prompt for '{preset['name']}'.")
        if run_now:
            self.action_submit_prompt()

    def _set_agent_team_selection(self, item: dict[str, Any] | None) -> None:
        detail = self._agent_team_detail
        if detail is None:
            return
        if item is None:
            self.state.agent_studio.team_selected_item_id = None
            detail.set_preview("(no team items)")
            detail.set_actions([])
            self._set_agent_team_form_values(None)
            return
        self.state.agent_studio.team_selected_item_id = str(item.get("id", "")).strip() or None
        detail.set_preview(render_agent_team_detail_text(item))
        detail.set_actions([])
        preset = normalize_team_preset(item.get("preset"))
        self._set_agent_team_form_values(preset)

    def _refresh_agent_tools_header(self) -> None:
        header = self._agent_tools_header
        if header is None:
            return
        effective = (
            self.state.agent_studio.tools_policy_lens.get("effective", {})
            if isinstance(self.state.agent_studio.tools_policy_lens.get("effective"), dict)
            else {}
        )
        overrides = (
            self.state.agent_studio.tools_policy_lens.get("session_overrides", {})
            if isinstance(self.state.agent_studio.tools_policy_lens.get("session_overrides"), dict)
            else {}
        )
        consent = str(effective.get("tool_consent", "ask")).strip().lower() or "ask"
        header.set_badges([f"consent {consent}", f"overrides {len(overrides)}"])

    def _refresh_agent_team_header(self) -> None:
        header = self._agent_team_header
        if header is None:
            return
        header.set_badges([f"presets {len(self.state.agent_studio.team_presets)}"])

    def _render_agent_tools_panel(self) -> None:
        self._refresh_agent_tools_policy_lens()
        self.state.agent_studio.tools_items = [
            dict(item) for item in build_agent_tools_safety_sidebar_items(self.state.agent_studio.tools_policy_lens)
        ]
        list_widget = self._agent_tools_list
        if list_widget is not None:
            selected_id = self.state.agent_studio.tools_selected_item_id
            if not selected_id and self.state.agent_studio.tools_items:
                selected_id = str(self.state.agent_studio.tools_items[0].get("id", "")).strip()
            list_widget.set_items(self.state.agent_studio.tools_items, selected_id=selected_id, emit=False)
            selected_id = list_widget.selected_id()
            selected_item = self._agent_tools_item_by_id(selected_id)
            if selected_item is None and self.state.agent_studio.tools_items:
                selected_item = self.state.agent_studio.tools_items[0]
                with contextlib.suppress(Exception):
                    list_widget.select_by_id(str(selected_item.get("id", "")), emit=False)
            self._set_agent_tools_selection(selected_item)
        else:
            self._set_agent_tools_selection(
                self.state.agent_studio.tools_items[0] if self.state.agent_studio.tools_items else None
            )
        self._refresh_agent_tools_header()
        self._set_agent_tools_override_form_values(self.state.agent_studio.session_safety_overrides)

    def _render_agent_team_panel(self) -> None:
        self.state.agent_studio.team_presets = normalize_team_presets(self.state.agent_studio.team_presets)
        self.state.agent_studio.team_items = [
            dict(item) for item in build_agent_team_sidebar_items(self.state.agent_studio.team_presets)
        ]
        list_widget = self._agent_team_list
        if list_widget is not None:
            selected_id = self.state.agent_studio.team_selected_item_id
            if not selected_id and self.state.agent_studio.team_items:
                selected_id = str(self.state.agent_studio.team_items[0].get("id", "")).strip()
            list_widget.set_items(self.state.agent_studio.team_items, selected_id=selected_id, emit=False)
            selected_id = list_widget.selected_id()
            selected_item = self._agent_team_item_by_id(selected_id)
            if selected_item is None and self.state.agent_studio.team_items:
                selected_item = self.state.agent_studio.team_items[0]
                with contextlib.suppress(Exception):
                    list_widget.select_by_id(str(selected_item.get("id", "")), emit=False)
            self._set_agent_team_selection(selected_item)
        else:
            self._set_agent_team_selection(
                self.state.agent_studio.team_items[0] if self.state.agent_studio.team_items else None
            )
        self._refresh_agent_team_header()

    def _set_agent_overview_status(self, message: str) -> None:
        widget = self._agent_overview_status
        if widget is None:
            return
        widget.update(message.strip() if isinstance(message, str) else "")

    def _set_agent_builder_status(self, message: str) -> None:
        widget = self._agent_builder_status
        if widget is None:
            return
        widget.update(message.strip() if isinstance(message, str) else "")

    def _refresh_agent_tool_catalog(self, tool_names: list[str] | None = None) -> None:
        incoming = [str(name).strip() for name in (tool_names or []) if str(name).strip()]
        if incoming:
            self.state.agent_studio.tool_catalog = sorted(set(incoming))
            return
        if self.state.agent_studio.tool_catalog:
            return
        with contextlib.suppress(Exception):
            self.state.agent_studio.tool_catalog = sorted(set(get_tools().keys()))

    def _agent_builder_item_by_id(self, item_id: str | None) -> dict[str, Any] | None:
        target = str(item_id or "").strip()
        if not target:
            return None
        for item in self.state.agent_studio.builder_items:
            if str(item.get("id", "")).strip() == target:
                return item
        return None

    def _set_agent_overview_selection(self, item: dict[str, Any] | None) -> None:
        detail = self._agent_overview_detail
        if detail is None:
            return
        if item is None:
            self.state.agent_studio.activated_selected_item_id = None
            detail.set_preview("(no activated agents)")
            detail.set_actions([])
            return
        self.state.agent_studio.activated_selected_item_id = str(item.get("id", "")).strip() or None
        detail.set_preview(render_activated_agent_detail_text(item))
        detail.set_actions([])

    def _set_agent_builder_form_values(self, agent_def: dict[str, Any] | None) -> None:
        normalized = normalize_agent_definition(agent_def) if isinstance(agent_def, dict) else None
        self.state.agent_studio.builder_form_syncing = True
        try:
            if self._agent_builder_agent_id_input is not None:
                self._agent_builder_agent_id_input.value = str(normalized.get("id", "")).strip() if normalized else ""
            if self._agent_builder_agent_name_input is not None:
                self._agent_builder_agent_name_input.value = (
                    str(normalized.get("name", "")).strip() if normalized else ""
                )
            if self._agent_builder_agent_summary_input is not None:
                self._agent_builder_agent_summary_input.value = (
                    str(normalized.get("summary", "")).strip() if normalized else ""
                )
            if self._agent_builder_agent_prompt_input is not None:
                self._agent_builder_agent_prompt_input.load_text(
                    str(normalized.get("prompt", "")).strip() if normalized else ""
                )
            if self._agent_builder_agent_provider_select is not None:
                provider = str(normalized.get("provider", "")).strip().lower() if normalized else ""
                with contextlib.suppress(Exception):
                    self._agent_builder_agent_provider_select.value = provider or "__inherit__"
            if self._agent_builder_agent_tier_select is not None:
                tier = str(normalized.get("tier", "")).strip().lower() if normalized else ""
                with contextlib.suppress(Exception):
                    self._agent_builder_agent_tier_select.value = tier or "__inherit__"
            if self._agent_builder_agent_tools_input is not None:
                tools = _normalized_tool_name_list(normalized.get("tool_names")) if normalized else []
                self._agent_builder_agent_tools_input.value = ", ".join(tools)
            if self._agent_builder_agent_sops_input is not None:
                sops = _normalized_tool_name_list(normalized.get("sop_names")) if normalized else []
                self._agent_builder_agent_sops_input.value = ", ".join(sops)
            if self._agent_builder_agent_kb_input is not None:
                self._agent_builder_agent_kb_input.value = (
                    str(normalized.get("knowledge_base_id", "")).strip() if normalized else ""
                )
            if self._agent_builder_agent_activated_checkbox is not None:
                self._agent_builder_agent_activated_checkbox.value = bool(normalized.get("activated")) if normalized else False
        finally:
            self.state.agent_studio.builder_form_syncing = False

    def _agent_builder_form_payload(self) -> dict[str, Any] | None:
        raw_name = str(getattr(self._agent_builder_agent_name_input, "value", "")).strip()
        if not raw_name:
            self._notify("Agent name is required.", severity="warning")
            return None
        raw_id = str(getattr(self._agent_builder_agent_id_input, "value", "")).strip()
        tools = self._parse_agent_tools_csv_list(str(getattr(self._agent_builder_agent_tools_input, "value", "")))
        sops = self._parse_agent_tools_csv_list(str(getattr(self._agent_builder_agent_sops_input, "value", "")))
        provider = str(getattr(self._agent_builder_agent_provider_select, "value", "")).strip().lower()
        tier = str(getattr(self._agent_builder_agent_tier_select, "value", "")).strip().lower()
        payload = normalize_agent_definition(
            {
                "id": raw_id or raw_name,
                "name": raw_name,
                "summary": str(getattr(self._agent_builder_agent_summary_input, "value", "")).strip(),
                "prompt": str(getattr(self._agent_builder_agent_prompt_input, "text", "")).strip(),
                "provider": None if provider == "__inherit__" else provider,
                "tier": None if tier == "__inherit__" else tier,
                "tool_names": tools,
                "sop_names": sops,
                "knowledge_base_id": str(getattr(self._agent_builder_agent_kb_input, "value", "")).strip() or None,
                "activated": bool(getattr(self._agent_builder_agent_activated_checkbox, "value", False)),
            }
        )
        if payload is None:
            self._notify("Agent draft is invalid.", severity="warning")
            return None
        return payload

    def _set_agent_builder_selection(self, item: dict[str, Any] | None) -> None:
        detail = self._agent_builder_detail
        if detail is None:
            return
        if item is None:
            self.state.agent_studio.builder_selected_item_id = None
            detail.set_preview("(no draft agents)")
            detail.set_actions([])
            self._set_agent_builder_form_values(None)
            return
        self.state.agent_studio.builder_selected_item_id = str(item.get("id", "")).strip() or None
        detail.set_preview(render_activated_agent_detail_text({"id": item.get("id"), "agent": item.get("agent")}))
        detail.set_actions([])
        self._set_agent_builder_form_values(item.get("agent") if isinstance(item.get("agent"), dict) else None)

    def _render_agent_overview_panel(self) -> None:
        items = build_activated_agent_sidebar_items(self.state.agent_studio.agents)
        self.state.agent_studio.activated_items = [dict(item) for item in items]
        list_widget = self._agent_overview_list
        selected_id = self.state.agent_studio.activated_selected_item_id
        if list_widget is not None:
            if not selected_id and items:
                selected_id = str(items[0].get("id", "")).strip()
            list_widget.set_items(items, selected_id=selected_id, emit=False)
            selected_overview_id = list_widget.selected_id()
            selected_item = next(
                (
                    item
                    for item in self.state.agent_studio.activated_items
                    if str(item.get("id", "")).strip() == str(selected_overview_id or "").strip()
                ),
                None,
            )
            self._set_agent_overview_selection(selected_item)
        else:
            self._set_agent_overview_selection(self.state.agent_studio.activated_items[0] if items else None)
        if self._agent_overview_header is not None:
            activated_count = sum(
                1 for item in self.state.agent_studio.agents if isinstance(item, dict) and bool(item.get("activated"))
            )
            self._agent_overview_header.set_badges([f"activated {activated_count}"])

    def _render_agent_builder_panel(self) -> None:
        self.state.agent_studio.agents = normalize_agent_definitions(self.state.agent_studio.agents)
        items: list[dict[str, Any]] = []
        for agent_def in self.state.agent_studio.agents:
            title = str(agent_def.get("name", "")).strip() or "Unnamed Agent"
            summary = str(agent_def.get("summary", "")).strip()
            provider = str(agent_def.get("provider", "")).strip()
            tier = str(agent_def.get("tier", "")).strip()
            model_label = "/".join(token for token in (provider, tier) if token) or "inherit"
            subtitle = summary or f"model: {model_label}"
            if summary:
                subtitle = f"{summary} | {model_label}"
            items.append(
                {
                    "id": str(agent_def.get("id", "")).strip(),
                    "title": title,
                    "subtitle": subtitle,
                    "state": "active" if bool(agent_def.get("activated")) else "default",
                    "agent": dict(agent_def),
                }
            )
        self.state.agent_studio.builder_items = items
        list_widget = self._agent_builder_list
        selected_id = self.state.agent_studio.builder_selected_item_id
        if list_widget is not None:
            if not selected_id and items:
                selected_id = str(items[0].get("id", "")).strip()
            list_widget.set_items(items, selected_id=selected_id, emit=False)
            selected_item = self._agent_builder_item_by_id(list_widget.selected_id())
            if selected_item is None and items:
                selected_item = items[0]
            self._set_agent_builder_selection(selected_item)
        else:
            self._set_agent_builder_selection(items[0] if items else None)
        if self._agent_builder_auto_delegate_checkbox is not None:
            self._agent_builder_auto_delegate_checkbox.value = bool(self.state.agent_studio.auto_delegate_assistive)
        self._render_agent_overview_panel()

    def _new_agent_builder_draft(self) -> None:
        payload = normalize_agent_definition(
            {
                "id": f"agent-{uuid.uuid4().hex[:8]}",
                "name": "New Agent",
                "summary": "",
                "prompt": "",
                "activated": False,
            }
        )
        if payload is None:
            return
        self.state.agent_studio.builder_selected_item_id = None
        self._set_agent_builder_form_values(payload)
        self._set_agent_builder_status("New agent draft.")
        self._set_agent_draft_dirty(True)

    def _save_agent_builder_draft(self) -> None:
        payload = self._agent_builder_form_payload()
        if payload is None:
            return
        selected_id = str(self.state.agent_studio.builder_selected_item_id or "").strip()
        payload_id = str(payload.get("id", "")).strip()
        next_agents = [dict(item) for item in normalize_agent_definitions(self.state.agent_studio.agents)]
        if selected_id and selected_id != payload_id:
            next_agents = [item for item in next_agents if str(item.get("id", "")).strip() != selected_id]
        replaced = False
        for idx, existing in enumerate(next_agents):
            if str(existing.get("id", "")).strip() == payload_id:
                next_agents[idx] = payload
                replaced = True
                break
        if not replaced:
            next_agents.append(payload)
        self.state.agent_studio.agents = normalize_agent_definitions(next_agents)
        self.state.agent_studio.builder_selected_item_id = payload_id
        self._render_agent_builder_panel()
        self._set_agent_builder_status(f"Saved agent '{payload['name']}' in draft.")
        self._set_agent_draft_dirty(True, note=f"Agent '{payload['name']}' saved in draft.")

    def _delete_selected_builder_agent(self) -> None:
        selected = self._agent_builder_item_by_id(self.state.agent_studio.builder_selected_item_id)
        if selected is None:
            self._notify("Select a draft agent first.", severity="warning")
            return
        selected_id = str(selected.get("id", "")).strip()
        next_agents = [
            item for item in self.state.agent_studio.agents if str(item.get("id", "")).strip() != selected_id
        ]
        self.state.agent_studio.agents = normalize_agent_definitions(next_agents)
        self.state.agent_studio.builder_selected_item_id = None
        self._render_agent_builder_panel()
        self._set_agent_builder_status(f"Deleted agent '{selected_id}' from draft.")
        self._set_agent_draft_dirty(True, note=f"Agent '{selected_id}' removed from draft.")

    def _insert_activated_agents_run_prompt(self, *, run_now: bool = False) -> None:
        prompt = build_activated_agents_run_prompt(self.state.agent_studio.agents)
        if not prompt:
            self._notify("No activated agents available.", severity="warning")
            return
        if not self._set_prompt_editor_text(prompt):
            self._notify("Prompt editor is unavailable.", severity="warning")
            return
        self._set_agent_builder_status("Inserted swarm prompt for activated agents.")
        if run_now:
            self.action_submit_prompt()

    def _set_agent_studio_view_mode(self, mode: str) -> None:
        normalized = normalize_agent_studio_view_mode(mode)
        self.state.agent_studio.view_mode = normalized

        overview_view = self._agent_overview_view
        builder_view = self._agent_builder_view
        if overview_view is not None:
            overview_view.styles.display = "block" if normalized == "overview" else "none"
        if builder_view is not None:
            builder_view.styles.display = "block" if normalized == "builder" else "none"

        overview_button = self._agent_view_overview_button
        builder_button = self._agent_view_builder_button
        if overview_button is not None:
            overview_button.variant = "primary" if normalized == "overview" else "default"
        if builder_button is not None:
            builder_button.variant = "primary" if normalized == "builder" else "default"

    def _kb_id_from_context_sources(self) -> str | None:
        for source in self._context_sources:
            source_type = str(source.get("type", "")).strip().lower()
            if source_type != "kb":
                continue
            kb_id = str(source.get("id", "")).strip()
            if kb_id:
                return kb_id
        return None

    def _session_effective_profile(self) -> AgentProfile:
        provider_name, tier_name, _model_id = choose_model_summary_parts(
            daemon_provider=self.state.daemon.provider,
            daemon_tier=self.state.daemon.tier,
            daemon_model_id=self.state.daemon.model_id,
            daemon_tiers=self.state.daemon.tiers,
            pending_value=self.state.daemon.pending_model_select_value,
            override_provider=self.state.daemon.model_provider_override,
            override_tier=self.state.daemon.model_tier_override,
        )
        current = self.state.agent_studio.effective_profile
        return AgentProfile(
            id=(current.id if current is not None else "session-effective"),
            name=(current.name if current is not None else "Session Effective"),
            provider=provider_name,
            tier=tier_name,
            system_prompt_snippets=(list(current.system_prompt_snippets) if current is not None else []),
            context_sources=self._context_sources_payload(),
            active_sops=sorted(self._active_sop_names),
            knowledge_base_id=(
                self._kb_id_from_context_sources() or (current.knowledge_base_id if current else None)
            ),
            agents=(normalize_agent_definitions(current.agents) if current is not None else []),
            auto_delegate_assistive=(
                bool(current.auto_delegate_assistive) if current is not None else True
            ),
            team_presets=(normalize_team_presets(current.team_presets) if current is not None else []),
        )

    def _set_agent_status(self, message: str) -> None:
        widget = self._agent_profile_status
        if widget is None:
            return
        text = message.strip() if isinstance(message, str) else ""
        widget.update(text)

    def _set_agent_draft_dirty(self, dirty: bool, *, note: str | None = None) -> None:
        self.state.agent_studio.draft_dirty = bool(dirty)
        if self.state.agent_studio.draft_dirty:
            base = "Draft changes pending."
        else:
            base = "Draft synced."
        if isinstance(note, str) and note.strip():
            self._set_agent_status(f"{base} {note.strip()}")
        else:
            self._set_agent_status(base)
        self._reload_saved_profiles()

    def _reload_saved_profiles(self, *, selected_id: str | None = None) -> None:
        self.state.agent_studio.saved_profiles = sorted(
            list_profiles(),
            key=lambda item: (item.name.lower(), item.id.lower()),
        )
        selector = self._agent_profile_select
        if selector is None:
            return

        options: list[tuple[str, str]] = [("Current draft / session", _AGENT_PROFILE_SELECT_NONE)]
        for profile in self.state.agent_studio.saved_profiles:
            model_summary = "/".join(
                part for part in [str(profile.provider or "").strip(), str(profile.tier or "").strip()] if part
            )
            label = profile.name
            if model_summary:
                label = f"{profile.name} ({model_summary})"
            options.append((label, profile.id))

        current_value = str(getattr(selector, "value", "")).strip()
        candidate = selected_id if selected_id else current_value
        saved_ids = {profile.id for profile in self.state.agent_studio.saved_profiles}
        if candidate == _AGENT_PROFILE_SELECT_NONE:
            pass
        elif candidate not in saved_ids:
            candidate = (
                self.state.agent_studio.saved_profiles[0].id
                if self.state.agent_studio.saved_profiles
                else _AGENT_PROFILE_SELECT_NONE
            )

        self.state.agent_studio.profile_select_syncing = True
        try:
            selector.set_options(options)
            selector.value = candidate
        finally:
            self.state.agent_studio.profile_select_syncing = False

    def _refresh_agent_summary(self) -> None:
        from swarmee_river.tui.widgets import render_agent_profile_summary_text

        summary = self._agent_summary
        if summary is None:
            return
        effective = self._session_effective_profile()
        self.state.agent_studio.effective_profile = effective
        summary.load_text(render_agent_profile_summary_text(effective.to_dict()))
        summary.scroll_home(animate=False)
        if not self.state.agent_studio.agents:
            self.state.agent_studio.agents = normalize_agent_definitions(effective.agents)
            self.state.agent_studio.auto_delegate_assistive = bool(effective.auto_delegate_assistive)
        self._render_agent_overview_panel()

    def _new_agent_profile_draft(self, *, announce: bool = True) -> None:
        snapshot = self._session_effective_profile()
        draft = AgentProfile(
            id=_default_profile_id(),
            name=_default_profile_name(),
            provider=snapshot.provider,
            tier=snapshot.tier,
            system_prompt_snippets=list(snapshot.system_prompt_snippets),
            context_sources=[dict(source) for source in snapshot.context_sources],
            active_sops=list(snapshot.active_sops),
            knowledge_base_id=snapshot.knowledge_base_id,
            agents=normalize_agent_definitions(snapshot.agents),
            auto_delegate_assistive=bool(snapshot.auto_delegate_assistive),
            team_presets=normalize_team_presets(snapshot.team_presets),
        )
        selector = self._agent_profile_select
        if selector is not None:
            self.state.agent_studio.profile_select_syncing = True
            try:
                selector.value = _AGENT_PROFILE_SELECT_NONE
            finally:
                self.state.agent_studio.profile_select_syncing = False
        self.state.agent_studio.team_presets = normalize_team_presets(draft.team_presets)
        self.state.agent_studio.team_selected_item_id = None
        self.state.agent_studio.agents = normalize_agent_definitions(draft.agents)
        self.state.agent_studio.builder_selected_item_id = None
        self.state.agent_studio.auto_delegate_assistive = bool(draft.auto_delegate_assistive)
        self._render_agent_builder_panel()
        self._set_agent_form_values(profile_id=draft.id, profile_name=draft.name)
        self._set_agent_draft_dirty(True, note=("New profile draft." if announce else None))

    def _load_profile_into_draft(self, profile: AgentProfile) -> None:
        selector = self._agent_profile_select
        if selector is not None:
            self.state.agent_studio.profile_select_syncing = True
            try:
                selector.value = profile.id
            finally:
                self.state.agent_studio.profile_select_syncing = False
        self.state.agent_studio.team_presets = normalize_team_presets(profile.team_presets)
        self.state.agent_studio.team_selected_item_id = None
        self.state.agent_studio.agents = normalize_agent_definitions(profile.agents)
        self.state.agent_studio.builder_selected_item_id = None
        self.state.agent_studio.auto_delegate_assistive = bool(profile.auto_delegate_assistive)
        self._render_agent_builder_panel()
        self._set_agent_form_values(profile_id=profile.id, profile_name=profile.name)
        self._set_agent_draft_dirty(False, note=f"Loaded profile '{profile.name}'.")

    def _profile_from_draft(self) -> AgentProfile:
        selected_profile = self._lookup_saved_profile(self._selected_agent_profile_id())
        seed = selected_profile if selected_profile is not None else self._session_effective_profile()

        raw_id = str(getattr(self._agent_profile_id_input, "value", "")).strip()
        raw_name = str(getattr(self._agent_profile_name_input, "value", "")).strip()
        profile_id = _sanitize_profile_token(raw_id or seed.id or _default_profile_id())
        profile_name = raw_name or seed.name or _default_profile_name()

        return AgentProfile.from_dict(
            {
                "id": profile_id,
                "name": profile_name,
                "provider": seed.provider,
                "tier": seed.tier,
                "system_prompt_snippets": list(seed.system_prompt_snippets),
                "context_sources": [dict(source) for source in seed.context_sources],
                "active_sops": list(seed.active_sops),
                "knowledge_base_id": seed.knowledge_base_id,
                "agents": normalize_agent_definitions(self.state.agent_studio.agents),
                "auto_delegate_assistive": bool(self.state.agent_studio.auto_delegate_assistive),
                "team_presets": normalize_team_presets(self.state.agent_studio.team_presets),
            }
        )

    def _save_agent_profile_draft(self) -> None:
        profile = self._profile_from_draft()
        saved = save_profile(profile)
        self._reload_saved_profiles(selected_id=saved.id)
        self._set_agent_form_values(profile_id=saved.id, profile_name=saved.name)
        self._set_agent_draft_dirty(False, note=f"Saved profile '{saved.name}'.")

    def _delete_selected_agent_profile(self) -> None:
        selected_id = self._selected_agent_profile_id()
        if not selected_id:
            self._notify("Select a saved profile first.", severity="warning")
            return
        removed = delete_profile(selected_id)
        if not removed:
            self._notify("Profile not found.", severity="warning")
            return
        self._reload_saved_profiles()
        if self.state.agent_studio.saved_profiles:
            self._load_profile_into_draft(self.state.agent_studio.saved_profiles[0])
            self._set_agent_draft_dirty(False, note=f"Deleted profile '{selected_id}'.")
        else:
            self._new_agent_profile_draft(announce=False)
            self._set_agent_draft_dirty(True, note=f"Deleted profile '{selected_id}'.")

    def _apply_agent_profile_draft(self) -> None:
        if self.state.daemon.query_active:
            self._write_transcript_line("[agent] cannot apply profile while a run is active.")
            return
        if not self.state.daemon.ready:
            self._write_transcript_line("[agent] daemon is not ready.")
            return
        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None:
            self._write_transcript_line("[agent] daemon is not running.")
            self.state.daemon.ready = False
            return
        profile = self._profile_from_draft()
        payload = {"cmd": "set_profile", "profile": profile.to_dict()}
        if not _transport_send_daemon_command(proc, payload):
            self._write_transcript_line("[agent] failed to send set_profile.")
            return
        self._set_agent_status(f"Applying profile '{profile.name}'...")
