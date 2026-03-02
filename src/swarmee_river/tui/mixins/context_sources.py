from __future__ import annotations

import contextlib
import re
import uuid
from typing import Any

from swarmee_river.profiles.models import ORCHESTRATOR_AGENT_ID, normalize_agent_definitions
from swarmee_river.tui.commands import _CONTEXT_USAGE_TEXT, _SOP_USAGE_TEXT
from swarmee_river.tui.sops import discover_available_sop_names, discover_available_sops
from swarmee_river.tui.transport import send_daemon_command as _transport_send_daemon_command

_CONTEXT_SOURCE_ICONS: dict[str, str] = {
    "file": "📄",
    "url": "🌐",
    "kb": "📚",
    "sop": "📋",
    "note": "📝",
}
_CONTEXT_SOURCE_TYPES = {"file", "note", "sop", "kb", "url"}
_CONTEXT_SOURCE_MAX_LABEL = 72
_CONTEXT_SELECT_PLACEHOLDER = "__context_select_none__"
_CONTEXT_INPUT_SOURCE_TYPES = {"file", "note", "kb"}
_CONTEXT_SOP_SOURCE_TYPE = "sop"


def _sanitize_context_source_id(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "").strip())
    return token.strip("-") or uuid.uuid4().hex[:12]


def _normalize_context_source(source: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(source, dict):
        return None
    source_type = str(source.get("type", "")).strip().lower()
    if source_type not in _CONTEXT_SOURCE_TYPES:
        return None

    normalized: dict[str, str] = {"type": source_type}
    source_id = str(source.get("id", "")).strip()
    if source_id:
        normalized["id"] = _sanitize_context_source_id(source_id)
    else:
        source_seed = (
            source.get("path", "") or source.get("text", "") or source.get("name", "") or source.get("kb_id", "")
        )
        seed = str(source_seed).strip() or uuid.uuid4().hex
        normalized["id"] = _sanitize_context_source_id(f"{source_type}-{seed}")

    if source_type == "file":
        path = str(source.get("path", "")).strip()
        if not path:
            return None
        normalized["path"] = path
    elif source_type == "note":
        text = str(source.get("text", "")).strip()
        if not text:
            return None
        normalized["text"] = text
    elif source_type == "sop":
        name = str(source.get("name", "")).strip()
        if not name:
            return None
        normalized["name"] = name
    elif source_type == "kb":
        kb_id = str(source.get("id", source.get("kb_id", ""))).strip()
        if not kb_id:
            return None
        normalized["id"] = kb_id
    elif source_type == "url":
        url = str(source.get("url", source.get("path", ""))).strip()
        if not url:
            return None
        normalized["url"] = url
    return normalized


def _normalize_context_sources(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for source in sources:
        normalized = _normalize_context_source(source)
        if normalized is not None:
            result.append(normalized)
    return result


class ContextSourcesMixin:
    def _set_engage_view_mode(self, mode: str) -> None:
        normalized = (mode or "").strip().lower()
        if normalized in {"execution", "planning"}:
            normalized = "plan"
        if normalized not in {"plan", "session"}:
            normalized = "plan"
        self.state.engage_view_mode = normalized
        if self._engage_plan_view:
            self._engage_plan_view.styles.display = "block" if normalized == "plan" else "none"
        if self._engage_session_view:
            self._engage_session_view.styles.display = "block" if normalized == "session" else "none"
        if self._engage_view_plan_button:
            self._engage_view_plan_button.variant = "primary" if normalized == "plan" else "default"
        if self._engage_view_session_button:
            self._engage_view_session_button.variant = "primary" if normalized == "session" else "default"

    def _set_tooling_view_mode(self, mode: str) -> None:
        normalized = (mode or "").strip().lower()
        if normalized not in {"prompts", "tools", "sops", "kbs"}:
            normalized = "tools"
        self.state.tooling.view_mode = normalized
        self.state.tooling_view_mode = normalized

        if self._tooling_prompts_view:
            self._tooling_prompts_view.styles.display = "block" if normalized == "prompts" else "none"
        if self._tooling_tools_view:
            self._tooling_tools_view.styles.display = "block" if normalized == "tools" else "none"
        if self._tooling_sops_view:
            self._tooling_sops_view.styles.display = "block" if normalized == "sops" else "none"
        if self._tooling_kbs_view:
            self._tooling_kbs_view.styles.display = "block" if normalized == "kbs" else "none"

        if self._tooling_view_prompts_button:
            self._tooling_view_prompts_button.variant = "primary" if normalized == "prompts" else "default"
        if self._tooling_view_tools_button:
            self._tooling_view_tools_button.variant = "primary" if normalized == "tools" else "default"
        if self._tooling_view_sops_button:
            self._tooling_view_sops_button.variant = "primary" if normalized == "sops" else "default"
        if self._tooling_view_kbs_button:
            self._tooling_view_kbs_button.variant = "primary" if normalized == "kbs" else "default"

    def _refresh_tooling_prompts_list(self) -> None:
        from swarmee_river.prompt_assets import ensure_prompt_assets_bootstrapped, load_prompt_assets
        from swarmee_river.tui.tooling_handlers import build_prompt_table_rows

        bootstrap_result = ensure_prompt_assets_bootstrapped()
        assets = [item.to_dict() for item in load_prompt_assets()]
        used_by_map = self._tooling_prompt_used_by_map()

        rows_payload: list[dict[str, Any]] = []
        for asset in assets:
            prompt_id = str(asset.get("id", "")).strip().lower()
            used_by = sorted(used_by_map.get(prompt_id, set()))
            row = dict(asset)
            row["id"] = prompt_id
            row["used_by"] = used_by
            rows_payload.append(row)

        self.state.tooling.prompt_assets = rows_payload
        if bootstrap_result.migrated and not getattr(self, "_prompt_assets_migration_notice_shown", False):
            legacy_paths = ", ".join(bootstrap_result.prompt_paths) if bootstrap_result.prompt_paths else "(none)"
            self._write_transcript_line(
                "[tooling] migrated legacy prompt templates to prompt assets; legacy files remain archived at "
                f"{legacy_paths}."
            )
            self._prompt_assets_migration_notice_shown = True

        table = self._tooling_prompts_table
        if table is None:
            return

        prev_selected = str(self.state.tooling.prompt_selected_id or "").strip() or None
        if not table.columns:
            table.add_column("Name", key="name")
            table.add_column("ID", key="id", width=24)
            table.add_column("Tags", key="tags", width=16)
            table.add_column("Used By", key="used_by", width=24)
            table.add_column("Preview", key="preview")

        table.clear()
        rows = build_prompt_table_rows(rows_payload)
        for prompt_id, name, id_text, tags, used_by_text, preview in rows:
            table.add_row(name, id_text, tags, used_by_text, preview, key=prompt_id)

        if prev_selected and rows:
            for idx, (prompt_id, _, _, _, _, _) in enumerate(rows):
                if prompt_id == prev_selected:
                    with contextlib.suppress(Exception):
                        table.move_cursor(row=idx)
                    self._tooling_select_prompt(prev_selected)
                    return
        if rows:
            with contextlib.suppress(Exception):
                table.move_cursor(row=0)
            self._tooling_select_prompt(rows[0][0])
            return
        self._tooling_select_prompt(None)

    def _tooling_select_prompt(self, selected_id: str | None) -> None:
        target_id = str(selected_id or "").strip()
        selected: dict[str, Any] | None = None
        if target_id:
            selected = next(
                (item for item in self.state.tooling.prompt_assets if str(item.get("id", "")).strip() == target_id),
                None,
            )
        self.state.tooling.prompt_selected_id = str(selected.get("id", "")).strip() if selected else None

        if self._tooling_prompt_content_input is not None:
            with contextlib.suppress(Exception):
                self._tooling_prompt_content_input.text = str((selected or {}).get("content", "")).strip()

    def _tooling_prompt_new(self) -> None:
        from swarmee_river.prompt_assets import PromptAsset, load_prompt_assets, save_prompt_assets

        assets = load_prompt_assets()
        existing_ids = {str(item.id).strip().lower() for item in assets}
        new_id = f"new_prompt_{uuid.uuid4().hex[:8]}".lower()
        while new_id in existing_ids:
            new_id = f"new_prompt_{uuid.uuid4().hex[:8]}".lower()
        new_asset = PromptAsset(
            id=new_id,
            name="New Prompt",
            content="",
            tags=[],
            source="project",
        )
        assets.append(new_asset)
        save_prompt_assets(assets)
        proc = self.state.daemon.proc
        if self.state.daemon.ready and proc is not None and proc.poll() is None and not self.state.daemon.query_active:
            _transport_send_daemon_command(proc, {"cmd": "set_prompt_asset", "asset": new_asset.to_dict()})
        self.state.tooling.prompt_selected_id = new_id
        self._refresh_tooling_prompts_list()
        if self._tooling_prompt_content_input is not None:
            with contextlib.suppress(Exception):
                self._tooling_prompt_content_input.focus()

    def _tooling_prompt_open_table_cell_editor(self, row_id: str, column_key: str) -> None:
        from swarmee_river.tui.widgets import TableCellEditScreen

        selected = next(
            (
                item
                for item in self.state.tooling.prompt_assets
                if str(item.get("id", "")).strip().lower() == str(row_id or "").strip().lower()
            ),
            None,
        )
        if selected is None:
            self._notify("Prompt row not found.", severity="warning")
            return
        key = str(column_key or "").strip().lower()
        if key not in {"name", "id", "tags"}:
            return
        initial_value = ""
        help_text = ""
        title = "Edit prompt metadata"
        if key == "name":
            initial_value = str(selected.get("name", "")).strip()
            help_text = "Prompt display name (required)."
            title = "Edit Prompt Name"
        elif key == "id":
            initial_value = str(selected.get("id", "")).strip()
            help_text = "Prompt ID (sanitized to lowercase and must be unique)."
            title = "Edit Prompt ID"
        elif key == "tags":
            tags = selected.get("tags") or []
            initial_value = ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
            help_text = "Comma-separated tags."
            title = "Edit Prompt Tags"

        self.push_screen(
            TableCellEditScreen(title, initial_value, help_text=help_text),
            callback=lambda result: self._tooling_prompt_apply_metadata_edit(
                str(row_id or "").strip().lower(),
                key,
                result,
            ),
        )

    def _tooling_prompt_apply_metadata_edit(self, row_id: str, column_key: str, new_value: str | None) -> None:
        from swarmee_river.prompt_assets import load_prompt_assets, save_prompt_assets

        if new_value is None:
            return
        target_id = str(row_id or "").strip().lower()
        if not target_id:
            return
        key = str(column_key or "").strip().lower()
        if key not in {"name", "id", "tags"}:
            return

        assets = load_prompt_assets()
        target = next((item for item in assets if str(item.id).strip().lower() == target_id), None)
        if target is None:
            self._notify("Prompt asset not found.", severity="warning")
            return

        raw_value = str(new_value).strip()
        new_selected_id = target_id
        if key == "name":
            if not raw_value:
                self._notify("Prompt name is required.", severity="warning")
                return
            target.name = raw_value
        elif key == "id":
            proposed_id = _sanitize_context_source_id(raw_value or target.id).lower()
            if not proposed_id:
                self._notify("Prompt ID is required.", severity="warning")
                return
            if proposed_id != target_id and any(str(item.id).strip().lower() == proposed_id for item in assets):
                self._notify(f"Prompt ID '{proposed_id}' already exists.", severity="warning")
                return
            old_id = str(target.id).strip().lower()
            target.id = proposed_id
            new_selected_id = proposed_id
            if old_id != proposed_id:
                self._tooling_prompt_rewrite_prompt_refs(old_id, proposed_id)
        elif key == "tags":
            tags: list[str] = []
            seen: set[str] = set()
            for token in raw_value.split(","):
                value = str(token).strip()
                lowered = value.lower()
                if not value or lowered in seen:
                    continue
                seen.add(lowered)
                tags.append(value)
            target.tags = tags

        save_prompt_assets(assets)
        proc = self.state.daemon.proc
        if self.state.daemon.ready and proc is not None and proc.poll() is None and not self.state.daemon.query_active:
            _transport_send_daemon_command(proc, {"cmd": "set_prompt_asset", "asset": target.to_dict()})
        self.state.tooling.prompt_selected_id = new_selected_id
        self._refresh_tooling_prompts_list()

    def _tooling_prompt_used_by_map(self) -> dict[str, set[str]]:
        usage: dict[str, set[str]] = {}
        normalized_agents = normalize_agent_definitions(self.state.agent_studio.agents)
        name_counts: dict[str, int] = {}
        for agent in normalized_agents:
            if not isinstance(agent, dict):
                continue
            name = str(agent.get("name", "")).strip() or str(agent.get("id", "")).strip() or "agent"
            key = name.casefold()
            name_counts[key] = name_counts.get(key, 0) + 1
        for agent in normalized_agents:
            if not isinstance(agent, dict):
                continue
            agent_id = str(agent.get("id", "")).strip()
            if not agent_id:
                continue
            name = str(agent.get("name", "")).strip() or agent_id
            display = name if name_counts.get(name.casefold(), 0) <= 1 else f"{name} ({agent_id})"
            refs = agent.get("prompt_refs") or []
            if not isinstance(refs, list):
                continue
            for prompt_ref in refs:
                prompt_id = str(prompt_ref or "").strip().lower()
                if not prompt_id:
                    continue
                usage.setdefault(prompt_id, set()).add(display)
        return usage

    def _tooling_prompt_used_by_choices(self) -> list[tuple[str, str]]:
        normalized_agents = normalize_agent_definitions(self.state.agent_studio.agents)
        name_counts: dict[str, int] = {}
        for agent in normalized_agents:
            name = str(agent.get("name", "")).strip() or str(agent.get("id", "")).strip() or "agent"
            key = name.casefold()
            name_counts[key] = name_counts.get(key, 0) + 1

        choices: list[tuple[str, str]] = []
        for agent in normalized_agents:
            agent_id = str(agent.get("id", "")).strip()
            if not agent_id:
                continue
            name = str(agent.get("name", "")).strip() or agent_id
            label_name = name if name_counts.get(name.casefold(), 0) <= 1 else f"{name} ({agent_id})"
            choices.append((agent_id, f"{label_name} ({agent_id})"))

        choices.sort(
            key=lambda item: (
                0 if str(item[0]).strip().lower() == ORCHESTRATOR_AGENT_ID else 1,
                str(item[1]).strip().lower(),
            )
        )
        return choices

    def _tooling_prompt_open_used_by_editor(self, prompt_id: str) -> None:
        from swarmee_river.tui.widgets import PromptUsedByEditScreen

        token = str(prompt_id or "").strip().lower()
        if not token:
            return
        normalized_agents = normalize_agent_definitions(self.state.agent_studio.agents)
        selected_agent_ids: list[str] = []
        for agent in normalized_agents:
            agent_id = str(agent.get("id", "")).strip()
            if not agent_id:
                continue
            refs = [str(item).strip().lower() for item in (agent.get("prompt_refs") or []) if str(item).strip()]
            if token in refs:
                selected_agent_ids.append(agent_id)
        self.push_screen(
            PromptUsedByEditScreen(self._tooling_prompt_used_by_choices(), selected_agent_ids),
            callback=lambda selected_ids: self._tooling_prompt_apply_used_by_edit(token, selected_ids),
        )

    def _tooling_prompt_rewrite_prompt_refs(self, old_prompt_id: str, new_prompt_id: str) -> None:
        old_token = str(old_prompt_id or "").strip().lower()
        new_token = str(new_prompt_id or "").strip().lower()
        if not old_token or not new_token or old_token == new_token:
            return
        changed = False
        updated_agents: list[dict[str, Any]] = []
        for agent in normalize_agent_definitions(self.state.agent_studio.agents):
            next_agent = dict(agent)
            refs = [str(item).strip().lower() for item in (next_agent.get("prompt_refs") or []) if str(item).strip()]
            rewritten: list[str] = []
            seen: set[str] = set()
            local_changed = False
            for ref in refs:
                token = new_token if ref == old_token else ref
                local_changed = local_changed or (token != ref)
                if token in seen:
                    continue
                seen.add(token)
                rewritten.append(token)
            if local_changed:
                changed = True
            next_agent["prompt_refs"] = rewritten
            updated_agents.append(next_agent)
        if not changed:
            return
        self.state.agent_studio.agents = normalize_agent_definitions(updated_agents)
        self._set_agent_draft_dirty(True, note=f"Updated prompt references: {old_token} -> {new_token}.")
        with contextlib.suppress(Exception):
            self._render_agent_builder_panel()
        with contextlib.suppress(Exception):
            self._render_agent_overview_panel()

    def _tooling_prompt_apply_used_by_edit(self, prompt_id: str, selected_agent_ids: list[str] | None) -> None:
        if selected_agent_ids is None:
            return
        token = str(prompt_id or "").strip().lower()
        if not token:
            return
        selected_ids = {str(agent_id).strip().lower() for agent_id in selected_agent_ids if str(agent_id).strip()}
        updated_agents: list[dict[str, Any]] = []
        changed = False
        for agent in normalize_agent_definitions(self.state.agent_studio.agents):
            next_agent = dict(agent)
            agent_id = str(next_agent.get("id", "")).strip().lower()
            refs = [str(item).strip().lower() for item in (next_agent.get("prompt_refs") or []) if str(item).strip()]
            rewritten: list[str] = []
            seen: set[str] = set()
            for ref in refs:
                if ref == token and agent_id not in selected_ids:
                    changed = True
                    continue
                if ref in seen:
                    continue
                seen.add(ref)
                rewritten.append(ref)
            if agent_id in selected_ids and token not in seen:
                rewritten.append(token)
                changed = True
            next_agent["prompt_refs"] = rewritten
            updated_agents.append(next_agent)
        if not changed:
            return
        self.state.agent_studio.agents = normalize_agent_definitions(updated_agents)
        self._set_agent_draft_dirty(True, note=f"Updated prompt assignments for '{token}'.")
        with contextlib.suppress(Exception):
            self._render_agent_builder_panel()
        with contextlib.suppress(Exception):
            self._render_agent_overview_panel()
        self._refresh_tooling_prompts_list()

    def _refresh_tooling_sops_table(self) -> None:
        from swarmee_river.tui.tooling_handlers import build_sop_table_rows

        self.state.tooling.sop_catalog = [dict(item) for item in self._sop_catalog]
        table = self._tooling_sops_table
        if table is None:
            return

        prev_selected = str(self.state.tooling.sop_selected_id or "").strip() or None
        if not table.columns:
            table.add_column("Name", key="name")
            table.add_column("Active", key="active", width=7)
            table.add_column("Source", key="source", width=18)
            table.add_column("Preview", key="preview")

        table.clear()
        rows = build_sop_table_rows(self.state.tooling.sop_catalog, set(self._active_sop_names))
        for name, active, source, preview in rows:
            table.add_row(name, active, source, preview, key=name)

        if prev_selected and rows:
            for idx, (name, _, _, _) in enumerate(rows):
                if name == prev_selected:
                    with contextlib.suppress(Exception):
                        table.move_cursor(row=idx)
                    self._tooling_select_sop(prev_selected)
                    return
        if rows:
            with contextlib.suppress(Exception):
                table.move_cursor(row=0)
            self._tooling_select_sop(rows[0][0])
            return
        self._tooling_select_sop(None)

    def _tooling_select_sop(self, selected_id: str | None) -> None:
        from swarmee_river.tui.tooling_handlers import render_sop_detail

        target_id = str(selected_id or "").strip()
        selected: dict[str, Any] | None = None
        if target_id:
            selected = next(
                (item for item in self.state.tooling.sop_catalog if str(item.get("name", "")).strip() == target_id),
                None,
            )
        self.state.tooling.sop_selected_id = str(selected.get("name", "")).strip() if selected else None
        detail = self._tooling_sops_detail
        if detail is not None:
            with contextlib.suppress(Exception):
                if selected is None:
                    detail.set_preview("Select an SOP to view details. Press Enter to activate/deactivate.")
                else:
                    sop_name = str(selected.get("name", "")).strip()
                    detail.set_preview(render_sop_detail(selected, active=(sop_name in self._active_sop_names)))

    def _refresh_tooling_kbs_table(self) -> None:
        from swarmee_river.tui.tooling_handlers import build_kb_table_rows

        table = self._tooling_kbs_table
        if table is None:
            return

        prev_selected = str(self.state.tooling.kb_selected_id or "").strip() or None
        if not table.columns:
            table.add_column("Name", key="name")
            table.add_column("ID", key="id", width=24)
            table.add_column("Description", key="description")

        table.clear()
        rows = build_kb_table_rows(self.state.tooling.kb_entries)
        for kb_id, name, description in rows:
            table.add_row(name, kb_id, description, key=kb_id)

        empty_state = None
        with contextlib.suppress(Exception):
            empty_state = self.query_one("#kbs_empty_state")
        if empty_state is not None:
            with contextlib.suppress(Exception):
                empty_state.styles.display = "none" if rows else "block"

        if prev_selected and rows:
            for idx, (kb_id, _, _) in enumerate(rows):
                if kb_id == prev_selected:
                    with contextlib.suppress(Exception):
                        table.move_cursor(row=idx)
                    self._tooling_select_kb(prev_selected)
                    return
        if rows:
            with contextlib.suppress(Exception):
                table.move_cursor(row=0)
            self._tooling_select_kb(rows[0][0])
            return
        self._tooling_select_kb(None)

    def _tooling_select_kb(self, selected_id: str | None) -> None:
        from swarmee_river.tui.tooling_handlers import render_kb_detail

        target_id = str(selected_id or "").strip()
        selected: dict[str, Any] | None = None
        if target_id:
            selected = next(
                (
                    item
                    for index, item in enumerate(self.state.tooling.kb_entries)
                    if str(item.get("id", item.get("name", f"kb-{index + 1}"))).strip() == target_id
                ),
                None,
            )
        self.state.tooling.kb_selected_id = target_id if selected is not None else None
        if self._kbs_detail is not None:
            with contextlib.suppress(Exception):
                if selected is None:
                    self._kbs_detail.set_preview("Select a knowledge base entry to view details.")
                else:
                    self._kbs_detail.set_preview(render_kb_detail(selected))

    def _tooling_prompt_save(self) -> None:
        from swarmee_river.prompt_assets import load_prompt_assets, save_prompt_assets

        content = (
            str(getattr(self._tooling_prompt_content_input, "text", "")).strip()
            if self._tooling_prompt_content_input
            else ""
        )
        assets = load_prompt_assets()
        selected_id = str(self.state.tooling.prompt_selected_id or "").strip().lower()
        if not selected_id:
            self._notify("Select a prompt asset first.", severity="warning")
            return
        saved_asset = next((item for item in assets if str(item.id).strip().lower() == selected_id), None)
        if saved_asset is None:
            self._notify("Prompt asset not found.", severity="warning")
            return
        saved_asset.content = content
        save_prompt_assets(assets)
        proc = self.state.daemon.proc
        if self.state.daemon.ready and proc is not None and proc.poll() is None and not self.state.daemon.query_active:
            _transport_send_daemon_command(proc, {"cmd": "set_prompt_asset", "asset": saved_asset.to_dict()})
        self._refresh_tooling_prompts_list()
        self._write_transcript_line(f"[tooling] saved prompt asset: {saved_asset.name}")

    def _tooling_prompt_delete(self) -> None:
        from swarmee_river.prompt_assets import load_prompt_assets, save_prompt_assets

        selected_id = str(self.state.tooling.prompt_selected_id or "").strip().lower()
        if not selected_id:
            self._notify("Select a prompt asset to delete.", severity="warning")
            return
        orchestrator = next(
            (
                agent
                for agent in normalize_agent_definitions(self.state.agent_studio.agents)
                if str(agent.get("id", "")).strip().lower() == ORCHESTRATOR_AGENT_ID
            ),
            None,
        )
        orchestrator_refs = {
            str(item).strip().lower() for item in ((orchestrator or {}).get("prompt_refs") or []) if str(item).strip()
        }
        if selected_id in orchestrator_refs:
            self._notify("Cannot delete a prompt assigned to the orchestrator agent.", severity="warning")
            return
        assets = load_prompt_assets()
        filtered = [item for item in assets if str(item.id).strip().lower() != selected_id]
        if len(filtered) == len(assets):
            self._notify("Prompt asset not found.", severity="warning")
            return
        save_prompt_assets(filtered)
        proc = self.state.daemon.proc
        if self.state.daemon.ready and proc is not None and proc.poll() is None and not self.state.daemon.query_active:
            _transport_send_daemon_command(proc, {"cmd": "delete_prompt_asset", "id": selected_id})
        self.state.tooling.prompt_selected_id = None
        self._refresh_tooling_prompts_list()
        self._tooling_prompt_new()
        self._write_transcript_line(f"[tooling] deleted prompt asset: {selected_id}")

    def _refresh_tooling_tools_list(self) -> None:
        from swarmee_river.tui.tool_metadata import discover_tools_with_metadata
        from swarmee_river.tui.tooling_handlers import build_tool_table_rows

        catalog = [item.to_dict() for item in discover_tools_with_metadata()]
        self.state.tooling.tool_catalog = catalog
        table = self._tooling_tools_table
        if table is None:
            return

        # Remember previously selected tool
        prev_selected = str(self.state.tooling.tool_selected_id or "").strip() or None

        # Add columns on first load
        if not table.columns:
            table.add_column("Name", key="name")
            table.add_column("Access", key="access", width=9)
            table.add_column("Source", key="source", width=10)
            table.add_column("Tags", key="tags")

        # Clear and repopulate rows
        table.clear()
        rows = build_tool_table_rows(catalog)
        for name, access, source, tags in rows:
            table.add_row(name, access, source, tags, key=name)

        # Restore cursor to previously selected tool
        if prev_selected and rows:
            for idx, (name, _, _, _) in enumerate(rows):
                if name == prev_selected:
                    with contextlib.suppress(Exception):
                        table.move_cursor(row=idx)
                    break
        elif rows:
            with contextlib.suppress(Exception):
                table.move_cursor(row=0)
            first_name = rows[0][0] if rows else None
            if first_name:
                self._tooling_select_tool(first_name)

    def _tooling_select_tool(self, selected_id: str | None) -> None:
        from swarmee_river.tui.tooling_handlers import render_tool_detail

        target_id = str(selected_id or "").strip()
        selected: dict[str, Any] | None = None
        if target_id:
            selected = next(
                (item for item in self.state.tooling.tool_catalog if str(item.get("name", "")).strip() == target_id),
                None,
            )
        self.state.tooling.tool_selected_id = str(selected.get("name", "")).strip() if selected else None
        if self._tooling_tools_detail is not None:
            with contextlib.suppress(Exception):
                if selected is None:
                    self._tooling_tools_detail.set_preview("Select a tool to view details.")
                else:
                    self._tooling_tools_detail.set_preview(render_tool_detail(selected))

    def _tooling_tools_open_tag_manager(self) -> None:
        from swarmee_river.tui.widgets import ToolTagManagerScreen

        catalog = [dict(item) for item in self.state.tooling.tool_catalog if isinstance(item, dict)]
        self.push_screen(
            ToolTagManagerScreen(catalog),
            callback=self._on_tool_tag_manager_complete,
        )

    def _on_tool_tag_manager_complete(self, result: dict[str, Any] | None) -> None:
        if result is None or not isinstance(result, dict):
            return
        tool_tags_raw = result.get("tool_tags")
        if not isinstance(tool_tags_raw, dict):
            return

        from swarmee_river.tui.tool_metadata import load_tool_metadata_overrides, save_tool_metadata_overrides

        selected_name = str(self.state.tooling.tool_selected_id or "").strip()
        catalog_names = {
            str(item.get("name", "")).strip()
            for item in self.state.tooling.tool_catalog
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }
        overrides = load_tool_metadata_overrides()

        for raw_name, raw_tags in tool_tags_raw.items():
            tool_name = str(raw_name).strip()
            if not tool_name or tool_name not in catalog_names:
                continue
            incoming_tags = raw_tags if isinstance(raw_tags, list) else []
            tags: list[str] = []
            seen: set[str] = set()
            for raw_tag in incoming_tags:
                tag = str(raw_tag).strip()
                lowered = tag.lower()
                if not tag or lowered in seen:
                    continue
                seen.add(lowered)
                tags.append(tag)

            record = dict(overrides.get(tool_name, {}))
            if tags:
                record["tags"] = tags
            else:
                record.pop("tags", None)

            if record:
                overrides[tool_name] = record
            else:
                overrides.pop(tool_name, None)

        save_tool_metadata_overrides(overrides)
        self._refresh_tooling_tools_list()
        if selected_name:
            self._tooling_select_tool(selected_name)
        self._write_transcript_line("[tooling] saved tag manager changes.")

    def _tooling_tool_open_tag_editor(self) -> None:
        from swarmee_river.tui.widgets import TagEditScreen

        selected_name = str(self.state.tooling.tool_selected_id or "").strip()
        if not selected_name:
            self._notify("Select a tool first.", severity="warning")
            return
        selected = next(
            (item for item in self.state.tooling.tool_catalog if str(item.get("name", "")).strip() == selected_name),
            None,
        )
        current_tags = ", ".join(str(tag).strip() for tag in ((selected or {}).get("tags") or []) if str(tag).strip())
        self.push_screen(
            TagEditScreen(selected_name, current_tags),
            callback=self._on_tag_edit_complete,
        )

    def _on_tag_edit_complete(self, result: str | None) -> None:
        if result is None:
            return
        from swarmee_river.tui.tool_metadata import load_tool_metadata_overrides, save_tool_metadata_overrides

        selected_name = str(self.state.tooling.tool_selected_id or "").strip()
        if not selected_name:
            return
        tags = [segment.strip() for segment in result.split(",") if segment.strip()]
        overrides = load_tool_metadata_overrides()
        record = dict(overrides.get(selected_name, {}))
        record["tags"] = tags
        overrides[selected_name] = record
        save_tool_metadata_overrides(overrides)
        self._refresh_tooling_tools_list()
        self._tooling_select_tool(selected_name)
        self._write_transcript_line(f"[tooling] updated tags for: {selected_name}")

    def _tooling_s3_import(self, target: str) -> None:
        from swarmee_river.tui.s3_assets import (
            import_kbs_from_s3,
            import_sops_from_s3,
            import_tools_config_from_s3,
        )
        from swarmee_river.tui.tool_metadata import load_tool_metadata_overrides, save_tool_metadata_overrides

        normalized = (target or "").strip().lower()
        try:
            if normalized == "tools":
                imported = import_tools_config_from_s3()
                overrides = load_tool_metadata_overrides()
                payload = imported.get("tools", imported) if isinstance(imported, dict) else {}
                if isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name", "")).strip()
                        if not name:
                            continue
                        current = dict(overrides.get(name, {}))
                        current.update(
                            {
                                "tags": [str(tag).strip() for tag in (item.get("tags") or []) if str(tag).strip()],
                                "access_read": bool(item.get("access_read", current.get("access_read", False))),
                                "access_write": bool(item.get("access_write", current.get("access_write", False))),
                                "access_execute": bool(
                                    item.get("access_execute", current.get("access_execute", False))
                                ),
                            }
                        )
                        if str(item.get("description", "")).strip():
                            current["description"] = str(item.get("description", "")).strip()
                        overrides[name] = current
                elif isinstance(payload, dict):
                    for name, item in payload.items():
                        if not isinstance(item, dict):
                            continue
                        tool_name = str(name).strip()
                        if not tool_name:
                            continue
                        current = dict(overrides.get(tool_name, {}))
                        current.update(item)
                        overrides[tool_name] = current
                save_tool_metadata_overrides(overrides)
                self._refresh_tooling_tools_list()
                self._write_transcript_line("[tooling] imported tool metadata from S3.")
                return

            if normalized == "sops":
                imported = import_sops_from_s3()
                self._refresh_sop_catalog()
                self._refresh_tooling_sops_table()
                self._write_transcript_line(
                    f"[tooling] fetched {len(imported)} SOP definitions from S3 (manual install may be required)."
                )
                return

            if normalized == "kbs":
                imported = import_kbs_from_s3()
                self.state.tooling.kb_entries = [dict(item) for item in imported if isinstance(item, dict)]
                self._refresh_tooling_kbs_table()
                self._write_transcript_line(f"[tooling] imported {len(imported)} knowledge base records from S3.")
                return

            self._write_transcript_line(f"[tooling] unsupported S3 import target: {target}")
        except Exception as exc:
            self._write_transcript_line(f"[tooling] S3 import failed for {normalized or target}: {exc}")

    def _context_source_label(self, source: dict[str, str]) -> str:
        source_type = source.get("type", "")
        if source_type == "file":
            return source.get("path", "")
        if source_type == "note":
            return source.get("text", "")
        if source_type == "sop":
            return source.get("name", "")
        if source_type == "kb":
            return source.get("id", "")
        if source_type == "url":
            return source.get("url", source.get("path", ""))
        return str(source)

    def _truncate_context_label(self, value: str, *, max_chars: int = _CONTEXT_SOURCE_MAX_LABEL) -> str:
        text = value.strip().replace("\n", " ")
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return text[:max_chars]
        return text[: max_chars - 1].rstrip() + "…"

    def _render_context_sources_panel(self) -> None:
        from textual.containers import VerticalScroll
        from textual.widgets import Button, Static

        container = self._context_sources_list
        if container is None:
            with contextlib.suppress(Exception):
                container = self.query_one("#context_sources_list", VerticalScroll)
                self._context_sources_list = container
        if container is None:
            return

        for child in list(container.children):
            with contextlib.suppress(Exception):
                child.remove()

        if not self._context_sources:
            container.mount(Static("[dim](no context sources attached)[/dim]"))
            return

        from textual.containers import Horizontal

        for index, source in enumerate(self._context_sources):
            source_type = source.get("type", "").strip().lower()
            icon = _CONTEXT_SOURCE_ICONS.get(source_type, "•")
            label = self._truncate_context_label(self._context_source_label(source))
            row = Horizontal(classes="context-source-row")
            container.mount(row)
            row.mount(Static(f"{icon} {label}", classes="context-source-label"))
            row.mount(
                Button(
                    "✕",
                    id=f"context_remove_{index}",
                    classes="context-remove-btn",
                    compact=True,
                    variant="error",
                )
            )

    def _refresh_context_sop_options(self) -> None:
        from textual.widgets import Select

        self._context_sop_names = discover_available_sop_names()
        selector = self._context_sop_select
        if selector is None:
            with contextlib.suppress(Exception):
                selector = self.query_one("#context_sop_select", Select)
                self._context_sop_select = selector
        if selector is None:
            return
        options: list[tuple[str, str]] = [("Select SOP...", _CONTEXT_SELECT_PLACEHOLDER)]
        options.extend((name, name) for name in self._context_sop_names)
        with contextlib.suppress(Exception):
            selector.set_options(options)
            selector.value = _CONTEXT_SELECT_PLACEHOLDER

    def _refresh_sop_catalog(self) -> None:
        self._sop_catalog = discover_available_sops()
        self.state.tooling.sop_catalog = [dict(item) for item in self._sop_catalog]
        available_names = {item.get("name", "").strip() for item in self._sop_catalog}
        self._active_sop_names = {name for name in self._active_sop_names if name in available_names}
        self._sops_ready_for_sync = bool(self._active_sop_names)

    def _lookup_sop(self, name: str) -> dict[str, str] | None:
        target = name.strip().lower()
        for item in self._sop_catalog:
            sop_name = str(item.get("name", "")).strip()
            if sop_name.lower() == target:
                return item
        return None

    def _sync_active_sops_with_daemon(self, *, notify_on_failure: bool = False) -> bool:
        proc = self.state.daemon.proc
        if not self.state.daemon.ready or proc is None or proc.poll() is not None:
            self._sops_ready_for_sync = bool(self._active_sop_names)
            return False
        for name in sorted(self._active_sop_names):
            record = self._lookup_sop(name)
            if record is None:
                continue
            content = str(record.get("content", "")).strip()
            if not content:
                continue
            payload = {"cmd": "set_sop", "name": name, "content": content}
            if not _transport_send_daemon_command(proc, payload):
                self._sops_ready_for_sync = True
                if notify_on_failure:
                    self._write_transcript_line("[sop] failed to sync active SOPs with daemon.")
                return False
        self._sops_ready_for_sync = False
        return True

    def _set_sop_active(self, name: str, active: bool, *, sync: bool = True, announce: bool = True) -> bool:
        record = self._lookup_sop(name)
        if record is None:
            self._write_transcript_line(f"[sop] unknown SOP: {name}")
            return False
        sop_name = str(record.get("name", "")).strip()
        if not sop_name:
            return False

        changed = False
        if active and sop_name not in self._active_sop_names:
            self._active_sop_names.add(sop_name)
            changed = True
        elif (not active) and sop_name in self._active_sop_names:
            self._active_sop_names.remove(sop_name)
            changed = True

        if not changed:
            return True

        self._refresh_tooling_sops_table()
        self._save_session()
        self._refresh_agent_summary()

        if sync:
            proc = self.state.daemon.proc
            if not self.state.daemon.ready or proc is None or proc.poll() is not None:
                self._sops_ready_for_sync = bool(self._active_sop_names)
            else:
                payload = {
                    "cmd": "set_sop",
                    "name": sop_name,
                    "content": (str(record.get("content", "")).strip() if active else None),
                }
                if not _transport_send_daemon_command(proc, payload):
                    self._sops_ready_for_sync = True
                    self._write_transcript_line("[sop] failed to sync with daemon.")
                else:
                    self._sops_ready_for_sync = False

        if announce:
            status = "activated" if active else "deactivated"
            self._write_transcript_line(f"[sop] {status}: {sop_name}")
        return True

    def _sop_list_lines(self) -> list[str]:
        if not self._sop_catalog:
            return ["[sop] no SOPs found."]
        lines: list[str] = ["[sop] available:"]
        for index, sop in enumerate(self._sop_catalog, start=1):
            name = str(sop.get("name", "")).strip()
            source = str(sop.get("source", "")).strip() or "unknown"
            marker = "✓" if name in self._active_sop_names else " "
            lines.append(f"{index}. [{marker}] {name} ({source})")
        return lines

    def _handle_sop_command(self, argument: str) -> bool:
        from swarmee_river.tui.widgets import render_assistant_message

        raw = (argument or "").strip()
        if not raw:
            self._write_transcript_line(_SOP_USAGE_TEXT)
            return True
        lowered = raw.lower()

        if lowered == "list":
            for line in self._sop_list_lines():
                self._write_transcript_line(line)
            return True

        if lowered.startswith("activate "):
            target = raw.split(maxsplit=1)[1].strip()
            if not target:
                self._write_transcript_line(_SOP_USAGE_TEXT)
                return True
            self._set_sop_active(target, True, sync=True, announce=True)
            return True

        if lowered.startswith("deactivate "):
            target = raw.split(maxsplit=1)[1].strip()
            if not target:
                self._write_transcript_line(_SOP_USAGE_TEXT)
                return True
            self._set_sop_active(target, False, sync=True, announce=True)
            return True

        if lowered.startswith("preview "):
            target = raw.split(maxsplit=1)[1].strip()
            if not target:
                self._write_transcript_line(_SOP_USAGE_TEXT)
                return True
            record = self._lookup_sop(target)
            if record is None:
                self._write_transcript_line(f"[sop] unknown SOP: {target}")
                return True
            name = str(record.get("name", target)).strip()
            source = str(record.get("source", "")).strip() or "unknown"
            content = str(record.get("content", "")).strip()
            if not content:
                self._write_transcript_line(f"[sop] no content available for {name}")
                return True
            markdown = f"# SOP: {name}\n\n[dim]Source: {source}[/dim]\n\n{content}"
            self._mount_transcript_widget(
                render_assistant_message(markdown),
                plain_text=f"SOP: {name}\n\n{content}",
            )
            return True

        self._write_transcript_line(_SOP_USAGE_TEXT)
        return True

    def _set_context_add_mode(self, mode: str | None) -> None:
        from textual.containers import Horizontal
        from textual.widgets import Input

        normalized = (mode or "").strip().lower() or None
        self._context_add_mode = normalized
        input_row = None
        sop_row = None
        with contextlib.suppress(Exception):
            input_row = self.query_one("#context_input_row", Horizontal)
        with contextlib.suppress(Exception):
            sop_row = self.query_one("#context_sop_row", Horizontal)
        input_widget = self._context_input
        if input_widget is None:
            with contextlib.suppress(Exception):
                input_widget = self.query_one("#context_input", Input)
                self._context_input = input_widget

        if normalized in _CONTEXT_INPUT_SOURCE_TYPES:
            if input_row is not None:
                input_row.styles.display = "block"
            if sop_row is not None:
                sop_row.styles.display = "none"
            if input_widget is not None:
                if normalized == "file":
                    input_widget.placeholder = "Enter file path"
                elif normalized == "kb":
                    input_widget.placeholder = "Enter knowledge base ID"
                else:
                    input_widget.placeholder = "Enter context note"
                with contextlib.suppress(Exception):
                    input_widget.value = ""
                    input_widget.focus()
            return

        if normalized == _CONTEXT_SOP_SOURCE_TYPE:
            if input_row is not None:
                input_row.styles.display = "none"
            if sop_row is not None:
                sop_row.styles.display = "block"
            self._refresh_context_sop_options()
            if self._context_sop_select is not None:
                with contextlib.suppress(Exception):
                    self._context_sop_select.focus()
            return

        if input_row is not None:
            input_row.styles.display = "none"
        if sop_row is not None:
            sop_row.styles.display = "none"

    def _context_sources_payload(self) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        for source in self._context_sources:
            source_type = source.get("type", "")
            if source_type == "file":
                path = source.get("path", "").strip()
                if path:
                    payload.append({"type": "file", "path": path, "id": source.get("id", "")})
            elif source_type == "note":
                text = source.get("text", "").strip()
                if text:
                    payload.append({"type": "note", "text": text, "id": source.get("id", "")})
            elif source_type == "sop":
                name = source.get("name", "").strip()
                if name:
                    payload.append({"type": "sop", "name": name, "id": source.get("id", "")})
            elif source_type == "kb":
                kb_id = source.get("id", "").strip()
                if kb_id:
                    payload.append({"type": "kb", "id": kb_id})
            elif source_type == "url":
                url = source.get("url", "").strip()
                if url:
                    payload.append({"type": "url", "url": url, "id": source.get("id", "")})
        return payload

    def _sync_context_sources_with_daemon(self, *, notify_on_failure: bool = False) -> bool:
        proc = self.state.daemon.proc
        if not self.state.daemon.ready or proc is None or proc.poll() is not None:
            self._context_ready_for_sync = True
            return False
        payload = {"cmd": "set_context_sources", "sources": self._context_sources_payload()}
        if _transport_send_daemon_command(proc, payload):
            self._context_ready_for_sync = False
            return True
        self._context_ready_for_sync = True
        if notify_on_failure:
            self._write_transcript_line("[context] failed to sync sources with daemon.")
        return False

    def _set_context_sources(self, sources: list[dict[str, str]], *, sync: bool = True) -> None:
        self._context_sources = _normalize_context_sources(sources)
        self._render_context_sources_panel()
        self._save_session()
        self._refresh_agent_summary()
        if sync:
            self._sync_context_sources_with_daemon(notify_on_failure=True)

    def _add_context_source(self, source: dict[str, str]) -> None:
        normalized = _normalize_context_source(source)
        if normalized is None:
            return
        updated = [*self._context_sources, normalized]
        self._set_context_sources(updated, sync=True)

    def _remove_context_source(self, index: int) -> None:
        if index < 0 or index >= len(self._context_sources):
            self._write_transcript_line(f"[context] invalid index: {index + 1}.")
            return
        updated = [item for i, item in enumerate(self._context_sources) if i != index]
        self._set_context_sources(updated, sync=True)

    def _clear_context_sources(self) -> None:
        self._set_context_sources([], sync=True)

    def _context_list_lines(self) -> list[str]:
        if not self._context_sources:
            return ["[context] no sources attached."]
        lines: list[str] = ["[context] active sources:"]
        for index, source in enumerate(self._context_sources, start=1):
            source_type = source.get("type", "unknown")
            label = self._truncate_context_label(self._context_source_label(source), max_chars=96)
            lines.append(f"{index}. {source_type}: {label}")
        return lines

    def _commit_context_add_from_ui(self) -> None:
        mode = (self._context_add_mode or "").strip().lower()
        if mode in _CONTEXT_INPUT_SOURCE_TYPES:
            if self._context_input is None:
                return
            raw_value = str(getattr(self._context_input, "value", "")).strip()
            if not raw_value:
                self._notify("Context value is required.", severity="warning")
                return
            source: dict[str, str]
            if mode == "file":
                source = {"type": "file", "path": raw_value, "id": _sanitize_context_source_id(uuid.uuid4().hex)}
            elif mode == "kb":
                source = {"type": "kb", "id": raw_value}
            else:
                source = {"type": "note", "text": raw_value, "id": _sanitize_context_source_id(uuid.uuid4().hex)}
            self._add_context_source(source)
            self._set_context_add_mode(None)
            self._write_transcript_line(f"[context] added {mode} source.")
            return

        if mode == _CONTEXT_SOP_SOURCE_TYPE:
            selector = self._context_sop_select
            selected = str(getattr(selector, "value", "")).strip() if selector is not None else ""
            if not selected or selected == _CONTEXT_SELECT_PLACEHOLDER:
                self._notify("Select an SOP first.", severity="warning")
                return
            self._add_context_source(
                {
                    "type": "sop",
                    "name": selected,
                    "id": _sanitize_context_source_id(f"sop-{selected}"),
                }
            )
            self._set_context_add_mode(None)
            self._write_transcript_line(f"[context] added sop source: {selected}")

    def _handle_context_command(self, argument: str) -> bool:
        raw = (argument or "").strip()
        if not raw:
            self._write_transcript_line(_CONTEXT_USAGE_TEXT)
            return True

        lowered = raw.lower()
        if lowered == "list":
            for line in self._context_list_lines():
                self._write_transcript_line(line)
            return True

        if lowered == "clear":
            self._clear_context_sources()
            self._write_transcript_line("[context] cleared all sources.")
            return True

        if lowered.startswith("remove "):
            token = raw.split(maxsplit=1)[1].strip()
            try:
                index = int(token) - 1
            except ValueError:
                self._write_transcript_line("Usage: /context remove <index>")
                return True
            self._remove_context_source(index)
            return True

        add_match = re.match(r"^add\s+(file|note|sop|kb)\s+(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
        if add_match is None:
            self._write_transcript_line(_CONTEXT_USAGE_TEXT)
            return True
        source_type = add_match.group(1).strip().lower()
        value = add_match.group(2).strip()
        if not value:
            self._write_transcript_line(_CONTEXT_USAGE_TEXT)
            return True

        if source_type == "file":
            self._add_context_source(
                {
                    "type": "file",
                    "path": value,
                    "id": _sanitize_context_source_id(uuid.uuid4().hex),
                }
            )
        elif source_type == "note":
            self._add_context_source(
                {
                    "type": "note",
                    "text": value,
                    "id": _sanitize_context_source_id(uuid.uuid4().hex),
                }
            )
        elif source_type == "sop":
            self._add_context_source({"type": "sop", "name": value, "id": _sanitize_context_source_id(f"sop-{value}")})
        elif source_type == "kb":
            self._add_context_source({"type": "kb", "id": value})
        else:
            self._write_transcript_line(_CONTEXT_USAGE_TEXT)
            return True

        self._write_transcript_line(f"[context] added {source_type} source.")
        return True

    def _request_context_compact(self) -> None:
        from swarmee_river.tui.transport import send_daemon_command

        if self.state.daemon.query_active:
            self._write_transcript_line("[compact] unavailable while a run is active.")
            return
        if not self.state.daemon.ready:
            self._write_transcript_line("[compact] daemon is not ready. Use /daemon restart.")
            return
        proc = self.state.daemon.proc
        if proc is None or proc.poll() is not None:
            self.state.daemon.ready = False
            self._write_transcript_line("[compact] daemon is not running. Use /daemon restart.")
            return
        self._notify("Compacting context...", severity="information", timeout=5.0)
        if send_daemon_command(proc, {"cmd": "compact"}):
            self._write_transcript_line("[compact] requested context compaction.")
        else:
            self._write_transcript_line("[compact] failed to send compact command.")
