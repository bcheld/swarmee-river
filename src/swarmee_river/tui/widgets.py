"""Custom Textual widgets for the conversation-style TUI transcript."""

from __future__ import annotations

import contextlib
import json as _json
import re
import textwrap
from typing import Any

from rich import box as rich_box
from rich.console import Group as RichGroup
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel as RichPanel
from rich.text import Text as RichText
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static, TextArea


def render_user_message(text: str, *, timestamp: str | None = None) -> RichPanel:
    """Render a user prompt card with a green left accent bar."""
    body = RichText()
    lines = str(text or "").splitlines() or [""]
    for index, line in enumerate(lines):
        if index:
            body.append("\n")
        body.append("▌ ", style="bold #6a9955")
        body.append(line)
    if isinstance(timestamp, str) and timestamp.strip():
        body.append(f"\n▌ {timestamp.strip()}", style="#6a9955")
    return RichPanel(
        body,
        title="You",
        border_style="#36523d",
        box=rich_box.ROUNDED,
        style="on #111b13",
        padding=(0, 1),
    )


def render_system_message(text: str) -> RichText:
    """Render a dimmed system/info line for RichLog."""
    return RichText(text, style="dim")


def render_thinking_message() -> RichText:
    """Render a compact thinking placeholder."""
    return RichText("thinking...", style="dim")


def render_thinking_indicator(
    *,
    char_count: int = 0,
    preview: str | None = None,
    elapsed_s: float = 0.0,
    frame_index: int = 0,
) -> RichText:
    """Dynamic thinking indicator with optional content preview."""
    frames = (".", "..", "...")
    suffix = frames[frame_index % len(frames)]
    rendered = RichText()
    rendered.append(f"thinking{suffix}", style="bold dim")
    if char_count > 0:
        rendered.append(f" ({char_count:,} chars)", style="dim")
    if elapsed_s > 0:
        rendered.append(f" · {elapsed_s:.0f}s", style="dim")
    if preview:
        truncated = str(preview).replace("\n", " ").strip()
        if truncated:
            if len(truncated) > 60:
                truncated = "…" + truncated[-59:]
            rendered.append(f"\n  ╰ {truncated}", style="dim italic")
    return rendered


def render_assistant_message(
    text: str,
    *,
    model: str | None = None,
    timestamp: str | None = None,
) -> RichMarkdown | RichGroup:
    """Render a finalized assistant response with optional metadata."""
    body = RichMarkdown(text)
    meta_parts: list[str] = []
    if isinstance(model, str) and model.strip():
        meta_parts.append(model.strip())
    if isinstance(timestamp, str) and timestamp.strip():
        meta_parts.append(timestamp.strip())
    if not meta_parts:
        return body
    return RichGroup(body, RichText(" · ".join(meta_parts), style="dim"))


def render_tool_start_line(
    tool_name: str,
    *,
    tool_use_id: str | None = None,
    tool_input: dict | None = None,
) -> RichText:
    """Render a compact tool start line."""
    return render_tool_start_line_with_input(tool_name, tool_use_id=tool_use_id, tool_input=tool_input)


def render_tool_start_line_with_input(
    tool_name: str,
    *,
    tool_input: dict | None = None,
    tool_use_id: str | None = None,
) -> RichText:
    """Render tool start with inline input summary."""
    _ = tool_use_id  # intentionally not shown in default transcript UI
    rendered = RichText()
    rendered.append("⚙ ", style="dim")
    rendered.append(tool_name, style="dim")
    summary = _format_tool_input_oneliner(tool_name, tool_input)
    if summary:
        rendered.append(f" — {summary}", style="dim")
    rendered.append(" ...", style="dim")
    return rendered


def render_tool_result_line(
    tool_name: str,
    *,
    status: str,
    duration_s: float,
    tool_input: dict | None = None,
    tool_use_id: str | None = None,
) -> RichText:
    """Render a compact tool result line with inline input summary."""
    _ = tool_use_id  # intentionally not shown in default transcript UI
    succeeded = status == "success"
    status_glyph = "✓" if succeeded else "✗"
    status_style = "green" if succeeded else "red"
    rendered = RichText()
    rendered.append(status_glyph, style=f"bold {status_style}")
    rendered.append(" ")
    rendered.append(tool_name)
    rendered.append(f" ({duration_s:.1f}s)")
    summary = _format_tool_input_oneliner(tool_name, tool_input)
    if summary:
        rendered.append(" — ", style="dim")
        rendered.append(summary, style="dim")
    if not succeeded:
        label = (status or "error").strip()
        rendered.append(f" ({label})", style="dim red")
    return rendered


def render_tool_progress_chunk(content: str, *, stream: str = "stdout") -> RichText:
    """Render streamed tool output as indented dim lines."""
    text = str(content or "")
    if not text:
        return RichText("", style="dim")
    prefix = "  │ "
    if stream == "stderr":
        prefix = "  │ ! "
    elif stream == "mixed":
        prefix = "  │ ~ "
    lines = text.splitlines()
    if text.endswith("\n"):
        lines.append("")
    rendered = RichText()
    for index, line in enumerate(lines):
        if index:
            rendered.append("\n")
        rendered.append(prefix, style="dim")
        rendered.append(line, style="dim")
    return rendered


def render_tool_heartbeat_line(tool_name: str, *, elapsed_s: float, tool_use_id: str | None = None) -> RichText:
    """Render a lightweight running heartbeat line for long-running tools."""
    _ = tool_use_id  # intentionally not shown in default transcript UI
    return RichText(f"⚙ {tool_name} running... ({elapsed_s:.1f}s)", style="dim")


def render_tool_details_panel(tool_record: dict[str, Any]) -> RichPanel:
    """Render expanded tool details for /expand output."""
    tool_name = str(tool_record.get("tool", "unknown"))
    tool_use_id = str(tool_record.get("tool_use_id", "")).strip()
    status = str(tool_record.get("status", "running"))
    duration_s = float(tool_record.get("duration_s", 0.0) or 0.0)
    tool_input = tool_record.get("input")
    lines = [
        f"tool: {tool_name}",
        f"id: {tool_use_id or '(none)'}",
        f"status: {status}",
        f"duration_s: {duration_s:.3f}",
    ]
    chars = tool_record.get("chars")
    if isinstance(chars, int):
        lines.append(f"chars: {chars}")
    if tool_input is not None:
        try:
            lines.append("")
            lines.append("input:")
            lines.append(_json.dumps(tool_input, indent=2, ensure_ascii=False))
        except Exception:
            lines.append("")
            lines.append("input:")
            lines.append(str(tool_input))
    output_preview = str(tool_record.get("output", "") or "").rstrip()
    if output_preview:
        lines.append("")
        lines.append("output (tail):")
        lines.append(output_preview)
    return RichPanel(RichText("\n".join(lines)), title="Tool Details", border_style="cyan")


def render_consent_panel(context: str, *, options: list[str] | None = None) -> RichPanel:
    """Render consent prompt as a panel."""
    consent_options = options or ["y", "n", "a", "v"]
    option_line = "  ".join(f"[{item}]" for item in consent_options)
    content = RichText()
    content.append(context.strip() if context.strip() else "Consent requested.")
    content.append("\n")
    content.append(option_line, style="bold")
    return RichPanel(content, title="Consent", border_style="yellow")


def render_plan_panel(plan_json: dict[str, Any]) -> RichPanel:
    """Render a plan payload as a panel."""
    return render_plan_panel_with_status(plan_json)


def render_plan_panel_with_status(
    plan_json: dict[str, Any],
    *,
    step_statuses: list[str] | None = None,
) -> RichPanel:
    """Render a plan payload with optional per-step statuses."""
    summary = str(plan_json.get("summary", plan_json.get("title", ""))).strip()
    steps = plan_json.get("steps", [])
    lines: list[str] = []
    if summary:
        lines.append(summary)
        lines.append("")
    if isinstance(steps, list) and steps:
        for index, step in enumerate(steps, start=1):
            if isinstance(step, str):
                desc = step
            elif isinstance(step, dict):
                desc = str(step.get("description", step.get("title", step))).strip()
            else:
                desc = str(step)
            marker = "☐"
            status = (
                step_statuses[index - 1] if isinstance(step_statuses, list) and index - 1 < len(step_statuses) else ""
            )
            if status == "in_progress":
                marker = "▶"
            elif status == "completed":
                marker = "☑"
            lines.append(f"{marker} {index}. {desc}")
    else:
        lines.append("(no steps)")
    return RichPanel(RichText("\n".join(lines)), title="Plan", border_style="green")


def render_agent_profile_summary_text(profile: dict[str, Any] | None) -> str:
    """Render a read-only summary for the currently effective session profile."""
    if not isinstance(profile, dict):
        return "(no effective profile yet)"

    profile_id = str(profile.get("id", "")).strip() or "session-effective"
    name = str(profile.get("name", "")).strip() or "Session Effective"
    provider = str(profile.get("provider", "")).strip() or "(auto)"
    tier = str(profile.get("tier", "")).strip() or "(auto)"
    snippets_raw = profile.get("system_prompt_snippets")
    snippets = snippets_raw if isinstance(snippets_raw, list) else []
    sources_raw = profile.get("context_sources")
    context_sources = sources_raw if isinstance(sources_raw, list) else []
    sops_raw = profile.get("active_sops")
    active_sops = [str(item).strip() for item in sops_raw] if isinstance(sops_raw, list) else []
    active_sops = [item for item in active_sops if item]
    kb_id = str(profile.get("knowledge_base_id", "")).strip() or "(none)"
    agents_raw = profile.get("agents")
    agents = agents_raw if isinstance(agents_raw, list) else []
    activated_agents = [
        item
        for item in agents
        if isinstance(item, dict) and bool(item.get("activated")) and str(item.get("name", "")).strip()
    ]
    auto_delegate = bool(profile.get("auto_delegate_assistive", True))

    lines = [
        f"Name: {name}",
        f"ID: {profile_id}",
        f"Model: {provider}/{tier}",
        f"KB: {kb_id}",
        f"Assistive delegation: {'on' if auto_delegate else 'off'}",
        f"Activated agents: {len(activated_agents)}",
        "",
        f"System snippets ({len(snippets)}):",
    ]
    if snippets:
        for snippet in snippets[:3]:
            one_line = _truncate_single_line(str(snippet), max_len=96)
            lines.append(f"- {one_line}")
        if len(snippets) > 3:
            lines.append(f"- ... +{len(snippets) - 3} more")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append(f"Context sources ({len(context_sources)}):")
    if context_sources:
        for source in context_sources[:5]:
            if not isinstance(source, dict):
                continue
            source_type = str(source.get("type", "unknown")).strip().lower() or "unknown"
            label = (
                str(source.get("path", "")).strip()
                or str(source.get("text", "")).strip()
                or str(source.get("name", "")).strip()
                or str(source.get("url", "")).strip()
                or str(source.get("id", "")).strip()
            )
            lines.append(f"- {source_type}: {_truncate_single_line(label, max_len=88)}")
        if len(context_sources) > 5:
            lines.append(f"- ... +{len(context_sources) - 5} more")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append(f"Activated agent roster ({len(activated_agents)}):")
    if activated_agents:
        for agent in activated_agents[:5]:
            name = str(agent.get("name", "")).strip() or str(agent.get("id", "")).strip() or "agent"
            summary = _truncate_single_line(str(agent.get("summary", "")).strip() or "(no summary)", max_len=72)
            lines.append(f"- {name}: {summary}")
        if len(activated_agents) > 5:
            lines.append(f"- ... +{len(activated_agents) - 5} more")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append(f"Active SOPs ({len(active_sops)}):")
    if active_sops:
        for sop_name in active_sops[:8]:
            lines.append(f"- {sop_name}")
        if len(active_sops) > 8:
            lines.append(f"- ... +{len(active_sops) - 8} more")
    else:
        lines.append("- (none)")

    return "\n".join(lines)


_CONSENT_TOOL_RE = re.compile(r"allow tool ['\"](?P<tool>[^'\"]+)['\"]", re.IGNORECASE)


def extract_consent_tool_name(context: str) -> str:
    """Best-effort extraction of tool name from consent context text."""
    match = _CONSENT_TOOL_RE.search(context)
    if match:
        name = match.group("tool").strip()
        if name:
            return name
    lowered = context.lower()
    if "shell" in lowered:
        return "shell"
    return "tool"


class ConsentPrompt(Vertical):
    """Interactive inline consent prompt with context and clickable choices."""

    DEFAULT_CSS = """
    ConsentPrompt {
        display: none;
        border: round yellow;
        background: $panel;
        padding: 0 1;
        margin: 0 0 1 0;
        height: auto;
        max-height: 7;
    }
    ConsentPrompt.-highlight {
        border: heavy yellow;
    }
    ConsentPrompt #consent_context {
        height: auto;
        color: $text;
    }
    ConsentPrompt #consent_separator {
        height: auto;
        color: $text-muted;
        margin: 0;
        padding: 0;
    }
    ConsentPrompt #consent_actions {
        layout: horizontal;
        height: auto;
        margin: 0;
        padding: 0;
    }
    ConsentPrompt #consent_actions Button {
        width: 1fr;
        min-width: 11;
        margin: 0 1 0 0;
    }
    """

    _CHOICE_IDS = ("y", "n", "a", "v")
    _highlight_timer: Any = None

    def compose(self):  # type: ignore[override]
        yield Static("", id="consent_context")
        yield Static("", id="consent_separator")
        with Horizontal(id="consent_actions"):
            yield Button("Yes (y)", id="consent_choice_y", variant="success", compact=True)
            yield Button("No (n)", id="consent_choice_n", variant="error", compact=True)
            yield Button("Always (a)", id="consent_choice_a", variant="primary", compact=True)
            yield Button("Never (v)", id="consent_choice_v", variant="warning", compact=True)

    def set_prompt(self, context: str, *, options: list[str] | None = None, alert: bool = True) -> None:
        """Show context + enabled options and reveal the prompt."""
        available = {opt.strip().lower() for opt in (options or self._CHOICE_IDS) if opt.strip()}
        if not available:
            available = set(self._CHOICE_IDS)

        tool_name = extract_consent_tool_name(context)
        summary = self._first_context_line(context)
        rich_context = RichText()
        rich_context.append("Consent required: ", style="bold yellow")
        rich_context.append(tool_name, style="bold")
        if summary:
            rich_context.append(" — ", style="bold yellow")
            rich_context.append("Command: ", style="bold")
            rich_context.append(summary, style="dim")
        self.query_one("#consent_context", Static).update(rich_context)
        self.query_one("#consent_separator", Static).update(RichText("─" * 28, style="dim"))
        self.query_one("#consent_actions", Horizontal).styles.display = "block"

        for choice in self._CHOICE_IDS:
            with contextlib.suppress(Exception):
                button = self.query_one(f"#consent_choice_{choice}", Button)
                is_available = choice in available
                button.disabled = not is_available
                button.styles.display = "block" if is_available else "none"

        self.styles.display = "block"
        if alert:
            self._signal_attention()
            self._flash_highlight()
        self.focus_first_choice()

    def hide_prompt(self) -> None:
        self._clear_highlight()
        self.styles.display = "none"
        with contextlib.suppress(Exception):
            self.query_one("#consent_actions", Horizontal).styles.display = "block"

    def focus_first_choice(self) -> None:
        for choice in self._CHOICE_IDS:
            with contextlib.suppress(Exception):
                button = self.query_one(f"#consent_choice_{choice}", Button)
                if not button.disabled and str(button.styles.display) != "none":
                    button.focus()
                    return

    def show_confirmation(self, message: str, *, approved: bool) -> None:
        """Show a compact post-choice confirmation before the prompt is hidden."""
        self._clear_highlight()
        status_style = "bold green" if approved else "bold red"
        self.query_one("#consent_context", Static).update(RichText(message, style=status_style))
        self.query_one("#consent_separator", Static).update("")
        with contextlib.suppress(Exception):
            self.query_one("#consent_actions", Horizontal).styles.display = "none"
        self.styles.display = "block"

    def _first_context_line(self, context: str) -> str:
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        if not lines:
            return ""
        for line in lines:
            lower = line.lower()
            if "allow tool" in lower or "consent" in lower or line.startswith("~"):
                continue
            return line[:120]
        return ""

    def _signal_attention(self) -> None:
        app = getattr(self, "app", None)
        if app is None:
            return
        with contextlib.suppress(Exception):
            app.notify("Tool consent required", severity="warning", timeout=10)
        with contextlib.suppress(Exception):
            app.bell()

    def _flash_highlight(self) -> None:
        timer = self._highlight_timer
        self._highlight_timer = None
        if timer is not None:
            with contextlib.suppress(Exception):
                timer.stop()
        self.add_class("-highlight")
        self._highlight_timer = self.set_timer(2.0, self._clear_highlight)

    def _clear_highlight(self) -> None:
        timer = self._highlight_timer
        self._highlight_timer = None
        if timer is not None:
            with contextlib.suppress(Exception):
                timer.stop()
        self.remove_class("-highlight")

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key not in {"left", "right", "up", "down"}:
            return
        buttons: list[Button] = []
        for choice in self._CHOICE_IDS:
            with contextlib.suppress(Exception):
                button = self.query_one(f"#consent_choice_{choice}", Button)
                if not button.disabled and str(button.styles.display) != "none":
                    buttons.append(button)
        if not buttons:
            return
        current = getattr(self.app, "focused", None)
        if current not in buttons:
            buttons[0].focus()
            event.stop()
            event.prevent_default()
            return
        idx = buttons.index(current)
        delta = -1 if key in {"left", "up"} else 1
        buttons[(idx + delta) % len(buttons)].focus()
        event.stop()
        event.prevent_default()


class ErrorActionPrompt(Vertical):
    """Inline error recovery actions (tool retry/skip, model escalation, continue)."""

    DEFAULT_CSS = """
    ErrorActionPrompt {
        display: none;
        border: round red;
        background: $panel;
        padding: 0 1;
        margin: 0 0 1 0;
        height: auto;
        max-height: 4;
    }
    ErrorActionPrompt #error_action_context {
        height: 2;
        color: $text;
    }
    ErrorActionPrompt #error_action_buttons {
        layout: horizontal;
        height: 1;
        margin: 0;
        padding: 0;
    }
    ErrorActionPrompt #error_action_buttons Button {
        width: 1fr;
        min-width: 10;
        margin: 0 1 0 0;
    }
    """

    _BUTTON_IDS = (
        "error_action_retry_tool",
        "error_action_skip_tool",
        "error_action_escalate",
        "error_action_continue",
    )

    def compose(self):  # type: ignore[override]
        yield Static("", id="error_action_context")
        with Horizontal(id="error_action_buttons"):
            yield Button("Retry", id="error_action_retry_tool", variant="primary")
            yield Button("Skip", id="error_action_skip_tool", variant="warning")
            yield Button("Escalate", id="error_action_escalate", variant="success")
            yield Button("Continue", id="error_action_continue", variant="default")

    def show_tool_error(self, *, tool_name: str, tool_use_id: str) -> None:
        context = RichText()
        context.append("Tool error: ", style="bold red")
        context.append(f"{tool_name} [{tool_use_id}]")
        context.append("\nRetry the tool call or skip it and continue.", style="dim")
        self.query_one("#error_action_context", Static).update(context)
        self._show_buttons({"error_action_retry_tool", "error_action_skip_tool"})
        self.styles.display = "block"
        self.focus_first_choice()

    def show_escalation(self, *, next_tier: str | None) -> None:
        context = RichText()
        context.append("Execution blocked by model/context limits.", style="bold yellow")
        if isinstance(next_tier, str) and next_tier.strip():
            context.append(f"\nEscalate to {next_tier.strip().lower()} or continue as-is.", style="dim")
            self.query_one("#error_action_escalate", Button).label = f"Escalate ({next_tier.strip().lower()})"
            self._show_buttons({"error_action_escalate", "error_action_continue"})
        else:
            context.append("\nNo higher tier available. Continue on current tier.", style="dim")
            self._show_buttons({"error_action_continue"})
        self.query_one("#error_action_context", Static).update(context)
        self.styles.display = "block"
        self.focus_first_choice()

    def hide_prompt(self) -> None:
        self.styles.display = "none"

    def focus_first_choice(self) -> None:
        for button_id in self._BUTTON_IDS:
            with contextlib.suppress(Exception):
                button = self.query_one(f"#{button_id}", Button)
                if not button.disabled and str(button.styles.display) != "none":
                    button.focus()
                    return

    def _show_buttons(self, visible: set[str]) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#error_action_escalate", Button).label = "Escalate"
        for button_id in self._BUTTON_IDS:
            with contextlib.suppress(Exception):
                button = self.query_one(f"#{button_id}", Button)
                show = button_id in visible
                button.disabled = not show
                button.styles.display = "block" if show else "none"


class SidebarHeader(Horizontal):
    """Compact header primitive with title, optional badges, and optional actions."""

    DEFAULT_CSS = """
    SidebarHeader {
        height: auto;
        layout: horizontal;
        margin: 0 0 1 0;
    }
    SidebarHeader .sidebar-header-title {
        width: 1fr;
        color: $text;
    }
    SidebarHeader .sidebar-header-badge {
        width: auto;
        color: $text-muted;
        margin: 0 1 0 0;
    }
    SidebarHeader .sidebar-header-action {
        width: auto;
        min-width: 8;
        margin: 0 0 0 1;
    }
    """

    def __init__(
        self,
        title: str,
        *,
        badges: list[str] | None = None,
        actions: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._title = str(title or "").strip()
        self._badges = [str(item).strip() for item in (badges or []) if str(item).strip()]
        self._actions = [dict(item) for item in (actions or []) if isinstance(item, dict)]

    def compose(self):  # type: ignore[override]
        yield Static(self._title, classes="sidebar-header-title")
        for badge in self._badges:
            yield Static(badge, classes="sidebar-header-badge")
        for action in self._actions:
            action_id = str(action.get("id", "")).strip()
            if not action_id:
                continue
            label = str(action.get("label", action_id)).strip() or action_id
            variant = str(action.get("variant", "default")).strip() or "default"
            compact = bool(action.get("compact", True))
            yield Button(
                label,
                id=action_id,
                variant=variant,
                compact=compact,
                classes="sidebar-header-action",
            )

    def set_title(self, title: str) -> None:
        self._title = str(title or "").strip()
        with contextlib.suppress(Exception):
            self.query_one(".sidebar-header-title", Static).update(self._title)

    def set_badges(self, badges: list[str] | None) -> None:
        self._badges = [str(item).strip() for item in (badges or []) if str(item).strip()]
        with contextlib.suppress(Exception):
            self.refresh(recompose=True)

    def set_actions(self, actions: list[dict[str, Any]] | None) -> None:
        self._actions = [dict(item) for item in (actions or []) if isinstance(item, dict)]
        with contextlib.suppress(Exception):
            self.refresh(recompose=True)


class SidebarListItem(Static):
    """Selectable sidebar list row with title/subtitle and visual state."""

    DEFAULT_CSS = """
    SidebarListItem {
        height: auto;
        padding: 0 1;
        margin: 0;
        color: $text;
    }
    SidebarListItem.-selected {
        background: $accent 20%;
    }
    SidebarListItem.-state-default {
        color: $text;
    }
    SidebarListItem.-state-active {
        color: green;
    }
    SidebarListItem.-state-warning {
        color: yellow;
    }
    SidebarListItem.-state-error {
        color: red;
    }
    SidebarListItem.-state-syncing {
        color: $text-muted;
    }
    """

    class Pressed(Message):
        def __init__(self, item: SidebarListItem, *, item_id: str) -> None:
            super().__init__()
            self.item = item
            self.item_id = item_id

    _ALLOWED_STATES = {"default", "active", "warning", "error", "syncing"}
    _STATE_CLASS_MAP = {
        "default": "-state-default",
        "active": "-state-active",
        "warning": "-state-warning",
        "error": "-state-error",
        "syncing": "-state-syncing",
    }

    def __init__(
        self,
        *,
        item_id: str,
        title: str,
        subtitle: str | None = None,
        state: str = "default",
        **kwargs: object,
    ) -> None:
        super().__init__("", **kwargs)
        self._item_id = str(item_id).strip()
        self._title = str(title).strip()
        self._subtitle = str(subtitle or "").strip()
        self._state = self._normalize_state(state)
        self._selected = False
        self.can_focus = True

    @property
    def item_id(self) -> str:
        return self._item_id

    @property
    def state(self) -> str:
        return self._state

    def on_mount(self) -> None:
        self._render_text()
        self._apply_classes()

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self._apply_classes()

    def set_state(self, state: str) -> None:
        self._state = self._normalize_state(state)
        self._apply_classes()

    def set_text(self, *, title: str, subtitle: str | None = None) -> None:
        self._title = str(title).strip()
        self._subtitle = str(subtitle or "").strip()
        self._render_text()

    def on_click(self, _event: Any) -> None:
        self.post_message(self.Pressed(self, item_id=self._item_id))

    def _normalize_state(self, state: str) -> str:
        normalized = str(state or "").strip().lower()
        return normalized if normalized in self._ALLOWED_STATES else "default"

    def _apply_classes(self) -> None:
        for class_name in self._STATE_CLASS_MAP.values():
            with contextlib.suppress(Exception):
                self.remove_class(class_name)
        with contextlib.suppress(Exception):
            self.add_class(self._STATE_CLASS_MAP[self._state])
        if self._selected:
            with contextlib.suppress(Exception):
                self.add_class("-selected")
        else:
            with contextlib.suppress(Exception):
                self.remove_class("-selected")

    def _render_text(self) -> None:
        rendered = RichText(self._title)
        if self._subtitle:
            rendered.append(f"\n{self._subtitle}", style="dim")
        self.update(rendered)


class SidebarList(Vertical):
    """Reusable list primitive with keyboard navigation and selection events."""

    DEFAULT_CSS = """
    SidebarList {
        border: round #3b3b3b;
        padding: 0;
        height: 1fr;
    }
    SidebarList #sidebar_list_scroll {
        height: 1fr;
    }
    SidebarList .sidebar-list-empty {
        color: $text-muted;
        padding: 0 1;
    }
    """

    class SelectionChanged(Message):
        def __init__(self, sidebar_list: SidebarList, *, item: dict[str, str]) -> None:
            super().__init__()
            self.sidebar_list = sidebar_list
            self.item = item
            self.item_id = str(item.get("id", "")).strip()

    def __init__(self, *, items: list[dict[str, Any]] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._items: list[dict[str, str]] = []
        self._selected_index: int = -1
        self.can_focus = True
        self.set_items(items or [], emit=False)

    def compose(self):  # type: ignore[override]
        yield VerticalScroll(id="sidebar_list_scroll")

    def set_items(
        self,
        items: list[dict[str, Any]],
        *,
        selected_id: str | None = None,
        emit: bool = False,
    ) -> None:
        normalized: list[dict[str, str]] = []
        for raw_item in items:
            item_id = str((raw_item or {}).get("id", "")).strip()
            title = str((raw_item or {}).get("title", "")).strip()
            if not item_id or not title:
                continue
            subtitle = str((raw_item or {}).get("subtitle", "")).strip()
            state = str((raw_item or {}).get("state", "default")).strip().lower() or "default"
            normalized.append(
                {
                    "id": item_id,
                    "title": title,
                    "subtitle": subtitle,
                    "state": state,
                }
            )

        self._items = normalized
        self._selected_index = -1
        if self._items:
            if selected_id:
                for index, item in enumerate(self._items):
                    if item["id"] == selected_id:
                        self._selected_index = index
                        break
            if self._selected_index < 0:
                self._selected_index = 0

        self._render_items()
        self._sync_item_selection_styles()
        self._emit_selection_changed(enabled=emit)

    def selected_item(self) -> dict[str, str] | None:
        if self._selected_index < 0 or self._selected_index >= len(self._items):
            return None
        return dict(self._items[self._selected_index])

    def selected_id(self) -> str | None:
        item = self.selected_item()
        if item is None:
            return None
        selected_id = str(item.get("id", "")).strip()
        return selected_id or None

    def select_by_id(self, item_id: str, *, emit: bool = True) -> bool:
        target = str(item_id or "").strip()
        if not target:
            return False
        for index, item in enumerate(self._items):
            if item.get("id") != target:
                continue
            self._selected_index = index
            self._sync_item_selection_styles()
            self._emit_selection_changed(enabled=emit)
            return True
        return False

    def move_selection(self, delta: int, *, emit: bool = True) -> None:
        if not self._items:
            return
        if self._selected_index < 0:
            self._selected_index = 0
        next_index = self._selected_index + int(delta)
        next_index = min(len(self._items) - 1, max(0, next_index))
        if next_index == self._selected_index:
            return
        self._selected_index = next_index
        self._sync_item_selection_styles()
        self._emit_selection_changed(enabled=emit)

    def on_focus(self) -> None:
        if self._items and self._selected_index < 0:
            self._selected_index = 0
            self._sync_item_selection_styles()

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).strip().lower()
        if key == "up":
            event.stop()
            event.prevent_default()
            self.move_selection(-1, emit=True)
            return
        if key == "down":
            event.stop()
            event.prevent_default()
            self.move_selection(1, emit=True)
            return
        if key == "home":
            event.stop()
            event.prevent_default()
            if self._items:
                self._selected_index = 0
                self._sync_item_selection_styles()
                self._emit_selection_changed(enabled=True)
            return
        if key == "end":
            event.stop()
            event.prevent_default()
            if self._items:
                self._selected_index = len(self._items) - 1
                self._sync_item_selection_styles()
                self._emit_selection_changed(enabled=True)
            return
        if key in {"enter", "return"}:
            event.stop()
            event.prevent_default()
            self._emit_selection_changed(enabled=True)

    def on_sidebar_list_item_pressed(self, event: SidebarListItem.Pressed) -> None:
        if self.select_by_id(event.item_id, emit=True):
            event.stop()

    def _emit_selection_changed(self, *, enabled: bool) -> None:
        if not enabled:
            return
        item = self.selected_item()
        if item is None:
            return
        self.post_message(self.SelectionChanged(self, item=item))

    def _render_items(self) -> None:
        if not getattr(self, "is_mounted", False):
            return
        container = self.query_one("#sidebar_list_scroll", VerticalScroll)
        for child in list(container.children):
            with contextlib.suppress(Exception):
                child.remove()

        if not self._items:
            container.mount(Static("(no items)", classes="sidebar-list-empty"))
            return

        for index, item in enumerate(self._items):
            entry = SidebarListItem(
                item_id=item["id"],
                title=item["title"],
                subtitle=item.get("subtitle", ""),
                state=item.get("state", "default"),
            )
            entry.set_selected(index == self._selected_index)
            container.mount(entry)

    def _sync_item_selection_styles(self) -> None:
        if not getattr(self, "is_mounted", False):
            return
        container = self.query_one("#sidebar_list_scroll", VerticalScroll)
        item_widgets = [child for child in container.children if isinstance(child, SidebarListItem)]
        for index, widget in enumerate(item_widgets):
            widget.set_selected(index == self._selected_index)
        if 0 <= self._selected_index < len(item_widgets):
            with contextlib.suppress(Exception):
                container.scroll_to_widget(item_widgets[self._selected_index], animate=False)


class SidebarDetail(Vertical):
    """Reusable detail panel with scrollable preview text and action buttons."""

    DEFAULT_CSS = """
    SidebarDetail {
        border: round #3b3b3b;
        padding: 0 1;
        height: 1fr;
    }
    SidebarDetail .sidebar-detail-scroll {
        height: 1fr;
        scrollbar-background: #2f2f2f;
        scrollbar-background-hover: #3a3a3a;
        scrollbar-background-active: #454545;
        scrollbar-color: #7f7f7f;
        scrollbar-color-hover: #999999;
        scrollbar-color-active: #b3b3b3;
    }
    SidebarDetail .sidebar-detail-preview {
        height: auto;
        color: $text;
    }
    SidebarDetail .sidebar-detail-actions {
        layout: horizontal;
        height: auto;
        margin: 1 0 0 0;
    }
    SidebarDetail .sidebar-detail-actions Button {
        width: 1fr;
        min-width: 8;
        margin: 0 1 0 0;
    }
    """

    class ActionSelected(Message):
        def __init__(self, detail: SidebarDetail, *, action_id: str) -> None:
            super().__init__()
            self.detail = detail
            self.action_id = action_id

    def __init__(
        self,
        *,
        preview: str = "",
        actions: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._preview = str(preview or "").strip()
        self._actions = [dict(item) for item in (actions or []) if isinstance(item, dict)]
        self._action_button_ids: dict[str, str] = {}

    def compose(self):  # type: ignore[override]
        text = self._preview if self._preview else "(no selection)"
        with VerticalScroll(classes="sidebar-detail-scroll"):
            yield Static(text, classes="sidebar-detail-preview")
        with Horizontal(classes="sidebar-detail-actions"):
            self._action_button_ids = {}
            for index, action in enumerate(self._actions):
                action_id = str(action.get("id", "")).strip()
                if not action_id:
                    continue
                button_id = f"sidebar_detail_action_{index}"
                self._action_button_ids[button_id] = action_id
                label = str(action.get("label", action_id)).strip() or action_id
                variant = str(action.get("variant", "default")).strip() or "default"
                compact = bool(action.get("compact", True))
                yield Button(label, id=button_id, variant=variant, compact=compact)

    def set_preview(self, preview: str) -> None:
        self._preview = str(preview or "").strip()
        text = self._preview if self._preview else "(no selection)"
        with contextlib.suppress(Exception):
            self.query_one(".sidebar-detail-preview", Static).update(text)

    def set_actions(self, actions: list[dict[str, Any]] | None) -> None:
        self._actions = [dict(item) for item in (actions or []) if isinstance(item, dict)]
        with contextlib.suppress(Exception):
            self.refresh(recompose=True)

    def on_button_pressed(self, event: Any) -> None:
        button_id = str(getattr(getattr(event, "button", None), "id", "")).strip()
        action_id = self._action_button_ids.get(button_id)
        if not action_id:
            return
        event.stop()
        self.post_message(self.ActionSelected(self, action_id=action_id))


class UserMessage(Static):
    """Displays a user prompt with distinct styling."""

    DEFAULT_CSS = """
    UserMessage {
        background: $surface;
        color: $text;
        padding: 0 1;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, text: str, timestamp: str | None = None, **kwargs: object) -> None:
        render_text = f"[bold cyan]YOU>[/bold cyan] {text}"
        if isinstance(timestamp, str) and timestamp.strip():
            render_text = f"{render_text}\n[dim]{timestamp.strip()}[/dim]"
        super().__init__(render_text, **kwargs)


class AssistantMessage(Static):
    """Accumulates text_delta events and renders as markdown via Rich."""

    DEFAULT_CSS = """
    AssistantMessage {
        padding: 0 1;
        margin: 0 0 0 0;
    }
    """

    def __init__(self, model: str | None = None, timestamp: str | None = None, **kwargs: object) -> None:
        super().__init__("", **kwargs)
        self._buffer: list[str] = []
        self._model = model.strip() if isinstance(model, str) and model.strip() else ""
        self._timestamp = timestamp.strip() if isinstance(timestamp, str) and timestamp.strip() else ""

    def append_delta(self, text: str) -> None:
        self._buffer.append(text)
        full = "".join(self._buffer)
        self.update(RichMarkdown(full))

    def finalize(self, *, model: str | None = None, timestamp: str | None = None) -> str:
        """Called on text_complete. Returns the full raw text."""
        if isinstance(model, str) and model.strip():
            self._model = model.strip()
        if isinstance(timestamp, str) and timestamp.strip():
            self._timestamp = timestamp.strip()
        full = "".join(self._buffer)
        meta_parts = [part for part in [self._model, self._timestamp] if part]
        if meta_parts:
            meta_line = " · ".join(meta_parts)
            self.update(RichGroup(RichMarkdown(full), RichText(meta_line, style="dim")))
        return full

    @property
    def full_text(self) -> str:
        return "".join(self._buffer)


class AssistantStreamBlock(Static):
    """Collapsible assistant response card for live streaming output."""

    DEFAULT_CSS = """
    AssistantStreamBlock {
        margin: 1 0 0 0;
        padding: 0;
        background: #141414;
        border-left: heavy #6a9955;
    }
    AssistantStreamBlock Collapsible {
        margin: 0;
        padding: 0 1;
    }
    AssistantStreamBlock .assistant-stream-body {
        padding: 0 0 0 1;
    }
    """

    def __init__(self, *, model: str | None = None, timestamp: str | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._buffer: list[str] = []
        self._model = model.strip() if isinstance(model, str) and model.strip() else ""
        self._timestamp = timestamp.strip() if isinstance(timestamp, str) and timestamp.strip() else ""

    def compose(self):  # type: ignore[override]
        with Collapsible(title=self._header_text(running=True), collapsed=False):
            yield Static(RichText("Waiting for response...", style="dim"), classes="assistant-stream-body")

    def _header_text(self, *, running: bool) -> str:
        if running:
            return "Assistant response (streaming...)"
        summary = self._summary()
        meta = self._meta_label()
        title = f"Assistant response{f' — {summary}' if summary else ''}"
        if meta:
            title = f"{title} · {meta}"
        return title

    def _summary(self) -> str:
        full = "".join(self._buffer).strip()
        if not full:
            return "(no text)"
        one_line = re.sub(r"\s+", " ", full)
        return one_line if len(one_line) <= 80 else f"{one_line[:79].rstrip()}…"

    def _meta_label(self) -> str:
        parts = [part for part in [self._model, self._timestamp] if part]
        return " · ".join(parts)

    def _refresh_header(self, *, running: bool) -> None:
        with contextlib.suppress(Exception):
            collapsible = self.query_one(Collapsible)
            collapsible.title = self._header_text(running=running)

    def _refresh_body(self) -> None:
        body = "".join(self._buffer)
        with contextlib.suppress(Exception):
            target = self.query_one(".assistant-stream-body", Static)
            if body.strip():
                target.update(RichMarkdown(body))
            else:
                target.update(RichText("Waiting for response...", style="dim"))

    def append_delta(self, text: str) -> None:
        chunk = str(text or "")
        if not chunk:
            return
        self._buffer.append(chunk)
        self._refresh_header(running=True)
        self._refresh_body()
        with contextlib.suppress(Exception):
            self.query_one(Collapsible).collapsed = False

    def finalize(self, *, model: str | None = None, timestamp: str | None = None) -> str:
        if isinstance(model, str) and model.strip():
            self._model = model.strip()
        if isinstance(timestamp, str) and timestamp.strip():
            self._timestamp = timestamp.strip()
        self._refresh_body()
        self._refresh_header(running=False)
        with contextlib.suppress(Exception):
            self.query_one(Collapsible).collapsed = True
        return "".join(self._buffer)

    def collapse(self) -> None:
        self._refresh_header(running=False)
        with contextlib.suppress(Exception):
            self.query_one(Collapsible).collapsed = True

    @property
    def full_text(self) -> str:
        return "".join(self._buffer)


class ReasoningBlock(Static):
    """Collapsible card for model reasoning/thinking output."""

    DEFAULT_CSS = """
    ReasoningBlock {
        margin: 1 0 0 0;
        padding: 0;
        background: #141414;
        border-left: heavy #6a9955;
    }
    ReasoningBlock Collapsible {
        margin: 0;
        padding: 0 1;
    }
    ReasoningBlock .reasoning-body {
        padding: 0 0 0 1;
    }
    """

    def __init__(self, *, timestamp: str | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._buffer: list[str] = []
        self._timestamp = timestamp.strip() if isinstance(timestamp, str) and timestamp.strip() else ""
        self._elapsed_s: float = 0.0

    def compose(self):  # type: ignore[override]
        with Collapsible(title=self._header_text(running=True), collapsed=False):
            yield Static(RichText("Thinking...", style="dim"), classes="reasoning-body")

    def _header_text(self, *, running: bool) -> str:
        if running:
            return "Reasoning (streaming...)"
        chars = sum(len(chunk) for chunk in self._buffer)
        title = f"Reasoning ({chars:,} chars, {self._elapsed_s:.1f}s)"
        if self._timestamp:
            title = f"{title} · {self._timestamp}"
        return title

    def _refresh_header(self, *, running: bool) -> None:
        with contextlib.suppress(Exception):
            self.query_one(Collapsible).title = self._header_text(running=running)

    def _refresh_body(self) -> None:
        text = "".join(self._buffer)
        with contextlib.suppress(Exception):
            body = self.query_one(".reasoning-body", Static)
            if text.strip():
                body.update(RichMarkdown(text))
            else:
                body.update(RichText("Thinking...", style="dim"))

    def append_delta(self, text: str) -> None:
        chunk = str(text or "")
        if not chunk:
            return
        self._buffer.append(chunk)
        self._refresh_body()
        self._refresh_header(running=True)
        with contextlib.suppress(Exception):
            self.query_one(Collapsible).collapsed = False

    def finalize(self, *, elapsed_s: float | None = None) -> str:
        if isinstance(elapsed_s, (int, float)):
            self._elapsed_s = max(0.0, float(elapsed_s))
        self._refresh_body()
        self._refresh_header(running=False)
        with contextlib.suppress(Exception):
            self.query_one(Collapsible).collapsed = True
        return "".join(self._buffer)

    def collapse(self) -> None:
        self._refresh_header(running=False)
        with contextlib.suppress(Exception):
            self.query_one(Collapsible).collapsed = True


def _format_tool_input(tool_name: str, tool_input: dict) -> str:
    """Format tool input for display in a collapsible body."""
    if not tool_input:
        return ""
    if tool_name == "shell":
        lines = []
        if "command" in tool_input:
            lines.append(f"Command: {tool_input['command']}")
        if "cwd" in tool_input:
            lines.append(f"CWD: {tool_input['cwd']}")
        return "\n".join(lines) if lines else _json.dumps(tool_input, indent=2)
    if tool_name in ("file_read", "read"):
        if "path" in tool_input:
            return f"Path: {tool_input['path']}"
    if tool_name in ("file_write", "write", "file_edit", "edit"):
        path = tool_input.get("path", "")
        if path:
            return f"Path: {path}"
    return _json.dumps(tool_input, indent=2)


def _truncate_single_line(text: str, *, max_len: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max(0, max_len - 1)].rstrip() + "…"


def _format_tool_input_oneliner(tool_name: str, tool_input: dict | None) -> str:
    """Format tool input as a compact one-line summary."""
    if not isinstance(tool_input, dict) or not tool_input:
        return ""

    canonical = str(tool_name or "").strip().lower()

    if canonical in {"shell", "bash", "python_repl"}:
        command = str(tool_input.get("command", tool_input.get("cmd", ""))).strip()
        if command:
            return _truncate_single_line(f"$ {command[:80]}", max_len=100)

    if canonical in {"file_read", "read"}:
        path = str(tool_input.get("path", "")).strip()
        if path:
            return _truncate_single_line(f"← {path}", max_len=100)

    if canonical in {"file_write", "write"}:
        path = str(tool_input.get("path", "")).strip()
        if path:
            return _truncate_single_line(f"→ {path}", max_len=100)

    if canonical in {"editor", "edit", "file_edit"}:
        path = str(tool_input.get("path", tool_input.get("file_path", ""))).strip()
        if path:
            return _truncate_single_line(f"✎ {path}", max_len=100)

    if canonical == "http_request":
        method = str(tool_input.get("method", "GET")).strip().upper() or "GET"
        url = str(tool_input.get("url", "")).strip()
        if url:
            return _truncate_single_line(f"{method} {url[:60]}", max_len=100)

    if canonical in {"glob", "file_search", "file_list"}:
        pattern = str(tool_input.get("pattern", tool_input.get("query", ""))).strip()
        if pattern:
            return _truncate_single_line(pattern[:60], max_len=100)

    if canonical == "retrieve":
        query = str(tool_input.get("query", "")).strip()
        if query:
            return _truncate_single_line(f"? {query[:60]}", max_len=100)

    for key, value in tool_input.items():
        val_str = str(value).strip()
        if val_str and len(val_str) < 80:
            return _truncate_single_line(f"{key}: {val_str[:60]}", max_len=100)
        break
    return ""


def format_tool_input_oneliner(tool_name: str, tool_input: dict | None) -> str:
    """Public wrapper for compact tool input summaries."""
    return _format_tool_input_oneliner(tool_name, tool_input)


class ToolCallBlock(Static):
    """Collapsible tool call block: shows header with status, expands to show input details."""

    DEFAULT_CSS = """
    ToolCallBlock {
        color: $text-muted;
        padding: 0;
        margin: 1 0 0 0;
        background: #141414;
        border-left: heavy #6a9955;
    }
    ToolCallBlock Collapsible {
        padding: 0 1;
        margin: 0;
        border: none;
    }
    ToolCallBlock .tool-details {
        color: $text-muted;
        padding: 0 0 0 1;
    }
    """

    def __init__(self, tool_name: str, tool_use_id: str, **kwargs: object) -> None:
        self._tool_name = tool_name
        self._tool_use_id = tool_use_id
        self._tool_input: dict = {}
        self._status_text = "running..."
        self._result_text: str | None = None
        self._output_preview: str = ""
        self._elapsed_s: float = 0.0
        super().__init__(**kwargs)

    def compose(self):  # type: ignore[override]
        with Collapsible(title=self._header_text(), collapsed=False):
            yield Static("", classes="tool-details")

    @property
    def _header_markup(self) -> str:
        if self._result_text is not None:
            return self._result_text
        return f"[dim]{self._icon} {self._tool_name} {self._status_text}[/dim]"

    @property
    def _icon(self) -> str:
        return "\u2699"  # ⚙

    def _header_text(self) -> str:
        if self._result_text is not None:
            return self._result_text
        return f"{self._icon} {self._tool_name} {self._status_text}"

    def _refresh_header(self) -> None:
        try:
            collapsible = self.query_one(Collapsible)
            collapsible.title = self._header_text()
        except Exception:
            pass

    def _refresh_details(self) -> None:
        try:
            details = self.query_one(".tool-details", Static)
            lines: list[str] = []
            formatted = _format_tool_input(self._tool_name, self._tool_input)
            if formatted:
                lines.append(formatted)
            if self._output_preview.strip():
                if lines:
                    lines.append("")
                lines.append("Output:")
                lines.append(self._output_preview.rstrip())
            details.update("\n".join(lines) if lines else "(waiting...)")
        except Exception:
            pass

    def set_input(self, tool_input: dict) -> None:
        self._tool_input = tool_input
        self._refresh_details()

    def update_progress(self, chars: int) -> None:
        self._status_text = f"({chars} chars)..."
        self._refresh_header()
        self._refresh_details()

    def set_elapsed(self, elapsed_s: float) -> None:
        self._elapsed_s = max(0.0, float(elapsed_s))
        self._status_text = f"(running {self._elapsed_s:.1f}s)..."
        self._refresh_header()

    def append_output(self, content: str, *, stream: str = "stdout") -> None:
        chunk = str(content or "")
        if not chunk:
            return
        prefix = ""
        if stream == "stderr":
            prefix = "[stderr] "
        elif stream == "mixed":
            prefix = "[mixed] "
        updated = (self._output_preview + prefix + chunk)[-4096:]
        self._output_preview = updated
        self._refresh_details()

    def set_result(self, status: str, duration_s: float) -> None:
        if status == "success":
            self._result_text = f"\u2713 {self._tool_name} ({duration_s:.1f}s)"
        else:
            self._result_text = f"\u2717 {self._tool_name} ({status}) ({duration_s:.1f}s)"
        self._refresh_header()
        with contextlib.suppress(Exception):
            self.query_one(Collapsible).collapsed = True

    def collapse(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one(Collapsible).collapsed = True


class ThinkingIndicator(Static):
    """Shows a thinking indicator, removed when text starts."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        color: $text-muted;
        padding: 0 1;
    }
    """

    _FRAMES = ["thinking.", "thinking..", "thinking..."]

    def __init__(self, **kwargs: object) -> None:
        super().__init__("[dim]thinking.[/dim]", **kwargs)

    def on_mount(self) -> None:
        self._frame_index = 0
        self._timer = self.set_interval(0.4, self._animate)

    def _animate(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(self._FRAMES)
        self.update(f"[dim]{self._FRAMES[self._frame_index]}[/dim]")


class ThinkingBar(Static):
    """Bottom-docked live thinking status with char count and preview."""

    DEFAULT_CSS = """
    ThinkingBar {
        dock: bottom;
        height: auto;
        max-height: 2;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
        display: none;
    }
    """

    def show_thinking(
        self,
        *,
        char_count: int,
        elapsed_s: float,
        preview: str = "",
        frame_index: int = 0,
    ) -> None:
        self.update(
            render_thinking_indicator(
                char_count=char_count,
                preview=preview,
                elapsed_s=elapsed_s,
                frame_index=frame_index,
            )
        )
        self.styles.display = "block"

    def hide_thinking(self) -> None:
        self.styles.display = "none"


class ConsentCard(Static):
    """Inline consent prompt with context and option hints."""

    DEFAULT_CSS = """
    ConsentCard {
        border: round $accent;
        padding: 0 1;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, context: str, options: list[str] | None = None, **kwargs: object) -> None:
        options = options or ["y", "n", "a", "v"]
        option_str = "  ".join(f"[bold][{o}][/bold]" for o in options)
        super().__init__(f"{context}\n{option_str}", **kwargs)


class PlanCard(Static):
    """Inline plan card with summary and numbered step checklist."""

    DEFAULT_CSS = """
    PlanCard {
        border: round $accent;
        padding: 0 1;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, plan_json: dict, **kwargs: object) -> None:
        self._plan_json = plan_json
        self._step_status: list[bool] = []
        self._step_in_progress: int | None = None
        content = self._render_plan(plan_json)
        super().__init__(content, **kwargs)

    def _render_plan(self, plan: dict) -> str:
        lines: list[str] = []
        summary = plan.get("summary", plan.get("title", ""))
        if summary:
            lines.append(f"[bold]Plan:[/bold] {summary}")
            lines.append("")

        steps = plan.get("steps", [])
        self._step_status = [False] * len(steps)
        for i, step in enumerate(steps):
            desc = step if isinstance(step, str) else step.get("description", step.get("title", str(step)))
            check = "\u2611" if self._step_status[i] else "\u2610"  # ☑ / ☐
            lines.append(f"  {check} {i + 1}. {desc}")
        return "\n".join(lines)

    def mark_step_complete(self, step_index: int) -> None:
        if 0 <= step_index < len(self._step_status):
            self._step_status[step_index] = True
            if self._step_in_progress == step_index:
                self._step_in_progress = None
            self.update(self._render_from_status())

    def mark_step_in_progress(self, step_index: int) -> None:
        if 0 <= step_index < len(self._step_status):
            if self._step_status[step_index]:
                return
            self._step_in_progress = step_index
            self.update(self._render_from_status())

    def _render_from_status(self) -> str:
        lines: list[str] = []
        plan = self._plan_json
        summary = plan.get("summary", plan.get("title", ""))
        if summary:
            lines.append(f"[bold]Plan:[/bold] {summary}")
            lines.append("")
        steps = plan.get("steps", [])
        for i, step in enumerate(steps):
            desc = step if isinstance(step, str) else step.get("description", step.get("title", str(step)))
            if i < len(self._step_status) and self._step_status[i]:
                check = "\u2611"
            elif self._step_in_progress == i:
                check = "\u25b6"
            else:
                check = "\u2610"
            lines.append(f"  {check} {i + 1}. {desc}")
        return "\n".join(lines)


class PlanStepRow(Vertical):
    """Interactive plan step: checkbox + detail line + expandable comment input."""

    DEFAULT_CSS = """
    PlanStepRow {
        height: auto;
        padding: 0;
        margin: 0 0 1 0;
    }
    PlanStepRow .plan-step-row {
        height: auto;
        layout: horizontal;
        margin: 0 0 0 0;
    }
    PlanStepRow .plan-step-toggle {
        width: auto;
        min-width: 10;
        margin: 0 1 0 0;
    }
    PlanStepRow .plan-step-description {
        width: 1fr;
        height: auto;
        color: $text;
        text-wrap: wrap;
        overflow-x: hidden;
    }
    PlanStepRow .plan-step-detail {
        height: auto;
        margin: 0 0 0 4;
        color: $text-muted;
        text-wrap: wrap;
        overflow-x: hidden;
    }
    PlanStepRow .plan-step-comment {
        height: auto;
        margin: 0 0 0 4;
        display: none;
    }
    """

    def __init__(
        self,
        *,
        step_index: int,
        description: str,
        files_to_edit: list[str] | None = None,
        files_to_read: list[str] | None = None,
        tools_expected: list[str] | None = None,
        risks: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._step_index = step_index
        self._description = description
        self._files_to_edit = files_to_edit or []
        self._files_to_read = files_to_read or []
        self._tools_expected = tools_expected or []
        self._risks = risks or []

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def is_included(self) -> bool:
        with contextlib.suppress(Exception):
            cb = self.query_one(f"#plan_step_cb_{self._step_index}", Checkbox)
            return bool(cb.value)
        return True

    @property
    def comment(self) -> str:
        with contextlib.suppress(Exception):
            inp = self.query_one(f"#plan_step_comment_{self._step_index}", Input)
            return (inp.value or "").strip()
        return ""

    def compose(self):  # type: ignore[override]
        with Horizontal(classes="plan-step-row"):
            yield Checkbox(
                "Include",
                id=f"plan_step_cb_{self._step_index}",
                value=True,
                classes="plan-step-toggle",
            )
            yield Static(
                RichText(
                    f"{self._step_index + 1}. {self._description}",
                    no_wrap=False,
                    overflow="fold",
                ),
                classes="plan-step-description",
            )
        detail_parts: list[str] = []
        if self._files_to_edit:
            detail_parts.append(f"edit: {', '.join(self._files_to_edit[:3])}")
        if self._files_to_read:
            detail_parts.append(f"read: {', '.join(self._files_to_read[:3])}")
        if self._risks:
            detail_parts.append(f"risks: {', '.join(self._risks[:2])}")
        if detail_parts:
            yield Static(
                RichText("\n".join(detail_parts), no_wrap=False, overflow="fold"),
                classes="plan-step-detail",
            )
        yield Input(
            placeholder="Add feedback for this step...",
            id=f"plan_step_comment_{self._step_index}",
            classes="plan-step-comment",
        )

    def toggle_comment_visibility(self) -> None:
        with contextlib.suppress(Exception):
            comment_input = self.query_one(f"#plan_step_comment_{self._step_index}", Input)
            if self.is_included:
                comment_input.styles.display = "none"
            else:
                comment_input.styles.display = "block"
                comment_input.focus()


class PlanQuestionRow(Vertical):
    """Interactive refinement question with an auto-expanding answer box."""

    DEFAULT_CSS = """
    PlanQuestionRow {
        height: auto;
        padding: 0;
        margin: 0 0 1 0;
    }
    PlanQuestionRow .plan-question-text {
        height: auto;
        color: $text-muted;
        margin: 0 0 0 0;
    }
    PlanQuestionRow .plan-question-answer {
        height: 3;
        min-height: 3;
        max-height: 7;
        margin: 0 0 0 0;
    }
    """

    _MIN_VIS_LINES = 1
    _MAX_VIS_LINES = 5

    def __init__(self, *, question_index: int, question: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._question_index = question_index
        self._question = str(question or "").strip()

    @property
    def question_index(self) -> int:
        return self._question_index

    @property
    def question(self) -> str:
        return self._question

    @property
    def answer(self) -> str:
        with contextlib.suppress(Exception):
            answer_box = self.query_one(f"#plan_question_answer_{self._question_index}", TextArea)
            return str(answer_box.text or "").strip()
        return ""

    def compose(self):  # type: ignore[override]
        yield Static(f"Q{self._question_index + 1}. {self._question}", classes="plan-question-text")
        yield TextArea(
            text="",
            id=f"plan_question_answer_{self._question_index}",
            classes="plan-question-answer",
            soft_wrap=True,
        )

    def on_mount(self) -> None:
        self._sync_height()

    def on_text_area_changed(self, event: Any) -> None:
        text_area = getattr(event, "text_area", None)
        text_area_id = str(getattr(text_area, "id", "")).strip()
        if text_area_id != f"plan_question_answer_{self._question_index}":
            return
        self._sync_height()

    def on_resize(self, event: Any) -> None:
        _ = event
        self._sync_height()

    def _wrapped_line_count(self, text: str, width: int) -> int:
        normalized = str(text or "")
        logical_lines = normalized.split("\n") or [""]
        total = 0
        for line in logical_lines:
            expanded = line.expandtabs(4)
            wrapped = textwrap.wrap(
                expanded,
                width=max(1, width),
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            total += max(1, len(wrapped))
        return max(1, total)

    def _sync_height(self) -> None:
        with contextlib.suppress(Exception):
            answer_box = self.query_one(f"#plan_question_answer_{self._question_index}", TextArea)
            width = int(getattr(getattr(answer_box, "content_region", None), "width", 0) or 0)
            if width <= 0:
                width = max(1, int(getattr(getattr(answer_box, "size", None), "width", 1) or 1) - 2)
            line_count = self._wrapped_line_count(str(answer_box.text or ""), width)
            visible_lines = max(self._MIN_VIS_LINES, min(self._MAX_VIS_LINES, line_count))
            # Add 2 rows for TextArea chrome.
            answer_box.styles.height = visible_lines + 2
            if line_count > self._MAX_VIS_LINES:
                answer_box.styles.overflow_y = "auto"
                with contextlib.suppress(Exception):
                    answer_box.scroll_end(animate=False)
            else:
                answer_box.styles.overflow_y = "hidden"
            with contextlib.suppress(Exception):
                answer_box.refresh(layout=True)


class PlanActions(Static):
    """Button row for plan actions in the sidebar."""

    DEFAULT_CSS = """
    PlanActions {
        layout: horizontal;
        height: auto;
        padding: 0;
        margin: 0;
    }
    PlanActions Button {
        width: 1fr;
        min-width: 8;
        margin: 0;
        padding: 0 1;
        background: transparent;
        color: $accent;
        border: round $accent;
    }
    PlanActions Button:hover {
        background: transparent;
        color: $accent;
        border: round $accent;
    }
    PlanActions Button.-active,
    PlanActions Button:focus {
        background: transparent;
        color: $accent;
        border: round $accent;
    }
    """

    def compose(self):  # type: ignore[override]
        yield Button("Approve", id="plan_action_approve", compact=True, variant="default")
        yield Button("Replan", id="plan_action_replan", compact=True, variant="default")
        yield Button("Clear", id="plan_action_clear", compact=True, variant="default")


class AgentProfileActions(Static):
    """Button row for profile CRUD/apply actions in the Agent tab."""

    DEFAULT_CSS = """
    AgentProfileActions {
        layout: horizontal;
        height: auto;
        padding: 0;
        margin: 0;
    }
    AgentProfileActions Button {
        width: 1fr;
        min-width: 10;
        margin: 0;
        padding: 0 1;
        background: transparent;
        color: $accent;
        border: round $accent;
    }
    AgentProfileActions Button:hover {
        background: transparent;
        color: $accent;
        border: round $accent;
    }
    AgentProfileActions Button.-active,
    AgentProfileActions Button:focus {
        background: transparent;
        color: $accent;
        border: round $accent;
    }
    """

    def compose(self):  # type: ignore[override]
        yield Button("New", id="agent_profile_new", compact=True, variant="default")
        yield Button("Save", id="agent_profile_save", compact=True, variant="default")
        yield Button("Delete", id="agent_profile_delete", compact=True, variant="default")
        yield Button("Apply", id="agent_profile_apply", compact=True, variant="default")


class ContextBudgetBar(Static):
    """Single-line context budget visualization with optional prompt estimate."""

    DEFAULT_CSS = """
    ContextBudgetBar {
        width: 1fr;
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)
        self._prompt_tokens_est: int | None = None
        self._budget_tokens: int | None = None
        self._prompt_input_tokens_est: int | None = None
        self._display_ratio: float = 0.0
        self._target_ratio: float = 0.0
        self._anim_timer: Any = None
        self._last_plain_text: str = ""

    def on_mount(self) -> None:
        self._render_bar()

    def set_context(self, *, prompt_tokens_est: int | None, budget_tokens: int | None, animate: bool = True) -> None:
        self._prompt_tokens_est = (
            prompt_tokens_est if isinstance(prompt_tokens_est, int) and prompt_tokens_est >= 0 else None
        )
        self._budget_tokens = budget_tokens if isinstance(budget_tokens, int) and budget_tokens > 0 else None
        ratio = self._ratio()
        self._target_ratio = ratio
        if animate and getattr(self, "is_mounted", False):
            self._ensure_anim_timer()
        else:
            self._display_ratio = ratio
            self._stop_anim_timer()
            self._render_bar()

    def set_prompt_input_estimate(self, tokens_est: int | None) -> None:
        self._prompt_input_tokens_est = tokens_est if isinstance(tokens_est, int) and tokens_est >= 0 else None
        self._render_bar()

    def _ratio(self) -> float:
        if (
            not isinstance(self._prompt_tokens_est, int)
            or not isinstance(self._budget_tokens, int)
            or self._budget_tokens <= 0
        ):
            return 0.0
        return max(0.0, min(1.0, float(self._prompt_tokens_est) / float(self._budget_tokens)))

    def _ensure_anim_timer(self) -> None:
        if self._anim_timer is not None:
            return
        self._anim_timer = self.set_interval(0.05, self._animate_step)

    def _stop_anim_timer(self) -> None:
        timer = self._anim_timer
        self._anim_timer = None
        if timer is not None:
            with contextlib.suppress(Exception):
                timer.stop()

    def _animate_step(self) -> None:
        target = self._target_ratio
        current = self._display_ratio
        delta = target - current
        if abs(delta) < 0.01:
            self._display_ratio = target
            self._stop_anim_timer()
            self._render_bar()
            return
        self._display_ratio = max(0.0, min(1.0, current + (delta * 0.4)))
        self._render_bar()

    def _format_tokens(self, value: int | None) -> str:
        if not isinstance(value, int):
            return "--"
        if value >= 1_000_000:
            rendered = f"{value / 1_000_000:.1f}m"
            return rendered.replace(".0m", "m")
        if value >= 1_000:
            rendered = f"{value / 1_000:.1f}k"
            return rendered.replace(".0k", "k")
        return str(value)

    @property
    def plain_text(self) -> str:
        return self._last_plain_text

    def _render_bar(self) -> None:
        budget_known = isinstance(self._budget_tokens, int) and self._budget_tokens > 0
        estimate_known = isinstance(self._prompt_tokens_est, int)
        ratio_actual = self._ratio() if budget_known and estimate_known else 0.0
        ratio_for_bar = self._display_ratio if budget_known and estimate_known else 0.0
        ratio_for_bar = max(0.0, min(1.0, ratio_for_bar))

        if ratio_for_bar < 0.5:
            fill_style = "green"
        elif ratio_for_bar <= 0.8:
            fill_style = "yellow"
        else:
            fill_style = "red"

        bar_width = 16
        filled = int(round(bar_width * ratio_for_bar))
        filled = max(0, min(bar_width, filled))

        rendered = RichText()
        rendered.append("[", style="dim")
        if filled > 0:
            rendered.append("█" * filled, style=fill_style)
        if filled < bar_width:
            rendered.append("░" * (bar_width - filled), style="dim")
        rendered.append("] ", style="dim")

        pct = int(round(ratio_actual * 100.0)) if budget_known and estimate_known else 0
        context_line = (
            f"Context: {self._format_tokens(self._prompt_tokens_est)} / "
            f"{self._format_tokens(self._budget_tokens)} ({pct}%)"
        )
        rendered.append(
            context_line,
            style="default",
        )
        if isinstance(self._prompt_input_tokens_est, int):
            rendered.append(f"  ~{self._format_tokens(self._prompt_input_tokens_est)} tokens", style="dim")

        tooltip_text: str | None = None
        if budget_known and estimate_known and ratio_actual > 0.8:
            rendered.append("  ⚠", style="bold yellow")
            tooltip_text = "Context nearly full. Consider /compact or /new."
        else:
            tooltip_text = None

        self.tooltip = tooltip_text
        self._last_plain_text = rendered.plain
        if getattr(self, "is_mounted", False):
            self.update(rendered)


class CommandPalette(Static):
    """Dropdown overlay showing matching slash commands."""

    DEFAULT_CSS = """
    CommandPalette {
        border: round $accent;
        padding: 0 1;
        margin: 0;
        background: $surface;
        layer: overlay;
        dock: bottom;
        display: none;
    }
    """

    TUI_COMMANDS: list[tuple[str, str]] = [
        ("/help", "Show available commands"),
        ("/plan", "Generate a plan"),
        ("/run", "Execute immediately"),
        ("/approve", "Execute pending plan"),
        ("/replan", "Regenerate plan"),
        ("/clearplan", "Clear plan"),
        ("/model", "Model settings"),
        ("/restore", "Restore previous session"),
        ("/new", "Start fresh session"),
        ("/context", "Manage context sources"),
        ("/compact", "Summarize context"),
        ("/text", "Toggle transcript text mode"),
        ("/thinking", "Show model reasoning"),
        ("/sop", "Browse and toggle SOPs"),
        ("/connect", "Connect provider auth (Copilot/AWS)"),
        ("/auth", "List/logout provider auth"),
        ("/diagnostics bundle", "Create support bundle"),
        ("/copy", "Copy transcript"),
        ("/copy plan", "Copy plan text"),
        ("/copy issues", "Copy issues"),
        ("/copy last", "Copy last response"),
        ("/copy all", "Copy everything"),
        ("/expand", "Expand tool details"),
        ("/open", "Open artifact by number"),
        ("/search", "Search transcript"),
        ("/consent", "Respond to consent"),
        ("/daemon restart", "Reconnect/restart daemon transport"),
        ("/daemon stop", "Shut down daemon/broker"),
        ("/broker stop", "Shut down shared runtime broker"),
        ("/stop", "Stop current run"),
        ("/exit", "Quit TUI"),
    ]

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)
        self._filtered: list[tuple[str, str]] = list(self.TUI_COMMANDS)
        self._selected_index: int = 0

    def filter(self, prefix: str) -> None:
        """Filter commands by prefix and update display. Empty prefix shows all."""
        prefix_lower = prefix.lower()
        self._filtered = [(cmd, desc) for cmd, desc in self.TUI_COMMANDS if cmd.startswith(prefix_lower)]
        self._selected_index = 0
        if self._filtered:
            self._render_items()
            self.styles.display = "block"
        else:
            self.styles.display = "none"

    def move_selection(self, delta: int) -> None:
        if not self._filtered:
            return
        self._selected_index = (self._selected_index + delta) % len(self._filtered)
        self._render_items()

    def get_selected(self) -> str | None:
        if not self._filtered:
            return None
        return self._filtered[self._selected_index][0]

    def hide(self) -> None:
        self.styles.display = "none"

    @property
    def is_visible(self) -> bool:
        return str(self.styles.display) != "none"

    def _render_items(self) -> None:
        lines: list[str] = []
        for i, (cmd, desc) in enumerate(self._filtered):
            marker = "[bold]>[/bold] " if i == self._selected_index else "  "
            lines.append(f"{marker}{cmd:<14} [dim]{desc}[/dim]")
        self.update("\n".join(lines))


class ActionSheet(Vertical):
    """Centered, context-sensitive action overlay."""

    DEFAULT_CSS = """
    ActionSheet {
        display: none;
        layer: overlay;
        width: 100%;
        height: 100%;
        align: center middle;
        background: transparent;
    }
    ActionSheet #action_sheet_panel {
        width: 72;
        max-width: 96%;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 0 1;
    }
    ActionSheet #action_sheet_title {
        height: 1;
        color: $text-muted;
    }
    ActionSheet #action_sheet_items {
        height: 10;
        max-height: 10;
        min-height: 1;
        margin: 0 0 1 0;
        scrollbar-background: #2f2f2f;
        scrollbar-color: #7f7f7f;
    }
    ActionSheet .action-sheet-row {
        height: 1;
        layout: horizontal;
        padding: 0 1;
    }
    ActionSheet .action-sheet-row.-selected {
        background: $accent 30%;
    }
    ActionSheet .action-sheet-left {
        width: 1fr;
    }
    ActionSheet .action-sheet-right {
        width: auto;
        color: $text-muted;
    }
    """

    class ActionSelected(Message):
        def __init__(self, sender: "ActionSheet", action_id: str) -> None:
            super().__init__()
            self.sender = sender
            self.action_id = action_id

    class Dismissed(Message):
        def __init__(self, sender: "ActionSheet") -> None:
            super().__init__()
            self.sender = sender

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._title: str = "Actions"
        self._actions: list[dict[str, str]] = []
        self._selected_index: int = 0

    def compose(self):  # type: ignore[override]
        with Vertical(id="action_sheet_panel"):
            yield Static("", id="action_sheet_title")
            yield VerticalScroll(id="action_sheet_items")

    @property
    def is_visible(self) -> bool:
        return str(self.styles.display) != "none"

    def show_sheet(self, *, focus: bool = True) -> None:
        self.styles.display = "block"
        self._render_items()
        if focus:
            self.focus()

    def hide_sheet(self) -> None:
        self.styles.display = "none"

    def set_actions(self, *, title: str, actions: list[dict[str, str]]) -> None:
        self._title = title
        self._actions = [dict(item) for item in actions if isinstance(item, dict)]
        if self._selected_index >= len(self._actions):
            self._selected_index = max(0, len(self._actions) - 1)
        if self._selected_index < 0:
            self._selected_index = 0
        self._render_items()

    def move_selection(self, delta: int) -> None:
        if not self._actions:
            return
        self._selected_index = (self._selected_index + delta) % len(self._actions)
        self._render_items()

    def selected_action_id(self) -> str | None:
        if not self._actions:
            return None
        if self._selected_index < 0 or self._selected_index >= len(self._actions):
            return None
        action_id = str(self._actions[self._selected_index].get("id", "")).strip()
        return action_id or None

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key in {"up", "down"}:
            event.stop()
            event.prevent_default()
            self.move_selection(-1 if key == "up" else 1)
            return
        if key in {"enter", "return"}:
            event.stop()
            event.prevent_default()
            action_id = self.selected_action_id()
            if action_id:
                self.post_message(self.ActionSelected(self, action_id))
            return
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.post_message(self.Dismissed(self))

    def _render_items(self) -> None:
        if not getattr(self, "is_mounted", False):
            return
        with contextlib.suppress(Exception):
            self.query_one("#action_sheet_title", Static).update(self._title)
        container: VerticalScroll | None = None
        with contextlib.suppress(Exception):
            container = self.query_one("#action_sheet_items", VerticalScroll)
        if container is None:
            return

        target_height = min(10, max(1, len(self._actions)))
        with contextlib.suppress(Exception):
            container.styles.height = target_height

        for child in list(container.children):
            with contextlib.suppress(Exception):
                child.remove()

        for idx, action in enumerate(self._actions):
            icon = str(action.get("icon", "•")).strip() or "•"
            label = str(action.get("label", "")).strip() or str(action.get("id", "action")).strip()
            shortcut = str(action.get("shortcut", "")).strip()
            row = Horizontal(classes="action-sheet-row")
            container.mount(row)
            if idx == self._selected_index:
                row.add_class("-selected")
            row.mount(Static(f"{icon} {label}", classes="action-sheet-left"))
            row.mount(Static(shortcut, classes="action-sheet-right"))

        if 0 <= self._selected_index < len(container.children):
            with contextlib.suppress(Exception):
                container.scroll_to_widget(container.children[self._selected_index], animate=False)


class StatusBar(Static):
    """One-line bar showing run state, model, tool count, elapsed time."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)
        self._state: str = "idle"
        self._model: str = ""
        self._tool_count: int = 0
        self._elapsed: float = 0.0
        self._warning_count: int = 0
        self._error_count: int = 0
        self._prompt_tokens_est: int | None = None
        self._budget_tokens: int | None = None
        self._usage: dict[str, Any] | None = None
        self._cost_usd: float | None = None
        self._step_current: int | None = None
        self._step_total: int | None = None
        self.refresh_display()

    def set_state(self, state: str) -> None:
        self._state = state
        self.refresh_display()

    def set_tool_count(self, n: int) -> None:
        self._tool_count = n
        self.refresh_display()

    def set_elapsed(self, secs: float) -> None:
        self._elapsed = secs
        self.refresh_display()

    def set_model(self, summary: str) -> None:
        self._model = summary
        self.refresh_display()

    def set_counts(self, *, warnings: int = 0, errors: int = 0) -> None:
        self._warning_count = warnings
        self._error_count = errors
        self.refresh_display()

    def set_context(self, *, prompt_tokens_est: int | None, budget_tokens: int | None) -> None:
        self._prompt_tokens_est = prompt_tokens_est
        self._budget_tokens = budget_tokens
        self.refresh_display()

    def set_usage(self, usage: dict[str, Any] | None, *, cost_usd: float | None = None) -> None:
        self._usage = usage
        self._cost_usd = cost_usd
        self.refresh_display()

    def set_plan_step(self, *, current: int | None, total: int | None) -> None:
        self._step_current = current
        self._step_total = total
        self.refresh_display()

    def _format_k(self, n: int) -> str:
        if n >= 100_000:
            return f"{n / 1000.0:.0f}k"
        if n >= 10_000:
            return f"{n / 1000.0:.1f}k"
        if n >= 1000:
            return f"{n / 1000.0:.2f}k"
        return str(n)

    def _extract_usage_tokens(self) -> tuple[int | None, int | None, int | None]:
        if not isinstance(self._usage, dict):
            return None, None, None
        usage = self._usage

        def _as_int(value: Any) -> int | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
            return None

        in_tokens = _as_int(usage.get("input_tokens")) or _as_int(usage.get("prompt_tokens"))
        out_tokens = _as_int(usage.get("output_tokens")) or _as_int(usage.get("completion_tokens"))
        cached = None
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = _as_int(details.get("cached_tokens"))
        cached = cached or _as_int(usage.get("cache_read_input_tokens"))
        return in_tokens, out_tokens, cached

    def refresh_display(self) -> None:
        icon = "\u25b6" if self._state == "running" else "\u23f8"  # ▶ / ⏸
        parts: list[str] = [f"{icon} {self._state}", self._model or "Model: ?"]
        context_ratio: float | None = None

        if isinstance(self._prompt_tokens_est, int):
            if isinstance(self._budget_tokens, int) and self._budget_tokens > 0:
                context_ratio = float(self._prompt_tokens_est) / float(self._budget_tokens)
                parts.append(f"ctx {self._format_k(self._prompt_tokens_est)}/{self._format_k(self._budget_tokens)}")
            else:
                parts.append(f"ctx {self._format_k(self._prompt_tokens_est)}")

        in_tokens, out_tokens, cached = self._extract_usage_tokens()
        if in_tokens is not None or out_tokens is not None:
            token_parts: list[str] = []
            if in_tokens is not None:
                token_parts.append(f"in {self._format_k(in_tokens)}")
            if out_tokens is not None:
                token_parts.append(f"out {self._format_k(out_tokens)}")
            if cached is not None and cached > 0:
                token_parts.append(f"cache {self._format_k(cached)}")
            parts.append(" ".join(token_parts))
        if isinstance(self._cost_usd, (int, float)):
            parts.append(f"${self._cost_usd:.4f}")
        if isinstance(self._step_total, int) and self._step_total > 0:
            if isinstance(self._step_current, int) and self._step_current > 0:
                parts.append(f"step {self._step_current}/{self._step_total}")
            else:
                parts.append(f"step 0/{self._step_total}")
        if isinstance(context_ratio, float) and context_ratio >= 0.9:
            parts.append("CTX HIGH")

        parts.append(f"tools {self._tool_count}")
        parts.append(f"{self._elapsed:.1f}s")
        if self._warning_count:
            parts.append(f"warn={self._warning_count}")
        if self._error_count:
            parts.append(f"err={self._error_count}")

        rendered = " | ".join(parts)
        width = getattr(getattr(self, "size", None), "width", None)
        if isinstance(width, int) and width > 0 and len(rendered) > width:
            # Drop least important segments until it fits.
            drop_prefixes = (
                "warn=",
                "err=",
            )
            prunable: list[int] = []
            for i, part in enumerate(parts):
                if part.startswith(drop_prefixes) or part.endswith("s") or part.startswith("tools "):
                    prunable.append(i)
            for idx in reversed(prunable):
                candidate = parts[:idx] + parts[idx + 1 :]
                if len(" | ".join(candidate)) <= width:
                    parts = candidate
                    rendered = " | ".join(parts)
                    break
            if len(rendered) > width:
                # As a last resort, truncate the model segment.
                model_idx = 1
                if len(parts) > model_idx and width > 10:
                    other = [p for i, p in enumerate(parts) if i != model_idx]
                    overhead = len(" | ".join(other)) + (3 * (len(parts) - 1))
                    room = max(5, width - overhead)
                    model = parts[model_idx]
                    if len(model) > room:
                        parts[model_idx] = model[: max(0, room - 1)] + "…"
                        rendered = " | ".join(parts)

        self.update(rendered)


class TagEditScreen(ModalScreen[str | None]):
    """Popup editor for tool tags — returns new comma-separated tag string or None."""

    DEFAULT_CSS = """
    TagEditScreen {
        align: center middle;
    }
    TagEditScreen #tag_edit_container {
        width: 60;
        max-width: 90%;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    TagEditScreen #tag_edit_title {
        height: auto;
        margin: 0 0 1 0;
        color: $text;
    }
    TagEditScreen #tag_edit_input {
        margin: 0 0 1 0;
    }
    TagEditScreen #tag_edit_buttons {
        layout: horizontal;
        height: auto;
    }
    TagEditScreen #tag_edit_buttons Button {
        width: 1fr;
        margin: 0 1 0 0;
    }
    """

    def __init__(self, tool_name: str, current_tags: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._current_tags = current_tags

    def compose(self):  # type: ignore[override]
        with Vertical(id="tag_edit_container"):
            yield Static(f"Edit tags for [bold]{self._tool_name}[/bold]", id="tag_edit_title")
            yield Input(
                value=self._current_tags,
                placeholder="Tags (comma-separated)",
                id="tag_edit_input",
            )
            with Horizontal(id="tag_edit_buttons"):
                yield Button("Save", id="tag_edit_save", variant="success")
                yield Button("Cancel", id="tag_edit_cancel", variant="default")

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#tag_edit_input", Input).focus()

    def on_button_pressed(self, event: Any) -> None:
        button_id = str(getattr(getattr(event, "button", None), "id", "")).strip()
        if button_id == "tag_edit_save":
            value = str(self.query_one("#tag_edit_input", Input).value or "").strip()
            self.dismiss(value)
        elif button_id == "tag_edit_cancel":
            self.dismiss(None)

    def on_input_submitted(self, event: Any) -> None:
        input_id = str(getattr(getattr(event, "input", None), "id", "")).strip()
        if input_id == "tag_edit_input":
            value = str(self.query_one("#tag_edit_input", Input).value or "").strip()
            self.dismiss(value)

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.dismiss(None)


class TableCellEditScreen(ModalScreen[str | None]):
    """Lightweight modal editor for table cell text values."""

    DEFAULT_CSS = """
    TableCellEditScreen {
        align: center middle;
    }
    TableCellEditScreen #table_cell_edit_container {
        width: 68;
        max-width: 92%;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    TableCellEditScreen #table_cell_edit_title {
        height: auto;
        margin: 0 0 1 0;
        color: $text;
    }
    TableCellEditScreen #table_cell_edit_help {
        height: auto;
        margin: 0 0 1 0;
        color: $text-muted;
    }
    TableCellEditScreen #table_cell_edit_input {
        margin: 0 0 1 0;
    }
    TableCellEditScreen #table_cell_edit_buttons {
        layout: horizontal;
        height: auto;
    }
    TableCellEditScreen #table_cell_edit_buttons Button {
        width: 1fr;
        margin: 0 1 0 0;
    }
    """

    def __init__(self, title: str, initial_value: str, *, help_text: str = "", **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._title = str(title).strip() or "Edit value"
        self._initial_value = str(initial_value or "")
        self._help_text = str(help_text or "").strip()

    def compose(self):  # type: ignore[override]
        with Vertical(id="table_cell_edit_container"):
            yield Static(self._title, id="table_cell_edit_title")
            if self._help_text:
                yield Static(self._help_text, id="table_cell_edit_help")
            yield Input(
                value=self._initial_value,
                id="table_cell_edit_input",
            )
            with Horizontal(id="table_cell_edit_buttons"):
                yield Button("Save", id="table_cell_edit_save", variant="success")
                yield Button("Cancel", id="table_cell_edit_cancel", variant="default")

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#table_cell_edit_input", Input).focus()

    def on_button_pressed(self, event: Any) -> None:
        button_id = str(getattr(getattr(event, "button", None), "id", "")).strip()
        if button_id == "table_cell_edit_save":
            value = str(self.query_one("#table_cell_edit_input", Input).value or "")
            self.dismiss(value)
        elif button_id == "table_cell_edit_cancel":
            self.dismiss(None)

    def on_input_submitted(self, event: Any) -> None:
        input_id = str(getattr(getattr(event, "input", None), "id", "")).strip()
        if input_id == "table_cell_edit_input":
            value = str(self.query_one("#table_cell_edit_input", Input).value or "")
            self.dismiss(value)

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.dismiss(None)


class AuthConnectScreen(ModalScreen[None]):
    """Persistent auth/connect status dialog for device-code flows."""

    DEFAULT_CSS = """
    AuthConnectScreen {
        align: center middle;
    }
    AuthConnectScreen #auth_connect_container {
        width: 96;
        max-width: 98%;
        max-height: 92%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    AuthConnectScreen #auth_connect_title {
        height: auto;
        margin: 0 0 1 0;
        color: $text;
    }
    AuthConnectScreen #auth_connect_body {
        height: 1fr;
        min-height: 10;
        border: round $panel;
        padding: 0 1;
        margin: 0 0 1 0;
        scrollbar-background: #2f2f2f;
        scrollbar-background-hover: #3a3a3a;
        scrollbar-background-active: #454545;
        scrollbar-color: #7f7f7f;
        scrollbar-color-hover: #999999;
        scrollbar-color-active: #b3b3b3;
    }
    AuthConnectScreen #auth_connect_buttons {
        layout: horizontal;
        height: auto;
    }
    AuthConnectScreen #auth_connect_buttons Button {
        width: 1fr;
    }
    """

    def __init__(self, *, title: str, lines: list[str] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._title = str(title or "").strip() or "Connect Provider"
        self._lines: list[str] = []
        for line in lines or []:
            text = str(line or "").strip()
            if text:
                self._lines.append(text)

    def compose(self):  # type: ignore[override]
        with Vertical(id="auth_connect_container"):
            yield Static(self._title, id="auth_connect_title")
            with VerticalScroll(id="auth_connect_body"):
                yield Static(self._render_lines(), id="auth_connect_text")
            with Horizontal(id="auth_connect_buttons"):
                yield Button("Close", id="auth_connect_close", variant="primary")

    def _render_lines(self) -> str:
        if not self._lines:
            return "Waiting for provider status..."
        return "\n".join(self._lines)

    def append_line(self, text: str) -> None:
        line = str(text or "").strip()
        if not line:
            return
        self._lines.append(line)
        with contextlib.suppress(Exception):
            target = self.query_one("#auth_connect_text", Static)
            target.update(self._render_lines())
        with contextlib.suppress(Exception):
            body = self.query_one("#auth_connect_body", VerticalScroll)
            body.scroll_end(animate=False)

    def on_button_pressed(self, event: Any) -> None:
        button_id = str(getattr(getattr(event, "button", None), "id", "")).strip()
        if button_id == "auth_connect_close":
            self.dismiss(None)

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.dismiss(None)


class PromptUsedByEditScreen(ModalScreen[list[str] | None]):
    """Modal editor for assigning a prompt asset to agent roster entries."""

    DEFAULT_CSS = """
    PromptUsedByEditScreen {
        align: center middle;
    }
    PromptUsedByEditScreen #prompt_used_by_container {
        width: 74;
        max-width: 94%;
        max-height: 90%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    PromptUsedByEditScreen #prompt_used_by_title {
        height: auto;
        margin: 0 0 1 0;
        color: $text;
    }
    PromptUsedByEditScreen #prompt_used_by_options {
        height: 1fr;
        min-height: 8;
        border: round $panel;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    PromptUsedByEditScreen #prompt_used_by_buttons {
        layout: horizontal;
        height: auto;
    }
    PromptUsedByEditScreen #prompt_used_by_buttons Button {
        width: 1fr;
        margin: 0 1 0 0;
    }
    """

    def __init__(self, choices: list[tuple[str, str]], selected_ids: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._choices = [
            (str(agent_id).strip(), str(label).strip()) for agent_id, label in choices if str(agent_id).strip()
        ]
        self._selected_ids = {str(item).strip() for item in selected_ids if str(item).strip()}
        self._checkbox_to_agent_id: dict[str, str] = {}

    def compose(self):  # type: ignore[override]
        with Vertical(id="prompt_used_by_container"):
            yield Static("Assign prompt to agents", id="prompt_used_by_title")
            with VerticalScroll(id="prompt_used_by_options"):
                for index, (agent_id, label) in enumerate(self._choices):
                    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", agent_id) or f"agent_{index}"
                    checkbox_id = f"prompt_used_by_cb_{index}_{safe_id}"
                    self._checkbox_to_agent_id[checkbox_id] = agent_id
                    yield Checkbox(
                        label=label or agent_id,
                        value=agent_id in self._selected_ids,
                        id=checkbox_id,
                    )
            with Horizontal(id="prompt_used_by_buttons"):
                yield Button("Save", id="prompt_used_by_save", variant="success")
                yield Button("Cancel", id="prompt_used_by_cancel", variant="default")

    def _selected_agent_ids(self) -> list[str]:
        selected: list[str] = []
        for checkbox_id, agent_id in self._checkbox_to_agent_id.items():
            with contextlib.suppress(Exception):
                checkbox = self.query_one(f"#{checkbox_id}", Checkbox)
                if bool(getattr(checkbox, "value", False)):
                    selected.append(agent_id)
        return selected

    def on_mount(self) -> None:
        first_id = next(iter(self._checkbox_to_agent_id), "")
        if first_id:
            with contextlib.suppress(Exception):
                self.query_one(f"#{first_id}", Checkbox).focus()

    def on_button_pressed(self, event: Any) -> None:
        button_id = str(getattr(getattr(event, "button", None), "id", "")).strip()
        if button_id == "prompt_used_by_save":
            self.dismiss(self._selected_agent_ids())
            return
        if button_id == "prompt_used_by_cancel":
            self.dismiss(None)

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.dismiss(None)


class CatalogMultiSelectScreen(ModalScreen[list[str] | None]):
    """Catalog-backed multi-select modal with optional custom CSV values."""

    DEFAULT_CSS = """
    CatalogMultiSelectScreen {
        align: center middle;
    }
    CatalogMultiSelectScreen #catalog_multi_container {
        width: 78;
        max-width: 94%;
        max-height: 90%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    CatalogMultiSelectScreen #catalog_multi_title {
        height: auto;
        margin: 0 0 1 0;
        color: $text;
    }
    CatalogMultiSelectScreen #catalog_multi_help {
        height: auto;
        margin: 0 0 1 0;
        color: $text-muted;
    }
    CatalogMultiSelectScreen #catalog_multi_options {
        height: 1fr;
        min-height: 8;
        border: round $panel;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    CatalogMultiSelectScreen #catalog_multi_custom {
        margin: 0 0 1 0;
    }
    CatalogMultiSelectScreen #catalog_multi_buttons {
        layout: horizontal;
        height: auto;
    }
    CatalogMultiSelectScreen #catalog_multi_buttons Button {
        width: 1fr;
        margin: 0 1 0 0;
    }
    """

    def __init__(
        self,
        *,
        title: str,
        options: list[str],
        selected_values: list[str],
        help_text: str = "",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._title = str(title or "").strip() or "Select values"
        self._help_text = str(help_text or "").strip()
        self._options = self._normalize_tokens(options)
        self._selected_values = self._normalize_tokens(selected_values)
        option_lowers = {item.lower() for item in self._options}
        self._custom_initial = ", ".join(
            item for item in self._selected_values if item.lower() not in option_lowers
        )
        self._checkbox_map: dict[str, str] = {}

    @staticmethod
    def _normalize_tokens(tokens: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            value = str(token).strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(value)
        return normalized

    def _selected_values_from_ui(self) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for checkbox_id, option in self._checkbox_map.items():
            with contextlib.suppress(Exception):
                checkbox = self.query_one(f"#{checkbox_id}", Checkbox)
                if bool(getattr(checkbox, "value", False)):
                    lowered = option.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    selected.append(option)
        custom_value = ""
        with contextlib.suppress(Exception):
            custom_value = str(self.query_one("#catalog_multi_custom", Input).value or "")
        for raw in custom_value.split(","):
            token = str(raw).strip()
            lowered = token.lower()
            if not token or lowered in seen:
                continue
            seen.add(lowered)
            selected.append(token)
        return selected

    def compose(self):  # type: ignore[override]
        with Vertical(id="catalog_multi_container"):
            yield Static(self._title, id="catalog_multi_title")
            if self._help_text:
                yield Static(self._help_text, id="catalog_multi_help")
            with VerticalScroll(id="catalog_multi_options"):
                if not self._options:
                    yield Static("(no catalog entries)")
                selected_lowers = {item.lower() for item in self._selected_values}
                for index, option in enumerate(self._options):
                    checkbox_id = f"catalog_multi_cb_{index}"
                    self._checkbox_map[checkbox_id] = option
                    yield Checkbox(
                        option,
                        value=(option.lower() in selected_lowers),
                        id=checkbox_id,
                    )
            yield Input(
                value=self._custom_initial,
                placeholder="Additional values (comma-separated)",
                id="catalog_multi_custom",
            )
            with Horizontal(id="catalog_multi_buttons"):
                yield Button("Save", id="catalog_multi_save", variant="success")
                yield Button("Clear", id="catalog_multi_clear", variant="warning")
                yield Button("Cancel", id="catalog_multi_cancel", variant="default")

    def on_mount(self) -> None:
        first_checkbox_id = next(iter(self._checkbox_map), "")
        if first_checkbox_id:
            with contextlib.suppress(Exception):
                self.query_one(f"#{first_checkbox_id}", Checkbox).focus()
                return
        with contextlib.suppress(Exception):
            self.query_one("#catalog_multi_custom", Input).focus()

    def on_button_pressed(self, event: Any) -> None:
        button_id = str(getattr(getattr(event, "button", None), "id", "")).strip()
        if button_id == "catalog_multi_save":
            self.dismiss(self._selected_values_from_ui())
            return
        if button_id == "catalog_multi_clear":
            self.dismiss([])
            return
        if button_id == "catalog_multi_cancel":
            self.dismiss(None)

    def on_input_submitted(self, event: Any) -> None:
        input_id = str(getattr(getattr(event, "input", None), "id", "")).strip()
        if input_id == "catalog_multi_custom":
            self.dismiss(self._selected_values_from_ui())

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.dismiss(None)


class CatalogSingleSelectScreen(ModalScreen[str | None]):
    """Catalog-backed single-select modal with optional custom value."""

    _INHERIT_VALUE = "__inherit__"

    DEFAULT_CSS = """
    CatalogSingleSelectScreen {
        align: center middle;
    }
    CatalogSingleSelectScreen #catalog_single_container {
        width: 74;
        max-width: 94%;
        max-height: 90%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    CatalogSingleSelectScreen #catalog_single_title {
        height: auto;
        margin: 0 0 1 0;
        color: $text;
    }
    CatalogSingleSelectScreen #catalog_single_help {
        height: auto;
        margin: 0 0 1 0;
        color: $text-muted;
    }
    CatalogSingleSelectScreen #catalog_single_select {
        margin: 0 0 1 0;
    }
    CatalogSingleSelectScreen #catalog_single_custom {
        margin: 0 0 1 0;
    }
    CatalogSingleSelectScreen #catalog_single_buttons {
        layout: horizontal;
        height: auto;
    }
    CatalogSingleSelectScreen #catalog_single_buttons Button {
        width: 1fr;
        margin: 0 1 0 0;
    }
    """

    def __init__(
        self,
        *,
        title: str,
        options: list[tuple[str, str]],
        selected_value: str | None,
        help_text: str = "",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._title = str(title or "").strip() or "Select value"
        self._help_text = str(help_text or "").strip()
        self._options = self._normalize_options(options)
        token = str(selected_value or "").strip()
        token_lower = token.lower()
        resolved_value = next(
            (value for _label, value in self._options if value.lower() == token_lower),
            self._INHERIT_VALUE,
        )
        self._initial_select_value = resolved_value
        self._initial_custom_value = "" if resolved_value != self._INHERIT_VALUE else token

    @staticmethod
    def _normalize_options(options: list[tuple[str, str]]) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for label, value in options:
            resolved_label = str(label).strip()
            resolved_value = str(value).strip()
            if not resolved_value:
                continue
            lowered = resolved_value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append((resolved_label or resolved_value, resolved_value))
        return normalized

    def _resolve_value(self) -> str:
        custom_value = ""
        with contextlib.suppress(Exception):
            custom_value = str(self.query_one("#catalog_single_custom", Input).value or "").strip()
        if custom_value:
            return custom_value
        selected_value = str(getattr(self.query_one("#catalog_single_select", Select), "value", "") or "").strip()
        if not selected_value or selected_value == self._INHERIT_VALUE:
            return ""
        return selected_value

    def compose(self):  # type: ignore[override]
        options: list[tuple[str, str]] = [("Inherit (none)", self._INHERIT_VALUE)]
        options.extend(self._options)
        with Vertical(id="catalog_single_container"):
            yield Static(self._title, id="catalog_single_title")
            if self._help_text:
                yield Static(self._help_text, id="catalog_single_help")
            yield Select(
                options,
                id="catalog_single_select",
                value=self._initial_select_value,
            )
            yield Input(
                value=self._initial_custom_value,
                placeholder="Custom value (optional)",
                id="catalog_single_custom",
            )
            with Horizontal(id="catalog_single_buttons"):
                yield Button("Save", id="catalog_single_save", variant="success")
                yield Button("Clear", id="catalog_single_clear", variant="warning")
                yield Button("Cancel", id="catalog_single_cancel", variant="default")

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#catalog_single_select", Select).focus()

    def on_button_pressed(self, event: Any) -> None:
        button_id = str(getattr(getattr(event, "button", None), "id", "")).strip()
        if button_id == "catalog_single_save":
            self.dismiss(self._resolve_value())
            return
        if button_id == "catalog_single_clear":
            self.dismiss("")
            return
        if button_id == "catalog_single_cancel":
            self.dismiss(None)

    def on_input_submitted(self, event: Any) -> None:
        input_id = str(getattr(getattr(event, "input", None), "id", "")).strip()
        if input_id == "catalog_single_custom":
            self.dismiss(self._resolve_value())

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.dismiss(None)


class ToolTagManagerScreen(ModalScreen[dict[str, Any] | None]):
    """Modal tag manager for editing tool-tag associations with explicit save."""

    _NO_TAG_VALUE = "__no_tag__"

    DEFAULT_CSS = """
    ToolTagManagerScreen {
        align: center middle;
    }
    ToolTagManagerScreen #tool_tag_manager_container {
        width: 112;
        max-width: 98%;
        max-height: 94%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    ToolTagManagerScreen #tool_tag_manager_title {
        height: auto;
        margin: 0 0 1 0;
        color: $text;
    }
    ToolTagManagerScreen #tool_tag_manager_controls {
        height: auto;
        layout: horizontal;
        margin: 0 0 1 0;
    }
    ToolTagManagerScreen #tool_tag_manager_select {
        width: 1fr;
        margin: 0 1 0 0;
    }
    ToolTagManagerScreen #tool_tag_manager_input {
        width: 1fr;
        margin: 0 1 0 0;
    }
    ToolTagManagerScreen #tool_tag_manager_add,
    ToolTagManagerScreen #tool_tag_manager_rename,
    ToolTagManagerScreen #tool_tag_manager_delete {
        width: auto;
        height: 1;
        min-height: 1;
        margin: 0 1 0 0;
    }
    ToolTagManagerScreen #tool_tag_manager_list_header {
        layout: horizontal;
        height: auto;
        padding: 0 1;
        background: #3a4350;
        color: $text;
        margin: 0 0 1 0;
    }
    ToolTagManagerScreen #tool_tag_manager_list {
        height: 1fr;
        min-height: 12;
        margin: 0 0 1 0;
        border: round #4a5461;
        scrollbar-background: #2f2f2f;
        scrollbar-background-hover: #3a3a3a;
        scrollbar-background-active: #454545;
        scrollbar-color: #7f7f7f;
        scrollbar-color-hover: #999999;
        scrollbar-color-active: #b3b3b3;
    }
    ToolTagManagerScreen .tool-tag-row {
        layout: horizontal;
        height: auto;
        padding: 0 1;
        margin: 0;
    }
    ToolTagManagerScreen .tool-tag-row-even {
        background: #222831;
    }
    ToolTagManagerScreen .tool-tag-row-odd {
        background: #293240;
    }
    ToolTagManagerScreen .tool-tag-row-associated {
        background: #1f3b2d;
    }
    ToolTagManagerScreen .tool-tag-cell {
        height: auto;
        padding: 0 1 0 0;
    }
    ToolTagManagerScreen .tool-tag-name {
        width: 2fr;
    }
    ToolTagManagerScreen .tool-tag-access {
        width: 9;
        color: $text-muted;
    }
    ToolTagManagerScreen .tool-tag-source {
        width: 10;
        color: $text-muted;
    }
    ToolTagManagerScreen .tool-tag-action-cell {
        width: 1fr;
    }
    ToolTagManagerScreen .tool-tag-action-tags {
        width: 1fr;
        color: $text-muted;
        padding: 0 1 0 0;
    }
    ToolTagManagerScreen .tool-tag-action {
        width: 9;
        min-width: 8;
        height: 1;
        margin: 0;
    }
    ToolTagManagerScreen #tool_tag_manager_status {
        height: auto;
        color: $text-muted;
        margin: 0 0 1 0;
    }
    ToolTagManagerScreen #tool_tag_manager_buttons {
        layout: horizontal;
        height: auto;
    }
    ToolTagManagerScreen #tool_tag_manager_buttons Button {
        width: 1fr;
        margin: 0 1 0 0;
    }
    """

    def __init__(self, catalog: list[dict[str, Any]], **kwargs: object) -> None:
        super().__init__(**kwargs)
        from swarmee_river.tui.tooling_handlers import build_tool_table_rows

        self._catalog = [dict(item) for item in catalog if isinstance(item, dict)]
        ordered_catalog: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for item in self._catalog:
            tool_name = str(item.get("name", "")).strip()
            if not tool_name or tool_name in seen_names:
                continue
            seen_names.add(tool_name)
            ordered_catalog.append(item)

        self._tool_names: list[str] = []
        self._tool_meta_by_name: dict[str, dict[str, str]] = {}
        self._tool_tags: dict[str, list[str]] = {}
        for tool_name, access, source, _ in build_tool_table_rows(ordered_catalog):
            resolved_name = str(tool_name or "").strip()
            if not resolved_name:
                continue
            entry = next((item for item in ordered_catalog if str(item.get("name", "")).strip() == resolved_name), {})
            self._tool_names.append(resolved_name)
            self._tool_tags[resolved_name] = self._normalize_tags(entry.get("tags", []))
            self._tool_meta_by_name[resolved_name] = {
                "access": str(access or ""),
                "source": str(source or ""),
            }
        self._all_tags = self._collect_all_tags()
        self._selected_tag = self._all_tags[0] if self._all_tags else ""
        self._row_button_to_tool: dict[str, str] = {}

    @staticmethod
    def _normalize_tags(raw_tags: Any) -> list[str]:
        tags = raw_tags if isinstance(raw_tags, list) else []
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in tags:
            tag = str(raw).strip()
            lowered = tag.lower()
            if not tag or lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(tag)
        return normalized

    def _collect_all_tags(self) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for tool_name in self._tool_names:
            for tag in self._tool_tags.get(tool_name, []):
                lowered = tag.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                tags.append(tag)
        tags.sort(key=str.lower)
        return tags

    def _selected_tag_lower(self) -> str:
        return str(self._selected_tag or "").strip().lower()

    def _lookup_tag_by_lower(self, lowered: str) -> str | None:
        for tag in self._all_tags:
            if tag.lower() == lowered:
                return tag
        return None

    def _set_status(self, text: str) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#tool_tag_manager_status", Static).update(str(text or "").strip())

    def _refresh_tag_select(self) -> None:
        selector = self.query_one("#tool_tag_manager_select", Select)
        options: list[tuple[str, str]] = []
        if self._all_tags:
            options = [(tag, tag) for tag in self._all_tags]
        else:
            options = [("(no tags)", self._NO_TAG_VALUE)]
        selector.set_options(options)
        if self._all_tags:
            target = self._lookup_tag_by_lower(self._selected_tag_lower())
            self._selected_tag = target or self._all_tags[0]
            selector.value = self._selected_tag
        else:
            self._selected_tag = ""
            selector.value = self._NO_TAG_VALUE

    def _refresh_tool_rows(self) -> None:
        container = self.query_one("#tool_tag_manager_list", VerticalScroll)
        self._row_button_to_tool = {}
        for child in list(container.children):
            with contextlib.suppress(Exception):
                child.remove()

        selected_lower = self._selected_tag_lower()
        for index, tool_name in enumerate(self._tool_names):
            tags = list(self._tool_tags.get(tool_name, []))
            has_selected_tag = bool(
                selected_lower and any(str(tag).strip().lower() == selected_lower for tag in tags)
            )
            row_classes = "tool-tag-row " + ("tool-tag-row-even" if index % 2 == 0 else "tool-tag-row-odd")
            if has_selected_tag:
                row_classes += " tool-tag-row-associated"

            row = Horizontal(classes=row_classes)
            container.mount(row)
            meta = self._tool_meta_by_name.get(tool_name, {"access": "", "source": ""})
            row.mount(Static(tool_name, classes="tool-tag-cell tool-tag-name", markup=False))
            row.mount(Static(meta.get("access", ""), classes="tool-tag-cell tool-tag-access", markup=False))
            row.mount(Static(meta.get("source", ""), classes="tool-tag-cell tool-tag-source", markup=False))
            tag_text = ", ".join(tags)
            action_cell = Horizontal(classes="tool-tag-cell tool-tag-action-cell")
            row.mount(action_cell)
            action_cell.mount(Static(tag_text, classes="tool-tag-action-tags", markup=False))

            button_id = f"tool_tag_toggle_{index}"
            self._row_button_to_tool[button_id] = tool_name
            button_label = "Remove" if has_selected_tag else "Add"
            button_variant = "warning" if has_selected_tag else "success"
            action_cell.mount(
                Button(
                    button_label,
                    id=button_id,
                    variant=button_variant,
                    compact=True,
                    classes="tool-tag-action",
                )
            )

    def add_tag(self, tag_name: str) -> bool:
        token = str(tag_name or "").strip()
        lowered = token.lower()
        if not token:
            return False
        existing = self._lookup_tag_by_lower(lowered)
        if existing is not None:
            self._selected_tag = existing
            return False
        self._all_tags.append(token)
        self._all_tags.sort(key=str.lower)
        self._selected_tag = token
        return True

    def rename_selected_tag(self, tag_name: str) -> bool:
        old_tag = str(self._selected_tag or "").strip()
        old_lower = old_tag.lower()
        new_tag = str(tag_name or "").strip()
        new_lower = new_tag.lower()
        if not old_tag or not new_tag:
            return False

        changed = False
        for tool_name in self._tool_names:
            current = self._tool_tags.get(tool_name, [])
            next_tags: list[str] = []
            seen: set[str] = set()
            for tag in current:
                lowered = str(tag).strip().lower()
                if lowered in {old_lower, new_lower}:
                    candidate = new_tag
                else:
                    candidate = str(tag).strip()
                candidate_lower = candidate.lower()
                if not candidate or candidate_lower in seen:
                    continue
                seen.add(candidate_lower)
                next_tags.append(candidate)
                if candidate != tag:
                    changed = True
            self._tool_tags[tool_name] = next_tags

        tags: list[str] = []
        seen_tags: set[str] = set()
        for tag in self._all_tags:
            lowered = tag.lower()
            candidate = new_tag if lowered in {old_lower, new_lower} else tag
            candidate_lower = candidate.lower()
            if candidate_lower in seen_tags:
                continue
            seen_tags.add(candidate_lower)
            tags.append(candidate)
        if new_lower not in seen_tags:
            tags.append(new_tag)
        tags.sort(key=str.lower)
        self._all_tags = tags
        self._selected_tag = new_tag
        return changed or old_lower != new_lower or old_tag != new_tag

    def delete_selected_tag(self) -> bool:
        old_tag = str(self._selected_tag or "").strip()
        old_lower = old_tag.lower()
        if not old_tag:
            return False

        changed = False
        for tool_name in self._tool_names:
            current = self._tool_tags.get(tool_name, [])
            filtered = [tag for tag in current if str(tag).strip().lower() != old_lower]
            if len(filtered) != len(current):
                changed = True
            self._tool_tags[tool_name] = filtered

        self._all_tags = [tag for tag in self._all_tags if tag.lower() != old_lower]
        self._selected_tag = self._all_tags[0] if self._all_tags else ""
        return changed

    def toggle_tool_for_selected_tag(self, tool_name: str) -> bool:
        selected = str(self._selected_tag or "").strip()
        selected_lower = selected.lower()
        target_tool = str(tool_name or "").strip()
        if not selected or not target_tool or target_tool not in self._tool_tags:
            return False

        tags = list(self._tool_tags.get(target_tool, []))
        if any(str(tag).strip().lower() == selected_lower for tag in tags):
            tags = [tag for tag in tags if str(tag).strip().lower() != selected_lower]
        else:
            tags.append(selected)
        self._tool_tags[target_tool] = self._normalize_tags(tags)

        if self._lookup_tag_by_lower(selected_lower) is None:
            self._all_tags.append(selected)
            self._all_tags.sort(key=str.lower)
        return True

    def build_result_payload(self) -> dict[str, Any]:
        return {"tool_tags": {name: list(tags) for name, tags in self._tool_tags.items()}}

    def _resolve_tag_for_row_action(self) -> str | None:
        selected = str(self._selected_tag or "").strip()
        return selected or None

    def compose(self):  # type: ignore[override]
        options: list[tuple[str, str]] = (
            [(tag, tag) for tag in self._all_tags] if self._all_tags else [("(no tags)", self._NO_TAG_VALUE)]
        )
        selected_value = self._selected_tag if self._selected_tag else self._NO_TAG_VALUE
        with Vertical(id="tool_tag_manager_container"):
            yield Static("Tag Manager", id="tool_tag_manager_title")
            with Horizontal(id="tool_tag_manager_controls"):
                yield Select(options, id="tool_tag_manager_select", value=selected_value)
                yield Input(placeholder="Type a tag name", id="tool_tag_manager_input")
                yield Button("Add", id="tool_tag_manager_add", variant="success", compact=True)
                yield Button("Rename", id="tool_tag_manager_rename", variant="warning", compact=True)
                yield Button("Delete", id="tool_tag_manager_delete", variant="error", compact=True)
            with Horizontal(id="tool_tag_manager_list_header"):
                yield Static("Name", classes="tool-tag-cell tool-tag-name")
                yield Static("Access", classes="tool-tag-cell tool-tag-access")
                yield Static("Source", classes="tool-tag-cell tool-tag-source")
                yield Static("Tags", classes="tool-tag-cell tool-tag-action-cell")
            yield VerticalScroll(id="tool_tag_manager_list")
            yield Static(
                "Select a tag and use row buttons to add/remove associations.",
                id="tool_tag_manager_status",
            )
            with Horizontal(id="tool_tag_manager_buttons"):
                yield Button("Save", id="tool_tag_manager_save", variant="success")
                yield Button("Cancel", id="tool_tag_manager_cancel", variant="default")

    def on_mount(self) -> None:
        self._refresh_tag_select()
        self._refresh_tool_rows()
        with contextlib.suppress(Exception):
            self.query_one("#tool_tag_manager_select", Select).focus()

    def on_select_changed(self, event: Any) -> None:
        select_id = str(getattr(getattr(event, "select", None), "id", "")).strip()
        if select_id != "tool_tag_manager_select":
            return
        value = str(getattr(event, "value", "")).strip()
        if value == self._NO_TAG_VALUE:
            self._selected_tag = ""
        else:
            self._selected_tag = value
        self._refresh_tool_rows()

    def on_button_pressed(self, event: Any) -> None:
        button_id = str(getattr(getattr(event, "button", None), "id", "")).strip()
        if button_id == "tool_tag_manager_add":
            tag_name = str(self.query_one("#tool_tag_manager_input", Input).value or "").strip()
            if not tag_name:
                self._set_status("Enter a tag name to add.")
                return
            created = self.add_tag(tag_name)
            self._refresh_tag_select()
            self._refresh_tool_rows()
            self.query_one("#tool_tag_manager_input", Input).value = ""
            if created:
                self._set_status(f"Added tag '{self._selected_tag}'.")
            else:
                self._set_status(f"Selected existing tag '{self._selected_tag}'.")
            return
        if button_id in self._row_button_to_tool:
            resolved_tag = self._resolve_tag_for_row_action()
            if not resolved_tag:
                self._set_status("Select a tag before toggling, or add one first.")
                return
            self._selected_tag = resolved_tag
            tool_name = self._row_button_to_tool[button_id]
            selected_lower = self._selected_tag_lower()
            already_associated = any(
                str(tag).strip().lower() == selected_lower for tag in self._tool_tags.get(tool_name, [])
            )
            if self.toggle_tool_for_selected_tag(tool_name):
                self._refresh_tool_rows()
                action = "Removed" if already_associated else "Added"
                self._set_status(f"{action} '{self._selected_tag}' for {tool_name}.")
            return
        if button_id == "tool_tag_manager_rename":
            tag_name = str(self.query_one("#tool_tag_manager_input", Input).value or "").strip()
            if not self._selected_tag:
                self._set_status("Select a tag to rename.")
                return
            if not tag_name:
                self._set_status("Enter the new tag name.")
                return
            if self.rename_selected_tag(tag_name):
                self._refresh_tag_select()
                self._refresh_tool_rows()
                self.query_one("#tool_tag_manager_input", Input).value = ""
                self._set_status(f"Renamed tag to '{tag_name}'.")
            return
        if button_id == "tool_tag_manager_delete":
            if not self._selected_tag:
                self._set_status("Select a tag to delete.")
                return
            deleted_tag = self._selected_tag
            if self.delete_selected_tag():
                self._refresh_tag_select()
                self._refresh_tool_rows()
                self._set_status(f"Deleted tag '{deleted_tag}'.")
            else:
                self._set_status(f"Tag '{deleted_tag}' removed.")
            return
        if button_id == "tool_tag_manager_save":
            self.dismiss(self.build_result_payload())
            return
        if button_id == "tool_tag_manager_cancel":
            self.dismiss(None)

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.dismiss(None)


class ModelConfigManagerScreen(ModalScreen[dict[str, Any] | None]):
    """Popup editor for model catalog entries (selector + form, explicit save)."""

    _PROVIDERS = ("bedrock", "openai", "ollama", "github_copilot")

    DEFAULT_CSS = """
    ModelConfigManagerScreen {
        align: center middle;
    }
    ModelConfigManagerScreen #model_manager_container {
        width: 96;
        max-width: 98%;
        max-height: 96%;
        border: round $accent;
        background: $surface;
        padding: 1;
        layout: vertical;
    }
    ModelConfigManagerScreen #model_manager_title {
        height: auto;
        margin: 0 0 1 0;
        color: $text;
    }
    ModelConfigManagerScreen #model_manager_body {
        height: 1fr;
        border: round #4a5461;
        padding: 1;
        margin: 0 0 1 0;
        scrollbar-background: #2f2f2f;
        scrollbar-background-hover: #3a3a3a;
        scrollbar-background-active: #454545;
        scrollbar-color: #7f7f7f;
        scrollbar-color-hover: #999999;
        scrollbar-color-active: #b3b3b3;
    }
    ModelConfigManagerScreen #model_manager_controls,
    ModelConfigManagerScreen #model_manager_select_row {
        height: auto;
        layout: horizontal;
        margin: 0 0 1 0;
    }
    ModelConfigManagerScreen #model_manager_provider_filter,
    ModelConfigManagerScreen #model_manager_search,
    ModelConfigManagerScreen #model_manager_model_select {
        width: 1fr;
        margin: 0 1 0 0;
    }
    ModelConfigManagerScreen #model_manager_editor {
        height: auto;
        border: round #4a5461;
        margin: 0 0 1 0;
        padding: 1;
    }
    ModelConfigManagerScreen .model-edit-row {
        height: auto;
        layout: horizontal;
        margin: 0 0 1 0;
    }
    ModelConfigManagerScreen .model-edit-row Input,
    ModelConfigManagerScreen .model-edit-row Select {
        width: 1fr;
        margin: 0 1 0 0;
    }
    ModelConfigManagerScreen #model_manager_summary {
        height: auto;
        color: $text-muted;
        margin: 0 0 1 0;
    }
    ModelConfigManagerScreen #model_manager_status {
        height: auto;
        color: $text-muted;
        margin: 0 0 1 0;
    }
    ModelConfigManagerScreen #model_manager_buttons {
        layout: horizontal;
        height: auto;
    }
    ModelConfigManagerScreen #model_manager_buttons Button {
        width: 1fr;
        margin: 0 1 0 0;
    }
    """

    def __init__(self, settings_payload: dict[str, Any], **kwargs: object) -> None:
        super().__init__(**kwargs)
        payload = dict(settings_payload) if isinstance(settings_payload, dict) else {}
        models_payload = payload.get("models")
        env_payload = payload.get("env")
        self._models: dict[str, Any] = _json.loads(
            _json.dumps(models_payload if isinstance(models_payload, dict) else {})
        )
        self._env: dict[str, Any] = _json.loads(
            _json.dumps(env_payload if isinstance(env_payload, dict) else {})
        )
        self._provider_filter: str = "__all__"
        self._search: str = ""
        self._selected_entry: tuple[str, str] | None = None

    @staticmethod
    def _tier_key(raw: str) -> str:
        return str(raw or "").strip().lower()

    def _provider_options(self) -> list[tuple[str, str]]:
        return [(provider, provider) for provider in self._PROVIDERS]

    def _models_providers(self) -> dict[str, Any]:
        providers = self._models.setdefault("providers", {})
        return providers if isinstance(providers, dict) else {}

    def _default_pair(self) -> tuple[str | None, str]:
        default_selection = self._models.get("default_selection")
        if isinstance(default_selection, dict):
            provider = str(default_selection.get("provider", "")).strip().lower() or None
            tier = self._tier_key(str(default_selection.get("tier", "")).strip() or "balanced")
            return provider, tier
        provider = str(self._models.get("provider", "")).strip().lower() or None
        tier = self._tier_key(str(self._models.get("default_tier", "")).strip() or "balanced")
        return provider, tier

    def _set_default_pair(self, provider: str | None, tier: str) -> None:
        normalized_provider = str(provider or "").strip().lower() or None
        normalized_tier = self._tier_key(tier) or "balanced"
        self._models["default_selection"] = {"provider": normalized_provider, "tier": normalized_tier}
        self._models["provider"] = normalized_provider
        self._models["default_tier"] = normalized_tier

    def _model_rows(self) -> list[tuple[str, str, dict[str, Any]]]:
        rows: list[tuple[str, str, dict[str, Any]]] = []
        providers = self._models_providers()
        for provider_name, provider_payload in providers.items():
            provider = str(provider_name or "").strip().lower()
            if not provider or self._provider_filter not in {"", "__all__", provider}:
                continue
            tiers = provider_payload.get("tiers", {}) if isinstance(provider_payload, dict) else {}
            if not isinstance(tiers, dict):
                continue
            for tier_name, tier_payload in tiers.items():
                tier = self._tier_key(tier_name)
                if not tier or not isinstance(tier_payload, dict):
                    continue
                if self._search:
                    needle = self._search.lower()
                    haystack = " ".join(
                        [
                            provider,
                            tier,
                            str(tier_payload.get("model_id", "")),
                            str(tier_payload.get("display_name", "")),
                            str(tier_payload.get("description", "")),
                        ]
                    ).lower()
                    if needle not in haystack:
                        continue
                rows.append((provider, tier, tier_payload))
        return sorted(rows, key=lambda item: (item[0], item[1]))

    def _model_select_options(self) -> list[tuple[str, str]]:
        options = [("Select model...", "__none__")]
        for provider, tier, payload in self._model_rows():
            model_id = str(payload.get("model_id", "") or "").strip() or "(unset)"
            options.append((f"{provider}/{tier}  [{model_id}]", f"{provider}|{tier}"))
        return options

    def _set_status(self, text: str) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#model_manager_status", Static).update(str(text or "").strip())

    def _refresh_summary(self) -> None:
        default_provider, default_tier = self._default_pair()
        row_count = len(self._model_rows())
        default_label = f"{default_provider}/{default_tier}" if default_provider else f"auto/{default_tier}"
        with contextlib.suppress(Exception):
            self.query_one("#model_manager_summary", Static).update(
                f"Models: {row_count} | Default: {default_label}"
            )

    def _refresh_editor_provider_options(self, selected_provider: str | None = None) -> None:
        selector = self.query_one("#model_edit_provider", Select)
        options = self._provider_options()
        selector.set_options(options)
        target = str(selected_provider or "").strip().lower()
        if target not in {value for _label, value in options}:
            target = "openai"
        selector.value = target

    def _load_editor(self, *, provider: str, tier: str, payload: dict[str, Any]) -> None:
        self._selected_entry = (provider, tier)
        self._refresh_editor_provider_options(provider)
        self.query_one("#model_edit_tier", Input).value = tier
        self.query_one("#model_edit_model_id", Input).value = str(payload.get("model_id", "") or "")
        self.query_one("#model_edit_display_name", Input).value = str(payload.get("display_name", "") or "")
        self.query_one("#model_edit_description", Input).value = str(payload.get("description", "") or "")
        provider_key = provider.upper()
        self.query_one("#model_edit_price_input", Input).value = str(
            self._env.get(f"SWARMEE_PRICE_{provider_key}_INPUT_PER_1M", "") or ""
        )
        self.query_one("#model_edit_price_output", Input).value = str(
            self._env.get(f"SWARMEE_PRICE_{provider_key}_OUTPUT_PER_1M", "") or ""
        )
        self.query_one("#model_edit_price_cached", Input).value = str(
            self._env.get(f"SWARMEE_PRICE_{provider_key}_CACHED_INPUT_PER_1M", "") or ""
        )

    def _set_model_select_value(self, provider: str | None, tier: str | None) -> None:
        target = "__none__"
        if provider and tier:
            target = f"{provider}|{tier}"
        selector = self.query_one("#model_manager_model_select", Select)
        with contextlib.suppress(Exception):
            selector.value = target

    def _refresh_model_select_options(self) -> None:
        selector = self.query_one("#model_manager_model_select", Select)
        options = self._model_select_options()
        selector.set_options(options)
        selected_value = "__none__"
        if self._selected_entry is not None:
            selected_value = f"{self._selected_entry[0]}|{self._selected_entry[1]}"
        values = {value for _label, value in options}
        if selected_value not in values:
            self._selected_entry = None
            selected_value = "__none__"
        with contextlib.suppress(Exception):
            selector.value = selected_value
        self._refresh_summary()

    def _clear_editor(self, *, keep_selection: bool = False) -> None:
        if not keep_selection:
            self._selected_entry = None
        self._refresh_editor_provider_options("openai")
        self.query_one("#model_edit_tier", Input).value = ""
        self.query_one("#model_edit_model_id", Input).value = ""
        self.query_one("#model_edit_display_name", Input).value = ""
        self.query_one("#model_edit_description", Input).value = ""
        self.query_one("#model_edit_price_input", Input).value = ""
        self.query_one("#model_edit_price_output", Input).value = ""
        self.query_one("#model_edit_price_cached", Input).value = ""
        self._set_model_select_value(None, None)

    def _load_selected_entry(self) -> None:
        selector = self.query_one("#model_manager_model_select", Select)
        raw = str(getattr(selector, "value", "__none__") or "__none__").strip().lower()
        if "|" not in raw:
            self._selected_entry = None
            self._set_status("Select an existing model or choose New Model.")
            return
        provider, tier = raw.split("|", 1)
        provider_payload = self._models_providers().get(provider, {})
        tiers = provider_payload.get("tiers", {}) if isinstance(provider_payload, dict) else {}
        payload = tiers.get(tier, {}) if isinstance(tiers, dict) else {}
        if not isinstance(payload, dict):
            self._selected_entry = None
            self._set_status("Selected model is unavailable.")
            return
        self._load_editor(provider=provider, tier=tier, payload=payload)
        self._set_status(f"Editing {provider}/{tier}.")

    def _save_editor_entry(self) -> bool:
        provider = str(self.query_one("#model_edit_provider", Select).value or "").strip().lower()
        tier = self._tier_key(str(self.query_one("#model_edit_tier", Input).value or ""))
        model_id = str(self.query_one("#model_edit_model_id", Input).value or "").strip()
        if provider not in self._PROVIDERS:
            self._set_status("Select a valid provider.")
            return False
        if not tier:
            self._set_status("Tier is required.")
            return False
        if not model_id:
            self._set_status("model_id is required.")
            return False
        providers = self._models_providers()
        provider_payload = providers.setdefault(provider, {})
        if not isinstance(provider_payload, dict):
            provider_payload = {}
            providers[provider] = provider_payload
        tiers = provider_payload.setdefault("tiers", {})
        if not isinstance(tiers, dict):
            tiers = {}
            provider_payload["tiers"] = tiers
        if (
            self._selected_entry is not None
            and (self._selected_entry[0] != provider or self._selected_entry[1] != tier)
        ):
            with contextlib.suppress(Exception):
                prev_provider = providers.get(self._selected_entry[0], {})
                prev_tiers = prev_provider.get("tiers", {}) if isinstance(prev_provider, dict) else {}
                if isinstance(prev_tiers, dict):
                    prev_tiers.pop(self._selected_entry[1], None)
        entry: dict[str, Any] = {
            "provider": provider,
            "model_id": model_id,
        }
        display_name = str(self.query_one("#model_edit_display_name", Input).value or "").strip()
        description = str(self.query_one("#model_edit_description", Input).value or "").strip()
        if display_name:
            entry["display_name"] = display_name
        if description:
            entry["description"] = description
        tiers[tier] = entry
        provider_key = provider.upper()
        for suffix, selector in (
            ("INPUT_PER_1M", "#model_edit_price_input"),
            ("OUTPUT_PER_1M", "#model_edit_price_output"),
            ("CACHED_INPUT_PER_1M", "#model_edit_price_cached"),
        ):
            env_key = f"SWARMEE_PRICE_{provider_key}_{suffix}"
            value = str(self.query_one(selector, Input).value or "").strip()
            if value:
                self._env[env_key] = value
            else:
                self._env.pop(env_key, None)
        self._selected_entry = (provider, tier)
        self._set_model_select_value(provider, tier)
        self._set_status(f"Saved draft entry for {provider}/{tier}.")
        return True

    def _delete_entry(self, provider: str, tier: str) -> bool:
        providers = self._models_providers()
        provider_payload = providers.get(provider, {})
        tiers = provider_payload.get("tiers", {}) if isinstance(provider_payload, dict) else {}
        removed = False
        if isinstance(tiers, dict):
            removed = tiers.pop(tier, None) is not None
            if not tiers:
                providers.pop(provider, None)
        default_provider, default_tier = self._default_pair()
        if provider == default_provider and tier == default_tier:
            self._set_default_pair(None, "balanced")
        if self._selected_entry == (provider, tier):
            self._selected_entry = None
            self._clear_editor(keep_selection=True)
        return removed

    def _set_current_entry_default(self) -> bool:
        if self._selected_entry is None:
            return False
        provider, tier = self._selected_entry
        self._set_default_pair(provider, tier)
        return True

    def _ensure_valid_default_pair(self) -> None:
        default_provider, default_tier = self._default_pair()
        rows = self._model_rows()
        for provider, tier, _payload in rows:
            if provider == default_provider and tier == default_tier:
                return
        if rows:
            provider, tier, _payload = rows[0]
            self._set_default_pair(provider, tier)
            return
        self._set_default_pair(None, "balanced")

    def build_result_payload(self) -> dict[str, Any]:
        self._ensure_valid_default_pair()
        hidden_raw = self._models.get("hidden_tiers")
        if isinstance(hidden_raw, list):
            existing = set()
            for provider, provider_payload in self._models_providers().items():
                tiers = provider_payload.get("tiers", {}) if isinstance(provider_payload, dict) else {}
                if not isinstance(tiers, dict):
                    continue
                for tier in tiers.keys():
                    existing.add((str(provider).strip().lower(), self._tier_key(str(tier))))
            filtered_hidden: list[str] = []
            seen: set[str] = set()
            for item in hidden_raw:
                token = str(item or "").strip().lower()
                if "|" not in token:
                    continue
                provider, tier = token.split("|", 1)
                key = (provider.strip(), self._tier_key(tier))
                normalized = f"{key[0]}|{key[1]}"
                if key not in existing or normalized in seen:
                    continue
                seen.add(normalized)
                filtered_hidden.append(normalized)
            self._models["hidden_tiers"] = filtered_hidden
        return {"models": self._models, "env": self._env}

    def compose(self):  # type: ignore[override]
        filter_options = [("All providers", "__all__")] + [(provider, provider) for provider in self._PROVIDERS]
        with Vertical(id="model_manager_container"):
            yield Static("Model Manager", id="model_manager_title")
            with VerticalScroll(id="model_manager_body"):
                with Horizontal(id="model_manager_controls"):
                    yield Select(filter_options, id="model_manager_provider_filter", value=self._provider_filter)
                    yield Input(placeholder="Search...", id="model_manager_search")
                with Horizontal(id="model_manager_select_row"):
                    yield Select(
                        [("Select model...", "__none__")],
                        id="model_manager_model_select",
                        allow_blank=False,
                    )
                    yield Button("New Model", id="model_manager_new_entry", variant="success", compact=True)
                yield Static("", id="model_manager_summary")
                with Vertical(id="model_manager_editor"):
                    with Horizontal(classes="model-edit-row"):
                        yield Select(self._provider_options(), id="model_edit_provider", value="openai")
                        yield Input(placeholder="tier", id="model_edit_tier")
                        yield Input(placeholder="model_id", id="model_edit_model_id")
                    with Horizontal(classes="model-edit-row"):
                        yield Input(placeholder="display_name", id="model_edit_display_name")
                        yield Input(placeholder="description", id="model_edit_description")
                    with Horizontal(classes="model-edit-row"):
                        yield Input(placeholder="input $ / 1M", id="model_edit_price_input")
                        yield Input(placeholder="output $ / 1M", id="model_edit_price_output")
                        yield Input(placeholder="cached input $ / 1M", id="model_edit_price_cached")
                    with Horizontal(classes="model-edit-row"):
                        yield Button("Save Entry", id="model_edit_save", variant="success", compact=True)
                        yield Button("Set Default", id="model_edit_set_default", variant="primary", compact=True)
                        yield Button("Delete", id="model_edit_delete", variant="error", compact=True)
                        yield Button("Clear", id="model_edit_clear", variant="default", compact=True)
                with Horizontal(classes="model-edit-row"):
                    yield Static("Esc to cancel without saving.", id="model_manager_status")
            with Horizontal(id="model_manager_buttons"):
                yield Button("Save", id="model_manager_save", variant="success")
                yield Button("Cancel", id="model_manager_cancel", variant="default")

    def on_mount(self) -> None:
        self._refresh_model_select_options()
        self._clear_editor()
        self._refresh_model_select_options()
        self._refresh_summary()
        with contextlib.suppress(Exception):
            self.query_one("#model_manager_model_select", Select).focus()

    def on_select_changed(self, event: Any) -> None:
        select_id = str(getattr(getattr(event, "select", None), "id", "")).strip()
        if select_id == "model_manager_provider_filter":
            self._provider_filter = str(getattr(event, "value", "__all__") or "__all__")
            self._refresh_model_select_options()
            return
        if select_id == "model_manager_model_select":
            self._load_selected_entry()

    def on_input_changed(self, event: Any) -> None:
        input_id = str(getattr(getattr(event, "input", None), "id", "")).strip()
        if input_id == "model_manager_search":
            self._search = str(getattr(event, "value", "") or "").strip()
            self._refresh_model_select_options()

    def on_button_pressed(self, event: Any) -> None:
        button_id = str(getattr(getattr(event, "button", None), "id", "")).strip()
        if button_id == "model_manager_new_entry":
            self._clear_editor()
            self._set_status("Ready for new model entry.")
            return
        if button_id == "model_edit_clear":
            self._clear_editor()
            self._set_status("Editor cleared.")
            return
        if button_id == "model_edit_save":
            if self._save_editor_entry():
                self._refresh_model_select_options()
            return
        if button_id == "model_edit_set_default":
            if self._set_current_entry_default():
                self._refresh_summary()
                self._set_status("Default model updated.")
            else:
                self._set_status("Select or save a model first.")
            return
        if button_id == "model_edit_delete":
            if self._selected_entry is None:
                self._set_status("Select a model to delete.")
                return
            provider, tier = self._selected_entry
            if self._delete_entry(provider, tier):
                self._refresh_model_select_options()
                self._set_status(f"Deleted {provider}/{tier}.")
            else:
                self._set_status("Selected model no longer exists.")
            return
        if button_id == "model_manager_save":
            self.dismiss(self.build_result_payload())
            return
        if button_id == "model_manager_cancel":
            self.dismiss(None)

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.dismiss(None)


class SystemMessage(Static):
    """TUI-internal messages (welcome, hints, status)."""

    DEFAULT_CSS = """
    SystemMessage {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, text: str, **kwargs: object) -> None:
        super().__init__(f"[dim]{text}[/dim]", **kwargs)
