"""Custom Textual widgets for the conversation-style TUI transcript."""

from __future__ import annotations

import contextlib
import json as _json
import re
from typing import Any

from rich.console import Group as RichGroup
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel as RichPanel
from rich.text import Text as RichText
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Collapsible, Static


def render_user_message(text: str, *, timestamp: str | None = None) -> RichText:
    """Render a user message line for RichLog."""
    rendered = RichText()
    rendered.append("YOU>", style="bold cyan")
    rendered.append(f" {text}")
    if isinstance(timestamp, str) and timestamp.strip():
        rendered.append(f"\n{timestamp.strip()}", style="dim")
    return rendered


def render_system_message(text: str) -> RichText:
    """Render a dimmed system/info line for RichLog."""
    return RichText(text, style="dim")


def render_thinking_message() -> RichText:
    """Render a compact thinking placeholder."""
    return RichText("thinking...", style="dim")


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


def render_tool_start_line(tool_name: str, *, tool_use_id: str | None = None) -> RichText:
    """Render a compact tool start line."""
    suffix = f" [{tool_use_id}]" if isinstance(tool_use_id, str) and tool_use_id.strip() else ""
    return RichText(f"⚙ {tool_name}{suffix} running...", style="dim")


def render_tool_result_line(
    tool_name: str,
    *,
    status: str,
    duration_s: float,
    tool_use_id: str | None = None,
) -> RichText:
    """Render a compact tool result line."""
    succeeded = status == "success"
    status_glyph = "✓" if succeeded else "✗"
    status_style = "green" if succeeded else "red"
    suffix = f" [{tool_use_id}]" if isinstance(tool_use_id, str) and tool_use_id.strip() else ""
    rendered = RichText()
    rendered.append("⚙ ")
    rendered.append(tool_name)
    rendered.append(suffix)
    rendered.append(f" ({duration_s:.1f}s) ")
    rendered.append(status_glyph, style=f"bold {status_style}")
    if not succeeded:
        rendered.append(f" ({status})", style="dim")
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
    suffix = f" [{tool_use_id}]" if isinstance(tool_use_id, str) and tool_use_id.strip() else ""
    return RichText(f"⚙ {tool_name}{suffix} running... ({elapsed_s:.1f}s)", style="dim")


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
            status = step_statuses[index - 1] if isinstance(step_statuses, list) and index - 1 < len(step_statuses) else ""
            if status == "in_progress":
                marker = "▶"
            elif status == "completed":
                marker = "☑"
            lines.append(f"{marker} {index}. {desc}")
    else:
        lines.append("(no steps)")
    lines.append("")
    lines.append("/approve  /replan  /clearplan")
    return RichPanel(RichText("\n".join(lines)), title="Plan", border_style="green")


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

    def finalize(self) -> str:
        """Called on text_complete. Returns the full raw text."""
        full = "".join(self._buffer)
        meta_parts = [part for part in [self._model, self._timestamp] if part]
        if meta_parts:
            meta_line = " · ".join(meta_parts)
            self.update(RichGroup(RichMarkdown(full), RichText(meta_line, style="dim")))
        return full

    @property
    def full_text(self) -> str:
        return "".join(self._buffer)


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


class ToolCallBlock(Static):
    """Collapsible tool call block: shows header with status, expands to show input details."""

    DEFAULT_CSS = """
    ToolCallBlock {
        color: $text-muted;
        padding: 0 1;
        margin: 0 0 0 0;
    }
    ToolCallBlock Collapsible {
        padding: 0;
        margin: 0;
        border: none;
    }
    ToolCallBlock .tool-details {
        color: $text-muted;
        padding: 0 0 0 2;
    }
    """

    def __init__(self, tool_name: str, tool_use_id: str, **kwargs: object) -> None:
        self._tool_name = tool_name
        self._tool_use_id = tool_use_id
        self._tool_input: dict = {}
        self._status_text = "running..."
        self._result_text: str | None = None
        super().__init__(**kwargs)

    def compose(self):  # type: ignore[override]
        with Collapsible(title=self._header_text(), collapsed=True):
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
            formatted = _format_tool_input(self._tool_name, self._tool_input)
            details.update(formatted if formatted else "(no input details)")
        except Exception:
            pass

    def set_input(self, tool_input: dict) -> None:
        self._tool_input = tool_input
        self._refresh_details()

    def update_progress(self, chars: int) -> None:
        self._status_text = f"({chars} chars)..."
        self._refresh_header()

    def set_result(self, status: str, duration_s: float) -> None:
        if status == "success":
            self._result_text = f"\u2713 {self._tool_name} ({duration_s:.1f}s)"
        else:
            self._result_text = f"\u2717 {self._tool_name} ({status}) ({duration_s:.1f}s)"
        self._refresh_header()


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
        super().__init__("[dim]thinking...[/dim]", **kwargs)

    def on_mount(self) -> None:
        self._frame_index = 0
        self._timer = self.set_interval(0.4, self._animate)

    def _animate(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(self._FRAMES)
        self.update(f"[dim]{self._FRAMES[self._frame_index]}[/dim]")


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

        if steps:
            lines.append("")
        lines.append("[dim]/approve  /replan  /clearplan[/dim]")
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
        if steps:
            lines.append("")
        lines.append("[dim]/approve  /replan  /clearplan[/dim]")
        return "\n".join(lines)


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
        self._prompt_tokens_est = prompt_tokens_est if isinstance(prompt_tokens_est, int) and prompt_tokens_est >= 0 else None
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
        if not isinstance(self._prompt_tokens_est, int) or not isinstance(self._budget_tokens, int) or self._budget_tokens <= 0:
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
            rendered = f"{value/1_000_000:.1f}m"
            return rendered.replace(".0m", "m")
        if value >= 1_000:
            rendered = f"{value/1_000:.1f}k"
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
        rendered.append(
            f"Context: {self._format_tokens(self._prompt_tokens_est)} / {self._format_tokens(self._budget_tokens)} ({pct}%)",
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
        ("/sop", "Browse and toggle SOPs"),
        ("/copy", "Copy transcript"),
        ("/copy plan", "Copy plan text"),
        ("/copy issues", "Copy issues"),
        ("/copy last", "Copy last response"),
        ("/copy all", "Copy everything"),
        ("/expand", "Expand tool details"),
        ("/open", "Open artifact by number"),
        ("/search", "Search transcript"),
        ("/consent", "Respond to consent"),
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
        self._filtered = [
            (cmd, desc) for cmd, desc in self.TUI_COMMANDS if cmd.startswith(prefix_lower)
        ]
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
            return f"{n/1000.0:.0f}k"
        if n >= 10_000:
            return f"{n/1000.0:.1f}k"
        if n >= 1000:
            return f"{n/1000.0:.2f}k"
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
            drop_prefixes = ("warn=", "err=",)
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
