"""Custom Textual widgets for the conversation-style TUI transcript."""

from __future__ import annotations

import json as _json

from rich.markdown import Markdown as RichMarkdown
from textual.widgets import Collapsible, Static


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

    def __init__(self, text: str, **kwargs: object) -> None:
        super().__init__(f"[bold cyan]YOU>[/bold cyan] {text}", **kwargs)


class AssistantMessage(Static):
    """Accumulates text_delta events and renders as markdown via Rich."""

    DEFAULT_CSS = """
    AssistantMessage {
        padding: 0 1;
        margin: 0 0 0 0;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)
        self._buffer: list[str] = []

    def append_delta(self, text: str) -> None:
        self._buffer.append(text)
        full = "".join(self._buffer)
        self.update(RichMarkdown(full))

    def finalize(self) -> str:
        """Called on text_complete. Returns the full raw text."""
        return "".join(self._buffer)

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
        border: round $warning;
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
            check = "\u2611" if (i < len(self._step_status) and self._step_status[i]) else "\u2610"
            lines.append(f"  {check} {i + 1}. {desc}")
        if steps:
            lines.append("")
        lines.append("[dim]/approve  /replan  /clearplan[/dim]")
        return "\n".join(lines)


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
        ("/copy", "Copy transcript"),
        ("/copy plan", "Copy plan text"),
        ("/copy issues", "Copy issues"),
        ("/copy last", "Copy last response"),
        ("/copy all", "Copy everything"),
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
        self._render()

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

    def refresh_display(self) -> None:
        icon = "\u25b6" if self._state == "running" else "\u23f8"  # ▶ / ⏸
        parts = [
            f"{icon} {self._state}",
            self._model or "Model: ?",
            f"Tools: {self._tool_count}",
            f"{self._elapsed:.1f}s",
        ]
        if self._warning_count:
            parts.append(f"warn={self._warning_count}")
        if self._error_count:
            parts.append(f"err={self._error_count}")
        self.update(" | ".join(parts))


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
