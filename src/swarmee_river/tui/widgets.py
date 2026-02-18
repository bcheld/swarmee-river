"""Custom Textual widgets for the conversation-style TUI transcript."""

from __future__ import annotations

from textual.widgets import Static


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
    """Accumulates text_delta events and renders as rich text."""

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
        self.update("".join(self._buffer))

    def finalize(self) -> str:
        """Called on text_complete. Returns the full text."""
        full_text = "".join(self._buffer)
        self.update(full_text)
        return full_text

    @property
    def full_text(self) -> str:
        return "".join(self._buffer)


class ToolCallBlock(Static):
    """Single-line status for a tool call: running → success/error."""

    DEFAULT_CSS = """
    ToolCallBlock {
        color: $text-muted;
        padding: 0 1;
        margin: 0 0 0 0;
    }
    """

    def __init__(self, tool_name: str, tool_use_id: str, **kwargs: object) -> None:
        self._tool_name = tool_name
        self._tool_use_id = tool_use_id
        super().__init__(f"[dim]⚙ {tool_name} running...[/dim]", **kwargs)

    def update_progress(self, chars: int) -> None:
        self.update(f"[dim]⚙ {self._tool_name} ({chars} chars)...[/dim]")

    def set_result(self, status: str, duration_s: float) -> None:
        if status == "success":
            self.update(f"[green]✓ {self._tool_name}[/green] [dim]({duration_s:.1f}s)[/dim]")
        else:
            self.update(f"[red]✗ {self._tool_name} ({status})[/red] [dim]({duration_s:.1f}s)[/dim]")


class ThinkingIndicator(Static):
    """Shows a thinking indicator, removed when text starts."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__("[dim]thinking...[/dim]", **kwargs)


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
