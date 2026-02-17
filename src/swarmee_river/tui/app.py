"""Optional Textual app scaffold for `swarmee tui`."""

from __future__ import annotations

import importlib
import sys
from typing import Any


def run_tui() -> int:
    """Run the full-screen TUI if Textual is installed."""
    try:
        textual_app = importlib.import_module("textual.app")
        textual_widgets = importlib.import_module("textual.widgets")
    except ImportError:
        print(
            "Textual is required for `swarmee tui`.\n"
            'Install with: pip install "swarmee-river[tui]"\n'
            'For editable installs in this repo: pip install -e ".[tui]"',
            file=sys.stderr,
        )
        return 1

    AppBase = textual_app.App
    Header = textual_widgets.Header
    Footer = textual_widgets.Footer
    Input = textual_widgets.Input
    RichLog = textual_widgets.RichLog

    class SwarmeeTUI(AppBase):
        CSS = """
        Screen {
            layout: vertical;
        }

        #transcript {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
        }

        #prompt {
            dock: bottom;
        }
        """

        def compose(self) -> Any:
            yield Header()
            yield RichLog(id="transcript", auto_scroll=True, wrap=True)
            yield Input(placeholder="Type here (/exit to quit)", id="prompt")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#prompt", Input).focus()
            transcript = self.query_one("#transcript", RichLog)
            transcript.write("Swarmee TUI scaffold ready. Type /exit or :exit to quit.")

        def on_input_submitted(self, event: Any) -> None:
            text = event.value.strip()
            event.input.value = ""
            if not text:
                return

            transcript = self.query_one("#transcript", RichLog)
            transcript.write(f"> {text}")
            if text in {"/exit", ":exit"}:
                self.exit(return_code=0)

    try:
        SwarmeeTUI().run()
    except KeyboardInterrupt:
        return 130
    return 0
