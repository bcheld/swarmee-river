import time
from threading import Event
from typing import Any

from colorama import Fore, Style, init
from halo import Halo
from rich.status import Status

# Initialize Colorama
init(autoreset=True)

# Configure spinner templates
SPINNERS = {
    "dots": {
        "interval": 80,
        "frames": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    }
}

# Tool state colors
TOOL_COLORS = {
    "running": Fore.GREEN,
    "success": Fore.GREEN,
    "error": Fore.RED,
    "info": Fore.CYAN,
}


def format_message(message: str, color: str | None = None, max_length: int = 50) -> str:
    """Format message with color and length control."""
    if len(message) > max_length:
        message = message[:max_length] + "..."
    return f"{color if color else ''}{message}{Style.RESET_ALL}"


class ToolSpinner:
    def __init__(self, text: str = "", color: str | int = TOOL_COLORS["running"]) -> None:
        self.spinner = Halo(
            text=text,
            spinner=SPINNERS["dots"],
            color="green",
            text_color="green",
            interval=80,
        )
        self.color = color
        self.current_text = text

    def start(self, text: str | None = None) -> None:
        if text:
            self.current_text = text
        print()  # Move to new line before starting spinner, prevents spinner from eating the previous line
        self.spinner.start(f"{self.color}{self.current_text}{Style.RESET_ALL}")

    def update(self, text: str) -> None:
        self.current_text = text
        self.spinner.text = f"{self.color}{text}{Style.RESET_ALL}"

    def succeed(self, text: str | None = None) -> None:
        if text:
            self.current_text = text
        self.spinner.succeed(f"{TOOL_COLORS['success']}{self.current_text}{Style.RESET_ALL}")

    def fail(self, text: str | None = None) -> None:
        if text:
            self.current_text = text
        self.spinner.fail(f"{TOOL_COLORS['error']}{self.current_text}{Style.RESET_ALL}")

    def info(self, text: str | None = None) -> None:
        if text:
            self.current_text = text
        self.spinner.info(f"{TOOL_COLORS['info']}{self.current_text}{Style.RESET_ALL}")

    def stop(self) -> None:
        self.spinner.stop()


class CallbackHandler:
    def __init__(self) -> None:
        self.thinking_spinner: Status | None = None

        # Tool tracking
        self.current_spinner: ToolSpinner | None = None
        self.current_tool: str | None = None
        self.tool_histories: dict[str, dict[str, Any]] = {}
        self.interrupt_event: Event | None = None

    def notify(self, title: str, message: str, sound: bool = True) -> None:
        """Send a native notification using mac_automation tool."""
        print(f"Notification: {title} - {message}")

    def callback_handler(
        self,
        reasoningText: str | bool = False,
        data: str = "",
        complete: bool = False,
        force_stop: bool = False,
        message: dict[str, Any] | None = None,
        current_tool_use: dict[str, Any] | None = None,
        init_event_loop: bool = False,
        start_event_loop: bool = False,
        event_loop_throttled_delay: int | None = None,
        console: Any = None,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[Any] | None = None,
        structured_output_prompt: str | None = None,
        result: Any = None,
        **extra_event_fields: Any,
    ) -> None:
        # Future-compatible callback shape support:
        # Strands may add event fields (or merge invocation_state keys into callback kwargs).
        # Keep these accepted and ignored unless needed by this handler.
        del invocation_state
        del structured_output_model
        del structured_output_prompt
        del result
        del extra_event_fields
        message = message or {}
        current_tool_use = current_tool_use or {}

        # Cleanup calls (e.g., from top-level exception handlers) should never re-raise an interrupt.
        if force_stop:
            if self.thinking_spinner:
                self.thinking_spinner.stop()
            if self.current_spinner:
                self.current_spinner.stop()
            return

        if self.interrupt_event is not None and self.interrupt_event.is_set():
            # Don't raise from inside the callback path. Interrupt cancellation is handled by the caller.
            # Raising here can cause noisy generator shutdown errors in upstream libraries.
            if self.thinking_spinner:
                self.thinking_spinner.stop()
            if self.current_spinner:
                self.current_spinner.stop()
            return

        try:
            # Concurrent thinking spinners are usual, which leads to:
            # "Only one live display may be active at once" error thrown,
            # This try except block ignore overlap of thinking spinners.
            if self.thinking_spinner and (data or current_tool_use):
                self.thinking_spinner.stop()

            if init_event_loop:
                self.thinking_spinner = Status(
                    "[blue] retrieving memories...[/blue]",
                    spinner="dots",
                    console=console,
                )
                self.thinking_spinner.start()

            if reasoningText:
                print(reasoningText, end="")

            if start_event_loop and self.thinking_spinner is not None:
                self.thinking_spinner.update("[blue] thinking...[/blue]")
        except BaseException:
            pass

        if event_loop_throttled_delay and console:
            if self.current_spinner:
                self.current_spinner.stop()
            console.print(
                f"[red]Throttled! Waiting [bold]{event_loop_throttled_delay} seconds[/bold] before retrying...[/red]"
            )

        # Handle regular output
        if data:
            # Print to stdout
            if complete:
                print(f"{Fore.WHITE}{data}{Style.RESET_ALL}")
            else:
                print(f"{Fore.WHITE}{data}{Style.RESET_ALL}", end="")

        # Handle tool input streaming
        if current_tool_use and current_tool_use.get("input"):
            raw_tool_id = current_tool_use.get("toolUseId")
            tool_id = raw_tool_id if isinstance(raw_tool_id, str) else None
            tool_name = current_tool_use.get("name")
            tool_input = current_tool_use.get("input", "")

            if tool_id is not None:
                # Check if this is a new tool execution
                if tool_id != self.current_tool:
                    # Stop previous spinner if exists
                    if self.current_spinner:
                        self.current_spinner.stop()

                    self.current_tool = tool_id

                    self.current_spinner = ToolSpinner(f"🛠️  {tool_name}: Preparing...", TOOL_COLORS["running"])
                    self.current_spinner.start()

                    # Record tool start
                    self.tool_histories[tool_id] = {
                        "name": tool_name,
                        "start_time": time.time(),
                        "input_size": 0,
                    }

                # Update tool progress
                if tool_id in self.tool_histories:
                    current_size = len(tool_input)
                    if current_size > self.tool_histories[tool_id]["input_size"]:
                        self.tool_histories[tool_id]["input_size"] = current_size
                        if self.current_spinner:
                            self.current_spinner.update(f"🛠️  {tool_name}: {current_size} chars")

        # Process messages
        if isinstance(message, dict):
            # Handle assistant messages (tool starts)
            if message.get("role") == "assistant":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_use = content.get("toolUse")
                        if tool_use:
                            tool_name = tool_use.get("name")
                            if self.current_spinner:
                                self.current_spinner.info(f"🔧 Starting {tool_name}...")

            # Handle user messages (tool results)
            elif message.get("role") == "user":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_result = content.get("toolResult")
                        if tool_result:
                            tool_id = tool_result.get("toolUseId")
                            status = tool_result.get("status")

                            if isinstance(tool_id, str) and tool_id in self.tool_histories:
                                tool_info = self.tool_histories[tool_id]
                                duration = round(time.time() - tool_info["start_time"], 2)

                                # Prepare notification message
                                if status == "success":
                                    status_message = f"{tool_info['name']} completed in {duration}s"
                                else:
                                    status_message = f"{tool_info['name']} failed after {duration}s"

                                # Update spinner only if not in Lambda
                                if self.current_spinner:
                                    if status == "success":
                                        self.current_spinner.succeed(status_message)
                                    else:
                                        self.current_spinner.fail(status_message)

                                # Send notification
                                # Uncomment for enabling notifications.
                                # self.notify(title, message, sound=(status != "success"))

                                # Cleanup
                                del self.tool_histories[tool_id]
                                self.current_spinner = None
                                self.current_tool = None


callback_handler_instance = CallbackHandler()
callback_handler = callback_handler_instance.callback_handler


def set_interrupt_event(event: Event | None) -> None:
    callback_handler_instance.interrupt_event = event
