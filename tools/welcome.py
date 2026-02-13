"""Tool for managing the Swarmee welcome text."""

from pathlib import Path

from strands.types.tools import ToolResult, ToolUse

# Default welcome text (kept intentionally small to avoid bloating the system prompt)
DEFAULT_WELCOME_TEXT = """# Swarmee

Enterprise analytics + coding assistant built on Strands Agents.

## First things to try
- Ask for a concrete task: "find the bug in <file>" or "add tests for <module>".
- Use `:help` to see all command shortcuts.
- Use `:tools` to list currently available tools.

## Useful commands
- `:plan` force the next prompt into plan-then-approve mode.
- `:approve` or `:cancel` handle a pending plan.
- `:session new|save|load|list` manage project-local sessions.
- `:status`, `:diff`, `:log tail`, `:config show` for diagnostics.
- `!<command>` run a one-off shell command.

## Workflow tips
- Tools in `./tools/*.py` are hot-loaded.
- Keep prompts focused and context lean for faster iterations.
- If configured, use `--kb <ID>` for Bedrock Knowledge Base retrieval/storage.

## Exit
Type `exit` or `quit`.
"""

TOOL_SPEC = {
    "name": "welcome",
    "description": (
        "Edit and manage Swarmee welcome text in cwd()/.welcome. Can also be used as a "
        "shared scratchpad for inter-session communication, status tracking, and coordination between "
        "multiple Swarmee instances."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["view", "edit"],
                    "description": "Action to perform: view or edit welcome text",
                },
                "content": {
                    "type": "string",
                    "description": "New welcome text content when action is edit",
                },
            },
            "required": ["action"],
        }
    },
}


def _welcome_path(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()) / ".welcome"


def read_welcome_text(*, cwd: Path | None = None) -> str:
    path = _welcome_path(cwd)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_WELCOME_TEXT


def write_welcome_text(content: str, *, cwd: Path | None = None) -> None:
    path = _welcome_path(cwd)
    path.write_text(content, encoding="utf-8")


def welcome(tool: ToolUse) -> ToolResult:
    """Tool implementation for managing welcome text.

    Beyond simple welcome text management, this tool can be used creatively as:
    1. Inter-session communication channel - Share information between different Swarmee sessions
    2. Status tracking - Monitor long-running tasks across multiple sessions
    3. Coordination mechanism - Establish handoffs between different instances
    4. Persistent scratchpad - Store temporary information that persists between sessions

    Since all Swarmee instances read from Path.cwd()/.welcome at startup, information stored
    here is immediately available to any new Swarmee session.
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    action = tool_input["action"]

    try:
        # Create file if doesn't exist
        if action == "edit":
            if "content" not in tool_input:
                raise ValueError("content is required for edit action")

            content = tool_input["content"]
            write_welcome_text(content)

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": "Welcome text updated successfully"}],
            }

        elif action == "view":
            content = read_welcome_text()

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": content}],
            }

        else:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}],
        }
