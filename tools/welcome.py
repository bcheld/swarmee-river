"""Tool for managing the Swarmee welcome text."""

import os
from pathlib import Path
from typing import Any

from strands.types.tools import ToolResult, ToolUse

# Default welcome text (kept intentionally small to avoid bloating the system prompt)
DEFAULT_WELCOME_TEXT = """# Swarmee

An enterprise analytics + coding assistant built on Strands Agents.

## Quick tips
- Tools in `./tools/*.py` are hot-loaded. Add a new tool and use it immediately.
- Prefer targeted context: read small file sections, summarize large outputs, and keep the prompt lean.
- If configured, you can `--kb <ID>` to retrieve/store context in an Amazon Bedrock Knowledge Base.

## Exiting
Type `exit` or `quit`.
"""

TOOL_SPEC = {
    "name": "welcome",
    "description": (
        "Edit and manage Swarmee welcome text with a backup in cwd()/.welcome. Can also be used as a "
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


def welcome(tool: ToolUse, **kwargs: Any) -> ToolResult:
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

    welcome_path = f"{Path.cwd()}/.welcome"
    action = tool_input["action"]

    try:
        # Create file if doesn't exist
        if action == "edit":
            if "content" not in tool_input:
                raise ValueError("content is required for edit action")

            content = tool_input["content"]
            # Write both to original and backup
            with open(welcome_path, "w") as f:
                f.write(content)

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": "Welcome text updated successfully"}],
            }

        elif action == "view":
            # Read from backup if exists, otherwise from default
            if os.path.exists(welcome_path):
                with open(welcome_path, "r") as f:
                    content = f.read()
                msg = "*.*"
            else:
                msg = "*welcome to swarmee!*"
                content = DEFAULT_WELCOME_TEXT

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"{msg}\n{content}"}],
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
