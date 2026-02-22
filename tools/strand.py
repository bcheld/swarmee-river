"""Strands wrapper tool for running a nested agent with selected tools/prompt."""

from contextlib import redirect_stdout
from io import StringIO

from strands import Agent, tool

from swarmee_river.utils.kb_utils import load_system_prompt


@tool
def strand(
    query: str,
    system_prompt: str | None = None,
    tool_names: list[str] | None = None,
) -> dict:
    """Run a nested Strands agent with optional prompt/tool overrides."""
    try:
        if not query:
            return {"status": "error", "content": [{"text": "No query provided to process."}]}

        captured_output = StringIO()

        with redirect_stdout(captured_output):
            from swarmee_river.tools import get_tools

            all_tools = get_tools()
            selected_tools = [all_tools[name] for name in (tool_names or []) if name in all_tools]
            if not selected_tools:
                selected_tools = list(all_tools.values())

            if not system_prompt:
                system_prompt = load_system_prompt()

            agent = Agent(tools=selected_tools, messages=[], system_prompt=system_prompt)
            response_text = str(agent(query) or "")

        output_text = captured_output.getvalue()
        result_text = (
            output_text + "\n" + response_text
            if output_text and response_text
            else output_text or response_text
        )

        return {"status": "success", "content": [{"text": f"Strands result:\n\n{result_text.strip()}"}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error running Strands: {e}"}]}
