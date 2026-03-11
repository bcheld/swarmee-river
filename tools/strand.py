"""Strands wrapper tool for running a nested agent with selected tools/prompt."""

from contextlib import redirect_stdout
from io import StringIO
from typing import Any

from strands import Agent, tool

from swarmee_river.tool_permissions import set_permissions
from swarmee_river.utils.agent_utils import extract_text, run_coroutine
from swarmee_river.utils.fork_utils import build_fork_invocation_state, create_shared_prefix_child_agent
from swarmee_river.utils.kb_utils import load_system_prompt


@tool
def strand(
    query: str,
    system_prompt: str | None = None,
    tool_names: list[str] | None = None,
    agent: Any | None = None,
) -> dict:
    """Run a nested Strands agent with optional prompt/tool overrides."""
    try:
        if not query:
            return {"status": "error", "content": [{"text": "No query provided to process."}]}

        if agent is not None and getattr(agent, "model", None) is not None:
            instruction_lines = [
                "You are running inside a shared-prefix strand fork.",
                "Answer the upcoming strand task directly and keep any tool use focused.",
            ]
            custom_prompt = str(system_prompt or "").strip()
            if custom_prompt:
                instruction_lines.append(f"Additional strand instructions:\n{custom_prompt}")
            requested_tool_names = [str(name).strip() for name in (tool_names or []) if str(name).strip()]
            if requested_tool_names:
                instruction_lines.append(
                    f"If tools are required, only use these tools: {', '.join(requested_tool_names)}."
                )
            child_agent, snapshot = create_shared_prefix_child_agent(
                parent_agent=agent,
                kind="strand",
                seed_instruction="\n\n".join(instruction_lines),
                tool_allowlist=requested_tool_names,
                callback_handler=None,
            )
            invocation_state = build_fork_invocation_state(
                snapshot,
                extra_prompt_chars=len(str(query or "").strip()) + len("\n\n".join(instruction_lines)),
            )
            result = run_coroutine(child_agent.invoke_async(query, invocation_state=invocation_state))
            result_text = extract_text(result) or str(result or "").strip()
            return {"status": "success", "content": [{"text": f"Strands result:\n\n{result_text.strip()}"}]}

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


set_permissions(strand, "execute")
