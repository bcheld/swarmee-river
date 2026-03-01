"""Knowledge base utility functions for Swarmee River."""

from typing import Any


def store_conversation_in_kb(
    agent: Any,
    user_input: str,
    response: Any | None = None,
    knowledge_base_id: str | None = None,
) -> None:
    """
    Store conversation between user and assistant in knowledge base asynchronously

    Args:
        agent: Strands agent instance
        user_input: User's input text
        response: Response from the agent (optional, may be None in command-line mode)
        knowledge_base_id: ID of the knowledge base to store in
    """
    if not knowledge_base_id:
        return

    try:
        # Initialize conversation content
        conversation_content = f"User: {user_input}\n\n"

        # If response is provided, extract details
        if response:
            try:
                # Initialize variables to store content
                reasoning_text = None
                assistant_text = None

                # Extract reasoning content and text from the response
                for item in response.message:
                    if (
                        "reasoningContent" in item
                        and "reasoningText" in item["reasoningContent"]
                        and "text" in item["reasoningContent"]["reasoningText"]
                    ):
                        reasoning_text = item["reasoningContent"]["reasoningText"]["text"]
                    elif "text" in item:
                        assistant_text = item["text"]

                # Create conversation content with both reasoning and response if available
                if reasoning_text and assistant_text:
                    conversation_content = (
                        f"User: {user_input}\n\n"
                        f"Assistant Reasoning: {reasoning_text}\n\n"
                        f"Assistant Response: {assistant_text}"
                    )
                elif assistant_text:
                    # Fallback to just the assistant text if no reasoning found
                    conversation_content = f"User: {user_input}\n\nAssistant: {assistant_text}"
                elif not response.message:  # Empty message list case
                    conversation_content = f"User: {user_input}\n\nAssistant: "
                else:
                    # Final fallback to string representation
                    conversation_content = f"User: {user_input}\n\nAssistant: {str(response)}"
            except Exception as e:
                # Fallback in case of any error in parsing
                print(f"Error parsing response structure: {str(e)}")
                try:
                    conversation_content = f"User: {user_input}\n\nAssistant: {str(response)}"
                except Exception:
                    # If str(response) also fails, use empty string
                    conversation_content = f"User: {user_input}\n\nAssistant: "
        else:
            # For command-line mode where response isn't available yet
            # Just store the user query to keep a record
            conversation_content = f"User: {user_input}"

        # Create a title based on the user input (truncate if too long)
        max_title_length = 50
        conversation_title = f"Conversation: {user_input[:max_title_length]}"
        if len(user_input) > max_title_length:
            conversation_title += "..."

        # Store in knowledge base asynchronously with record_direct_tool_call=False
        agent.tool.store_in_kb(
            content=conversation_content,
            title=conversation_title,
            knowledge_base_id=knowledge_base_id,
            record_direct_tool_call=False,
        )
    except Exception as e:
        print(f"Error storing conversation in knowledge base: {str(e)}")


def load_system_prompt() -> str:
    """
    Load orchestrator system prompt from prompt assets.

    Priority:
    1. ``orchestrator_base`` prompt asset
    2. Default prompt
    """
    try:
        from swarmee_river.prompt_assets import resolve_orchestrator_prompt_from_agent

        return resolve_orchestrator_prompt_from_agent(None)
    except Exception:
        pass

    # Return default prompt
    return "You are a helpful assistant."
