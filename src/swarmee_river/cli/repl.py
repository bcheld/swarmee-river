from __future__ import annotations

from typing import Any, Callable

from swarmee_river.cli.commands import CLIContext, CommandRegistry

def run_repl(
    *,
    ctx: CLIContext,
    registry: CommandRegistry,
    get_user_input: Callable[..., str],
    classify_intent: Callable[[str], str],
    render_goodbye_message: Callable[[], None],
) -> None:
    while True:
        try:
            user_input = get_user_input("\n~ ", default="", keyboard_interrupt_return_default=False)

            if (user_input or "").lower() in {"exit", "quit"}:
                render_goodbye_message()
                break

            dispatch = registry.dispatch(ctx, user_input)
            if dispatch.handled:
                if dispatch.should_exit:
                    render_goodbye_message()
                    break
                continue

            if user_input.startswith("!"):
                shell_command = user_input[1:]
                ctx.output(f"$ {shell_command}")
                try:
                    ctx.agent.tool.shell(
                        command=shell_command,
                        user_message_override=user_input,
                        non_interactive_mode=True,
                        record_direct_tool_call=False,
                    )
                    ctx.output("")  # newline after command execution
                except Exception as e:
                    ctx.output(f"Shell command execution error: {e}")
                continue

            if not user_input.strip():
                continue

            if ctx.knowledge_base_id:
                ctx.agent.tool.retrieve(text=user_input, knowledgeBaseId=ctx.knowledge_base_id)

            ctx.refresh_system_prompt()

            intent = "work" if ctx.force_plan_next else classify_intent(user_input)
            ctx.force_plan_next = False

            if intent == "work":
                ctx.pending_request = user_input
                ctx.pending_plan = ctx.generate_plan(user_input)
                ctx.output(ctx.render_plan(ctx.pending_plan))
                if ctx.auto_approve:
                    ctx.output("\nAuto-approving plan (--yes / SWARMEE_AUTO_APPROVE).")
                    ctx.last_plan = ctx.pending_plan
                    response = ctx.execute_with_plan(user_input, ctx.pending_plan)
                    ctx.pending_plan = None
                    ctx.pending_request = None
                    if ctx.knowledge_base_id:
                        ctx.store_conversation(user_input, response)
                else:
                    ctx.output(ctx.pending_plan.confirmation_prompt)
                continue

            response = ctx.run_agent(user_input, invocation_state={"swarmee": {"mode": "execute"}})
            if ctx.knowledge_base_id:
                ctx.store_conversation(user_input, response)
        except (KeyboardInterrupt, EOFError):
            render_goodbye_message()
            break
        except Exception as e:
            ctx.stop_spinners()
            error_text = str(e)
            ctx.output(f"\nError: {error_text}")
            if "unable to locate credentials" in error_text.lower():
                if (ctx.selected_provider or "").strip().lower() == "bedrock":
                    ctx.output(
                        "Hint: Bedrock requires AWS credentials (AWS_PROFILE or AWS_ACCESS_KEY_ID/"
                        "AWS_SECRET_ACCESS_KEY). Or run with `--model-provider openai`."
                    )
                elif ctx.knowledge_base_id:
                    ctx.output(
                        "Hint: Knowledge Base operations require AWS credentials even when model "
                        "provider is OpenAI."
                    )
