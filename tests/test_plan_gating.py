import sys
from unittest import mock

from swarmee_river import swarmee
from swarmee_river.planning import PlanStep, WorkPlan


def test_work_prompt_generates_plan_and_waits_for_approval(
    mock_agent,
    mock_bedrock,
    mock_load_prompt,
    mock_user_input,
    mock_welcome_message,
    mock_goodbye_message,
    monkeypatch,
):
    plan = WorkPlan(summary="Fix a bug", steps=[PlanStep(description="Inspect failing test", tools_expected=["file_read"])])
    mock_agent.invoke_async = mock.AsyncMock(return_value=mock.MagicMock(structured_output=plan, message=[]))

    mock_user_input.side_effect = ["fix the bug in swarmee.py", ":n", "exit"]
    monkeypatch.setattr(sys, "argv", ["swarmee"])

    swarmee.main()

    assert mock_agent.invoke_async.call_count == 1
    call = mock_agent.invoke_async.call_args
    assert call.args[0] == "fix the bug in swarmee.py"
    assert call.kwargs["invocation_state"]["swarmee"]["mode"] == "plan"
    assert call.kwargs["structured_output_model"] is WorkPlan


def test_yes_flag_auto_approves_plan_and_executes(
    mock_agent,
    mock_bedrock,
    mock_load_prompt,
    mock_user_input,
    mock_welcome_message,
    mock_goodbye_message,
    monkeypatch,
):
    plan = WorkPlan(summary="Do a refactor", steps=[PlanStep(description="Edit code", tools_expected=["editor"])])
    plan_result = mock.MagicMock(structured_output=plan, message=[])
    exec_result = mock.MagicMock(structured_output=None, message=[])
    mock_agent.invoke_async = mock.AsyncMock(side_effect=[plan_result, exec_result])

    mock_user_input.side_effect = ["refactor swarmee.py", "exit"]
    monkeypatch.setattr(sys, "argv", ["swarmee", "--yes"])

    swarmee.main()

    assert mock_agent.invoke_async.call_count == 2
    plan_call = mock_agent.invoke_async.call_args_list[0]
    exec_call = mock_agent.invoke_async.call_args_list[1]

    assert plan_call.kwargs["invocation_state"]["swarmee"]["mode"] == "plan"
    assert plan_call.kwargs["structured_output_model"] is WorkPlan

    assert exec_call.kwargs["invocation_state"]["swarmee"]["mode"] == "execute"
    assert exec_call.kwargs["invocation_state"]["swarmee"]["enforce_plan"] is True
    assert "editor" in exec_call.kwargs["invocation_state"]["swarmee"]["allowed_tools"]

