import os

import pytest
from strands import Agent

from swarmee_river.utils.env_utils import load_env_file
from swarmee_river.utils.model_utils import load_model, load_path


@pytest.mark.skipif(os.getenv("SWARMEE_RUN_LIVE_API_TESTS") != "1", reason="Live API tests disabled")
@pytest.mark.parametrize("model_id", ["gpt-5-nano", "gpt-5-mini", "gpt-5.2"])
def test_openai_connectivity(model_id: str):
    load_env_file()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    provider_path = load_path("openai")
    model = load_model(
        provider_path,
        {
            "model_id": model_id,
            "client_args": {"api_key": os.getenv("OPENAI_API_KEY")},
            "params": {"max_output_tokens": 32},
        },
    )
    agent = Agent(model=model, tools=None, system_prompt="Reply with exactly: PONG", callback_handler=None)
    result = agent("PING")
    assert "PONG" in str(result).upper()


@pytest.mark.skipif(os.getenv("SWARMEE_RUN_LIVE_API_TESTS") != "1", reason="Live API tests disabled")
def test_openai_gpt52_multi_turn_connectivity():
    load_env_file()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    provider_path = load_path("openai")
    model = load_model(
        provider_path,
        {
            "model_id": "gpt-5.2",
            "client_args": {"api_key": os.getenv("OPENAI_API_KEY")},
            "params": {"max_output_tokens": 64},
        },
    )
    agent = Agent(
        model=model,
        tools=None,
        system_prompt="Reply with exactly: ORANGE",
        callback_handler=None,
    )

    first = agent("Say the word.")
    second = agent("Repeat your previous answer exactly.")

    assert "ORANGE" in str(first).upper()
    assert "ORANGE" in str(second).upper()
