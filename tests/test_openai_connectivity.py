import os

import pytest
from strands import Agent

from swarmee_river.utils.env_utils import load_env_file
from swarmee_river.utils.model_utils import load_model, load_path


@pytest.mark.skipif(os.getenv("SWARMEE_RUN_LIVE_API_TESTS") != "1", reason="Live API tests disabled")
def test_openai_connectivity():
    load_env_file()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    provider_path = load_path("openai")
    model = load_model(
        provider_path,
        {
            "model_id": "gpt-5-nano",
            "client_args": {"api_key": os.getenv("OPENAI_API_KEY")},
            "params": {"max_completion_tokens": 32},
        },
    )
    agent = Agent(model=model, tools=None, system_prompt="Reply with exactly: PONG", callback_handler=None)
    result = agent("PING")
    assert "PONG" in str(result).upper()
