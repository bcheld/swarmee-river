from __future__ import annotations

from types import SimpleNamespace

from swarmee_river.context.prompt_cache import PromptCacheState
from swarmee_river.utils.agent_utils import capture_agent_turn_state, restore_prompt_cache_turn_state


def test_capture_agent_turn_state_snapshots_messages_state_and_prompt_cache() -> None:
    prompt_cache = PromptCacheState(
        sent_hashes={"context": "hash-1"},
        pending=["reminder one", "reminder two"],
    )
    agent = SimpleNamespace(
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        state={"foo": {"bar": 1}},
    )

    snapshot = capture_agent_turn_state(agent, prompt_cache=prompt_cache)

    agent.messages[0]["content"][0]["text"] = "changed"
    agent.state["foo"]["bar"] = 2
    prompt_cache.pending.append("mutated")
    prompt_cache.sent_hashes["context"] = "hash-2"

    assert snapshot.messages[0]["content"][0]["text"] == "hello"
    assert snapshot.state["foo"]["bar"] == 1
    assert snapshot.prompt_cache_pending == ["reminder one", "reminder two"]
    assert snapshot.prompt_cache_sent_hashes == {"context": "hash-1"}


def test_restore_prompt_cache_turn_state_restores_pending_and_hashes() -> None:
    prompt_cache = PromptCacheState(
        sent_hashes={"context": "hash-1"},
        pending=["reminder one"],
    )
    agent = SimpleNamespace(messages=[], state=None)
    snapshot = capture_agent_turn_state(agent, prompt_cache=prompt_cache)

    prompt_cache.pending[:] = ["other"]
    prompt_cache.sent_hashes.clear()
    prompt_cache.sent_hashes["context"] = "hash-2"

    restore_prompt_cache_turn_state(prompt_cache, snapshot)

    assert prompt_cache.pending == ["reminder one"]
    assert prompt_cache.sent_hashes == {"context": "hash-1"}
