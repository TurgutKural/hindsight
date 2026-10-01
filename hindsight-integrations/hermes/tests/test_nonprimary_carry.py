"""PR4952 regression cases relocated because current main appended new tests."""

import json
from typing import Any, Callable

import pytest
from conftest import FakeClient
from test_provider import _retain_item, _turns_of


@pytest.mark.parametrize("agent_context", ["cron", "subagent", "flush"])
@pytest.mark.parametrize("retain_every_n_turns", [1, 3])
def test_nonprimary_turns_are_not_retained_on_sync_or_switch(
    provider: Callable[..., Any], agent_context: str, retain_every_n_turns: int
) -> None:
    indicators: list[str] = []
    instance, fake = provider(
        {"retain_every_n_turns": retain_every_n_turns},
        agent_context=agent_context,
        status_callback=indicators.append,
    )
    instance.sync_turn("scheduled task", "automated output")
    instance.on_session_switch("session-2", reset=True)
    instance.sync_turn("another automated task", "more output")
    instance.shutdown()

    assert fake.retains == []
    assert indicators == []


@pytest.mark.parametrize("agent_context", ["primary", None])
def test_primary_and_legacy_hosts_still_auto_retain(provider: Callable[..., Any], agent_context: str | None) -> None:
    kwargs = {} if agent_context is None else {"agent_context": agent_context}
    instance, fake = provider({}, **kwargs)
    instance.sync_turn("remember my preference", "noted")
    instance.shutdown()
    assert _turns_of(fake) == [["User: remember my preference", "Assistant: noted"]]


@pytest.mark.parametrize("agent_context", ["cron", "subagent", "flush"])
def test_nonprimary_context_keeps_explicit_memory_tools(provider: Callable[..., Any], agent_context: str) -> None:
    instance, fake = provider({}, agent_context=agent_context, client=FakeClient(recall_texts=["Ada likes tea"]))
    instance.sync_turn("automated task", "not a user conversation")
    instance.handle_tool_call("hindsight_retain", {"content": "explicitly saved result"})
    recalled = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "Ada"}))
    instance.shutdown()

    assert len(fake.retains) == 1
    assert _retain_item(fake)["content"] == "explicitly saved result"
    assert recalled["result"] == "1. Ada likes tea"
