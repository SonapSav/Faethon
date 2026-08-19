"""Clearing the conversation buffer by voice.

The requirement with a sharp edge is that the wipe includes the exchange that
asked for it. The router records every turn *after* the skill has run, so the
naive implementation leaves "clear the buffer" / "Memory is cleared" sitting as
the first entry in the freshly emptied buffer -- useless as context, and
confusing if you then ask what was just said.

No network and no audio: the LLM leg is stubbed, and the skill itself only
returns a sentence. The clearing is the router's job, since skills hold no
reference to memory.
"""

from __future__ import annotations

import pytest

from faethon.config import load_config
from faethon.memory import Memory
from faethon.providers.llm import LLMReply, ToolCall
from faethon.router import Router
from faethon.skills.memory_skill import SKILL as CLEAR
from faethon.skills.registry import Registry
from faethon.skills.time_skill import SKILL as TIME


@pytest.fixture
def memory():
    return Memory(10)


@pytest.fixture
def router(memory, monkeypatch):
    cfg = load_config()
    r = Router(cfg, client=object(), registry=Registry([CLEAR, TIME]), memory=memory)
    r.replies: list[LLMReply] = []

    def fake_complete(client, messages, **kwargs):
        return r.replies.pop(0) if r.replies else LLMReply(text="fallback")

    monkeypatch.setattr("faethon.router.llm_mod.complete", fake_complete)
    return r


def fill(memory, n=4):
    for i in range(n):
        memory.add(f"question {i}", f"answer {i}")
    return memory


# -- the requirement ---------------------------------------------------------


def test_clearing_empties_the_buffer(router, memory):
    fill(memory)
    list(router.handle_streaming("Clear the buffer"))
    assert len(memory) == 0


def test_the_clearing_exchange_is_not_itself_remembered(router, memory):
    """0, not 1. Otherwise the empty buffer's first entry is the request to
    empty it, and "what did I just ask you?" answers with the wipe."""
    fill(memory)
    list(router.handle_streaming("Clear the buffer"))
    assert len(memory) == 0, "the clearing turn was recorded into the fresh buffer"

    prompt = memory.messages("SYS", "what did I just ask you")
    assert [m["role"] for m in prompt] == ["system", "user"]
    assert "clear" not in prompt[0]["content"].lower()


def test_it_says_so(router):
    assert list(router.handle_streaming("Clear the buffer")) == ["Memory is cleared."]


def test_clearing_an_already_empty_buffer_is_fine(router, memory):
    list(router.handle_streaming("Clear the buffer"))
    assert list(router.handle_streaming("Clear the buffer")) == ["Memory is cleared."]
    assert len(memory) == 0


# -- it must not swallow anything else ---------------------------------------


def test_the_next_turn_is_recorded_normally(router, memory):
    """The suppression lasts one turn.

    A flag left set would silently stop recording everything afterwards, and
    the symptom -- Faethon forgetting mid-conversation -- looks nothing like
    its cause.
    """
    fill(memory)
    list(router.handle_streaming("Clear the buffer"))
    memory.add("after", "wards")
    assert len(memory) == 1


def test_an_abandoned_clear_does_not_swallow_the_next_turn(router, memory):
    """Barge-in closes the generator before the recording step.

    The flag is reset per turn rather than after use precisely so an abandoned
    generator cannot leave it set.
    """
    fill(memory)
    stream = router.handle_streaming("Clear the buffer")
    next(stream)
    stream.close()
    assert len(memory) == 0

    list(router.handle_streaming("what time is it"))
    assert len(memory) == 1, "the turn after an abandoned clear was swallowed"


def test_an_ordinary_skill_still_records(router, memory):
    list(router.handle_streaming("what time is it"))
    assert len(memory) == 1


def test_the_clear_happens_before_the_reply_is_spoken(router, memory):
    """The wipe is done when the skill runs, not when the turn finishes.

    So barging in over "Memory is cleared" cannot leave it half-done: by the
    time there is a sentence to speak, the buffer is already empty.

    (A generator that is never advanced runs no code at all, so nothing is
    cleared in that case either -- correctly, since no turn took place.)
    """
    fill(memory)
    stream = router.handle_streaming("Clear the buffer")
    first = next(stream)
    assert len(memory) == 0, "still held turns when the reply was ready to speak"
    assert first == "Memory is cleared."
    stream.close()
    assert len(memory) == 0


# -- both routing paths ------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "clear the buffer",
    "Clear the buffer.",
    "clear the memory",
    "clear your memory",
    "forget the conversation",
    "forget everything",
    "start a new conversation",
])
def test_phrasings_that_reach_it(phrase):
    assert CLEAR.match(phrase) is not None, f"no pattern matched {phrase!r}"


@pytest.mark.parametrize("phrase", [
    "what time is it",
    "clear the table",
    "I forget things",
])
def test_phrasings_that_should_not(phrase):
    assert CLEAR.match(phrase) is None, f"unexpectedly matched {phrase!r}"


def test_the_model_can_clear_it_too(router, memory):
    """Unanticipated phrasing goes through tool-calling, which must clear
    as thoroughly as the regex path does."""
    fill(memory)
    router.replies = [LLMReply(tool_calls=[ToolCall(name="clear_memory", arguments={})])]
    assert router.handle("wipe whatever you have on me") == "Memory is cleared."
    assert len(memory) == 0
