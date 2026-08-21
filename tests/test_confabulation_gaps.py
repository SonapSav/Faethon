"""Phrasings that must never depend on the model choosing to call a tool.

Every case here was taken from the journal, where the model answered directly
-- `llm streamed (stop)` rather than `tool_calls` -- and produced something
that was either wrong or unverifiable.

Falling through to the model is not itself a bug: it has the tools and usually
calls them. The bug is that "usually" is not a property you can rely on for a
figure someone acts on, or an action they believe happened. Asked "What is my
budget with OpenRouter", it said "19 dollars and 34 cents" against a real
$1.35, and repeated it twice. Asked "Maximum Volume", it said "Volume is set to
100%" and the volume did not move.

So the common phrasings are pinned to the regex path, and the prompt covers the
ones nobody thought of.
"""

from __future__ import annotations

import pytest

from faethon.skills.registry import Registry

REGISTRY = Registry.discover()

# (phrase, the skill that must handle it)
MUST_ROUTE = [
    # Money. Wrong in the reassuring direction at the worst moment.
    ("what is my budget with openrouter", "get_credit_balance"),
    ("what is my budget", "get_credit_balance"),
    ("how much is left on openrouter", "get_credit_balance"),

    # Actions claimed but not performed -- both observed.
    ("maximum volume", "set_volume"),
    ("max volume", "set_volume"),
    ("full volume", "set_volume"),
    ("volume to maximum", "set_volume"),
    ("turn the volume all the way up", "set_volume"),
    ("turn up the volume", "set_volume"),
    ("minimum volume", "set_volume"),
    ("what is your volume", "set_volume"),
    ("how loud are you", "set_volume"),

    # Clearing is destructive: "Memory is cleared" without a call leaves every
    # word still in the buffer.
    ("forget what we said", "clear_memory"),
    ("wipe your memory", "clear_memory"),
    ("erase your memory", "clear_memory"),
    ("forget everything", "clear_memory"),

    # Readings the model answered with specific, unverifiable values.
    ("are you connected to a wifi network", "get_health"),
    ("is the wifi working", "get_health"),
    ("are you online", "get_health"),
    ("how is your cpu", "get_health"),
    ("what is the forecasted temperature for tomorrow", "get_weather"),
    ("how hot will it be tomorrow", "get_weather"),
    ("what is the high tomorrow", "get_weather"),
    ("is it hot outside", "get_weather"),
    ("what is it like outside", "get_weather"),
    ("how bad is the pollution", "get_air_quality"),
    ("is the air safe to breathe", "get_air_quality"),
    ("how late is it", "get_time"),
    ("how much time is left on the timer", "set_timer"),
]


@pytest.mark.parametrize("phrase,skill", MUST_ROUTE)
def test_reaches_the_skill_not_the_model(phrase, skill):
    hit = REGISTRY.match(phrase)
    assert hit is not None, f"{phrase!r} falls through to the model"
    assert hit[0].name == skill, f"{phrase!r} -> {hit[0].name}, wanted {skill}"


# Widening patterns is how one skill starts eating another's questions.
MUST_NOT_FIRE = [
    "how much money is a raspberry pi",
    "what is the budget of the film",
    "how much does a pizza cost",
    "what is the high score",
    "is it hot in here",
    "turn up",                       # ambiguous: not maximum, not a step
    "i had to restart my laptop this morning",
    "restart my router",
    "tell me about budgeting",
]


@pytest.mark.parametrize("phrase", MUST_NOT_FIRE)
def test_ordinary_english_stays_quiet(phrase):
    hit = REGISTRY.match(phrase)
    assert hit is None, f"{phrase!r} wrongly matched {hit[0].name if hit else ''}"


def test_bare_restart_is_still_refused():
    """The one place a false positive costs more than a false claim.

    An unwanted restart drops the conversation and costs ~20s of booting. A
    model merely saying "restarting now" without doing it is confusing and
    harmless. So restart stays narrow while everything else widened.
    """
    assert REGISTRY.match("restart") is None


def test_the_prompt_covers_what_the_patterns_cannot():
    from faethon.config import load_config

    prompt = load_config().llm.system_prompt.lower()
    assert "cannot know any live figure" in prompt
    assert "never state one from memory" in prompt
    assert "never say you have done one" in prompt
