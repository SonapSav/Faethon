"""`faethon-misses` -- finding the phrasings that reached no skill.

The failure it exists for is silent. A phrasing the regex misses does not
error: it goes to the model, which usually answers well enough that nobody
notices. "Radio volume at 7" fell through for a day and surfaced only because
somebody said the radio was ignoring them.

No journald here: the log lines are synthetic.
"""

from __future__ import annotations

import pytest

from faethon import misses

ROUTED = [
    "INFO    faethon: heard: play 94.9",
    "INFO    faethon.router: regex -> control_radio({'freq': '94.9'})",
]
RESCUED = [
    "INFO    faethon: heard: what's my budget",
    "INFO    faethon.providers.llm: llm streamed (tool_calls): tool=['get_credit_balance']",
    "INFO    faethon.router: tool -> get_credit_balance({})",
]
ANSWERED = [
    "INFO    faethon: heard: tell me a joke",
    "INFO    faethon.providers.llm: llm streamed (stop): 'No.'",
]
NOTHING = ["INFO    faethon: heard: mumble mumble"]


def test_classify_separates_the_four_outcomes():
    turns = misses.classify(ROUTED + RESCUED + ANSWERED + NOTHING)
    assert [t[1] for t in turns] == ["routed", "rescued", "answered", "nothing"]
    assert turns[0][2] == "control_radio"


def test_a_tool_call_after_the_model_is_rescued_not_routed():
    """The distinction that matters is whether a *pattern* could have caught
    it -- a tool call the model made still cost a round trip."""
    turns = misses.classify(RESCUED)
    assert turns[0][1] == "rescued"


def test_lines_before_the_first_heard_are_ignored():
    noise = ["INFO    faethon: Faethon is listening -- say the wake word"]
    assert misses.classify(noise + ROUTED) == [("play 94.9", "routed", "control_radio")]


# -- the vocabulary, derived from the patterns themselves ---------------------

def test_vocabulary_finds_the_words_that_matter():
    v = misses.vocabulary()
    assert v.get("radio") == "control_radio"
    assert v.get("volume") == "set_volume"
    assert v.get("timer") == "set_timer"
    assert v.get("weather") == "get_weather"
    assert v.get("dust") == "get_air_quality"


def test_regex_syntax_does_not_become_vocabulary():
    """A naive word grab turns `\\bwhat` into "bwhat" and splits a group into
    "thon", and both look like real words."""
    v = misses.vocabulary()
    for artifact in ("bwhat", "bhow", "bradio", "bset", "bturn", "thon"):
        assert artifact not in v, artifact


@pytest.mark.parametrize("word", ["about", "yourself", "time", "going", "outside"])
def test_ordinary_english_does_not_become_vocabulary(word):
    """A wrong hint is worse than none. "about" comes from "forget what we
    talked about" and pointed a joke about fatherhood at clear_memory."""
    assert word not in misses.vocabulary(), word


def test_a_word_two_skills_share_is_no_hint_at_all():
    v = misses.vocabulary()
    assert "clear" not in v, "clear_memory and get_air_quality both look for it"


def test_hint_only_when_one_skill_is_implicated():
    v = {"radio": "control_radio", "volume": "set_volume"}
    assert misses.looks_like_a_command("turn the radio on", v) == "control_radio"
    assert misses.looks_like_a_command("radio volume up", v) == ""
    assert misses.looks_like_a_command("tell me a joke", v) == ""


def test_normalise_strips_what_whisper_adds():
    assert misses.normalise(".  Radio Vol. 7. ") == "radio vol 7"


# -- not reporting what is already fixed --------------------------------------

def test_something_the_patterns_now_catch_is_not_reported():
    """The journal reaches further back than the last fix. Six of the first
    nine findings had been fixed hours earlier, and a list of solved problems
    is a list nobody reads twice."""
    assert misses.matches_now("play 94.9") is True
    assert misses.matches_now("maximum volume") is True


def test_something_still_missing_is_reported():
    assert misses.matches_now("prestar the service") is False
