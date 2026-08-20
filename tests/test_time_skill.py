"""The clock, and the questions that only look like they're about the clock.

The pattern used to end at \\b after "time", which a following space satisfies
-- so "what is the time complexity of quicksort" answered with the clock, and
so did "what is the date of the meeting". Both are questions for the model, and
a skill that intercepts them is worse than one that never existed: the user
gets a confident, irrelevant answer and no sign anything went wrong.
"""

from __future__ import annotations

import pytest

from faethon.skills.time_skill import SKILL


@pytest.mark.parametrize("heard", [
    "what time is it",
    "What time is it?",
    "what is the time",
    "what's the time",
    "whats the date",
    "what is the date today",
    "what's today's date",
    "tell me the time please",
    "tell me the date",
    "current time",
    "what day is it?",
])
def test_questions_actually_about_the_clock(heard):
    assert SKILL.match(heard) is not None, f"no longer matches {heard!r}"


@pytest.mark.parametrize("heard", [
    "what is the time complexity of quicksort",
    "whats the time zone here",
    "what is the date of the meeting",
    "what time is it in Tokyo",
    "what is the time signature of this song",
    "what is the date format in Germany",
])
def test_questions_that_only_start_the_same_way(heard):
    """These belong to the model. The skill answering them is a wrong answer
    delivered confidently, with nothing to indicate it happened."""
    assert SKILL.match(heard) is None, f"intercepted {heard!r}"


def test_the_kind_is_extracted(heard=None):
    assert SKILL.match("what time is it") == {"kind": "time"}
    assert SKILL.match("what is the date") == {"kind": "date"}
    assert SKILL.match("what day is it") == {"kind": "day"}


def test_it_still_answers():
    assert SKILL.run(kind="time").startswith("It's")
    assert SKILL.run(kind="date").startswith("It's")


def test_the_clock_is_not_offered_to_the_model():
    """Unusual for a harmless read-only skill, and the reason is structural: a
    tool result ends the turn, since it is yielded straight to speech and never
    fed back for a second pass. Asked "how long until Christmas?", the model
    called this as a *step* and the user got "It's Thursday, August 20" as the
    whole answer.

    The system prompt now carries the date on every request, so the model has
    no need of the tool. The regex path still answers instantly and free.
    """
    from faethon.skills.registry import Registry

    offered = [t["function"]["name"] for t in Registry.discover().tool_schemas()]
    assert "get_time" not in offered
    assert SKILL.match("what time is it") is not None, "regex path must still work"


def test_the_timer_description_disclaims_date_questions():
    """Same trap one skill over: the model reached for set_timer(check) on
    "how long until Christmas" because "how long" looks like a timer query,
    and the tool's answer ended the turn."""
    from faethon.skills.timer_skill import SKILL as TIMER

    assert "not for how long until a date" in TIMER.description.lower()
