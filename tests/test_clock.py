"""Telling the model what day it is.

Ungrounded, it does not say it doesn't know -- it retrieves a date from its
training data and answers confidently from that:

    Q: what is the date today
    A: Today is Friday, March 14, 2025.

which was nineteen months out, and "how long until Christmas" was computed off
it and came back 318 days when the answer was 127. Same shape as the identity
confabulation: no grounding fact, so it reaches for the nearest thing it knows.

The catch is that this Pi has no battery-backed clock, so for the first couple
of minutes after a cold boot the wall clock is wrong. A wrong date is worse
than no date, because it is exactly as confident as a right one.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from faethon import clock


@pytest.fixture
def synced(monkeypatch):
    monkeypatch.setattr(clock, "is_synced", lambda: True)


def test_it_names_the_date_and_time(synced):
    when = datetime(2026, 8, 20, 20, 51)
    line = clock.grounding(when)
    assert "Thursday" in line
    assert "20 August 2026" in line
    assert "20:51" in line


def test_it_says_nothing_when_the_clock_is_not_trusted(monkeypatch):
    """After a cold boot the wall clock is whatever was restored from disk,
    until NTP steps it. Telling the model that would be worse than silence."""
    monkeypatch.setattr(clock, "is_synced", lambda: False)
    assert clock.grounding(datetime(2026, 8, 20)) == ""


def test_it_asks_for_the_answer_not_the_working(synced):
    """Reasoning is off, so the model thinks in the visible reply. Without
    this it read out the month-by-month arithmetic -- 400 characters of TTS
    for a two-word answer."""
    assert "never the arithmetic" in clock.grounding(datetime(2026, 8, 20))


def test_the_router_grounds_every_request(monkeypatch):
    """Rebuilt per request, not at startup: a service up for a week would
    otherwise still be telling the model it is Monday."""
    from faethon.config import load_config
    from faethon.memory import Memory
    from faethon.router import Router
    from faethon.skills.registry import Registry

    monkeypatch.setattr(clock, "is_synced", lambda: True)
    r = Router(load_config(), client=object(), registry=Registry([]), memory=Memory(3))
    prompt = r._system_prompt()
    assert "Right now it is" in prompt
    assert prompt.startswith(load_config().llm.system_prompt)


def test_an_unsynced_clock_leaves_the_configured_prompt_alone(monkeypatch):
    from faethon.config import load_config
    from faethon.memory import Memory
    from faethon.router import Router
    from faethon.skills.registry import Registry

    monkeypatch.setattr(clock, "is_synced", lambda: False)
    r = Router(load_config(), client=object(), registry=Registry([]), memory=Memory(3))
    assert r._system_prompt() == load_config().llm.system_prompt
