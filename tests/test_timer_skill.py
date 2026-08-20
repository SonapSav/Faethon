"""Timers: the first thing Faethon does without being asked, and the first
thing it writes down.

Relative only, and that is a design decision rather than a shortcut. The Pi has
no battery-backed clock, so the wall clock is wrong for the first couple of
minutes after a boot and then steps when NTP corrects it -- measured, a 60s
step moved time.time() by 62.3s and time.monotonic() by 2.3s. So a running
timer counts on monotonic, which cannot jump, while persistence uses the wall
clock, which is the only one that means anything across a reboot.

No clock is slept on here: both are injected.
"""

from __future__ import annotations

import pytest

from faethon.skills.timer_skill import (
    STALE_AFTER_SEC,
    TimerSkill,
    parse_duration,
    say_duration,
)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A timer skill on a fake disk, a fake clock, and a trusted clock flag."""
    from faethon import state
    from faethon.skills import timer_skill

    monkeypatch.setattr(state, "state_dir", lambda: tmp_path)
    # The sync flag lives in faethon.clock now, shared with the prompt
    # grounding. Path takes no attributes, so patch the function.
    monkeypatch.setattr(timer_skill.clock, "is_synced", lambda: True)

    clock = {"wall": 1_000_000.0, "mono": 500.0}
    monkeypatch.setattr(timer_skill.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(timer_skill.time, "monotonic", lambda: clock["mono"])

    class Rig:
        def __init__(self):
            self.skill = TimerSkill()
            self.clock = clock

        def advance(self, seconds, wall_only=False, mono_only=False):
            if not mono_only:
                clock["wall"] += seconds
            if not wall_only:
                clock["mono"] += seconds

        def say(self, text):
            m = self.skill.match(text)
            assert m is not None, f"no pattern matched {text!r}"
            return self.skill.run(**m)

        def restart(self):
            """A new process reading the same disk."""
            self.skill = TimerSkill()
            return self.skill

    return Rig()


# -- reading the duration ----------------------------------------------------


@pytest.mark.parametrize("said,seconds", [
    ("10 minutes", 600),
    ("ten minutes", 600),
    ("1 hour", 3600),
    ("an hour", 3600),
    ("90 seconds", 90),
    ("five mins", 300),
    ("30 secs", 30),
    ("half an hour", 1800),
    ("quarter of an hour", 900),
    ("fifteen minutes", 900),
    ("two hours thirty minutes", 9000),
    ("2 hours and 30 minutes", 9000),
    ("three and a half minutes", 210),
    ("one and a half hours", 5400),
    ("an hour and a half", 5400),
])
def test_durations_people_actually_say(said, seconds):
    assert parse_duration(said) == seconds


@pytest.mark.parametrize("said", ["the weather", "a timer", "bananas", "", "for"])
def test_things_that_are_not_durations(said):
    assert parse_duration(said) is None


def test_both_orderings_of_a_half():
    """"An hour and a half" is far commoner than "one and a half hours", and
    an earlier pattern only caught the second -- silently giving 3600."""
    assert parse_duration("an hour and a half") == parse_duration("one and a half hours")


@pytest.mark.parametrize("seconds,spoken", [
    (600, "10 minutes"), (60, "1 minute"), (3600, "1 hour"),
    (5400, "1 hour and 30 minutes"), (30, "30 seconds"),
])
def test_it_says_durations_the_way_people_do(seconds, spoken):
    assert say_duration(seconds) == spoken


# -- setting, checking, cancelling -------------------------------------------


def test_setting_and_checking(rig):
    assert "10 minutes" in rig.say("set a timer for 10 minutes")
    rig.advance(60)
    assert "9 minutes" in rig.say("how long left")


def test_a_bare_check_with_no_timers(rig):
    assert "aren't any timers" in rig.say("how long left")


def test_several_at_once(rig):
    rig.say("set a pasta timer for 8 minutes")
    rig.say("set a rice timer for 12 minutes")
    both = rig.say("what timers")
    assert "pasta" in both and "rice" in both


def test_a_named_timer_can_be_asked_about(rig):
    rig.say("set a pasta timer for 8 minutes")
    rig.say("set a rice timer for 12 minutes")
    assert "pasta" in rig.say("how long left on the pasta timer")


def test_cancelling_by_name(rig):
    rig.say("set a pasta timer for 8 minutes")
    assert "Cancelled" in rig.say("cancel the pasta timer")
    assert "aren't any" in rig.say("how long left")


def test_an_ambiguous_cancel_asks_rather_than_guessing(rig):
    """Cancelling the wrong timer is a silent, unrecoverable mistake."""
    rig.say("set a pasta timer for 8 minutes")
    rig.say("set a rice timer for 12 minutes")
    assert "Which one" in rig.say("cancel the timer")
    assert "pasta" in rig.say("what timers")


def test_cancelling_the_only_timer_needs_no_name(rig):
    rig.say("set a timer for 10 minutes")
    assert "Cancelled" in rig.say("cancel the timer")


def test_an_unknown_name_says_so(rig):
    rig.say("set a pasta timer for 8 minutes")
    assert "don't have a rice timer" in rig.say("cancel the rice timer")


def test_a_missing_duration_is_not_guessed(rig):
    assert "didn't catch how long" in rig.skill.run(duration="bananas")


# -- firing ------------------------------------------------------------------


def test_it_fires_when_due(rig):
    rig.say("set a pasta timer for 10 minutes")
    rig.advance(599)
    assert rig.skill.tick() is None
    rig.advance(2)
    assert "pasta timer is up" in (rig.skill.tick() or "")


def test_it_fires_once(rig):
    rig.say("set a timer for 1 minute")
    rig.advance(61)
    assert rig.skill.tick() is not None
    assert rig.skill.tick() is None


def test_a_clock_step_does_not_move_a_running_timer(rig):
    """The whole reason a running timer counts on monotonic. NTP correcting a
    Pi with no RTC steps the wall clock by minutes."""
    rig.say("set a timer for 10 minutes")
    rig.advance(3600, wall_only=True)      # NTP jumps the wall clock an hour
    assert rig.skill.tick() is None, "a wall-clock jump fired the timer early"
    rig.advance(601)
    assert rig.skill.tick() is not None


def test_ticking_with_nothing_set_is_free(rig):
    assert rig.skill.tick() is None


# -- surviving a restart -----------------------------------------------------


def test_a_timer_survives_a_restart(rig):
    """The case this persistence exists for: a service restart takes ten
    seconds and must not cost you the dinner."""
    rig.say("set a pasta timer for 10 minutes")
    rig.advance(30)
    rig.restart()
    left = rig.say("how long left on the pasta timer")
    assert "9 minutes" in left


def test_time_spent_restarting_still_counts(rig):
    rig.say("set a timer for 10 minutes")
    rig.advance(300)                        # away for five minutes
    rig.restart()
    assert "5 minutes" in rig.say("how long left")


def test_one_that_came_due_while_away_fires_late(rig):
    """Better late and explained than silently dropped."""
    rig.say("set a pasta timer for 5 minutes")
    rig.advance(400)                        # due 100s ago
    rig.restart()
    said = rig.skill.tick() or ""
    assert "pasta" in said and "while I was away" in said


def test_one_long_past_is_dropped_not_announced(rig):
    """A pasta timer from last night announcing itself at breakfast is worse
    than losing it."""
    rig.say("set a pasta timer for 5 minutes")
    rig.advance(300 + STALE_AFTER_SEC + 60)
    rig.restart()
    assert rig.skill.tick() is None
    assert "aren't any" in rig.say("how long left")


def test_nothing_is_restored_until_the_clock_is_trusted(monkeypatch, rig):
    """After a cold boot the wall clock is wrong for a couple of minutes. A
    timer restored then would be off by however wrong it still was.
    """
    from faethon.skills import timer_skill

    rig.say("set a timer for 10 minutes")
    monkeypatch.setattr(timer_skill.clock, "is_synced", lambda: False)
    fresh = rig.restart()
    assert fresh.tick() is None
    assert "aren't any timers" in rig.say("how long left")


def test_a_corrupt_state_file_does_not_stop_it(rig, tmp_path):
    (tmp_path / "timers.json").write_text("{ not json")
    fresh = rig.restart()
    assert fresh.tick() is None
    assert "10 minutes" in rig.say("set a timer for 10 minutes")
