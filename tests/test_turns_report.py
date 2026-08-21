"""Summarising the turn log.

This was a script under scripts/, which meant nothing tested it -- and it
shipped with a defect that mattered: run against two minutes of data it
printed "$292.21 a month at this rate". A projection from a window that short
says more about when you ran it than about anything else, and it is exactly the
sort of number that would have someone panic-topping-up an account.

Moving it into the package makes it an entry point and makes it testable, which
were the same change.
"""

from __future__ import annotations

import json
import time

import pytest

from faethon import state, turns
from faethon.turnlog import NAME


@pytest.fixture
def log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(state, "state_dir", lambda: tmp_path)

    class Rig:
        path = tmp_path / NAME

        def write(self, rows):
            self.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        def report(self, *argv):
            monkeypatch.setattr("sys.argv", ["faethon-turns", *argv])
            turns.main()
            return capsys.readouterr().out

    return Rig()


def row(**kw):
    base = {"at": time.time(), "route": "llm", "wake": True, "audio_s": 1.0,
            "stt_s": 1.0, "reply_s": 2.0, "total_s": 3.0, "cost": 0.0001,
            "chars": 20, "said_chars": 60, "held": 1, "interrupted": False}
    return {**base, **kw}


# -- percentiles -------------------------------------------------------------


def test_percentiles_describe_the_tail_not_the_average():
    """An average hides a 19-second outlier among fifteen good turns, which is
    the one thing anyone reading this wants to see."""
    values = [1.0] * 9 + [19.0]
    assert turns.percentile(values, 0.5) == 1.0
    assert turns.percentile(values, 0.9) == 19.0


def test_percentile_of_nothing_is_not_a_crash():
    assert turns.percentile([], 0.5) == 0.0


# -- the projection guard ----------------------------------------------------


def test_it_refuses_to_project_a_month_from_minutes(log):
    """The defect this shipped with. Two minutes of testing became "$292.21 a
    month", which is both absurd and alarming."""
    now = time.time()
    log.write([row(at=now), row(at=now + 90)])
    out = log.report()
    assert "too short a window" in out
    assert "a month at this rate" not in out


def test_it_projects_from_a_long_enough_window(log):
    now = time.time()
    log.write([row(at=now - 86400), row(at=now)])
    assert "a month at this rate" in log.report()


# -- reading the log ---------------------------------------------------------


def test_it_counts_where_turns_went(log):
    log.write([row(route="llm"), row(route="llm"), row(route="regex:get_time")])
    out = log.report()
    assert "llm" in out and "regex:get_time" in out
    assert "3 turns" in out


def test_days_filters_by_age(log):
    now = time.time()
    log.write([row(at=now - 10 * 86400), row(at=now)])
    assert "1 turns" in log.report("--days", "2")
    assert "2 turns" in log.report()


def test_a_corrupt_line_is_skipped_not_fatal(log):
    """The log is appended to by a process that can be killed mid-write. One
    bad line should cost one row, not the whole report."""
    good = json.dumps(row())
    log.path.write_text(good + "\n{ truncated\n" + good + "\n")
    assert "2 turns" in log.report()


def test_no_log_at_all_says_so(log):
    with pytest.raises(SystemExit, match="no turn log"):
        log.report()


def test_an_empty_window_says_so(log):
    log.write([row(at=time.time() - 30 * 86400)])
    with pytest.raises(SystemExit, match="no turns in that window"):
        log.report("--days", "1")


def test_it_separates_wake_words_from_follow_ups(log):
    """Turns per conversation is what the conversation cap should eventually
    be tuned from."""
    log.write([row(wake=True), row(wake=False), row(wake=False)])
    out = log.report()
    assert "1 started by a wake word" in out
    assert "2 follow-ups" in out
