"""The turn log: metadata about turns, never their contents.

Every number in this project that turned out right was measured, and each of
those datasets had to be collected by hand for the occasion. The ones that were
not measured show it -- the conversation cap is 20 turns and 5 minutes because
three conversations happened to be in the journal at the time.

Two properties matter more than the format: it never contains a transcript, and
it never breaks a turn.
"""

from __future__ import annotations

import json

import pytest

from faethon.turnlog import NAME, TurnLog


@pytest.fixture
def log(tmp_path, monkeypatch):
    from faethon import state

    monkeypatch.setattr(state, "state_dir", lambda: tmp_path)

    class Rig(TurnLog):
        path = tmp_path / NAME

        def records(self):
            if not self.path.exists():
                return []
            return [json.loads(x) for x in self.path.read_text().splitlines()]

    return Rig()


def test_a_turn_becomes_a_line(log):
    log.append(route="regex:get_time", total_s=3.4, cost=0.00004)
    rows = log.records()
    assert len(rows) == 1
    assert rows[0]["route"] == "regex:get_time"
    assert rows[0]["cost"] == 0.00004


def test_every_record_is_stamped(log):
    log.append(route="llm")
    assert log.records()[0]["at"] > 0


def test_lines_accumulate(log):
    for i in range(5):
        log.append(route="llm", n=i)
    assert [r["n"] for r in log.records()] == [0, 1, 2, 3, 4]


def test_it_can_be_turned_off(log):
    log.enabled = False
    log.append(route="llm")
    assert log.records() == []


# -- the two properties that matter ------------------------------------------


def test_it_never_writes_a_transcript(log):
    """journald already keeps every transcript, and whether it should is a
    live decision. Recording them again here would answer it by accident."""
    log.append(route="llm", chars=42, said_chars=88)
    line = log.path.read_text()
    assert "chars" in line
    for word in ("text", "transcript", "heard", "said\":"):
        assert word not in line


def test_a_broken_log_does_not_break_a_turn(log, monkeypatch):
    """A log that can take the assistant down is worse than no log."""
    from faethon import state

    monkeypatch.setattr(state, "state_dir", lambda: (_ for _ in ()).throw(OSError("read-only")))
    log.append(route="llm")          # must not raise


def test_unserialisable_fields_are_survived(log):
    log.append(route="llm", weird=object())
    assert log.records() == []       # dropped, not crashed
    log.append(route="llm", fine=1)
    assert len(log.records()) == 1


# -- rotation ----------------------------------------------------------------


def test_it_rotates_rather_than_growing_forever(log):
    """An SD card is the part of a Pi that wears out."""
    log.max_bytes = 200
    for i in range(40):
        log.append(route="llm", n=i)
    assert log.path.stat().st_size < 400
    assert log.path.with_suffix(".jsonl.1").exists()


def test_only_one_generation_is_kept(log):
    """More would need a policy for pruning it, which is how logs eat cards."""
    log.max_bytes = 120
    for i in range(60):
        log.append(route="llm", n=i)
    siblings = list(log.path.parent.glob("turns.jsonl*"))
    assert len(siblings) == 2


def test_zero_disables_rotation(log):
    log.max_bytes = 0
    for i in range(50):
        log.append(route="llm", n=i)
    assert not log.path.with_suffix(".jsonl.1").exists()
