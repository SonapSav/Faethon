"""The 0-10 volume scale shared by the speaker and the radio.

Shared rather than duplicated because the same logic in two places is one
place and a stale copy: plural() was fixed in the spoken answer while the
report kept printing "1 times", and nothing failed to point it out.
"""

from __future__ import annotations

import pytest

from faethon import levels


@pytest.mark.parametrize("level,want", [(0, "0%"), (5, "50%"), (10, "100%")])
def test_percent(level, want):
    assert levels.percent(level) == want


@pytest.mark.parametrize("number,unit,want", [
    (5, "", 5),          # a level
    (10, "", 10),
    (0, "", 0),
    (50, "", 5),         # too big to be a level, so it is a percentage
    (50, "%", 5),        # said with a unit
    (50, "percent", 5),
    (45, "%", 5),
    (3, "%", 1),         # never rounds a quiet request down into a mute
    (0, "%", 0),
    (200, "%", 10),
    (-5, "", 0),
])
def test_as_level(number, unit, want):
    assert levels.as_level(number, unit) == want


@pytest.mark.parametrize("value,want", [
    (0, 0), (3, 0), (5, 1), (39, 4), (45, 5), (50, 5), (55, 6), (100, 10),
])
def test_from_percent(value, want):
    assert levels.from_percent(value) == want


def test_rounding_is_half_up_not_bankers():
    """Python's round() is banker's rounding: round(4.5) is 4 and round(5.5)
    is 6. On this scale that alternates -- 45% would read as level 4 while 55%
    reads as 6 -- so the same "turn it up" moves by 15 points or by 5
    depending where it started. Found by writing a docstring claiming 45 was
    level 5 and then measuring that it was not.
    """
    assert round(4.5) == 4, "the built-in still behaves as this test assumes"
    assert levels.from_percent(45) == 5
    assert levels.from_percent(25) == 3
    assert levels.from_percent(35) == 4
    assert levels.as_level(45, "%") == 5


def test_a_level_survives_a_round_trip():
    for level in range(levels.MIN_LEVEL, levels.MAX_LEVEL + 1):
        assert levels.from_percent(levels.to_percent(level)) == level
