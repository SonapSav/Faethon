"""One volume scale, for everything Faethon can make louder.

Levels 0 to 10, spoken as percentages. Two things use this now -- the speaker
in the room and the radio on the other Pi -- and they must agree, because
somebody who has learned that "volume 5" means half will say "radio volume 5"
and mean the same thing.

It lives here rather than in either skill for the reason the plural helper did:
the same logic in two places is the same logic in one place and a stale copy in
the other. That happened with plural() -- the spoken answer was fixed and the
report kept printing "1 times" -- and it was invisible because nothing failed.

The interesting part is `as_level`, and the trap it exists for is written into
its docstring: announcing percentages teaches people to say percentages back.
"""

from __future__ import annotations

import math

MIN_LEVEL, MAX_LEVEL = 0, 10
#: Each level is a tenth, so the spoken percentage is the level times this.
PERCENT_PER_LEVEL = 10


#: Whisper writes small numbers either way, and which one you get is not
#: stable: the same person asking twice produced "radio volume at six" and
#: "Radio volume at 7." within a minute of each other. A pattern that only
#: accepts digits works most of the time, which is the worst amount.
WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

#: For dropping into a pattern's capture group. Longest first so "seven" is
#: not matched as "se" by some future shorter alternative.
SPOKEN = r"\d{1,3}|" + "|".join(sorted(WORDS, key=len, reverse=True))


def to_number(text: object) -> int | None:
    """A spoken quantity as an integer, whether it arrived as digits or words."""
    if isinstance(text, int):
        return text
    word = str(text).strip().lower()
    if word in WORDS:
        return WORDS[word]
    try:
        return int(word)
    except ValueError:
        return None


def _half_up(value: float) -> int:
    """Round .5 away from zero, which is what a person means by "rounded".

    Not the built-in round(), which is banker's rounding: round(4.5) is 4 while
    round(5.5) is 6. On this scale that alternates -- 45% reads as level 4 but
    55% reads as level 6, and 25% as 2 while 35% is 4 -- so the same "turn it
    up" moves by 15 points or 5 depending on where it started. Measured after
    writing a docstring that claimed 45 was level 5; it was not.
    """
    return math.floor(value + 0.5)


def percent(level: int) -> str:
    return f"{level * PERCENT_PER_LEVEL}%"


def as_level(number: int, unit: str = "") -> int:
    """Turn a spoken number into a level, whichever scale it was said in.

    Faethon announces "Volume is set to 30%", so people say percentages back --
    and "30" read as a level would clamp to 10, i.e. every such request landing
    on maximum. A number is read as a percentage when it carries a unit, or
    when it is simply too big to be a level: "70" can only mean 70%.

    Any positive percentage gives at least level 1. Rounding 3% down to 0 would
    mute the speaker when somebody asked for something quiet, which is a
    different request and needs a different answer.
    """
    if unit or number > MAX_LEVEL:
        if number <= 0:
            return MIN_LEVEL
        return max(1, min(MAX_LEVEL, _half_up(number / PERCENT_PER_LEVEL)))
    return max(MIN_LEVEL, min(MAX_LEVEL, number))


def to_percent(level: int) -> int:
    """A level as the 0-100 number an API wants."""
    return max(MIN_LEVEL, min(MAX_LEVEL, level)) * PERCENT_PER_LEVEL


def from_percent(value: int) -> int:
    """A 0-100 reading as the nearest level.

    Nearest rather than truncated: a radio sitting at 45 is level 5 rather than
    level 4, so "turn it up" from there lands on 60 instead of appearing to do
    nothing much.
    """
    return max(MIN_LEVEL, min(MAX_LEVEL, _half_up(value / PERCENT_PER_LEVEL)))
