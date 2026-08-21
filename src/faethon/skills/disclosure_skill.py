"""What left this house today.

The answer people actually want is not a byte count, it is whether the thing
in the corner is listening when they did not ask it to. So the spoken sentence
leads with the two facts that bear on that -- how much of the room went out,
and how much of it went without a wake word -- and leaves scale to the report.

It cannot say what was in any of it, and that is the design rather than a
limitation. A ledger able to recite your conversations back would have to be
keeping your conversations. Host, path, kind, and whether a person asked: that
is enough to answer honestly and not enough to be worth stealing.

`faethon-sent` prints the full breakdown, because none of this fits in a
sentence anybody wants read aloud.
"""

from __future__ import annotations

from .. import disclosure
from .base import Skill

DAY = 86400.0


def _plural(n: int, word: str) -> str:
    """Spoken, so "1 requests" is not acceptable. Only regular plurals -- an
    irregular one is written out where it is used rather than guessed here."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _minutes(seconds: float) -> str:
    """Spoken duration. Nobody wants "one hundred and forty eight seconds"."""
    if seconds < 90:
        return f"{round(seconds)} seconds"
    minutes = seconds / 60
    if minutes < 10:
        return f"{minutes:.1f} minutes".replace(".0 ", " ")
    return f"{round(minutes)} minutes"


class DisclosureSkill(Skill):
    name = "get_disclosures"
    tag = "utility"
    description = (
        "Report what data has left this machine for the cloud today -- how "
        "many requests, to whom, how much audio, and how much went out "
        "without the user starting it. Use for questions about privacy, what "
        "was sent, what is being shared, or who is being contacted."
    )

    _END = r"(?:\s+(?:please|for me))*[^\w]*$"
    _WHEN = r"(?:\s+(?:today|so far|this session|recently))?"

    patterns = [
        rf"\bwhat (?:did|have) you sen[dt](?: to)?(?: the)? ?(?:cloud|internet)?{_WHEN}{_END}",
        rf"\bwhat(?:'s|s| is) (?:been )?(?:sent|shared|left)(?: to)?(?: the)? ?"
        rf"(?:cloud|internet)?{_WHEN}{_END}",
        r"\bwhat data (?:did|have) you (?:send|sent|share|shared)\b",
        r"\bwhat (?:are|were) you sending\b",
        r"\bwho (?:did|have) you (?:talk|talked|spoken|speak)(?:ed)? to\b",
        r"\bwhat leaves? (?:this |the )?(?:house|machine|room)\b",
        rf"\bhow much (?:data|audio) (?:did|have) you sen[dt]{_WHEN}{_END}",
        r"\bare you (?:recording|listening to|spying on) (?:me|us)\b",
    ]

    parameters: dict = {"type": "object", "properties": {}, "required": []}

    def run(self, **params: object) -> str:
        rows = disclosure.LEDGER.read(since_seconds=DAY)
        if not rows:
            return "Nothing has left this machine today that I have a record of."

        s = disclosure.summarise(rows)
        voice = s["by_kind"].get(disclosure.VOICE, 0)
        location = s["by_kind"].get(disclosure.LOCATION, 0)

        parts = [f"{_plural(s['calls'], 'request')} went out today"]

        # Voice first: it is the one people mean.
        if voice:
            parts.append(f"{voice} of them carrying audio from this room")
        if s["unasked"]:
            parts.append(f"{s['unasked']} with nobody asking")
        said = ", ".join(parts) + "."

        if location:
            times = "once" if location == 1 else f"{location} times"
            said += f" Your location went to the weather service {times}."
        if s["withheld"]:
            # The reassuring half, and the only part with no other trace.
            n = s["withheld"]
            said += (
                f" I also heard {n} stretch{'' if n == 1 else 'es'} of silence "
                f"and sent {'nothing' if n == 1 else 'none of them'}."
            )
        return said


SKILL = DisclosureSkill()
