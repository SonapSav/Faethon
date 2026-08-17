"""Demo skill: the current time and date.

Deliberately small -- it exists to prove both routing paths reach a skill and
to serve as the template for real ones.
"""

from __future__ import annotations

from datetime import datetime

from .base import Skill


class TimeSkill(Skill):
    name = "get_time"
    tag = "utility"
    description = (
        "Get the current local time, the current date, or both. "
        "Use when the user asks what time or day it is."
    )

    patterns = [
        r"\bwhat(?:'s|s| is) the (?P<kind>time|date)\b",
        r"\bwhat (?P<kind>time|day) is it\b",
        r"\bwhat(?:'s|s| is) (?:today'?s? )?(?P<kind>date|day)\b",
        r"\btell me the (?P<kind>time|date)\b",
        r"\bcurrent (?P<kind>time|date)\b",
    ]

    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["time", "date", "both"],
                "description": "Which to report. Defaults to time.",
            }
        },
        "required": [],
    }

    def run(self, **params: str) -> str:
        kind = (params.get("kind") or "time").lower()
        now = datetime.now()

        # Strip the leading zero from the hour: "9:05", not "09:05". Spoken
        # aloud, the zero becomes an audible "oh".
        time_str = now.strftime("%I:%M %p").lstrip("0")
        date_str = now.strftime("%A, %B ") + str(now.day)

        if kind == "date":
            return f"It's {date_str}."
        if kind in ("day",):
            return f"It's {now.strftime('%A')}."
        if kind == "both":
            return f"It's {time_str} on {date_str}."
        return f"It's {time_str}."


SKILL = TimeSkill()
