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
    # Kept out of the LLM's tool list, which is unusual for a harmless
    # read-only skill. The reason is that a tool result ends the turn: it is
    # yielded straight to speech and never fed back for a second pass. So when
    # the model reached for this as a *step* -- "how long until Christmas?"
    # -- the user got "It's Thursday, August 20" and nothing else.
    #
    # Since the system prompt now carries the date and time on every request
    # (see faethon/clock.py), the model has no need of the tool anyway. The
    # regex path still answers "what time is it" instantly and for nothing.
    regex_only = True
    description = (
        "Get the current local time, the current date, or both. "
        "Use when the user asks what time or day it is."
    )

    #: The question has to *end* here. Without it, "what is the time complexity
    #: of quicksort" and "what is the date of the meeting" both answered with
    #: the clock -- \b after "time" is satisfied by the following space, so the
    #: pattern matched a fragment of a longer, unrelated question.
    #: A few trailing pleasantries are allowed, since people say them.
    _END = r"(?:\s+(?:now|today|right now|please|for me))*[^\w]*$"

    patterns = [
        rf"\bwhat(?:'s|s| is) the (?P<kind>time|date){_END}",
        rf"\bwhat (?P<kind>time|day) is it{_END}",
        rf"\bwhat(?:'s|s| is) (?:today'?s? )?(?P<kind>date|day){_END}",
        rf"\btell me the (?P<kind>time|date){_END}",
        rf"\bcurrent (?P<kind>time|date|year){_END}",
        # The year was reaching the model, which answers it correctly from the
        # prompt grounding but costs a call for a fact already on the machine.
        rf"\bwhat(?:'s|s| is) (?:the |this )?(?:current )?(?P<kind>year){_END}",
        rf"\bwhat (?P<kind>year) is it{_END}",
        r"\bhow late is it\b(?P<kind>)",
    ]

    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["time", "date", "year", "both"],
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

        if kind == "year":
            return f"It's {now.year}."
        if kind == "date":
            return f"It's {date_str}."
        if kind in ("day",):
            return f"It's {now.strftime('%A')}."
        if kind == "both":
            return f"It's {time_str} on {date_str}."
        return f"It's {time_str}."


SKILL = TimeSkill()
