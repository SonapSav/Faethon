"""What time it is, and whether that can be believed.

The model has no idea what day it is. Left ungrounded it does not say so -- it
retrieves a date from its training data and answers confidently from that:

    Q: what is the date today
    A: Today is Friday, March 14, 2025.

which was nineteen months out, and "how long until Christmas" was computed off
it. Same shape as the identity confabulation: no grounding fact, so it reaches
for the nearest thing it knows.

The catch is that this Pi has no battery-backed clock. For the first couple of
minutes after a cold boot the wall clock is whatever systemd-timesyncd restored
from disk, and it steps when NTP corrects it -- measured at 2.5 minutes on this
machine. Telling the model a wrong date is worse than telling it nothing, since
a wrong date is exactly as confident as a right one. So the grounding is
omitted entirely until systemd says the clock has been corrected.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

#: systemd-timesyncd creates this once it has corrected the clock.
SYNC_FLAG = Path("/run/systemd/timesync/synchronized")


def is_synced() -> bool:
    """Whether the wall clock has been corrected since boot."""
    return SYNC_FLAG.exists()


def grounding(now: datetime | None = None) -> str:
    """A line telling the model the date and time, or "" if it can't be trusted.

    Costs about twenty tokens on every request -- fractions of a cent a month
    against answering "how long until Christmas" at all.
    """
    if not is_synced():
        return ""
    now = now or datetime.now().astimezone()
    return (
        f"\nRight now it is {now:%A %-d %B %Y}, {now:%H:%M} "
        f"local time ({now:%Z}). Use this for anything about dates, days of "
        f"the week, or how long until something -- and give only the answer, "
        f"never the arithmetic."
    )
