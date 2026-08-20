"""One line per turn, so the next threshold can be measured instead of guessed.

Every number in this project that turned out right was measured: the wake
threshold from scoring a real voice, the idle window from plotting 71 real
gaps, the barge-in threshold from a mixing experiment, the volume curve from
playing tones. Each of those datasets had to be manufactured for the occasion,
by hand, once.

And the ones that were not measured show it. The conversation cap is 20 turns
and 5 minutes because three conversations happened to be in the journal at the
time. The STT tail was characterised from eight calls. Both are defensible and
neither is measured.

So: append-only JSONL, written after every turn, in the state directory beside
the timers.

Deliberately no transcripts. Only what was routed where, how long each leg
took, what it cost, and how long the text was. journald already keeps every
transcript, and whether it should is a live decision -- recording them a
second time here would answer it by accident.

Rotates on size rather than time. An SD card is the one part of a Pi that
wears out, and a log nobody prunes is a slow way to fill it.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from . import state

log = logging.getLogger(__name__)

NAME = "turns.jsonl"


class TurnLog:
    """Appends turn records, rotating when the file gets large."""

    def __init__(self, enabled: bool = True, max_bytes: int = 5_000_000) -> None:
        self.enabled = enabled
        self.max_bytes = max_bytes

    def append(self, **fields: Any) -> None:
        """Write one record. Never raises: a log that breaks a turn is worse
        than no log."""
        if not self.enabled:
            return
        try:
            path = state.state_dir() / NAME
            if self.max_bytes and path.exists() and path.stat().st_size >= self.max_bytes:
                # One generation back, then gone. Keeping more would need a
                # policy for pruning it, which is how logs eat cards.
                path.replace(path.with_suffix(".jsonl.1"))
            record = {"at": round(time.time(), 3), **fields}
            with path.open("a") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except (OSError, TypeError, ValueError) as e:
            log.warning("could not write turn log: %s", e)
