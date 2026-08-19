"""Short conversational memory: the last N exchanges, in RAM only.

Deliberately not persisted. Faethon forgets on restart, which is the right
default for an always-on microphone in a home.

It also forgets after a stretch of silence. Without that, context has no end:
ask about France in the morning and "how big is it?" in the afternoon still
answers about France, which is wrong in a way nothing announces. Measured over
71 real interactions on this Pi, the gap between turns has a clear shape --
median 21s, with 53 of 70 gaps under a minute, then almost nothing between 5
and 10 minutes, then a band of clearly-separate sessions beyond that. The
default window sits in that trough.

Expiry is checked from the wake-word loop rather than run on a timer thread.
That loop already ticks every 80ms while idle, so the buffer is genuinely
wiped at the deadline -- not merely ignored next time someone speaks, which
would leave the text sitting in RAM indefinitely for a feature whose whole
point is that it stops being there.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


class Memory:
    def __init__(
        self,
        max_turns: int = 10,
        idle_timeout_sec: float = 600.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        # One turn is a user message plus Faethon's reply.
        self._turns: deque[tuple[str, str]] = deque(maxlen=max_turns)
        #: Seconds of silence after which the conversation is dropped. 0 or
        #: less keeps it until something else clears it.
        self.idle_timeout_sec = idle_timeout_sec
        self._now = now
        self._touched = now()

    def add(self, user: str, assistant: str) -> None:
        if user and assistant:
            self._turns.append((user, assistant))
            self._touched = self._now()

    def clear(self) -> None:
        self._turns.clear()
        self._touched = self._now()

    @property
    def idle_sec(self) -> float:
        return self._now() - self._touched

    def expire_if_idle(self) -> bool:
        """Drop the conversation if nobody has spoken for the idle window.

        Returns whether it cleared, so the caller can say so in the log. Cheap
        enough to call on every audio frame: an empty buffer -- the usual state
        while idle -- short-circuits before reading the clock.
        """
        if not self._turns or self.idle_timeout_sec <= 0:
            return False
        if self.idle_sec < self.idle_timeout_sec:
            return False
        log.info(
            "forgetting %d turn(s) after %.0f min idle",
            len(self._turns), self.idle_sec / 60,
        )
        self.clear()
        return True

    def __len__(self) -> int:
        return len(self._turns)

    def messages(self, system_prompt: str, user: str) -> list[dict[str, Any]]:
        """Build the chat payload: system, history, then the new message."""
        out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for prev_user, prev_assistant in self._turns:
            out.append({"role": "user", "content": prev_user})
            out.append({"role": "assistant", "content": prev_assistant})
        out.append({"role": "user", "content": user})
        return out
