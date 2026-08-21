"""Telling the user when Faethon can't do its job.

Every failure state here is silent today, and for the same structural reason:
the mechanism for speaking is the thing that has broken. Synthesising "I can't
reach the network" over the network does not work, and the code currently tries
exactly that -- it catches a failed transcription and calls cloud TTS to
apologise for it.

So these are pre-rendered clips, played straight to the speaker. No network, no
API key, no credit, no LLM. Render them with scripts/make_speech.py.

Two behaviours matter as much as the clips:

* **Say it once.** An outage lasts as long as it lasts, and repeating the same
  sentence at every attempt turns information into nagging. A status is
  announced once and then suppressed until something works again.
* **Don't try the doomed call first.** Reaching for cloud TTS before falling
  back costs the whole retry-and-backoff sequence, so the user waits ten
  seconds to be told there is no network.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path

import numpy as np

from .audio import playback
from .config import ASSETS_DIR
from .providers.client import OpenRouterError

log = logging.getLogger(__name__)

NO_NETWORK = "no-network"
NO_CREDIT = "no-credit"
NO_MIC = "no-mic"


def _parse_hhmm(value: str) -> dt_time | None:
    """"22:30" -> time(22, 30), or None if it cannot be read."""
    try:
        hh, mm = value.strip().split(":")
        return dt_time(int(hh), int(mm))
    except (ValueError, AttributeError):
        return None
MIC_BACK = "mic-back"
LOW_CREDIT = "low-credit"

#: 402 is the whole reason these are separate clips: "check the router" and
#: "top up the account" are different instructions, and guessing wrong sends
#: someone the wrong way.
PAYMENT_REQUIRED = 402


def classify(error: OpenRouterError) -> str:
    """Which status a failed OpenRouter call represents."""
    return NO_CREDIT if error.status == PAYMENT_REQUIRED else NO_NETWORK


class Announcer:
    """Plays a status clip, at most once until something succeeds."""

    def __init__(self, device: str, assets: Path = ASSETS_DIR) -> None:
        self._device = device
        self._assets = assets
        self._said: set[str] = set()

    def say(self, status: str) -> bool:
        """Announce `status` unless it has already been announced.

        Returns whether anything was played, which is only really useful to
        tests -- the caller has nothing to do differently either way.
        """
        if status in self._said:
            return False
        clip = self._assets / f"{status}.wav"
        if not clip.exists():
            log.warning("no clip for %s at %s -- run scripts/make_speech.py",
                        status, clip)
            self._said.add(status)      # don't warn on every frame
            return False
        log.info("status: %s", status)
        self._said.add(status)
        playback.play_wav(clip, self._device)
        return True

    def recovered(self) -> None:
        """Something worked. Let the next failure be heard again."""
        self._said.clear()

    def forget(self, status: str) -> bool:
        """Drop one status, and report whether it had actually been announced.

        The caller uses the answer to decide whether recovery is worth saying
        out loud. Telling someone the microphone is back, when they never
        heard it go, is just noise -- and at startup the common case is that
        nothing was ever wrong.
        """
        if status in self._said:
            self._said.discard(status)
            return True
        return False


class CreditWatch:
    """Warns once when the account balance crosses a threshold going down.

    Its suppression cannot use the Announcer's, and that is the whole reason it
    exists as a separate thing. `recovered()` clears every status and is called
    after each successful turn -- so a credit warning routed through it would be
    un-suppressed within seconds and fire again on every check, which is the
    nagging status.py was written to avoid.

    So: warn once per crossing, and re-arm only when the balance climbs back
    above the line. That only happens on a deliberate top-up, so it cannot
    flap.

    The balance lookup is injected rather than taken as a client, so this is
    testable without a network, and so a failing lookup is this class's problem
    rather than the caller's.
    """

    def __init__(
        self,
        warn_below: float,
        check_every_sec: float,
        balance: Callable[[], float | None],
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.warn_below = warn_below
        self.check_every_sec = check_every_sec
        self._balance = balance
        self._now = now
        self._warned = False
        #: Far enough in the past that the first turn checks.
        self._last_check = now() - check_every_sec

    def check(self) -> bool:
        """Look if it is time to. True means say something now."""
        if self.warn_below <= 0:
            return False
        if self._now() - self._last_check < self.check_every_sec:
            return False
        self._last_check = self._now()

        balance = self._balance()
        if balance is None:
            # Could not be read. A network problem is already announced by
            # whichever leg failed; a second voice for the same outage is noise.
            return False

        if balance > self.warn_below:
            self._warned = False        # topped up: re-arm
            return False
        if self._warned:
            return False
        self._warned = True
        log.info("credit low: $%.2f, below $%.2f", balance, self.warn_below)
        return True


class SilenceWatch:
    """Notices that the microphone has stopped producing anything at all.

    The failure this exists for raises no error. A wireless microphone whose
    transmitter is off or flat still enumerates, still opens, and still hands
    over frames -- of digital silence. Nothing fails, nothing logs, and Faethon
    sits there deaf indefinitely looking perfectly healthy.

    A live microphone in a silent room does not do this: measured on this Pi,
    a quiet room still peaks at 2-4 per frame and never once at zero. So frames
    that are literally all-zero mean the stream is dead rather than the house.
    """

    def __init__(self, frame_ms: int, after_sec: float = 120.0) -> None:
        self._frame_ms = frame_ms
        self._after_ms = after_sec * 1000
        self._silent_ms = 0.0

    @property
    def dead(self) -> bool:
        return self._silent_ms >= self._after_ms

    def feed(self, frame: bytes) -> bool:
        """Add a frame. Returns True the moment the stream looks dead."""
        was_dead = self.dead
        if np.frombuffer(frame, dtype=np.int16).any():
            self._silent_ms = 0.0
        else:
            self._silent_ms += self._frame_ms
        return self.dead and not was_dead

    def reset(self) -> None:
        self._silent_ms = 0.0


class Quiet:
    """Whether Faethon may speak unprompted right now.

    The one place that sees every unprompted voice at once. Without it each
    source decided alone to say its piece once, which reads fine individually
    and worse with every source added -- and nothing knew what time it was, so
    a latched under-voltage flag would announce itself at three in the morning.

    Anything said *during* a turn is exempt and never comes through here. If
    transcription fails and Faethon says so, the user is standing there waiting
    for a reply; that is an answer, not an announcement.
    """

    def __init__(
        self,
        quiet_start: str = "22:30",
        quiet_end: str = "07:30",
        min_gap_seconds: float = 30.0,
        now: Callable[[], float] = time.monotonic,
        clock: Callable[[], dt_time] | None = None,
    ) -> None:
        start, end = _parse_hhmm(quiet_start), _parse_hhmm(quiet_end)
        if start is None or end is None:
            # Disabled outright rather than half-applied. Collapsing only the
            # unreadable end to midnight would enforce a window nobody wrote --
            # a typo in quiet_start silently becoming 00:00 to 07:30 is worse
            # than no quiet hours, because it is wrong rather than absent.
            log.warning(
                "unreadable quiet hours (%r to %r) -- quiet hours disabled",
                quiet_start, quiet_end,
            )
            start = end = dt_time(0, 0)
        self.start, self.end = start, end
        self.min_gap_seconds = min_gap_seconds
        self._now = now
        self._clock = clock or (lambda: datetime.now().time())
        self._last_spoke: float | None = None

    @property
    def in_quiet_hours(self) -> bool:
        if self.start == self.end:
            return False        # equal times disable it
        now = self._clock()
        if self.start < self.end:
            return self.start <= now < self.end
        # Crosses midnight, which is the normal case for a night window.
        return now >= self.start or now < self.end

    def allows(self, urgency: str = "informational") -> bool:
        """Whether an announcement of this urgency may be spoken now."""
        if self._last_spoke is not None:
            if self._now() - self._last_spoke < self.min_gap_seconds:
                return False
        if urgency == "requested":
            return True
        return not self.in_quiet_hours

    def spoke(self) -> None:
        """Record that something was said, starting the gap."""
        self._last_spoke = self._now()
