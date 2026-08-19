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
from pathlib import Path

import numpy as np

from .audio import playback
from .config import ASSETS_DIR
from .providers.client import OpenRouterError

log = logging.getLogger(__name__)

NO_NETWORK = "no-network"
NO_CREDIT = "no-credit"
NO_MIC = "no-mic"

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
