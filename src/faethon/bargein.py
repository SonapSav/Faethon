"""Listening for the wake word while Faethon is talking, so it can be cut off.

The obvious way to interrupt a voice assistant -- stop as soon as anyone
speaks -- needs acoustic echo cancellation, because the microphone hears the
assistant far more loudly than it hears the room. Listening for a specific
*phrase* does not: a wake-word model is not asking "is someone talking", it is
asking "was that 'hey rhasspy'", and Faethon never says its own wake word.

Measured on this Pi by playing a 30s reply through the speaker while recording
the microphone, then scoring every frame through the real detector:

    median 0.0000   p95 0.0000   max 0.0014   threshold 0.7   ->  0 triggers

Three orders of magnitude of headroom, over a reply that twice contained the
word "stop" for good measure. So barge-in needs no echo cancellation, no second
model, and no new dependency -- only that something keeps feeding the detector
while the speaker is busy.

It does need a *lower threshold* than waking does, which is the one thing that
is not obvious. Faethon's voice does not trigger the detector, but it does mask
yours. Playing a reply and the user's own recorded wake word through the
speaker together, at the equal loudness they were measured to have at the
microphone, and scoring what came back:

    wake word, quiet room            0.9999
    wake word, over Faethon talking  0.3681
    Faethon talking, no wake word    0.0002

So the waking threshold of 0.7 never fires while Faethon speaks -- the feature
works or does not entirely on this number. 0.1 sits three and a half times
below the masked phrase and five hundred times above the self-audio floor.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


class BargeInListener:
    """Feeds the mic to the wake detector on a thread, and calls back on a hit.

    Used as a context manager around the speaking call. Exiting joins the
    thread, which matters more than it looks: the main loop reads the same
    capture stream, and two threads pulling frames from one arecord pipe would
    split the audio between them.
    """

    def __init__(self, stream, detector, on_detect, threshold=None) -> None:
        self._stream = stream
        self._detector = detector
        self._on_detect = on_detect
        self._threshold = threshold
        self._restore: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fired = False
        #: A capture failure on the listener thread, re-raised by the caller.
        self.error: BaseException | None = None

    def __enter__(self) -> "BargeInListener":
        # Clear the model's buffer so the wake word that opened this turn
        # cannot fire again, but leave it able to trigger immediately -- an
        # interruption a second into the reply is entirely reasonable.
        self._detector.reset(arm_refractory=False)
        if self._threshold is not None:
            # Borrow the detector at a lower bar for the duration. It is ours
            # alone while Faethon speaks, and it goes back before the main loop
            # uses it again for waking.
            self._restore = self._detector.threshold
            self._detector.threshold = self._threshold
        self._thread = threading.Thread(
            target=self._run, name="bargein", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._restore is not None:
            self._detector.threshold = self._restore
            self._restore = None
        if self._thread is not None:
            # One frame is 80ms, so this is the longest we ever wait; the
            # timeout is only a backstop against a wedged capture stream.
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                log.warning("barge-in listener did not stop; mic may be shared")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self._stream()
                if self._stop.is_set():
                    # Stopped while we were blocked on that read. Whatever it
                    # holds belongs to the next stage, not to us.
                    return
                if self._detector.process(frame) is not None:
                    self.fired = True
                    log.info("barge-in: stopping mid-reply")
                    self._on_detect()
                    return
        except BaseException as e:  # noqa: BLE001 - surfaced by the caller
            # Typically CaptureError: the USB mic went away mid-reply. Record
            # it rather than dying silently on a daemon thread.
            self.error = e
