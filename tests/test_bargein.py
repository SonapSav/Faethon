"""Cutting a reply off mid-sentence.

Barge-in works without echo cancellation because the trigger is a *phrase*
rather than speech in general -- Faethon never says its own wake word. That
part is measured against the hardware, not here (see the module docstring in
faethon/bargein.py). What these cover is the machinery around it: that the
listener stops the speaker, that it hands the microphone back cleanly, and that
an interrupted reply still reaches memory.

Fully offline: the capture stream, the detector and the audio device are all
stubbed.
"""

from __future__ import annotations

import threading
import time

import pytest

from faethon.bargein import BargeInListener


class FakeStream:
    """Yields frames on demand, blocking when empty like a real microphone."""

    def __init__(self, fire_after: int | None = None) -> None:
        self.fire_after = fire_after
        self.reads = 0
        self.released = threading.Event()

    def __call__(self) -> bytes:
        self.reads += 1
        time.sleep(0.005)
        return b"\x00\x00" * 1280


class FakeDetector:
    """Fires once, on the nth frame."""

    def __init__(self, fire_on: int | None = None) -> None:
        self.fire_on = fire_on
        self.seen = 0
        self.resets: list[bool] = []

    def reset(self, arm_refractory: bool = True) -> None:
        self.resets.append(arm_refractory)

    def process(self, frame: bytes):
        self.seen += 1
        return 0.98 if self.fire_on is not None and self.seen == self.fire_on else None


def test_the_wake_word_stops_the_reply():
    stopped = threading.Event()
    det = FakeDetector(fire_on=3)
    with BargeInListener(FakeStream(), det, stopped.set) as listener:
        assert stopped.wait(timeout=2.0), "listener never fired"
    assert listener.fired


def test_silence_leaves_the_reply_alone():
    stopped = threading.Event()
    with BargeInListener(FakeStream(), FakeDetector(), stopped.set) as listener:
        time.sleep(0.1)
    assert not listener.fired
    assert not stopped.is_set()


def test_the_detector_is_armed_to_fire_immediately():
    """The wake word that opened the turn started the refractory clock.

    Inheriting it would make Faethon deaf to an interruption for the first two
    seconds of a reply -- which is exactly when a too-long answer is most
    obviously going to be too long.
    """
    det = FakeDetector()
    with BargeInListener(FakeStream(), det, lambda: None):
        pass
    assert det.resets == [False]


def test_the_microphone_is_handed_back_before_speaking_resumes():
    """Two threads reading one arecord pipe would split the audio between them.

    The listener must be finished with the stream by the time the context
    manager returns, so the recorder that runs next gets whole frames.
    """
    stream = FakeStream()
    with BargeInListener(stream, FakeDetector(), lambda: None):
        time.sleep(0.05)
    after = stream.reads
    time.sleep(0.1)
    assert stream.reads == after, "listener kept reading after it was stopped"


def test_a_dead_microphone_is_reported_not_swallowed():
    """It runs on a daemon thread, where an exception would vanish silently."""

    class Broken:
        def __call__(self) -> bytes:
            raise RuntimeError("capture stream ended")

    with BargeInListener(Broken(), FakeDetector(), lambda: None) as listener:
        time.sleep(0.1)
    assert isinstance(listener.error, RuntimeError)
    assert not listener.fired


def test_it_listens_at_a_lower_bar_than_waking_does():
    """Faethon's voice masks yours, so the waking threshold never fires.

    Measured through the speaker and mic at equal loudness: the wake word
    scores 0.368 over Faethon talking, against 0.9999 in a quiet room. At the
    0.7 used for waking, barge-in would simply never work.
    """
    seen = []

    class Recording(FakeDetector):
        def process(self, frame):
            # What bar was in force at the moment a frame was judged.
            seen.append(self.threshold)
            return None

    rec = Recording()
    rec.threshold = 0.7
    with BargeInListener(FakeStream(), rec, lambda: None, threshold=0.1):
        time.sleep(0.05)
    assert seen and all(t == 0.1 for t in seen), f"listened at {set(seen)}"


def test_the_waking_threshold_is_put_back():
    """The detector is shared with the main loop, which wakes at the high bar.

    Leaving it lowered would make Faethon wake on almost anything for the rest
    of the session.
    """
    rec = FakeDetector()
    rec.threshold = 0.7
    with BargeInListener(FakeStream(), rec, lambda: None, threshold=0.1):
        pass
    assert rec.threshold == 0.7


def test_no_threshold_given_leaves_the_detector_alone():
    rec = FakeDetector()
    rec.threshold = 0.55
    with BargeInListener(FakeStream(), rec, lambda: None):
        time.sleep(0.02)
    assert rec.threshold == 0.55
