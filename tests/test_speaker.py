"""Speaker pipeline: ordering, pre-buffering, and error handling.

Synthesis runs concurrently so that sentence N+1 is ready before sentence N
finishes playing -- otherwise the sound card underruns at every full stop.
Concurrency means completion order is not request order, so the ordering
guarantee below is the thing that stops a reply coming out backwards.

Fully offline: both the TTS call and the audio device are stubbed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import pytest

from faethon import speech
from faethon.config import load_config
from faethon.providers.client import OpenRouterError


@pytest.fixture
def rig(monkeypatch):
    """Stub TTS and playback; record what reaches the speaker, in order."""
    written: list[bytes] = []
    opened: list[int] = []

    @contextmanager
    def fake_stream(device, rate):
        opened.append(rate)
        yield written.append

    monkeypatch.setattr(speech.playback, "stream", fake_stream)

    class Rig:
        delays: dict[str, float] = {}
        fail: set[str] = set()
        rate: int = 44100

        @property
        def opened_at(self) -> list[int]:
            return opened

        def audio(self) -> bytes:
            return b"".join(written)

        @property
        def device_opens(self) -> int:
            return len(opened)

    rig = Rig()

    def fake_synth(client, text, *, model, voice, on_rate=None, cost_per_1k_chars=0.0):
        time.sleep(rig.delays.get(text, 0.0))
        if text in rig.fail:
            raise OpenRouterError(f"synthetic failure for {text!r}")
        if on_rate is not None:
            on_rate(rig.rate)
        # One byte pair per character, tagged so order is checkable.
        for ch in text:
            yield ch.encode() * 2

    monkeypatch.setattr(speech.tts_mod, "synthesize_stream", fake_synth)
    return rig


@pytest.fixture
def config():
    return load_config()


def spoken(rig) -> str:
    """Reconstruct the text from the tagged PCM bytes."""
    raw = rig.audio()
    return bytes(raw[i] for i in range(0, len(raw), 2)).decode()


def test_sentences_play_in_order_despite_concurrent_synthesis(rig, config):
    # The first sentence is slowest, so without ordering the later ones would
    # overtake it and the reply would come out scrambled.
    rig.delays = {"AAAA": 0.30, "BBBB": 0.10, "CCCC": 0.0}
    speech.speak_streaming(object(), config, iter(["AAAA", "BBBB", "CCCC"]))
    assert spoken(rig) == "AAAABBBBCCCC"


def test_synthesis_actually_overlaps(rig, config):
    # Three 0.2s synthesises must take well under 0.6s, or they're sequential
    # and the underrun problem is back.
    rig.delays = {t: 0.20 for t in ("AAAA", "BBBB", "CCCC")}
    t0 = time.monotonic()
    speech.speak_streaming(object(), config, iter(["AAAA", "BBBB", "CCCC"]))
    assert time.monotonic() - t0 < 0.55


def test_device_is_opened_once_for_the_whole_reply(rig, config):
    # A separate aplay per sentence would put an audible stutter between them.
    speech.speak_streaming(object(), config, iter(["AAAA", "BBBB", "CCCC"]))
    assert rig.device_opens == 1


def test_short_reply_shorter_than_the_prebuffer_still_plays(rig, config):
    # The pre-roll must not swallow replies smaller than itself.
    speech.speak_streaming(object(), config, iter(["hi"]))
    assert spoken(rig) == "hi"
    assert rig.device_opens == 1


def test_nothing_to_say_opens_no_device(rig, config):
    speech.speak_streaming(object(), config, iter([]))
    assert rig.device_opens == 0


def test_whitespace_chunks_are_skipped(rig, config):
    speech.speak_streaming(object(), config, iter(["  ", "\n"]))
    assert rig.device_opens == 0


def test_a_failing_sentence_does_not_hang_the_turn(rig, config):
    # A dead synthesis worker must still release the player, or Faethon freezes.
    rig.fail = {"BBBB"}
    t0 = time.monotonic()
    speech.speak_streaming(object(), config, iter(["AAAA", "BBBB", "CCCC"]))
    assert time.monotonic() - t0 < 5.0
    # What did synthesise still gets spoken.
    assert "AAAA" in spoken(rig)


def test_returns_the_text_it_spoke(rig, config):
    out = speech.speak_streaming(object(), config, iter(["AAAA", "BBBB"]))
    assert out == "AAAA BBBB"


# -- sample rate detection ---------------------------------------------------

@pytest.mark.parametrize(
    "content_type,expected",
    [
        ("audio/pcm;rate=44100;channels=1", 44100),
        ("audio/pcm;rate=24000;channels=1", 24000),
        ("audio/pcm; rate=16000", 16000),
        ("audio/mpeg", None),
        ("", None),
        ("audio/pcm;rate=notanumber", None),
    ],
)
def test_rate_is_read_from_the_response(content_type, expected):
    # Models disagree: Fish Audio returns 44.1kHz, Kokoro and Deepgram 24kHz.
    # Playing 44.1k audio at 24k runs it at 0.54x speed.
    from faethon.providers.tts import rate_from_content_type
    assert rate_from_content_type(content_type) == expected


def test_device_opens_at_the_rate_the_model_returned(rig, config):
    # The detected rate has to reach aplay, not just be parsed and dropped.
    rig.rate = 44100
    speech.speak_streaming(object(), config, iter(["AAAA"]))
    assert rig.opened_at == [44100]


def test_a_different_model_rate_is_honoured_too(rig, config):
    rig.rate = 24000
    speech.speak_streaming(object(), config, iter(["AAAA"]))
    assert rig.opened_at == [24000]


# -- barge-in ----------------------------------------------------------------
# Stopping a reply is not the same as ending one: ending closes the pipe and
# lets aplay play out everything buffered, which on this Pi measured 9.1s of
# talking after the decision to stop. Aborting kills it instead, at 234ms.


def test_stopping_aborts_playback_rather_than_draining_it(config, monkeypatch):
    aborted = []

    class AbortableSink:
        def __call__(self, chunk): pass
        def abort(self): aborted.append(True)

    @contextmanager
    def fake_stream(device, rate):
        yield AbortableSink()

    monkeypatch.setattr(speech.playback, "stream", fake_stream)
    monkeypatch.setattr(speech, "PREBUFFER_MS", 0)   # open on the first chunk
    monkeypatch.setattr(
        speech.tts_mod, "synthesize_stream",
        lambda *a, on_rate=None, **kw: iter([b"\x00\x01" * 64]),
    )

    speaker = speech._Speaker(object(), config)
    speaker.start()
    speaker.send("AAAA")
    for _ in range(300):                     # wait for the device to open
        if speaker._sink is not None:
            break
        time.sleep(0.01)
    assert speaker._sink is not None, "device never opened"
    speaker.stop()
    speaker.finish()
    assert aborted, "stop() did not abort the playback stream"


def test_stopping_before_any_audio_opens_no_device(config, monkeypatch):
    """Interrupting during the pre-roll must not start the speaker up.

    The reply was cut off before a word of it was audible, so opening the
    device to play the pre-roll out would be Faethon talking after being told
    to stop.
    """
    opened = []

    @contextmanager
    def fake_stream(device, rate):
        opened.append(rate)
        yield lambda chunk: None

    monkeypatch.setattr(speech.playback, "stream", fake_stream)

    speaker = speech._Speaker(object(), config)
    speaker.start()
    speaker.stop()
    speaker.send("AAAA")
    speaker.finish()
    assert not opened


def test_stop_is_idempotent(rig, config):
    speaker = speech._Speaker(object(), config)
    speaker.start()
    speaker.stop()
    speaker.stop()
    assert speaker.stopped
    speaker.finish()


def test_an_interrupted_reply_stops_pulling_from_the_model(rig, config):
    """The rest of the reply is neither wanted nor free."""
    pulled: list[str] = []

    def chunks():
        for c in ["AAAA", "BBBB", "CCCC", "DDDD"]:
            pulled.append(c)
            yield c

    def stop_after_first(speaker):
        speaker.stop()

    out = speech.speak_streaming(
        object(), config, chunks(), on_start=stop_after_first
    )
    assert out.interrupted
    assert len(pulled) < 4, f"kept pulling after the stop: {pulled}"


def test_an_uninterrupted_reply_is_not_marked_interrupted(rig, config):
    out = speech.speak_streaming(object(), config, iter(["AAAA", "BBBB"]))
    assert not out.interrupted
    assert out == "AAAA BBBB"


def test_the_generator_is_closed_so_memory_still_records(rig, config):
    """Closing unwinds the router's generator, whose finally records what was
    said. Leaving it open would lose an interrupted turn from memory."""
    closed = []

    def chunks():
        try:
            yield "AAAA"
            yield "BBBB"
        finally:
            closed.append(True)

    speech.speak_streaming(object(), config, chunks())
    assert closed
