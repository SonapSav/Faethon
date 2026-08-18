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

    def fake_synth(client, text, *, model, voice, on_rate=None):
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
