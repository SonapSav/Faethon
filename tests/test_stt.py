"""Transcription: the leg that turns a microphone into words.

Untested until now, and it carries one setting with a documented failure
behind it. Whisper detects the language per request from the audio alone, and
on a short, accented or noisy clip it guesses wrong -- then transcribes *into*
that language, so an English question comes back as Greek. Pinning `language`
removes the failure outright, and nothing was guarding that it still gets sent.

The base64 detail matters too: OpenRouter wants bare base64 in `input_audio`,
not a `data:` URI. Getting that wrong is a 400 on every single utterance.

Offline: the client is a stub.
"""

from __future__ import annotations

import base64
import io
import wave

import pytest

from faethon.providers.stt import pcm_to_wav, transcribe

RATE = 16000
PCM = b"\x01\x00\x02\x00" * 8000        # 1 second of nonsense at 16 kHz


class FakeClient:
    def __init__(self, reply=None):
        self.reply = reply if reply is not None else {"text": " hello there "}
        self.path = None
        self.payload = None

    def post_json(self, path, payload):
        self.path = path
        self.payload = payload
        return self.reply


# -- the WAV wrapper ---------------------------------------------------------


def test_the_pcm_is_wrapped_as_a_real_wav():
    """Whisper is sent a container, not raw samples."""
    with wave.open(io.BytesIO(pcm_to_wav(PCM, RATE))) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == RATE
        assert w.readframes(w.getnframes()) == PCM


def test_the_declared_rate_is_the_one_it_was_given():
    """Sent at the wrong rate, Whisper transcribes a chipmunk."""
    with wave.open(io.BytesIO(pcm_to_wav(PCM, 44100))) as w:
        assert w.getframerate() == 44100


# -- the request -------------------------------------------------------------


def test_it_posts_to_the_transcriptions_endpoint():
    c = FakeClient()
    transcribe(c, PCM, model="whisper", sample_rate=RATE)
    assert c.path == "/audio/transcriptions"


def test_the_audio_is_bare_base64_not_a_data_uri():
    """OpenRouter rejects a data: URI here, and it would fail on every single
    utterance rather than intermittently."""
    c = FakeClient()
    transcribe(c, PCM, model="whisper", sample_rate=RATE)
    data = c.payload["input_audio"]["data"]
    assert not data.startswith("data:")
    assert base64.b64decode(data)[:4] == b"RIFF"
    assert c.payload["input_audio"]["format"] == "wav"


def test_the_language_is_pinned_when_set():
    """The whole reason the setting exists: unpinned, Whisper guesses the
    language from a short clip and transcribes English into Greek."""
    c = FakeClient()
    transcribe(c, PCM, model="whisper", sample_rate=RATE, language="en")
    assert c.payload["language"] == "en"


def test_an_empty_language_restores_auto_detection():
    """Deliberate, for a multilingual house -- so it must send no key at all
    rather than an empty one, which Whisper would reject."""
    c = FakeClient()
    transcribe(c, PCM, model="whisper", sample_rate=RATE, language="")
    assert "language" not in c.payload


def test_the_model_and_temperature_are_passed():
    c = FakeClient()
    transcribe(c, PCM, model="openai/whisper-large-v3-turbo",
               sample_rate=RATE, temperature=0.4)
    assert c.payload["model"] == "openai/whisper-large-v3-turbo"
    assert c.payload["temperature"] == 0.4


def test_temperature_defaults_to_deterministic():
    """0 makes decoding reproducible; higher values let it wander, which on a
    short command is how you get a different transcript each time."""
    c = FakeClient()
    transcribe(c, PCM, model="whisper", sample_rate=RATE)
    assert c.payload["temperature"] == 0.0


# -- the reply ---------------------------------------------------------------


def test_the_text_comes_back_stripped():
    """Whisper pads with whitespace, and the router matches regexes anchored
    at the start."""
    assert transcribe(FakeClient({"text": "  what time is it  "}),
                      PCM, model="w", sample_rate=RATE) == "what time is it"


@pytest.mark.parametrize("reply", [{}, {"text": None}, {"text": ""}, {"text": "   "}])
def test_nothing_heard_is_an_empty_string_not_a_crash(reply):
    """Silence, a false wake, or a clipped recording all land here. The main
    loop treats "" as "nothing to say" and moves on."""
    assert transcribe(FakeClient(reply), PCM, model="w", sample_rate=RATE) == ""


def test_a_failing_client_is_not_swallowed():
    """The caller announces it -- a network failure has its own clip -- so
    this must not quietly return an empty transcript instead."""
    from faethon.providers.client import OpenRouterError

    class Broken:
        def post_json(self, path, payload):
            raise OpenRouterError("connection refused")

    with pytest.raises(OpenRouterError):
        transcribe(Broken(), PCM, model="w", sample_rate=RATE)
