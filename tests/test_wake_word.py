"""The wake word, against recordings of the owner's actual voice.

Every other test in this suite exercises logic. This one exercises the
*configuration*: whether the model and thresholds in config.yaml can still hear
the person who has to use them. That is not a question the code can answer, and
it is the one that has gone wrong here twice.

The first time, a threshold of 0.78 was chosen from a custom model's scores
against synthesised speech, where it reached 0.84-0.91. On a real voice through
the real microphone it peaked at 0.52-0.55, so nothing ever triggered it. The
second time, barge-in was built against a threshold of 0.7 that a wake word
spoken over Faethon's own voice scores 0.368 on, so it would never once have
fired.

Both were configuration failures that no unit test could have caught, because
in both cases the code did exactly what it was told.

The fixtures are real recordings through the real microphone:

    hey-rhasspy-0038.wav        the wake word            peak 0.9999
    mixed-jarvis-roxy-0025.wav  "hey jarvis", "hey roxy" peak 0.0831

The second matters as much as the first. It is the same voice and the same
microphone saying phrases of the same shape that are *not* the wake word, so it
catches a model or threshold loose enough to fire on anything vaguely similar.

Offline, but it needs two things that are not in the repository.

The pretrained model is fetched by openWakeWord on first run, so a fresh clone
skips these until Faethon has been started once rather than quietly downloading
during a test run.

The recordings are absent on purpose: this is a public repository, and a
recording of someone's voice is not the kind of thing to publish for the
convenience of a fixture. Supply your own to get this coverage back -- 16 kHz
mono WAV, one saying the wake word and one saying similar phrases that are not
it -- and these will pick them up.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import openwakeword
import pytest

from faethon.config import PROJECT_ROOT, load_config
from faethon.wake import WakeWordDetector

SAMPLES = PROJECT_ROOT / "training-samples"
SPOKEN_WAKE_WORD = SAMPLES / "hey-rhasspy-0038.wav"
SPOKEN_OTHER_WAKE_WORDS = SAMPLES / "mixed-jarvis-roxy-0025.wav"


def _model_on_disk(name: str) -> bool:
    """Mirror of wake._ensure_model's lookup, without the download."""
    if Path(name).exists():
        return True
    stem_to_key = {
        Path(meta["download_url"]).stem: key
        for key, meta in openwakeword.MODELS.items()
    }
    meta = openwakeword.MODELS.get(stem_to_key.get(name, name))
    return bool(meta) and Path(meta["model_path"]).exists()


@pytest.fixture(scope="module")
def config():
    if not (SPOKEN_WAKE_WORD.exists() and SPOKEN_OTHER_WAKE_WORDS.exists()):
        pytest.skip(
            f"no voice recordings in {SAMPLES.name}/ -- see this module's "
            "docstring for what to put there"
        )
    return load_config()


@pytest.fixture(scope="module")
def detector(config):
    if not _model_on_disk(config.wake.model):
        pytest.skip(f"wake model {config.wake.model!r} not downloaded yet")
    # Refractory suppression would hide repeat detections we want to count.
    return WakeWordDetector(
        config.wake.model, threshold=config.wake.threshold, refractory_sec=0.0
    )


def _scan(detector, config, path: Path) -> float:
    """Highest score over the clip, the way the running assistant would see it."""
    with wave.open(str(path)) as w:
        assert w.getframerate() == config.audio.sample_rate
        assert w.getnchannels() == 1
        pcm = w.readframes(w.getnframes())

    detector.reset(arm_refractory=False)
    frame = config.audio.frame_bytes
    best = 0.0
    for i in range(0, len(pcm) - frame, frame):
        raw = detector._model.predict(np.frombuffer(pcm[i:i + frame], dtype=np.int16))
        best = max(best, max(raw.values()) if raw else 0.0)
    return best


# Scanned once each: inference over ten seconds of audio is the slow part here,
# and every test below wants the same two numbers.
@pytest.fixture(scope="module")
def wake_peak(detector, config) -> float:
    return _scan(detector, config, SPOKEN_WAKE_WORD)


@pytest.fixture(scope="module")
def other_peak(detector, config) -> float:
    return _scan(detector, config, SPOKEN_OTHER_WAKE_WORDS)


def test_the_owner_can_actually_wake_it(wake_peak, config):
    """The configured model and threshold must fire on the real wake word.

    If this fails, Faethon is deaf to the person it belongs to, whatever the
    rest of the suite says.
    """
    assert wake_peak >= config.wake.threshold, (
        f"peak {wake_peak:.4f} is under wake.threshold {config.wake.threshold}: "
        "the wake word would never trigger"
    )


def test_waking_has_room_to_spare(wake_peak, config):
    """A threshold that only just clears is one bad day from not clearing.

    Measured at 0.9999 against 0.7. Failing this means the margin has eroded --
    a model change, or a threshold raised too far -- long before the assistant
    stops working outright.
    """
    assert wake_peak >= config.wake.threshold + 0.15, (
        f"peak {wake_peak:.4f} clears {config.wake.threshold} by too little"
    )


def test_other_wake_words_do_not_trigger_it(other_peak, config):
    """Same voice, same mic, phrases of the same shape, but not the wake word.

    Guards the other direction from the test above: a threshold low enough to
    always hear you is also low enough to fire on the television.
    """
    assert other_peak < config.wake.threshold, (
        f"'hey jarvis' / 'hey roxy' scored {other_peak:.4f} against a threshold of "
        f"{config.wake.threshold} -- Faethon wakes on the wrong phrase"
    )


def test_other_wake_words_do_not_interrupt_a_reply(other_peak, config):
    """Barge-in listens at a much lower bar, so it needs checking separately.

    This is the tighter of the two margins by a long way: the wrong-phrase peak
    is 0.0831 against a barge-in threshold of 0.1. It holds, but not by much,
    and lowering barge_in_threshold to chase a missed interruption would cross
    it -- at which point saying almost anything would cut Faethon off.
    """
    assert other_peak < config.conversation.barge_in_threshold, (
        f"wrong-phrase peak {other_peak:.4f} reaches barge_in_threshold "
        f"{config.conversation.barge_in_threshold} -- replies would be cut off "
        "by phrases that are not the wake word"
    )


def test_barge_in_listens_below_the_waking_threshold(config):
    """Faethon's own voice masks yours: measured 0.368 for a wake word spoken
    over a reply, against 0.9999 in a quiet room. Set equal to wake.threshold,
    barge-in silently never fires."""
    assert config.conversation.barge_in_threshold < config.wake.threshold


def test_a_failure_here_does_not_print_the_api_key(config):
    """These tests assert on config objects, and pydantic reprs the whole tree.

    With a plain str that put the OpenRouter key in the output of every failing
    test, and so into CI logs and any traceback anyone pastes into an issue.
    Found the honest way: by reading it in this file's own failure output.
    """
    assert "sk-or" not in repr(config)
    assert "sk-or" not in str(config.settings)


# -- the startup greeting ----------------------------------------------------


GREETING = PROJECT_ROOT / "assets" / "greeting.wav"


def test_the_greeting_stays_under_the_wake_threshold(detector, config):
    """Saying the assistant's name is most of saying the wake word.

    The current wording avoids it and scores ~0.0001. An earlier draft opening
    "Hi, I am Rhasspy" measured 0.5036 through the speaker and mic. This test
    is what catches a rewording that drifts back toward the wake phrase --
    which would otherwise only show up as Faethon waking itself on every start.
    """
    if not GREETING.exists():
        pytest.skip("greeting not rendered; run scripts/make_speech.py")

    with wave.open(str(GREETING)) as w:
        raw = w.readframes(w.getnframes())
        rate = w.getframerate()

    # The file is at the TTS provider's rate; the model wants 16 kHz.
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    step = rate / config.audio.sample_rate
    idx = (np.arange(int(len(audio) / step)) * step).astype(int)
    resampled = audio[idx].astype(np.int16)

    detector.reset(arm_refractory=False)
    frame = config.audio.frame_bytes
    pcm = resampled.tobytes()
    peak = max(
        max(detector._model.predict(
            np.frombuffer(pcm[i:i + frame], dtype=np.int16)).values() or [0.0])
        for i in range(0, len(pcm) - frame, frame)
    )
    assert peak < config.wake.threshold, (
        f"the greeting scores {peak:.4f} against a wake threshold of "
        f"{config.wake.threshold} -- Faethon would wake itself on startup if "
        "this were ever played with the microphone open"
    )
