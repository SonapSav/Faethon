"""Utterance-boundary tests.

Real speech/non-speech judgement is exercised by scripts/check-audio.sh; these
cover the state machine around it, using noise as a stand-in for voice (which
webrtcvad reliably reports as speech).
"""

from __future__ import annotations

import numpy as np
import pytest

from faethon.audio.vad import Status, UtteranceRecorder

RATE = 16000
FRAME_SAMPLES = 1280                                   # 80 ms
SILENCE = b"\x00\x00" * FRAME_SAMPLES

_rng = np.random.default_rng(0)


def voiced(ms: int = 80) -> bytes:
    n = RATE * ms // 1000
    return (_rng.standard_normal(n) * 6000).astype(np.int16).tobytes()


def blip(voiced_ms: int = 20) -> bytes:
    """One 80 ms frame containing a short burst, the rest silence."""
    n = RATE * voiced_ms // 1000
    return (
        (_rng.standard_normal(n) * 6000).astype(np.int16).tobytes()
        + b"\x00\x00" * (FRAME_SAMPLES - n)
    )


def make(**kw) -> UtteranceRecorder:
    opts = dict(sample_rate=RATE, silence_ms=600, min_ms=400, max_ms=15000,
                start_timeout_ms=5000, speech_onset_ms=120)
    opts.update(kw)
    return UtteranceRecorder(**opts)


def run(rec, frames):
    """Feed frames until the recorder reaches a terminal status."""
    status = Status.LISTENING
    fed = 0
    for f in frames:
        fed += 1
        status = rec.feed(f)
        if status is not Status.LISTENING:
            break
    return status, fed * 80


def test_rejects_rates_webrtcvad_cannot_handle():
    with pytest.raises(ValueError, match="44100"):
        UtteranceRecorder(44100)


# -- the reported bug --------------------------------------------------------

def test_noise_blip_does_not_end_the_turn():
    # Regression: a single 20ms blip used to latch "speech started", and the
    # turn was discarded ~600ms later while the user was still thinking.
    rec = make()
    status, ms = run(rec, [SILENCE, blip(20)] + [SILENCE] * 30)
    assert status is Status.LISTENING or ms >= 5000, (
        f"gave up after {ms}ms on a noise blip"
    )


def test_user_can_pause_before_speaking():
    rec = make(start_timeout_ms=5000)
    # 3 seconds of thinking, then a real request.
    frames = [SILENCE] * 37 + [voiced()] * 15 + [SILENCE] * 10
    status, _ = run(rec, frames)
    assert status is Status.DONE
    assert rec.result(status).usable


def test_gives_up_only_after_start_timeout():
    rec = make(start_timeout_ms=2000)
    status, ms = run(rec, [SILENCE] * 200)
    assert status is Status.NO_SPEECH
    assert 2000 <= ms <= 2200, f"gave up at {ms}ms, expected ~2000ms"


def test_false_start_then_real_speech_still_works():
    rec = make()
    frames = [blip(20)] + [SILENCE] * 10 + [voiced()] * 15 + [SILENCE] * 10
    status, _ = run(rec, frames)
    assert status is Status.DONE
    assert rec.result(status).usable


# -- normal operation --------------------------------------------------------

def test_sustained_speech_then_silence_completes():
    rec = make()
    status, _ = run(rec, [voiced()] * 15 + [SILENCE] * 10)
    assert status is Status.DONE
    assert rec.result(status).usable


def test_max_duration_is_enforced_once_talking():
    rec = make(max_ms=800)
    status, _ = run(rec, [voiced()] * 40)
    assert status is Status.TIMED_OUT


def test_silence_only_is_not_usable():
    rec = make(start_timeout_ms=500)
    status, _ = run(rec, [SILENCE] * 50)
    assert rec.result(status).usable is False


def test_leading_silence_is_trimmed():
    # Waiting shouldn't inflate the audio sent to Whisper.
    rec = make()
    status, _ = run(rec, [SILENCE] * 25 + [voiced()] * 15 + [SILENCE] * 10)
    assert status is Status.DONE
    kept = len(rec.result(status).pcm) / 2 / RATE
    assert kept < 2.0, f"kept {kept:.2f}s; leading silence not trimmed"


def test_preroll_is_kept_so_first_phoneme_survives():
    rec = make()
    status, _ = run(rec, [SILENCE] * 25 + [voiced()] * 15 + [SILENCE] * 10)
    speech_only = 15 * 0.08
    kept = len(rec.result(status).pcm) / 2 / RATE
    assert kept > speech_only, "trimmed into the start of the speech"


def test_reset_clears_state():
    rec = make()
    for _ in range(5):
        rec.feed(voiced())
    rec.reset()
    assert rec._elapsed_ms == 0
    assert rec._chunks == []
    assert rec._heard_speech is False


def test_partial_frames_are_carried_over():
    # 50 ms doesn't divide into 20 ms VAD frames; the remainder must be kept,
    # or timing drifts over a long utterance.
    rec = make()
    half = b"\x00\x00" * 800     # 50 ms
    rec.feed(half)
    assert len(rec._pending) == (800 - 640) * 2
    rec.feed(half)
    assert rec._elapsed_ms == 100


# -- per-recording start timeout ---------------------------------------------
# The wake-word window and the post-reply follow-up window are different waits
# through the same state machine, so the budget is set per reset(), not once.


def test_reset_can_shorten_the_wait_for_this_recording_only():
    rec = make(start_timeout_ms=5000)
    rec.reset(1000)
    status, ms = run(rec, [SILENCE] * 100)
    assert status is Status.NO_SPEECH
    assert 1000 <= ms <= 1100, f"gave up at {ms}ms, expected ~1000ms"

    # Back to the configured budget when nothing is asked for.
    rec.reset()
    status, ms = run(rec, [SILENCE] * 100)
    assert status is Status.NO_SPEECH
    assert 5000 <= ms <= 5100, f"gave up at {ms}ms, expected ~5000ms"


def test_a_short_window_still_lets_a_started_sentence_finish():
    """The follow-up budget is for starting to speak, not for speaking.

    Someone who answers within the window must not be cut off at 5s just
    because the window that let them in was 5s long.
    """
    rec = make(start_timeout_ms=5000)
    rec.reset(1000)
    frames = [SILENCE] * 5 + [voiced()] * 100 + [SILENCE] * 10
    status, _ = run(rec, frames)
    assert status is Status.DONE
    assert rec.result(status).usable
