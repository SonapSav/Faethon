"""Saying so when Faethon can't do its job.

Every state here was previously silent, and for one structural reason: the
mechanism for speaking is the thing that has broken. The code caught a failed
transcription -- which almost always means the network is down -- and called
cloud TTS to apologise for it. So these are pre-rendered clips, played straight
to the speaker with no network, key, credit or model involved.

Two behaviours carry as much weight as the clips: not repeating, and telling
apart the failures whose fixes differ.
"""

from __future__ import annotations

import pytest

from faethon.providers.client import OpenRouterError, _http_error
from faethon.status import (
    NO_CREDIT,
    NO_MIC,
    NO_NETWORK,
    Announcer,
    SilenceWatch,
    classify,
)


@pytest.fixture
def announcer(tmp_path, monkeypatch):
    played: list[str] = []
    monkeypatch.setattr(
        "faethon.status.playback.play_wav",
        lambda path, device: played.append(path.name),
    )
    for name in (NO_NETWORK, NO_CREDIT, NO_MIC):
        (tmp_path / f"{name}.wav").write_bytes(b"RIFF")
    a = Announcer("fake-device", assets=tmp_path)
    a.played = played
    return a


# -- telling the failures apart ----------------------------------------------


def test_out_of_credit_is_not_reported_as_a_network_problem():
    """Different instructions: top up the account, versus check the router.

    Guessing wrong sends someone to reboot a router that was working fine.
    """
    assert classify(_http_error("/chat", 402, "insufficient credits")) == NO_CREDIT


def test_a_connection_failure_is_a_network_problem():
    assert classify(OpenRouterError("connection refused")) == NO_NETWORK


def test_a_server_error_is_reported_as_a_network_problem():
    """Not out of credit, and not something the user can act on differently."""
    assert classify(_http_error("/chat", 503, "upstream")) == NO_NETWORK


# -- not nagging -------------------------------------------------------------


def test_a_status_is_announced_once(announcer):
    """An outage lasts as long as it lasts. Saying it at every attempt turns
    information into nagging."""
    assert announcer.say(NO_NETWORK) is True
    assert announcer.say(NO_NETWORK) is False
    assert announcer.played == ["no-network.wav"]


def test_a_different_status_still_gets_through(announcer):
    announcer.say(NO_NETWORK)
    assert announcer.say(NO_CREDIT) is True
    assert announcer.played == ["no-network.wav", "no-credit.wav"]


def test_recovery_re_arms_it(announcer):
    """A later outage is news again."""
    announcer.say(NO_NETWORK)
    announcer.recovered()
    assert announcer.say(NO_NETWORK) is True
    assert announcer.played == ["no-network.wav", "no-network.wav"]


def test_a_missing_clip_does_not_loop_or_crash(tmp_path, monkeypatch):
    """A fresh clone before make_speech.py has been run. It must warn once,
    not once per audio frame."""
    calls: list[str] = []
    monkeypatch.setattr(
        "faethon.status.playback.play_wav", lambda p, d: calls.append(p.name)
    )
    a = Announcer("fake-device", assets=tmp_path)
    assert a.say(NO_MIC) is False
    assert a.say(NO_MIC) is False
    assert calls == []


# -- the microphone that fails without failing -------------------------------


def frames(n: int, live: bool) -> list[bytes]:
    quiet = b"\x00\x00" * 1280
    noisy = b"\x03\x00" + b"\x00\x00" * 1279       # one sample at the noise floor
    return [noisy if live else quiet for _ in range(n)]


def test_a_dead_stream_is_noticed():
    """The failure that raises nothing: a wireless mic whose transmitter is
    off still enumerates, still opens, and still hands over frames -- of
    digital silence, forever, looking perfectly healthy."""
    watch = SilenceWatch(frame_ms=80, after_sec=1.0)
    fired = [watch.feed(f) for f in frames(20, live=False)]
    assert sum(fired) == 1, "should announce exactly once, not every frame"
    assert watch.dead


def test_a_quiet_room_is_not_a_dead_stream():
    """Measured on this Pi: a live mic in a silent room peaks at 2-4 per frame
    and never at zero. That noise floor is the whole discriminator."""
    watch = SilenceWatch(frame_ms=80, after_sec=1.0)
    assert not any(watch.feed(f) for f in frames(200, live=True))
    assert not watch.dead


def test_one_live_frame_clears_it():
    watch = SilenceWatch(frame_ms=80, after_sec=1.0)
    for f in frames(10, live=False):
        watch.feed(f)
    watch.feed(frames(1, live=True)[0])
    assert not watch.dead


def test_it_waits_before_deciding():
    """A few silent frames are just a pause in the room, not a broken mic."""
    watch = SilenceWatch(frame_ms=80, after_sec=120.0)
    assert not any(watch.feed(f) for f in frames(100, live=False))


# -- TTS cost estimation -----------------------------------------------------
# The speech endpoint returns raw audio and no usage body, so unlike every
# other leg there is nothing authoritative to read. Left uncounted, a turn
# reported several times less than it cost -- speaking is the most expensive
# thing Faethon does.


def test_speaking_cost_scales_with_the_text_sent():
    """Billed on input text, not audio produced. Measured: five calls
    totalling 345 characters cost $0.001107, i.e. $0.0032 per thousand."""
    from faethon.providers.tts import estimate_cost

    assert estimate_cost("x" * 1000, 0.0032) == pytest.approx(0.0032)
    assert estimate_cost("x" * 345, 0.0032) == pytest.approx(0.001104, abs=1e-6)
    assert estimate_cost("", 0.0032) == 0.0


def test_a_zero_rate_disables_the_estimate_rather_than_guessing():
    from faethon.providers.tts import estimate_cost

    assert estimate_cost("a long spoken reply", 0.0) == 0.0


def test_speaking_is_billed_to_the_client(monkeypatch):
    """Without this the turn line omits the dominant cost of the turn."""
    from faethon.providers import tts as tts_mod

    class FakeClient:
        spent = 0.0

        def record_usage(self, usage):
            self.spent += usage["cost"]

        def post_stream(self, path, payload):
            from contextlib import contextmanager

            @contextmanager
            def cm():
                class R:
                    headers = {"content-type": "audio/pcm;rate=44100"}

                    def iter_bytes(self, n):
                        return iter([b"\x00\x01"])

                yield R()

            return cm()

    c = FakeClient()
    list(tts_mod.synthesize_stream(
        c, "x" * 500, model="m", voice="alloy", cost_per_1k_chars=0.0032
    ))
    assert c.spent == pytest.approx(0.0016)


def test_nothing_is_billed_for_empty_text(monkeypatch):
    """Whitespace never reaches the network, so it must not be charged for."""
    from faethon.providers import tts as tts_mod

    class Client:
        spent = 0.0

        def record_usage(self, usage):
            self.spent += usage["cost"]

    c = Client()
    assert list(tts_mod.synthesize_stream(
        c, "   ", model="m", voice="", cost_per_1k_chars=0.0032
    )) == []
    assert c.spent == 0.0


# -- recovery ----------------------------------------------------------------
# A USB microphone can take minutes to appear after a cold boot -- measured at
# 2m45s on this Pi. The retry loop handled that correctly and silently, so the
# journal ended on an error from long before everything started working, and
# the last thing heard in the room was "I can't hear the microphone".


def test_forget_reports_whether_anyone_heard_the_failure(announcer):
    """Announcing that the microphone is back, to someone who never heard it
    go, is noise -- and at startup nothing was usually wrong in the first
    place."""
    assert announcer.forget(NO_MIC) is False, "nothing was announced yet"
    announcer.say(NO_MIC)
    assert announcer.forget(NO_MIC) is True
    assert announcer.forget(NO_MIC) is False, "should only report it once"


def test_forgetting_one_status_leaves_the_others_alone(announcer):
    announcer.say(NO_MIC)
    announcer.say(NO_NETWORK)
    announcer.forget(NO_MIC)
    assert announcer.say(NO_MIC) is True, "should be sayable again"
    assert announcer.say(NO_NETWORK) is False, "unrelated status was cleared"
