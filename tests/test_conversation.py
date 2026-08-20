"""Multi-turn conversation: who has to say the wake word, and when.

Only the first turn of a conversation is opened by the wake word. Every turn
after it is opened by the follow-up window, and silence through that window
ends the conversation.

No audio hardware and no network: `_handle_turn` is stubbed, so what's under
test here is purely the loop around it -- which chime plays when, which
listening budget each turn gets, and whether Faethon's own voice is dropped
from the mic before it listens again.
"""

from __future__ import annotations

import pytest

from faethon.__main__ import ACK_SOUND, CLOSE_SOUND, Faethon
from faethon.config import load_config
from faethon.status import Announcer, SilenceWatch


class FakeStream:
    """Stands in for a capture stream. Records drains against the event log."""

    def __init__(self, log: list) -> None:
        self._log = log

    def __call__(self) -> bytes:
        raise AssertionError("_converse must not read frames itself")

    def drain(self) -> int:
        self._log.append(("drain",))
        return 32000


@pytest.fixture
def rig(monkeypatch):
    """A Faethon whose turns are scripted, with every chime and drain logged."""
    events: list = []

    def fake_play_async(path, device):
        events.append(("chime", path.name))

    def fake_play(path, device):
        events.append(("chime", path.name))

    monkeypatch.setattr("faethon.__main__.playback.play_wav_async", fake_play_async)
    monkeypatch.setattr("faethon.__main__.playback.play_wav", fake_play)

    class Rig:
        def __init__(self) -> None:
            # Built without __init__: a real one loads the wake model and opens
            # an API client, neither of which this loop touches.
            self.faethon = Faethon.__new__(Faethon)
            self.faethon.config = load_config()
            self.faethon._running = True
            self.faethon.announcer = Announcer(
                self.faethon.config.audio.output_device
            )
            self.faethon.silence = SilenceWatch(
                self.faethon.config.audio.frame_ms
            )
            self.faethon._capture_failures = 0
            self.events = events

        def script(self, replies: list[bool]) -> None:
            """Queue what each turn returns, then run the conversation."""
            remaining = list(replies)

            def fake_turn(stream, start_timeout_ms=None):
                events.append(("turn", start_timeout_ms))
                if not remaining:
                    raise AssertionError("more turns than the script allowed")
                return remaining.pop(0)

            self.faethon._handle_turn = fake_turn
            self.faethon._converse(FakeStream(events))

        @property
        def chimes(self) -> list[str]:
            return [name for kind, *rest in self.events
                    if kind == "chime" for name in rest]

        @property
        def windows(self) -> list:
            return [w for kind, *rest in self.events
                    if kind == "turn" for w in rest]

    return Rig()


def test_a_reply_opens_another_turn_without_the_wake_word(rig):
    """Two answered turns then silence: three turns off one wake word."""
    rig.script([True, True, False])
    assert rig.windows == [None, 5000, 5000]


def test_first_turn_uses_the_wake_budget_and_the_rest_the_follow_up(rig):
    """None means the recorder's configured start_timeout_ms.

    The two budgets are separate on purpose: someone who just said the wake
    word is expected to speak, someone who was merely spoken to is not.
    """
    rig.script([True, False])
    first, second = rig.windows
    assert first is None
    assert second == rig.faethon.config.conversation.follow_up_ms


def test_silence_after_a_reply_closes_with_the_falling_chime(rig):
    rig.script([True, False])
    assert rig.chimes == ["ack.wav", "ack.wav", "done.wav"]
    assert ACK_SOUND.name == "ack.wav" and CLOSE_SOUND.name == "done.wav"


def test_a_false_wake_closes_silently(rig):
    """Nobody spoke after the wake word, so there is no conversation to end.

    Playing the closing chime here would answer every misfire with a noise in
    a quiet room -- exactly when a false trigger is most annoying.
    """
    rig.script([False])
    assert rig.chimes == ["ack.wav"]


def test_faethons_own_voice_is_dropped_before_it_listens_again(rig):
    """The drain must land between the reply and the next listening window.

    arecord runs throughout, so without this the follow-up window opens onto a
    buffer holding the reply Faethon just spoke, and it answers itself.
    """
    rig.script([True, True, False])
    assert rig.events == [
        ("chime", "ack.wav"), ("turn", None), ("drain",),
        ("chime", "ack.wav"), ("turn", 5000), ("drain",),
        ("chime", "ack.wav"), ("turn", 5000),
        ("chime", "done.wav"),
    ]


def test_a_silent_turn_is_not_drained(rig):
    """Nothing was said, so nothing of Faethon's is in the buffer to drop."""
    rig.script([False])
    assert ("drain",) not in rig.events


def test_follow_up_off_gives_back_the_old_single_turn_behaviour(rig):
    rig.faethon.config.conversation.follow_up = False
    rig.script([True])
    assert rig.windows == [None]
    assert rig.chimes == ["ack.wav"]
    # Still drained: the stale reply would otherwise be fed to the wake model.
    assert ("drain",) in rig.events


def test_shutdown_mid_conversation_stops_the_loop(rig):
    """SIGTERM during a turn must not buy another follow-up window."""
    def fake_turn(stream, start_timeout_ms=None):
        rig.events.append(("turn", start_timeout_ms))
        rig.faethon._running = False
        return True

    rig.faethon._handle_turn = fake_turn
    rig.faethon._converse(FakeStream(rig.events))
    assert rig.windows == [None]


# -- idle expiry wiring ------------------------------------------------------


def test_faethon_builds_memory_with_the_configured_idle_window(rig):
    """The config value has to reach Memory in seconds, not minutes."""
    from faethon.memory import Memory

    cfg = rig.faethon.config
    mem = Memory(
        cfg.llm.history_turns,
        idle_timeout_sec=cfg.llm.history_idle_minutes * 60,
    )
    assert mem.idle_timeout_sec == cfg.llm.history_idle_minutes * 60
    assert mem.idle_timeout_sec > 60, "minutes were passed where seconds were wanted"


def test_the_wake_loop_checks_for_expiry(rig, monkeypatch):
    """Nothing else ticks while Faethon is idle.

    If this call goes missing the buffer is never wiped on time, and nothing
    fails -- it just quietly keeps yesterday's conversation.
    """
    from contextlib import contextmanager

    from faethon import __main__ as main_mod

    checks = []
    frames = [0]

    class FakeMemory:
        def expire_if_idle(self):
            checks.append(True)
            return False

    class FakeDetector:
        def process(self, frame):
            frames[0] += 1
            if frames[0] >= 3:
                rig.faethon._running = False
            return None

    @contextmanager
    def fake_stream(device, rate, frame_bytes):
        yield lambda: b"\x00\x00" * 1280

    monkeypatch.setattr(main_mod.capture, "open_stream", fake_stream)
    rig.faethon.memory = FakeMemory()
    rig.faethon.detector = FakeDetector()
    rig.faethon._listen_once()

    assert len(checks) == frames[0], "expiry is not checked once per frame"
    assert checks, "the wake loop never checked whether memory had gone stale"


# -- startup greeting --------------------------------------------------------


class FakeAsset:
    """Stands in for the greeting Path. Path itself takes no attributes."""

    name = "greeting.wav"

    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists


def test_the_greeting_plays_before_the_microphone_opens(rig, monkeypatch):
    """Load-bearing ordering, not tidiness.

    The greeting says the assistant's own name, and measured through the
    speaker and microphone it scores 0.5036 on the wake model -- under the 0.7
    wake threshold, but far over the 0.1 barge-in uses. If it played while the
    capture stream was open, Faethon would hear itself say its name on every
    single start.
    """
    from contextlib import contextmanager

    from faethon import __main__ as main_mod

    order = []

    @contextmanager
    def fake_stream(device, rate, frame_bytes):
        order.append("mic opened")
        rig.faethon._running = False
        yield lambda: b"\x00\x00" * 1280

    monkeypatch.setattr(main_mod.capture, "open_stream", fake_stream)
    monkeypatch.setattr(
        main_mod.playback, "play_wav",
        lambda path, device: order.append(f"played {path.name}"),
    )
    monkeypatch.setattr(main_mod, "GREETING_SOUND", FakeAsset(exists=True))

    rig.faethon.run()
    assert order == ["played greeting.wav", "mic opened"], order


def test_the_greeting_can_be_turned_off(rig, monkeypatch):
    from faethon import __main__ as main_mod

    played = []
    monkeypatch.setattr(
        main_mod.playback, "play_wav", lambda path, device: played.append(path)
    )
    rig.faethon.config.conversation.greet_on_start = False
    rig.faethon._greet()
    assert played == []


def test_a_missing_greeting_file_is_survivable(rig, monkeypatch):
    """A fresh clone before make_greeting.py has been run."""
    from faethon import __main__ as main_mod

    played = []
    monkeypatch.setattr(
        main_mod.playback, "play_wav", lambda path, device: played.append(path)
    )
    monkeypatch.setattr(main_mod, "GREETING_SOUND", FakeAsset(exists=False))
    rig.faethon._greet()
    assert played == []


def test_the_turn_line_reports_this_turn_not_just_the_running_total():
    """Printing only the process total made every turn look dearer than the
    last, which reads exactly like context growing without bound. It isn't:
    the buffer evicts at history_turns and the prompt plateaus."""
    import inspect

    from faethon.__main__ import Faethon

    src = inspect.getsource(Faethon._handle_turn)
    assert "spent_before" in src, "no per-turn cost is captured"
    assert "this turn" in src, "the log line does not distinguish the two"


# -- capture recovery --------------------------------------------------------


def test_a_recovered_microphone_says_so(rig, monkeypatch):
    """The failure was announced, so the recovery should be too."""
    from faethon import __main__ as main_mod
    from faethon.status import MIC_BACK, NO_MIC

    played: list[str] = []
    monkeypatch.setattr(
        main_mod.playback, "play_wav", lambda path, device: played.append(path.stem)
    )
    rig.faethon.announcer.say(NO_MIC)          # the mic went away, audibly
    rig.faethon._capture_failures = 55
    rig.faethon._capture_ready()

    assert played == [NO_MIC, MIC_BACK]
    assert rig.faethon._capture_failures == 0


def test_a_clean_start_says_nothing_about_the_microphone(rig, monkeypatch):
    """Nothing failed, so there is nothing to recover from."""
    from faethon import __main__ as main_mod

    played: list[str] = []
    monkeypatch.setattr(
        main_mod.playback, "play_wav", lambda path, device: played.append(path.stem)
    )
    rig.faethon._capture_ready()
    assert played == []


def test_recovery_is_silent_if_the_failure_never_was_announced(rig, monkeypatch):
    """It failed and recovered before the clip could be played -- saying "I can
    hear you again" to someone who noticed nothing is just noise."""
    from faethon import __main__ as main_mod

    played: list[str] = []
    monkeypatch.setattr(
        main_mod.playback, "play_wav", lambda path, device: played.append(path.stem)
    )
    rig.faethon._capture_failures = 2          # failed, but nothing announced
    rig.faethon._capture_ready()
    assert played == []


def test_a_new_stream_restarts_the_silence_clock(rig):
    """The old count was frames from a device that has since gone and come
    back; carrying it over would report a dead microphone that is now fine."""
    dead = b"\x00\x00" * 1280
    for _ in range(20):
        rig.faethon.silence.feed(dead)
    before = rig.faethon.silence._silent_ms
    rig.faethon._capture_ready()
    assert before > 0 and rig.faethon.silence._silent_ms == 0
