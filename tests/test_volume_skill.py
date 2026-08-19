"""Volume control on a 0-10 scale.

The thing worth guarding here is the mapping. The obvious implementation --
level 5 means "50%" to amixer -- is wrong in a way that reads fine: the Pi's
PCM control is scaled in dB and amixer's percentage is a linear position in
that range, not a loudness. Measured on the hardware:

    amixer 96% =   0 dB = rms 103
    amixer 68% = -30 dB = rms   3
    amixer 49% = -50 dB = rms   2   <- silence

So "50%" is inaudible. These pin the dB mapping and, in particular, that a
percentage is never sent at all.

No audio hardware: amixer is stubbed, and the fake keeps enough state to
answer the queries the skill makes of it.
"""

from __future__ import annotations

import pytest

from faethon.skills import volume_skill
from faethon.skills.volume_skill import MAX_LEVEL, MIN_LEVEL, VolumeSkill


class FakeMixer:
    """Enough of amixer to drive the skill, plus a log of what was asked."""

    def __init__(self, db: float = -14.89, max_db: float = 4.0, muted: bool = False):
        self.db = db
        self.max_db = max_db
        self.muted = muted
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, *args: str, card: str | None = None) -> str:
        self.commands.append(args)
        if args[0] == "scontrols":
            return "Simple mixer control 'PCM',0\n"
        if "mute" in args:
            self.muted = args[args.index("mute")] == "mute" and "unmute" not in args
        if "unmute" in args:
            self.muted = False
        for a in args:
            if a.endswith("dB"):
                self.db = float(a[:-2])
        state = "off" if self.muted else "on"
        return (
            "Simple mixer control 'PCM',0\n"
            "  Capabilities: pvolume pswitch\n"
            f"  Limits: Playback -10239 - {int(self.max_db * 100)}\n"
            f"  Mono: Playback 0 [96%] [{self.db:.2f}dB] [{state}]\n"
        )

    @property
    def volume_sets(self) -> list[str]:
        """Every value passed to a set, in order."""
        return [a for cmd in self.commands for a in cmd
                if a.endswith("dB") or a.endswith("%")]


@pytest.fixture
def skill(monkeypatch):
    mixer = FakeMixer()
    monkeypatch.setattr(volume_skill, "_amixer", mixer)
    s = VolumeSkill()
    s._control = ("Headphones", "PCM")
    s._looked = True
    s.mixer = mixer
    return s


def say(skill, text):
    """Route text the way the regex path would, then run the skill."""
    params = skill.match(text)
    assert params is not None, f"no pattern matched {text!r}"
    return skill.run(**params)


# -- the mapping -------------------------------------------------------------


def test_it_never_sends_a_percentage(skill):
    """A percentage is a position in a -102..+4 dB range, not a loudness.

    Sending "50%" would set roughly -49 dB, which measured at the microphone's
    noise floor. If this fails, the scale is silent in its lower half and
    nothing else in this file will tell you why.
    """
    for level in range(1, MAX_LEVEL + 1):
        skill._apply(level)
    assert skill.mixer.volume_sets, "nothing was set"
    assert all(v.endswith("dB") for v in skill.mixer.volume_sets), (
        f"a percentage reached amixer: {skill.mixer.volume_sets}"
    )


def test_every_level_reads_back_as_itself(skill):
    """Round trip: setting level N and asking must give N again.

    Stepping up and down repeatedly is the normal use, and each step reads the
    current level first -- so a mapping that rounds badly would drift or stick.
    """
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        skill._apply(level)
        db, max_db, muted = skill._state()
        assert skill._level_of(db, max_db, muted) == level


def test_the_steps_are_evenly_spaced_in_db(skill):
    """Even in dB is roughly even in perceived loudness.

    Measured on the hardware at 1.53-1.55x amplitude per step across the whole
    scale, which is what makes one "volume up" feel like the same amount
    wherever you are on the dial.
    """
    seen = []
    for level in range(1, MAX_LEVEL + 1):
        skill._apply(level)
        seen.append(skill._state()[0])
    gaps = [b - a for a, b in zip(seen, seen[1:])]
    step = volume_skill.USABLE_RANGE_DB / (MAX_LEVEL - 1)
    # abs=0.02 absorbs the two decimals amixer is given; anything larger would
    # be a real step of a different size.
    assert gaps == pytest.approx([step] * len(gaps), abs=0.02), f"uneven: {gaps}"


def test_the_top_of_the_scale_is_the_mixer_maximum(skill):
    skill._apply(MAX_LEVEL)
    db, max_db, _ = skill._state()
    assert db == pytest.approx(max_db)


def test_zero_mutes_rather_than_setting_a_quiet_level(skill):
    """0 is defined as muted, and a very low dB is not the same thing.

    -102 dB is still technically playing, so the switch is what gives silence.
    """
    skill._apply(MIN_LEVEL)
    assert skill._state()[2] is True


# -- the spoken commands -----------------------------------------------------


def test_volume_up_ascends_by_one(skill):
    skill._apply(5)
    assert say(skill, "volume up") == "Volume is set to 60%."
    assert say(skill, "volume up") == "Volume is set to 70%."


def test_volume_down_descends_by_one(skill):
    skill._apply(5)
    assert say(skill, "volume down") == "Volume is set to 40%."
    assert say(skill, "volume down") == "Volume is set to 30%."


def test_the_level_is_spoken_as_a_percentage_of_the_dial(skill):
    """One level is one tenth of the dial, so 7 is announced as 70%.

    The raw level would be ambiguous aloud -- "volume seven" gives no sense of
    how much further it goes -- and the percentage matches how the steps were
    specified.
    """
    for level, expected in [(1, "10%"), (5, "50%"), (10, "100%")]:
        skill._apply(level)
        assert expected in say(skill, "what's the volume")


@pytest.mark.parametrize("phrase", [
    "volume up", "turn the volume up", "turn it up", "louder",
])
def test_phrasings_that_mean_louder(skill, phrase):
    skill._apply(5)
    assert say(skill, phrase) == "Volume is set to 60%."


@pytest.mark.parametrize("phrase", [
    "volume down", "turn the volume down", "turn it down", "quieter",
])
def test_phrasings_that_mean_quieter(skill, phrase):
    skill._apply(5)
    assert say(skill, phrase) == "Volume is set to 40%."


def test_it_stops_at_the_top(skill):
    skill._apply(MAX_LEVEL)
    assert say(skill, "volume up") == "Volume is already at 100%."
    assert skill._state()[0] == pytest.approx(4.0), "level should not have moved"


def test_it_stops_at_the_bottom(skill):
    skill._apply(MIN_LEVEL)
    assert say(skill, "volume down") == "Volume is already at 0%, muted."


def test_muting_says_so_as_well_as_the_number(skill):
    """0% alone leaves it unclear whether the speaker is silent or just low."""
    assert say(skill, "mute") == "Volume is set to 0%, muted."


def test_stepping_up_from_muted_unmutes(skill):
    """Otherwise "volume up" on a muted speaker changes a number and stays silent."""
    skill._apply(MIN_LEVEL)
    assert say(skill, "volume up") == "Volume is set to 10%."
    assert skill._state()[2] is False


def test_unmuting_does_not_land_on_silence(skill):
    """Unmuting back to 0 would look exactly like the command failing."""
    skill._apply(MIN_LEVEL)
    say(skill, "unmute")
    assert skill._state()[2] is False
    db, max_db, muted = skill._state()
    assert skill._level_of(db, max_db, muted) > MIN_LEVEL


def test_an_absolute_level_can_be_set(skill):
    assert say(skill, "set the volume to 8") == "Volume is set to 80%."
    assert say(skill, "volume 3") == "Volume is set to 30%."


def test_asking_does_not_change_anything(skill):
    skill._apply(6)
    before = skill._state()[0]
    assert say(skill, "what's the volume") == "Volume is set to 60%."
    assert skill._state()[0] == before


def test_out_of_range_is_clamped_not_rejected(skill):
    """The model can pass anything through the tool-calling path."""
    assert skill.run(level=99) == "Volume is set to 100%."
    assert skill.run(level=-4) == "Volume is set to 0%, muted."


def test_a_missing_mixer_explains_itself(monkeypatch):
    """An unavailable skill is hidden from the LLM and speaks up on the regex
    path, rather than failing silently."""
    monkeypatch.setattr(volume_skill, "_find_control", lambda: None)
    s = VolumeSkill()
    assert not s.available
    assert "can't find an audio device" in s.run(action="up")


def test_a_broken_mixer_does_not_crash_the_turn(skill, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("amixer: no such control")

    monkeypatch.setattr(volume_skill, "_amixer", boom)
    assert "couldn't change the volume" in skill.run(action="up")


# -- percentages -------------------------------------------------------------
# Faethon announces "Volume is set to 30%", so people say percentages back.
# Read as a level, "30" clamps to 10 and every request lands on maximum --
# which is exactly what happened in use. Whisper transcribes the sign
# literally, so these are the real transcripts from that session.


@pytest.mark.parametrize("heard,expected", [
    ("Volume 20%.", "Volume is set to 20%."),
    ("Set the volume to 20%.", "Volume is set to 20%."),
    ("Set the volume to 30%.", "Volume is set to 30%."),
    ("set the volume to 40%", "Volume is set to 40%."),
    ("set the volume to 100%", "Volume is set to 100%."),
])
def test_a_spoken_percentage_is_not_read_as_a_level(skill, heard, expected):
    assert say(skill, heard) == expected


def test_the_word_percent_works_too(skill):
    """Whisper writes the sign, but not always."""
    assert say(skill, "set the volume to 60 percent") == "Volume is set to 60%."


def test_a_bare_number_too_big_to_be_a_level_is_a_percentage(skill):
    """Nobody means level 70, and clamping it to maximum is the bug this fixes."""
    assert say(skill, "set the volume to 70") == "Volume is set to 70%."


def test_the_two_scales_agree_where_they_overlap(skill):
    """"7" and "70" are the same request said two ways."""
    assert say(skill, "set the volume to 7") == say(skill, "set the volume to 70")


def test_a_small_percentage_does_not_silence_it(skill):
    """3% rounds to level 0 arithmetically, but the user asked for quiet, not
    off. Only an explicit zero mutes."""
    assert say(skill, "set the volume to 3%") == "Volume is set to 10%."
    assert skill._state()[2] is False
    assert say(skill, "set the volume to 0%") == "Volume is set to 0%, muted."
    assert skill._state()[2] is True


def test_the_model_can_still_pass_a_plain_level(skill):
    """The tool-calling path sends 0-10, with no unit."""
    assert skill.run(level=4) == "Volume is set to 40%."
