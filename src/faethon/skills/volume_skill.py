"""Volume control, on a spoken 0-10 scale.

    0   muted
    1   quietest still audible
    10  the mixer's maximum

The obvious implementation -- level 5 means "50%" to amixer -- does not work,
and fails in a way you would not catch by reading the code. The Pi's PCM
control is scaled in dB, and amixer's percentage is a linear position in that
range rather than a loudness. Measured on this Pi, playing a tone and recording
the result:

    amixer   dB      loudness at the mic
      96%     0        103
      87%   -10         24
      77%   -20          8
      68%   -30          3
      49%   -50          2   <- indistinguishable from silence

So "50%" is inaudible, and the whole useful range lives in the top third of the
percentage scale. The levels here are spaced evenly in dB instead, which is
roughly how loudness is perceived, so each step sounds like the same size step.

USABLE_RANGE_DB is the measured part: about 34 dB below maximum, the tone had
fallen to the microphone's noise floor and further reduction changed nothing
audible. Spending levels below that would mean several settings that all sound
like silence. Lower it if level 1 is still too loud in a quiet room; raise it
if the bottom of the scale is unusably quiet.
"""

from __future__ import annotations

import logging
import re
import subprocess

from .base import Skill

log = logging.getLogger(__name__)

MIN_LEVEL, MAX_LEVEL = 0, 10
#: Spoken as a percentage: one level is one tenth of the dial, so 7 -> "70%".
#: The literal "%" is deliberate -- Fish Audio S1 pronounces it, measured at
#: 2.04s of audio against 2.09s for the word spelled out and 1.72s with the
#: sign removed, so it is being said rather than skipped.
PERCENT_PER_LEVEL = 10
#: Levels 1..MAX_LEVEL span this many dB below the control's maximum.
USABLE_RANGE_DB = 34.0

_DB = re.compile(r"\[(-?\d+(?:\.\d+)?)dB\]")
_SWITCH = re.compile(r"\[(on|off)\]")
_LIMITS = re.compile(r"Limits: Playback (-?\d+) - (-?\d+)")


def _amixer(*args: str, card: str | None = None) -> str:
    cmd = ["amixer"]
    if card:
        cmd += ["-c", card]
    cmd += list(args)
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"amixer failed: {' '.join(cmd)}")
    return out.stdout


def _percent(level: int) -> str:
    return f"{level * PERCENT_PER_LEVEL}%"


def _spoken(level: int) -> str:
    """What Faethon says after a change, and when asked.

    Muting says both, because "0%" alone leaves it ambiguous whether the
    speaker is silent or merely turned all the way down.
    """
    if level <= MIN_LEVEL:
        return f"Volume is set to {_percent(MIN_LEVEL)}, muted."
    return f"Volume is set to {_percent(level)}."


def _find_control() -> tuple[str, str] | None:
    """First card with a playback volume control, as (card, control).

    Discovered rather than hardcoded so this works on a Pi whose audio sits on
    a different card -- the index moves when USB devices are re-plugged, which
    is why the rest of Faethon addresses ALSA by name too.
    """
    try:
        listing = subprocess.run(["aplay", "-l"], capture_output=True, text=True,
                                 timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    for name in re.findall(r"^card \d+: (\S+)", listing, re.MULTILINE):
        try:
            controls = _amixer("scontrols", card=name)
        except (RuntimeError, OSError, subprocess.SubprocessError):
            continue
        for control in re.findall(r"Simple mixer control '([^']+)'", controls):
            try:
                detail = _amixer("sget", control, card=name)
            except (RuntimeError, OSError, subprocess.SubprocessError):
                continue
            if "pvolume" in detail:
                return name, control
    return None


class VolumeSkill(Skill):
    name = "set_volume"
    tag = "utility"
    description = (
        "Change the speaker volume, on a scale where 0 is muted and 10 is "
        "maximum. Use for making Faethon louder or quieter, muting, unmuting, "
        "or setting a specific level."
    )

    patterns = [
        r"\b(?:turn (?:the )?)?volume (?P<action>up|down)\b",
        r"\bturn (?:it|the (?:volume|sound|music)) (?P<action>up|down)\b",
        r"\bset (?:the )?volume to (?P<level>\d+)\b",
        r"\bvolume (?:to |at )?(?P<level>\d+)\b",
        r"\b(?P<action>louder)\b",
        r"\b(?P<action>quieter|softer)\b",
        r"\b(?P<action>unmute)\b",
        r"\b(?P<action>mute)\b",
        r"\bwhat(?:'s|s| is) the (?P<action>volume)\b",
    ]

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["up", "down", "mute", "unmute", "volume"],
                "description": (
                    "up/down step by one level. 'volume' just reports the "
                    "current level without changing it."
                ),
            },
            "level": {
                "type": "integer",
                "minimum": MIN_LEVEL,
                "maximum": MAX_LEVEL,
                "description": "Set an absolute level, 0 (muted) to 10 (maximum).",
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        super().__init__()
        self._control: tuple[str, str] | None = None
        self._looked = False

    # -- mixer ------------------------------------------------------------

    @property
    def control(self) -> tuple[str, str] | None:
        if not self._looked:
            self._looked = True
            self._control = _find_control()
            if self._control:
                log.info("volume: using %s / %s", *self._control)
            else:
                log.warning("volume: no ALSA playback control found")
        return self._control

    @property
    def available(self) -> bool:
        return self.control is not None

    @property
    def unavailable_reason(self) -> str:
        return "I can't find an audio device to set the volume on."

    def _state(self) -> tuple[float, float, bool]:
        """Current dB, the control's maximum dB, and whether it's muted."""
        card, control = self.control
        out = _amixer("sget", control, card=card)

        db_values = [float(v) for v in _DB.findall(out)]
        current = db_values[0] if db_values else 0.0

        limits = _LIMITS.search(out)
        # Limits are in hundredths of a dB.
        max_db = float(limits.group(2)) / 100 if limits else 0.0

        switch = _SWITCH.search(out)
        muted = bool(switch) and switch.group(1) == "off"
        return current, max_db, muted

    def _level_of(self, db: float, max_db: float, muted: bool) -> int:
        if muted:
            return 0
        step = USABLE_RANGE_DB / (MAX_LEVEL - 1)
        floor = max_db - USABLE_RANGE_DB
        level = round((db - floor) / step) + 1
        return max(1, min(MAX_LEVEL, level))

    def _apply(self, level: int) -> None:
        card, control = self.control
        _, max_db, _ = self._state()

        if level <= MIN_LEVEL:
            _amixer("-q", "sset", control, "mute", card=card)
            return

        step = USABLE_RANGE_DB / (MAX_LEVEL - 1)
        db = max_db - USABLE_RANGE_DB + (level - 1) * step
        # "--" or amixer reads a negative dB value as a command-line flag.
        _amixer("-q", "sset", control, "--", f"{db:.2f}dB", card=card)
        _amixer("-q", "sset", control, "unmute", card=card)

    # -- skill ------------------------------------------------------------

    def run(self, **params: object) -> str:
        if not self.available:
            return self.unavailable_reason

        try:
            db, max_db, muted = self._state()
            current = self._level_of(db, max_db, muted)

            raw_level = params.get("level")
            action = str(params.get("action") or "").lower()

            if action == "volume" and raw_level is None:
                return self._describe(current)

            if raw_level is not None:
                try:
                    target = int(raw_level)
                except (TypeError, ValueError):
                    return "I didn't catch what level you wanted."
                target = max(MIN_LEVEL, min(MAX_LEVEL, target))
            elif action in ("up", "louder"):
                target = current + 1
            elif action in ("down", "quieter", "softer"):
                target = current - 1
            elif action == "mute":
                target = MIN_LEVEL
            elif action == "unmute":
                # Unmuting into silence would look like it failed.
                target = current if current > MIN_LEVEL else 5
            else:
                return self._describe(current)

            if target > MAX_LEVEL:
                return f"Volume is already at {_percent(MAX_LEVEL)}."
            if target < MIN_LEVEL:
                return f"Volume is already at {_percent(MIN_LEVEL)}, muted."

            self._apply(target)
        except (RuntimeError, OSError, subprocess.SubprocessError, IndexError) as e:
            log.error("volume: %s", e)
            return "Sorry, I couldn't change the volume."

        return _spoken(target)

    def _describe(self, level: int) -> str:
        return _spoken(level)


SKILL = VolumeSkill()
