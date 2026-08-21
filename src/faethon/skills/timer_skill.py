"""Kitchen timers: the first thing Faethon does without being asked.

Relative only -- "in ten minutes", never "at seven". That is not a limitation
so much as the design: the Pi has no battery-backed clock (`RTC time: n/a`), so
the wall clock is wrong for the first couple of minutes after a boot and then
steps when NTP corrects it. Measured on this Pi: stepping the clock 60 seconds
moved time.time() by 62.3s and time.monotonic() by 2.3s, the real elapsed. So a
running timer counts on the monotonic clock, which cannot jump.

Persistence needs the other clock, because monotonic resets on reboot and means
nothing across one. So each timer carries a wall-clock deadline on disk and a
monotonic deadline in memory, each used for what it is good at. On load the
wall deadline is converted back, and only once systemd says the clock has been
corrected -- otherwise a timer restored during those first two minutes would be
off by however wrong the clock still was.

Three outcomes when restoring, by how stale the deadline is:

    still ahead        resumes, counting the time it was away
    just missed        fires at once, saying it is late
    long past          dropped -- a pasta timer from last night announcing
                       itself at breakfast is worse than losing it
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from .. import clock, state
from .base import Skill

log = logging.getLogger(__name__)

STATE_NAME = "timers"
#: A restored timer this far past its deadline is dropped rather than fired.
STALE_AFTER_SEC = 300.0
MAX_TIMERS = 10

_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "ninety": 90,
}
_UNITS = {"hour": 3600, "hr": 3600, "minute": 60, "min": 60, "second": 1, "sec": 1}
_NUM = r"(?:\d+|" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")"
_UNIT = r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)"

# "half an hour", "quarter of an hour" -- said far too often to leave to the
# generic path, and neither has a number in it to find.
_FRACTIONS = [
    (re.compile(r"\bhalf an? (hour|minute)\b", re.I), {"hour": 1800, "minute": 30}),
    (re.compile(r"\bquarter (?:of an? )?(hour|minute)\b", re.I), {"hour": 900, "minute": 15}),
]
# Both orderings, because people say both: "one and a half hours" and, far
# more often, "an hour and a half". The second is why this is not one pattern.
_AND_A_HALF = re.compile(rf"\b({_NUM})\s+and\s+a\s+half\s+({_UNIT})\b", re.I)
_UNIT_THEN_HALF = re.compile(rf"\b({_NUM})\s+({_UNIT})\s+and\s+a\s+half\b", re.I)
_PAIR = re.compile(rf"\b({_NUM})\s*({_UNIT})\b", re.I)


def _value(token: str) -> int:
    return int(token) if token.isdigit() else _WORDS[token.lower()]


def _unit_seconds(token: str) -> int:
    t = token.lower().rstrip("s").rstrip(".")
    for stem, secs in _UNITS.items():
        if t.startswith(stem):
            return secs
    return 0


def parse_duration(text: str) -> int | None:
    """Seconds, or None if there is no duration in `text`.

    Sums every quantity it finds, so "two hours thirty minutes" works without
    a rule for that ordering specifically.
    """
    total = 0
    found = False
    rest = text

    for pattern, table in _FRACTIONS:
        for m in pattern.finditer(rest):
            total += table[m.group(1).lower()]
            found = True
        rest = pattern.sub(" ", rest)

    for pattern in (_AND_A_HALF, _UNIT_THEN_HALF):
        for m in pattern.finditer(rest):
            unit = _unit_seconds(m.group(2))
            total += _value(m.group(1)) * unit + unit // 2
            found = True
        rest = pattern.sub(" ", rest)

    for m in _PAIR.finditer(rest):
        total += _value(m.group(1)) * _unit_seconds(m.group(2))
        found = True

    return total if found and total > 0 else None


def say_duration(seconds: int) -> str:
    """Spoken form: "ten minutes", "an hour and a half", "ninety seconds"."""
    parts = []
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    if secs and not hours:
        parts.append(f"{secs} second" + ("s" if secs != 1 else ""))
    if not parts:
        return "no time at all"
    if len(parts) == 1:
        return parts[0]
    return " and ".join([", ".join(parts[:-1]), parts[-1]]) if len(parts) > 2 \
        else " and ".join(parts)


@dataclass
class Timer:
    name: str            # "" when unnamed
    seconds: int         # what was asked for, for the spoken confirmation
    due_wall: float      # epoch, what goes on disk and survives a restart
    due_mono: float      # what it actually counts on, immune to clock steps

    @property
    def spoken(self) -> str:
        return f"the {self.name} timer" if self.name else "your timer"


class TimerSkill(Skill):
    name = "set_timer"
    tag = "utility"
    # You asked for it. An eight-hour timer set at seven in the evening should
    # go off at three in the morning, which is the whole point of setting it.
    announce_urgency = "requested"
    description = (
        "Set, check or cancel a countdown timer that Faethon is running. "
        "Timers can be named, several can run at once, and Faethon announces "
        "them when they come due. Durations are relative -- 'in ten minutes', "
        "not 'at seven o'clock'. "
        "ONLY for Faethon's own timers. Not for how long until a date or an "
        "event -- answer those yourself from the current date."
    )

    patterns = [
        r"\b(?:set|start)\s+(?:an?\s+)?(?P<name>[\w ]{0,20}?)\s*timer\s+"
        r"(?:for\s+)?(?P<duration>.+)",
        r"\btimer\s+for\s+(?P<duration>.+)",
        r"\b(?:remind me|wake me)\s+in\s+(?P<duration>.+)",
        r"\b(?P<check>how) (?:long|much time)(?:'s| is)?\s*(?:left|remaining)"
        r"(?:\s+on\s+(?:the\s+)?(?P<query>[\w ]{0,20}?)\s*timer)?\b",
        r"\b(?P<cancel>cancel|stop|clear)\s+(?:the\s+|my\s+)?"
        r"(?P<cancel_name>[\w ]{0,20}?)\s*timer\b",
        r"\b(?P<list>what|which) timers?\b",
    ]

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "check", "cancel"],
                "description": "Defaults to set when a duration is given.",
            },
            "seconds": {"type": "integer", "description": "Duration for a new timer."},
            "name": {"type": "string", "description": "Optional timer name."},
        },
        "required": [],
    }

    def __init__(self) -> None:
        super().__init__()
        self._timers: list[Timer] = []
        self._loaded = False

    # -- persistence ------------------------------------------------------

    @staticmethod
    def _clock_trusted() -> bool:
        return clock.is_synced()

    def _ensure_loaded(self) -> None:
        """Restore from disk once, and only once the clock can be believed."""
        if self._loaded or not self._clock_trusted():
            return
        self._loaded = True
        now_wall, now_mono = time.time(), time.monotonic()
        restored, late, dropped = [], [], 0
        for row in state.load(STATE_NAME, []):
            try:
                t = Timer(str(row["name"]), int(row["seconds"]),
                          float(row["due_wall"]), 0.0)
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue
            remaining = t.due_wall - now_wall
            if remaining > 0:
                t.due_mono = now_mono + remaining
                restored.append(t)
            elif -remaining <= STALE_AFTER_SEC:
                t.due_mono = now_mono          # fire immediately, and say so
                late.append(t)
            else:
                dropped += 1
        self._timers = restored + late
        if self._timers or dropped:
            log.info(
                "timers restored: %d running, %d overdue, %d dropped as stale",
                len(restored), len(late), dropped,
            )

    def _persist(self) -> None:
        state.save(STATE_NAME, [
            {"name": t.name, "seconds": t.seconds, "due_wall": t.due_wall}
            for t in self._timers
        ])

    # -- the unprompted half ----------------------------------------------

    def tick(self) -> str | None:
        self._ensure_loaded()
        if not self._timers:
            return None
        now = time.monotonic()
        for t in list(self._timers):
            if now >= t.due_mono:
                self._timers.remove(t)
                self._persist()
                overdue = time.time() - t.due_wall
                if overdue > 60:
                    return (
                        f"{t.spoken.capitalize()} finished "
                        f"{say_duration(int(overdue))} ago, while I was away."
                    )
                return f"{t.spoken.capitalize()} is up."
        return None

    # -- the asked-for half -----------------------------------------------

    def run(self, **params: object) -> str:
        self._ensure_loaded()
        action = str(params.get("action") or "").lower()

        if "cancel" in params or action == "cancel":
            return self._cancel(str(params.get("cancel_name") or params.get("name") or ""))
        if {"query", "check", "list"} & params.keys() or action == "check":
            return self._report(str(params.get("query") or params.get("name") or ""))

        seconds = params.get("seconds")
        if seconds is None:
            raw = str(params.get("duration") or "")
            seconds = parse_duration(raw)
            if seconds is None:
                return "I didn't catch how long for."
        seconds = int(seconds)
        if seconds <= 0:
            return "That's not a length of time I can count."
        if len(self._timers) >= MAX_TIMERS:
            return f"I'm already running {MAX_TIMERS} timers. Cancel one first."

        name = self._clean(str(params.get("name") or ""))
        self._timers = [t for t in self._timers if t.name != name or not name]
        t = Timer(name, seconds, time.time() + seconds, time.monotonic() + seconds)
        self._timers.append(t)
        self._persist()
        log.info("timer set: %r for %ds", name or "(unnamed)", seconds)
        label = f" {name}" if name else ""
        return f"Right, a{label} timer for {say_duration(seconds)}."

    @staticmethod
    def _clean(name: str) -> str:
        name = re.sub(r"\b(a|an|the|new|another|my)\b", " ", name, flags=re.I)
        return " ".join(name.split()).lower()

    def _find(self, name: str) -> Timer | None:
        name = self._clean(name)
        if not name:
            return self._timers[0] if len(self._timers) == 1 else None
        for t in self._timers:
            if t.name == name:
                return t
        return None

    def _cancel(self, name: str) -> str:
        if not self._timers:
            return "There aren't any timers running."
        target = self._find(name)
        if target is None:
            if not self._clean(name) and len(self._timers) > 1:
                return f"There are {len(self._timers)} timers. Which one?"
            return f"I don't have a {self._clean(name)} timer."
        self._timers.remove(target)
        self._persist()
        return f"Cancelled {target.spoken}."

    def _report(self, name: str) -> str:
        if not self._timers:
            return "There aren't any timers running."
        now = time.monotonic()
        target = self._find(name)
        if target is not None:
            left = max(0, int(target.due_mono - now))
            return f"{say_duration(left).capitalize()} left on {target.spoken}."
        if self._clean(name):
            return f"I don't have a {self._clean(name)} timer."
        parts = [
            f"{say_duration(max(0, int(t.due_mono - now)))} on {t.spoken}"
            for t in self._timers
        ]
        if len(parts) == 1:
            return f"There's {parts[0]}."
        listed = " and ".join([", ".join(parts[:-1]), parts[-1]])
        return f"There's {listed}."


SKILL = TimerSkill()
