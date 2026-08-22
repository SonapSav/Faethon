"""Internet radio, running on the other Pi.

RadioHost is a plain HTTP/JSON API on the LAN with no auth -- the same trust
model as the phone app that normally drives it. The integration is four
endpoints and about twenty lines. All the design is in one question: how does a
spoken request become a station id.

**By frequency, not by name.** Measured, because the obvious approach fails
badly here. `stt.language` is pinned to "en", and Whisper renders Greek station
names as an unpredictable mix of transliteration and translation:

    Ρυθμός    -> "rithmos"     transliterated, recoverable
    Μέντα     -> "Menta"       fine
    Λάμψη     -> "lampsy"      mangled
    Δρόμος    -> "the road"    TRANSLATED
    Μελωδία   -> "melody"      TRANSLATED

No string comparison recovers "the road" into "Δρόμος", and no single rule
covers a set that is sometimes transliterated and sometimes translated.

Frequencies have none of that problem. All nineteen stations carry one, they
are unique, and digits are language-neutral -- "Play ninety four point nine"
arrives as `94.9` and "Βάλε το ενενήντα τέσσερα κόμμα εννιά" arrives as `94,9`.
Both parse. It is also how people refer to radio in the first place.

Names still reach the model as a fallback, which handles the Latin-script
twelve and can resolve "the road" to Δρόμος using priors no matcher has. That
is the same split as the weather skill: the regex path never captures a name,
and named things go through the model.

No ducking. The obvious worry with a radio in the same room is that it feeds
the microphone continuously and re-opens the follow-up window on every turn --
which is exactly what the conversation cap exists to survive. Measured
instead of assumed: 45s of room audio with the radio playing scored a maximum
of 0.0026 against a wake threshold of 0.7, and the mic sees the radio at about
-88 dBFS where speech arrives at -20 to -30. Verified that the measurement can
see the radio at all by sweeping its volume (39/60/80 -> -87.9/-78.1/-69.0
dBFS), so the null result means "far too quiet to matter", not "nothing was
playing". Different placement would need re-measuring; the same 45s recording
answers it.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import time

import httpx

from .. import disclosure, levels, state
from ..config import load_config
from .base import Skill

log = logging.getLogger(__name__)

#: A frequency inside a station name: "94,9", "102,2", "88".
_IN_NAME = re.compile(r"(\d{2,3})(?:[.,](\d))?")
#: How long an unreachable host is remembered, so a dead Pi is not retried on
#: every single turn while still recovering without a restart.
UNAVAILABLE_FOR = 60.0
#: How many frequencies to read out before summarising instead. Nineteen takes
#: about half a minute to say, which is already at the edge of what anyone
#: wants read back; a list twice that long stops being an answer and becomes a
#: recital you cannot interrupt without the wake word.
MAX_SPOKEN = 20
#: Remembers a duck across a crash. Without it, a process that dies mid-turn
#: leaves the radio at 15 until somebody notices and reaches for their phone.
DUCK_STATE = "radio_duck"


def parse_frequency(text: str) -> float | None:
    """Spoken frequency to a number, tolerating both decimal marks.

    English STT gives "94.9" and Greek gives "94,9" -- the same station, so
    both marks are accepted. A bare "94 9" with no mark at all parses as 94,
    deliberately: guessing that a loose digit is a decimal would turn "play 94
    9" into a different station than "play 94", and the FM-band check below is
    the only thing standing between this and every other number in a sentence.
    """
    m = re.search(r"\b(\d{2,3})\s*(?:[.,]\s*(\d))?\b", text)
    if not m:
        return None
    whole = int(m.group(1))
    if not 80 <= whole <= 110:          # FM band; anything else is another number
        return None
    return whole + (int(m.group(2)) / 10 if m.group(2) else 0.0)


def frequency_of(name: str) -> float | None:
    m = _IN_NAME.search(name)
    if not m:
        return None
    whole = int(m.group(1))
    if not 80 <= whole <= 110:
        return None
    return whole + (int(m.group(2)) / 10 if m.group(2) else 0.0)


def say_frequency(freq: float) -> str:
    """Spoken back the way it is said, not the way it is stored."""
    return f"{freq:.1f}".rstrip("0").rstrip(".")


class RadioSkill(Skill):
    name = "control_radio"
    tag = "utility"
    description = (
        "Control the internet radio on the other Raspberry Pi: start it, stop "
        "it, change station, change ITS volume, say what is playing, or list "
        "which stations exist. "
        "Stations are chosen by FM frequency, e.g. 94.9. Give `station` only "
        "if the user named one rather than giving a frequency; the station "
        "list is on the device, so a name you do not recognise is still worth "
        "passing through."
    )

    # Every pattern needs an explicit radio marker -- the word "radio", a
    # frequency, or "station". This skill sorts before set_volume and
    # set_timer in the registry, so a loose pattern here would silently steal
    # "turn it up" and "stop" from skills that already own them.
    _END = r"(?:\s+(?:please|for me))*[^\w]*$"

    patterns = [
        r"\b(?:play|put on|switch to|tune (?:in )?to|go to)\s+"
        r"(?P<freq>\d{2,3}(?:\s*[.,]\s*\d)?)\b",
        r"\b(?:play|put on|start|turn on)\s+(?:the\s+)?(?P<resume>radio)\b",
        # Word order the other way round: "put the radio on".
        r"\bput\s+(?:the\s+)?(?P<resume>radio)\s+(?:back\s+)?on\b",
        r"\b(?:stop|turn off|shut off|kill)\s+(?:the\s+)?(?P<stop>radio)\b",
        rf"\b(?P<step>next|previous|last)\s+(?:radio\s+)?station{_END}",
        r"\bturn\s+(?:the\s+)?radio\s+(?P<vol>up|down)\b",
        r"\bturn\s+(?P<vol>up|down)\s+the\s+radio\b",
        # The unit group is what stops "radio volume 50" becoming level 50
        # clamped to maximum. This skill announces percentages, so people say
        # percentages back -- the trap volume_skill hit, which is why both now
        # share faethon.levels rather than each having their own scale.
        r"\bradio\s+volume\s+(?:to\s+)?(?P<level>\d{1,3})\s*(?P<unit>%|percent)?",
        r"\bset\s+(?:the\s+)?radio\s+volume\s+(?:to\s+)?(?P<level>\d{1,3})"
        r"\s*(?P<unit>%|percent)?",
        rf"\bwhat(?:'s|s| is)\s+(?:playing|on the radio){_END}",
        rf"\bwhat\s+station\s+is\s+(?:this|it|on){_END}",
        # Listing is a different question from "what's playing", and the two
        # phrasings are close enough that the distinction has to be explicit.
        # The empty group is the marker: an empty match dict is
        # indistinguishable from "what's playing", which is the next pattern
        # along and answers a different question entirely.
        rf"\bwhat\s+stations?\s+(?:do you have|have you got|are (?:there|available)"
        rf"|can i (?:pick|choose|select|have))(?P<list>){_END}",
        rf"\b(?:list|name|tell me)\s+(?:the\s+|your\s+|all (?:the|your)\s+)?"
        rf"(?:radio\s+)?stations(?P<list>){_END}",
        rf"\bwhich\s+stations?\s+(?:do you have|are (?:there|available)"
        rf"|can i (?:pick|choose|select))(?P<list>){_END}",
        rf"\bwhat(?:'s|s| is)\s+(?:on\s+)?(?:the\s+)?(?P<list>station list){_END}",
    ]

    parameters = {
        "type": "object",
        "properties": {
            "frequency": {
                "type": "number",
                "description": "FM frequency, e.g. 94.9. Preferred over a name.",
            },
            "station": {
                "type": "string",
                "description": "Station name, only if no frequency was given.",
            },
            "action": {
                "type": "string",
                "enum": ["play", "stop", "next", "previous", "status", "list",
                         "volume_up", "volume_down"],
                "description": "Defaults to play when a station is given.",
            },
            "level": {
                "type": "integer",
                "minimum": levels.MIN_LEVEL,
                "maximum": levels.MAX_LEVEL,
                "description": (
                    "Radio volume as a level from 0 to 10, the same scale as "
                    "Faethon's own volume. 5 is half."
                ),
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        super().__init__()
        self._config = None
        self._stations: tuple[float, list[dict]] | None = None
        self._down_until = 0.0
        self._duck_thread: threading.Thread | None = None
        self._duck_lock = threading.Lock()

    @property
    def config(self):
        if self._config is None:
            self._config = load_config().radio
        return self._config

    @property
    def available(self) -> bool:
        """False while the other Pi is known to be down.

        Remembered for a minute rather than probed per turn: a switched-off Pi
        would otherwise cost a timeout on every question, and the timeout is
        inside a spoken turn.
        """
        return time.monotonic() >= self._down_until

    @property
    def unavailable_reason(self) -> str:
        return "I can't reach the radio right now."

    # -- the network ------------------------------------------------------

    def _call(self, method: str, path: str, **kw) -> dict | None:
        url = f"{self.config.base_url}{path}"
        # LAN, not cloud. It left this machine, which is why it is recorded,
        # but it did not leave the house -- a different thing to disclose than
        # the sound of the room going to OpenRouter.
        disclosure.LEDGER.record(httpx.URL(url).host, path, disclosure.LAN)
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                r = client.request(method, url, **kw)
                r.raise_for_status()
                return r.json() if r.content else {}
        except httpx.HTTPError as e:
            log.warning("radio %s %s failed: %s", method, path, e)
            self._down_until = time.monotonic() + UNAVAILABLE_FOR
            return None

    def stations(self, fresh: bool = False) -> list[dict]:
        if not fresh and self._stations:
            age = time.monotonic() - self._stations[0]
            if age < self.config.cache_seconds:
                return self._stations[1]
        data = self._call("GET", "/api/stations")
        if data is None:
            return self._stations[1] if self._stations else []
        rows = data if isinstance(data, list) else data.get("stations", [])
        self._stations = (time.monotonic(), rows)
        return rows

    def status(self) -> dict | None:
        return self._call("GET", "/api/player/status")

    # -- picking a station ------------------------------------------------

    def by_frequency(self, freq: float) -> dict | None:
        for s in self.stations():
            if frequency_of(s.get("name", "")) == freq:
                return s
        return None

    def by_name(self, text: str) -> dict | None:
        """Loose match, for the Latin-script names the model passes through."""
        want = text.strip().lower()
        if not want:
            return None
        rows = self.stations()
        for s in rows:
            if s.get("name", "").lower() == want:
                return s
        for s in rows:
            if want in s.get("name", "").lower():
                return s
        return None

    def _ordered(self) -> list[dict]:
        return sorted(self.stations(), key=lambda s: s.get("sort_order", 0))

    # -- doing things -----------------------------------------------------

    def run(self, **params: object) -> str:
        if not self.available:
            return self.unavailable_reason

        if "list" in params or params.get("action") == "list":
            return self._list_stations()
        if "stop" in params:
            return self._stop()
        if "resume" in params:
            return self._resume()
        if "step" in params:
            return self._step(str(params["step"]).lower())
        if "vol" in params:
            return self._nudge(str(params["vol"]).lower())
        if "level" in params:
            return self._set_volume(params["level"], str(params.get("unit") or ""))
        if "freq" in params:
            heard = str(params["freq"])
            freq = parse_frequency(heard)
            if freq is None:
                # The digits were caught fine, they are just not an FM
                # frequency. Saying "I didn't catch that" would send someone
                # off repeating themselves at a number that can never work.
                return f"{heard.strip()} isn't a frequency I have a station on."
            return self._play_frequency(freq)

        action = str(params.get("action") or "").lower()
        if action == "stop":
            return self._stop()
        if action in ("next", "previous"):
            return self._step(action)
        if action == "volume_up":
            return self._nudge("up")
        if action == "volume_down":
            return self._nudge("down")
        if action == "status" or (not action and not params):
            return self._now_playing()
        if params.get("level") is not None:
            return self._set_volume(params["level"], str(params.get("unit") or ""))
        if params.get("frequency") is not None:
            try:
                return self._play_frequency(float(params["frequency"]))
            except (TypeError, ValueError):
                return self._nothing_matched()
        if params.get("station"):
            if not self.stations():
                return self.unavailable_reason
            found = self.by_name(str(params["station"]))
            if found is None:
                return f"I don't have a station called {params['station']}."
            return self._play(found)
        return self._now_playing()

    def _nothing_matched(self) -> str:
        return "I didn't catch which frequency you wanted."

    def _play_frequency(self, freq: float) -> str:
        if not self.stations():
            # An empty list means the list could not be fetched, not that the
            # station does not exist. Saying "I don't have a station on 94.9"
            # when the other Pi is simply off is a confident wrong answer:
            # it sends someone looking for a station that is really there.
            return self.unavailable_reason
        found = self.by_frequency(freq)
        if found is None:
            return f"I don't have a station on {say_frequency(freq)}."
        return self._play(found)

    def _play(self, station: dict) -> str:
        if self._call("POST", f"/api/player/play/{station['id']}") is None:
            return self.unavailable_reason
        return f"Playing {self._spoken_name(station)}."

    def _spoken_name(self, station: dict) -> str:
        """Say the frequency, since that is how it was asked for -- and the
        Greek names would be read aloud by an English voice."""
        freq = frequency_of(station.get("name", ""))
        return say_frequency(freq) if freq else station.get("name", "the radio")

    def _stop(self) -> str:
        if self._call("POST", "/api/player/stop") is None:
            return self.unavailable_reason
        return "Radio off."

    def _resume(self) -> str:
        """Whatever played last. The commonest intent and it needs no choice."""
        now = self.status()
        if now is None:
            return self.unavailable_reason
        if now.get("playing"):
            return f"The radio is already on, {self._spoken_name(now.get('station') or {})}."
        last = now.get("station") or self._most_recent()
        if not last:
            return "I don't have a station to start."
        return self._play(last)

    def _most_recent(self) -> dict | None:
        rows = [s for s in self.stations() if s.get("last_played")]
        return max(rows, key=lambda s: s["last_played"]) if rows else None

    def _step(self, direction: str) -> str:
        rows = self._ordered()
        if not rows:
            return self.unavailable_reason
        now = self.status()
        if now is None:
            return self.unavailable_reason
        current = (now.get("station") or {}).get("id")
        ids = [s["id"] for s in rows]
        if current in ids:
            i = ids.index(current) + (1 if direction == "next" else -1)
        else:
            i = 0
        return self._play(rows[i % len(rows)])

    def _pending_duck(self) -> dict | None:
        """The duck in force, if there is one.

        Anything that changes the volume during a conversation has to go
        through this. The live reading is the *ducked* volume, not the
        listener's, and acting on it twice produced two bugs: "turn the radio
        up" from a real 60 read the ducked 15 and went to 30, losing the
        setting entirely; and setting a volume outright cancelled the duck, so
        the confirmation played over a radio back at full volume.
        """
        thread = self._duck_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)        # it writes the state before the POST
        saved = state.load(DUCK_STATE, None)
        return saved if isinstance(saved, dict) else None

    def _nudge(self, direction: str) -> str:
        """One level up or down, landing on a level boundary either way.

        Reading the current percentage back as the nearest level is what makes
        this predictable: a radio left at 45 on somebody's phone goes to 60
        rather than 55, so after one nudge it sits on the scale Faethon speaks
        in rather than between two of its steps.
        """
        pending = self._pending_duck()
        if pending is not None:
            # Step from where the listener had it, not from where we put it.
            level = levels.from_percent(int(pending.get("was", 0)))
        else:
            now = self.status()
            if now is None:
                return self.unavailable_reason
            level = levels.from_percent(int(now.get("volume", 0)))
        return self._apply(level + (1 if direction == "up" else -1))

    def _set_volume(self, number: object, unit: str = "") -> str:
        try:
            level = levels.as_level(int(number), unit)   # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "I didn't catch what volume you wanted."
        return self._apply(level)

    def _apply(self, level: int) -> str:
        level = max(levels.MIN_LEVEL, min(levels.MAX_LEVEL, level))
        pending = self._pending_duck()
        if pending is not None:
            # Ducked: change what we will come back up to, and stay down. The
            # radio would otherwise jump to the new level immediately and the
            # confirmation would play over it -- which is the thing ducking
            # exists to prevent, defeated by the one command that is about
            # volume. The new level lands when the conversation closes.
            state.save(DUCK_STATE, {**pending, "was": levels.to_percent(level)})
        elif self._call("POST", "/api/player/volume",
                        json={"level": levels.to_percent(level)}) is None:
            return self.unavailable_reason
        if level <= levels.MIN_LEVEL:
            # 0% alone leaves it unclear whether the radio is off or merely
            # turned right down -- the wording volume_skill settled on.
            return f"Radio volume is set to {levels.percent(level)}, muted."
        return f"Radio volume is set to {levels.percent(level)}."

    # -- getting out of the way -------------------------------------------

    @contextlib.contextmanager
    def ducked(self):
        """Turn the radio down for the duration, then put it back.

        Wraps the whole conversation rather than just the reply: you are
        talking over the music too, and a radio that dips only while Faethon
        speaks would lurch up and down through a conversation.

        Nothing in here may raise. Ducking is a courtesy, and a courtesy that
        can break a turn is worse than no courtesy -- so a dead RadioHost, a
        timeout or a surprising response body all just mean the radio stays
        where it is.
        """
        self._duck()
        try:
            yield
        finally:
            self._unduck()

    def _duck(self) -> None:
        """Start turning the radio down. Returns immediately.

        In a thread because it must never delay the ack chime. Measured the
        blocking version at a median of 0ms and a worst case of **4094ms** --
        one call hit the timeout, and that is four seconds of silence before
        the "go ahead" sound, on the one path where somebody is standing there
        waiting to speak.
        """
        if not self.config.duck or not self.available:
            return
        with self._duck_lock:
            if self._duck_thread and self._duck_thread.is_alive():
                return
            self._duck_thread = threading.Thread(target=self._duck_now, daemon=True)
            self._duck_thread.start()

    def _duck_now(self) -> None:
        try:
            now = self.status()
            if not now or not now.get("playing"):
                return                            # nothing to get out of the way of
            was = int(now.get("volume", 0))
            if was <= self.config.duck_to:
                return                            # already quieter than we would set it
            # Written BEFORE the request, not after. A POST that times out may
            # still have arrived -- that is exactly how the radio ended up
            # stuck at 15 with nothing recorded to restore it from. Saving
            # first makes the restore self-correcting: if the change never
            # landed, the volume will not match and the restore stands down.
            state.save(DUCK_STATE, {"was": was, "set": self.config.duck_to})
            self._call("POST", "/api/player/volume", json={"level": self.config.duck_to})
            log.debug("radio ducked %d -> %d", was, self.config.duck_to)
        except Exception as e:
            # Deliberately broad, and in a thread, where an escaping exception
            # is logged by the interpreter and lost rather than handled.
            # Ducking is a courtesy; a courtesy that can raise anywhere near a
            # turn is worse than no courtesy.
            log.debug("could not duck the radio: %s", e)

    def _unduck(self) -> None:
        try:
            # The duck may still be in flight on a slow link. Wait briefly
            # rather than racing it; the saved state makes a miss harmless.
            thread = self._duck_thread
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
            saved = state.load(DUCK_STATE, None)
            if not isinstance(saved, dict):
                return
            now = self.status()
            if now is None:
                return                            # try again at the next restore
            current = int(now.get("volume", -1))
            if current != saved.get("set"):
                # Either somebody moved it while we were talking -- almost
                # certainly the user, through this very skill, and their
                # instruction is the more recent one -- or the duck never
                # actually landed. Both mean: leave it alone.
                log.debug("radio at %d, not the %d we set; leaving it",
                          current, saved.get("set"))
            else:
                self._call("POST", "/api/player/volume", json={"level": saved["was"]})
            state.save(DUCK_STATE, None)
        except Exception as e:
            log.debug("could not restore the radio: %s", e)

    def restore_after_crash(self) -> None:
        """Called at startup. A process that died mid-turn left the radio low."""
        if isinstance(state.load(DUCK_STATE, None), dict):
            log.info("radio was left ducked by a previous run; restoring")
            self._unduck()

    def _list_stations(self) -> str:
        """What is on the dial, fetched fresh every time.

        Deliberately not cached: the list changes when someone edits it on the
        other Pi, and the whole point of asking is to find out whether it did.
        A cached answer to "what stations do you have" is the one answer that
        is never worth giving.

        Read as frequencies rather than names, for the same two reasons
        selection is by frequency: half the names are Greek and an English
        voice reading them produces noise, and the frequency is what you say
        back to choose one.
        """
        rows = self.stations(fresh=True)
        if not rows:
            return self.unavailable_reason

        freqs = sorted({f for f in (frequency_of(r.get("name", "")) for r in rows)
                        if f is not None})
        if not freqs:
            return f"I have {len(rows)} stations, but none of them list a frequency."

        spoken = [say_frequency(f) for f in freqs[:MAX_SPOKEN]]
        said = f"I have {len(freqs)} stations: " + ", ".join(spoken)
        if len(freqs) > MAX_SPOKEN:
            said += f", and {len(freqs) - MAX_SPOKEN} more"
        return said + "."

    def _now_playing(self) -> str:
        now = self.status()
        if now is None:
            return self.unavailable_reason
        if not now.get("playing"):
            return "The radio is off."
        where = self._spoken_name(now.get("station") or {})
        track = (now.get("now_playing") or "").strip()
        return f"{where} is playing {track}." if track else f"{where} is playing."


SKILL = RadioSkill()
