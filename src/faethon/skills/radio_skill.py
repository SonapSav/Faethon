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

import logging
import re
import time

import httpx

from .. import disclosure
from ..config import load_config
from .base import Skill

log = logging.getLogger(__name__)

#: A frequency inside a station name: "94,9", "102,2", "88".
_IN_NAME = re.compile(r"(\d{2,3})(?:[.,](\d))?")
#: How long an unreachable host is remembered, so a dead Pi is not retried on
#: every single turn while still recovering without a restart.
UNAVAILABLE_FOR = 60.0


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
        "it, change station, change ITS volume, or say what is playing. "
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
        r"\bradio\s+volume\s+(?:to\s+)?(?P<level>\d{1,3})\b",
        r"\bset\s+(?:the\s+)?radio\s+volume\s+(?:to\s+)?(?P<level>\d{1,3})\b",
        rf"\bwhat(?:'s|s| is)\s+(?:playing|on the radio){_END}",
        rf"\bwhat\s+station\s+is\s+(?:this|it|on){_END}",
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
                "enum": ["play", "stop", "next", "previous", "status",
                         "volume_up", "volume_down"],
                "description": "Defaults to play when a station is given.",
            },
            "level": {
                "type": "integer",
                "description": "Radio volume, 0 to 100.",
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        super().__init__()
        self._config = None
        self._stations: tuple[float, list[dict]] | None = None
        self._down_until = 0.0

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

        if "stop" in params:
            return self._stop()
        if "resume" in params:
            return self._resume()
        if "step" in params:
            return self._step(str(params["step"]).lower())
        if "vol" in params:
            return self._nudge(str(params["vol"]).lower())
        if "level" in params:
            return self._set_volume(params["level"])
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
            return self._set_volume(params["level"])
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

    def _nudge(self, direction: str) -> str:
        now = self.status()
        if now is None:
            return self.unavailable_reason
        step = self.config.volume_step * (1 if direction == "up" else -1)
        return self._set_volume(int(now.get("volume", 0)) + step)

    def _set_volume(self, level: object) -> str:
        try:
            wanted = max(0, min(100, int(level)))       # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "I didn't catch what volume you wanted."
        if self._call("POST", "/api/player/volume", json={"level": wanted}) is None:
            return self.unavailable_reason
        if wanted == 0:
            return "Radio volume is at zero, muted."
        return f"Radio volume is {wanted} percent."

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
