"""Air quality and UV, from Open-Meteo -- the same keyless provider as weather.

One skill rather than two, for the reason health_skill is one rather than five:
Registry.match is first-match-wins in import order, and "is it safe to go
outside" or "should I cover up" belong to both. They also share an endpoint, a
cache, a location and a config section, so splitting them would duplicate all
of that to gain nothing.

Dust is why this exists. It is the weather that changes your day here and is
completely invisible to a temperature forecast -- measured on the afternoon
this was written, the weather skill said "45 and clear in Abu Dhabi" while dust
sat at 310 micrograms per cubic metre and the European index read very poor.
Both sentences were true.

Answers carry their meaning rather than a bare reading: "very poor, mostly
dust" rather than "PM10 is 222", the same move volume_skill makes with dB and
health_skill with thermals. A number nobody can place is not an answer.

The forecast half is lopsided, because the API is. UV has a real daily
aggregate (uv_index_max) that the model produces itself. Dust and the air index
are hourly only -- there is no european_aqi_max, and asking for one is a 400 --
so their daily figures are aggregated here from 24 hourly values.

Peak, not mean, for dust: what matters is whether it gets bad at some point
today, not what it averaged while you were asleep.

Three days ahead, no further. The API will serve seven, but a dust forecast
that far out is guesswork dressed up as a number, and nobody asks. The decay
measured while this was written -- 321, 307, 219, 132, 112 -- is real signal
through about day three and thinning after.
"""

from __future__ import annotations

import logging
import time
from datetime import date as _date

import httpx

from ..config import load_config
from .base import Skill

log = logging.getLogger(__name__)

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
#: Short: this runs inside a spoken turn.
TIMEOUT_SEC = 10.0
#: Readings are hourly, so anything shorter refetches the same numbers.
CACHE_SEC = 900.0
#: Today plus three. See the module docstring on why not the seven on offer.
FORECAST_DAYS = 4

#: European AQI bands, in the words the index itself uses.
_AQI_BANDS = [(20, "good"), (40, "fair"), (60, "moderate"),
              (80, "poor"), (100, "very poor")]
#: WHO's UV scale.
_UV_BANDS = [(3, "low"), (6, "moderate"), (8, "high"), (11, "very high")]
#: Above this share of PM10, dust is worth naming as the cause rather than
#: leaving someone to wonder what the poor air actually is.
_DUST_SHARE = 0.5


def band(value: float, bands: list[tuple[float, str]], above: str) -> str:
    for limit, word in bands:
        if value < limit:
            return word
    return above


def describe_air(aqi: float) -> str:
    return band(aqi, _AQI_BANDS, "extremely poor")


def describe_uv(uv: float) -> str:
    return band(uv, _UV_BANDS, "extreme")


def daily_peaks(hourly: dict) -> dict[str, dict[str, float]]:
    """Collapse 24 hourly readings per day into that day's peak.

    Peak rather than mean: the question behind "will it be dusty tomorrow" is
    whether it gets bad at any point, not what it averaged overnight.
    """
    out: dict[str, dict[str, float]] = {}
    times = hourly.get("time") or []
    for field in ("dust", "european_aqi"):
        values = hourly.get(field) or []
        for stamp, value in zip(times, values):
            number = _number(value)
            if number is None:
                continue
            day = out.setdefault(stamp[:10], {})
            day[field] = max(day.get(field, number), number)
    return out


def day_name(iso: str, offset: int) -> str:
    """What to call a day out loud.

    Named from the API's own date strings rather than the local clock -- the Pi
    has no RTC, and a weekday is exactly the kind of thing it would get wrong
    for the first few seconds after a cold boot.
    """
    if offset == 0:
        return "today"
    if offset == 1:
        return "tomorrow"
    try:
        return _date.fromisoformat(iso).strftime("%A")
    except ValueError:
        return f"in {offset} days"


class AirSkill(Skill):
    name = "get_air_quality"
    tag = "utility"
    announce_urgency = "informational"
    description = (
        "Air quality, dust, or UV index -- now, tomorrow, or whether it is "
        "improving. Give a location only if the user named one; leave it out "
        "for here. Use for questions about air, pollution, dust, haze, the UV "
        "index, or sun strength. Only today and the next three days exist; "
        "say so rather than guessing if asked about further ahead."
    )

    _WHERE = r"(?:\s+(?:outside|out there|today|right now|now|(?P<tomorrow>tomorrow)))?"
    _END = r"(?:\s+(?:please|for me))*[^\w]*$"

    patterns = [
        # Trend first: "is the air going to clear" must not be read as
        # "is the air ok", which is a different question with a worse answer.
        r"\b(?:is|are|will)(?: the)? (?P<trend>dust|air|haze|dust storm)"
        r"(?:\s+(?:going to|gonna|due to))?\s+"
        r"(?:clear|clearing|lift|ease|settle|improve|get better|get worse|die down|dying down|settling)",
        r"\bis(?: the)? (?P<trend>dust|air|haze) getting (?:better|worse)",
        rf"\bwhat(?:'s|s| is) the air(?: quality)? (?:going to be |)like{_WHERE}{_END}",

        rf"\bwhat(?:'s|s| is) the air(?: quality)?{_WHERE}{_END}",
        rf"\bhow(?:'s|s| is) the air{_WHERE}{_END}",
        rf"\bis the air (?:ok|okay|bad|clean|safe){_WHERE}{_END}",
        rf"\b(?:is|will) it (?:be )?(?P<dust>dusty|hazy){_WHERE}{_END}",
        rf"\bhow much (?P<dust>dust)(?: is there| will there be)?{_WHERE}{_END}",
        rf"\bwhat(?:'s|s| is) the (?P<uv>u\.?v\.?)(?: index)?{_WHERE}{_END}",
        r"\bhow strong is the (?P<uv>sun)\b",
        r"\bdo i need (?:any )?(?P<uv>sunscreen|sun cream)\b",
    ]

    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Place name. Omit entirely for the user's home.",
            },
            "kind": {
                "type": "string",
                "enum": ["air", "dust", "uv"],
                "description": "Defaults to overall air quality.",
            },
            "when": {
                "type": "string",
                "enum": ["today", "tomorrow", "trend"],
                "description": (
                    "Use 'trend' for whether conditions are improving over the "
                    "next few days. Only today and the next three days exist."
                ),
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        super().__init__()
        self._cache: tuple[float, dict] | None = None
        self._config = None
        self._air_config = None
        self._last_check = 0.0
        self._warned: set[str] = set()

    @property
    def config(self):
        if self._config is None:
            self._config = load_config().weather      # same coordinates
        return self._config

    @property
    def air(self):
        if self._air_config is None:
            self._air_config = load_config().air
        return self._air_config

    # -- the network ------------------------------------------------------

    def _fetch(self, lat: float, lon: float, use_cache: bool = True) -> dict:
        if use_cache and self._cache and time.monotonic() - self._cache[0] < CACHE_SEC:
            return self._cache[1]
        with httpx.Client(timeout=TIMEOUT_SEC) as client:
            r = client.get(AIR_URL, params={
                "latitude": lat, "longitude": lon,
                "current": "pm10,pm2_5,dust,uv_index,european_aqi",
                "hourly": "dust,pm10,european_aqi",
                "daily": "uv_index_max",
                "timezone": "auto",
                "forecast_days": FORECAST_DAYS,
            })
            r.raise_for_status()
            data = r.json()
        if use_cache:
            self._cache = (time.monotonic(), data)
        return data

    # -- speaking up unasked ----------------------------------------------

    def tick(self) -> str | None:
        """Warn once when dust or UV crosses, and re-arm when it drops back.

        The fetch costs about 550ms, which is why it happens at most every half
        hour. It does stall the wake loop for that long, but arecord buffers
        the microphone, so detection is delayed rather than lost -- measured at
        0.56s of backlog, consumed at 1.28x real time.
        """
        now = time.monotonic()
        if now - self._last_check < self.air.check_every_minutes * 60:
            return None
        self._last_check = now

        try:
            current = self._fetch(
                self.config.latitude, self.config.longitude, use_cache=False
            )["current"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
            log.debug("air check failed: %s", e)
            return None
        self._cache = (now, {"current": current})

        dust = _number(current.get("dust"))
        uv = _number(current.get("uv_index"))

        if dust is not None:
            if dust >= self.air.dust_warn and "dust" not in self._warned:
                self._warned.add("dust")
                log.info("dust %.0f over %.0f", dust, self.air.dust_warn)
                return (
                    f"There's a lot of dust outside, {round(dust)} "
                    "micrograms. Worth keeping the windows shut."
                )
            if dust < self.air.dust_warn:
                self._warned.discard("dust")

        if uv is not None:
            if uv >= self.air.uv_warn and "uv" not in self._warned:
                self._warned.add("uv")
                log.info("uv %.1f over %.1f", uv, self.air.uv_warn)
                return (
                    f"The UV index is {round(uv)}, which is "
                    f"{describe_uv(uv)}. Worth covering up outside."
                )
            if uv < self.air.uv_warn:
                self._warned.discard("uv")
        return None

    # -- the asked-for half -----------------------------------------------

    def run(self, **params: object) -> str:
        when = str(params.get("when") or "").lower()
        if not when:
            if "trend" in params:
                when = "trend"
            elif "tomorrow" in params:
                when = "tomorrow"
            else:
                when = "today"

        kind = str(params.get("kind") or "").lower()
        if not kind:
            if "uv" in params:
                kind = "uv"
            elif "dust" in params:
                kind = "dust"
            else:
                kind = "air"

        place = str(params.get("location") or "").strip()
        try:
            if place:
                found = _geocode(place)
                if found is None:
                    return f"I couldn't find a place called {place}."
                lat, lon, label = found
                payload = self._fetch(lat, lon, use_cache=False)
            else:
                lat, lon = self.config.latitude, self.config.longitude
                label = self.config.place_name
                payload = self._fetch(lat, lon)
            current = payload["current"]
        except httpx.HTTPError as e:
            log.error("air lookup failed: %s", e)
            return "Sorry, I couldn't reach the air quality service."
        except (KeyError, TypeError, ValueError) as e:
            log.error("unexpected air response: %s", e)
            return "The air quality service gave me an answer I didn't understand."

        if when == "today":
            return self._describe(current, label, kind)
        if when == "trend":
            return self._describe_trend(payload, label, kind)
        return self._describe_day(payload, label, kind, offset=1)

    def _describe(self, current: dict, place: str, kind: str) -> str:
        dust = _number(current.get("dust"))
        uv = _number(current.get("uv_index"))
        pm10 = _number(current.get("pm10"))
        aqi = _number(current.get("european_aqi"))

        if kind == "uv":
            if uv is None:
                return f"I don't have a UV reading for {place}."
            return f"The UV index is {round(uv)} in {place}, which is {describe_uv(uv)}."

        if kind == "dust":
            if dust is None:
                return f"I don't have a dust reading for {place}."
            if dust < 50:
                return f"Hardly any dust in {place} right now."
            return f"Dust is {round(dust)} micrograms in {place}."

        if aqi is None:
            return f"I don't have an air quality reading for {place}."
        said = f"The air is {describe_air(aqi)} in {place}"
        # Name dust when it is most of what is in the air, rather than leaving
        # someone to wonder what "very poor" actually consists of.
        if dust and pm10 and dust >= pm10 * _DUST_SHARE and aqi >= 40:
            said += ", mostly dust"
        return said + "."


    # -- the next few days -------------------------------------------------

    def _days(self, payload: dict) -> list[str]:
        """Forecast dates in order, from whichever block carries them."""
        daily = payload.get("daily") or {}
        if daily.get("time"):
            return list(daily["time"])
        return sorted(daily_peaks(payload.get("hourly") or {}))

    def _series(self, payload: dict, kind: str) -> list[float | None]:
        """One value per forecast day, for whichever thing was asked about."""
        days = self._days(payload)
        if kind == "uv":
            values = (payload.get("daily") or {}).get("uv_index_max") or []
            return [_number(v) for v in values][: len(days)]
        peaks = daily_peaks(payload.get("hourly") or {})
        field = "dust" if kind == "dust" else "european_aqi"
        return [peaks.get(day, {}).get(field) for day in days]

    def _describe_day(self, payload: dict, place: str, kind: str, offset: int) -> str:
        days = self._days(payload)
        series = self._series(payload, kind)
        when = day_name(days[offset], offset) if offset < len(days) else "then"

        if offset >= len(series) or series[offset] is None:
            return f"I don't have a forecast for {place} {when}."
        value = series[offset]

        if kind == "uv":
            return (
                f"The UV index should peak around {round(value)} {when} "
                f"in {place}, which is {describe_uv(value)}."
            )
        if kind == "dust":
            if value < 50:
                return f"Very little dust expected in {place} {when}."
            return f"Dust should peak around {round(value)} micrograms {when} in {place}."
        return f"The air should be {describe_air(value)} in {place} {when}."

    def _describe_trend(self, payload: dict, place: str, kind: str) -> str:
        """Whether this is getting better -- the question worth asking mid-event.

        "Is the dust going to clear" wants a direction and a rough when, not a
        table of numbers. So: where it stands, where it lands, and the day.
        """
        if kind == "air":
            kind = "dust"                       # "is the air going to clear"
        days = self._days(payload)
        series = self._series(payload, kind)
        known = [(i, v) for i, v in enumerate(series) if v is not None]
        if len(known) < 2:
            return f"I don't have enough forecast to tell for {place}."

        first_value = known[0][1]
        last_index, last_value = known[-1]
        when = day_name(days[last_index], last_index)
        noun = "UV" if kind == "uv" else "dust"
        unit = "" if kind == "uv" else " micrograms"

        if first_value <= 0:
            change = 0.0
        else:
            change = (last_value - first_value) / first_value

        if change <= -0.3:
            return (
                f"It should ease off. {noun.capitalize()} is around "
                f"{round(first_value)}{unit} today, down to about "
                f"{round(last_value)}{unit} by {when}."
            )
        if change >= 0.3:
            return (
                f"It looks like getting worse. {noun.capitalize()} is around "
                f"{round(first_value)}{unit} today, up to about "
                f"{round(last_value)}{unit} by {when}."
            )
        return (
            f"Not much change expected. {noun.capitalize()} stays around "
            f"{round(first_value)}{unit} through {when}."
        )


def _number(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _geocode(place: str) -> tuple[float, float, str] | None:
    """Shared with the weather skill: the model resolves the name, never the
    regex -- Whisper returns Redding for Reading and both are real places."""
    from .weather_skill import SKILL as weather

    return weather._geocode(place)


SKILL = AirSkill()
