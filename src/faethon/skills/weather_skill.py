"""The weather, from Open-Meteo -- no key, no account, no sign-up.

Keyless matters more than it sounds. A skill with an API key can be
half-configured: `available` has to be plumbed, an unavailable_reason written,
and a fresh clone answers "I don't have a key for that". This one either works
or the network is down, and if the network were down the transcript would never
have been produced.

The decision that shapes everything else is that **the regex path never
captures a place name.** That is credit_skill's lesson at a different order of
magnitude. "OpenRouter" is one coined word with a handful of spellings and it
still needed a loose pattern; a place name is an open set of proper nouns,
many foreign, and Whisper will hand back "Redding" for Reading -- both of which
the geocoder resolves, one in England and one in California. The failure is not
an error. It is the forecast for the wrong continent, spoken with total
confidence.

So the two routing paths split along exactly that line:

    regex      home only, no location captured at all. The daily phrasings --
               "what's the weather", "will it rain", "do I need a jacket" --
               are about here, and this path is structurally incapable of
               getting the place wrong.
    tool call  named places. The model has far better priors than a regex for
               turning a mangled transcript into a real place, and it can see
               the conversation. One API call is the right price for the
               harder problem.
"""

from __future__ import annotations

import logging
import time

import httpx

from ..config import load_config
from .base import Skill

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
#: Short: this runs inside a spoken turn. The default would leave someone
#: standing in silence wondering whether it heard them.
TIMEOUT_SEC = 10.0
#: A follow-up conversation about the weather asks three times for data that
#: changes hourly.
CACHE_SEC = 600.0
#: Mention how it feels only when that differs enough to be worth saying. In
#: Dubai in August the gap is routinely ten degrees.
FEELS_LIKE_GAP = 4
#: Below this, "rain" is not worth a sentence.
RAIN_MENTION_PCT = 25

#: WMO codes, in the words someone would actually use.
_CONDITIONS = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy",
    51: "drizzling", 53: "drizzling", 55: "drizzling",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "raining", 63: "raining", 65: "raining heavily",
    66: "freezing rain", 67: "freezing rain",
    71: "snowing", 73: "snowing", 75: "snowing heavily", 77: "snowing",
    80: "showery", 81: "showery", 82: "heavy showers",
    85: "snow showers", 86: "snow showers",
    95: "thundery", 96: "thundery with hail", 99: "thundery with hail",
}

_WHEN = r"(?:\s+(?P<when>today|tomorrow|tonight|right now|now))?"
_END = r"(?:\s+(?:please|for me|outside))*[^\w]*$"


def condition(code: object) -> str:
    try:
        return _CONDITIONS.get(int(code), "hard to describe")
    except (TypeError, ValueError):
        return "hard to describe"


class WeatherSkill(Skill):
    name = "get_weather"
    tag = "utility"
    description = (
        "Current weather or the forecast for today or tomorrow. Give a "
        "location only if the user named one; leave it out for here. Use for "
        "questions about weather, temperature, rain, or what to wear."
    )

    patterns = [
        rf"\bwhat(?:'s|s| is) (?:the )?weather(?: like)?{_WHEN}{_END}",
        rf"\b(?:is|will) it (?:be |going to )?(?P<rain>rain)(?:ing)?{_WHEN}{_END}",
        r"\bdo i need (?:an? )?(?P<need>umbrella|jacket|coat)\b"
        r"(?:\s+(?P<when>today|tomorrow|tonight))?",
        rf"\bhow (?:hot|cold|warm) is it{_WHEN}{_END}",
        rf"\bwhat(?:'s|s| is) (?:the )?(?:temperature|forecast){_WHEN}{_END}",
    ]

    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Place name. Omit entirely for the user's home.",
            },
            "when": {
                "type": "string",
                "enum": ["now", "today", "tomorrow"],
                "description": "Defaults to now.",
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple, tuple[float, dict]] = {}
        self._config = None

    @property
    def config(self):
        # Lazily, so importing this module needs no config file -- the registry
        # imports every skill at startup and the test suite imports them bare.
        if self._config is None:
            self._config = load_config().weather
        return self._config

    # -- the network ------------------------------------------------------

    def _get(self, url: str, params: dict) -> dict:
        with httpx.Client(timeout=TIMEOUT_SEC) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()

    def _geocode(self, place: str) -> tuple[float, float, str] | None:
        data = self._get(GEOCODE_URL, {"name": place, "count": 1, "language": "en"})
        results = data.get("results") or []
        if not results:
            return None
        top = results[0]
        return float(top["latitude"]), float(top["longitude"]), str(top["name"])

    def _forecast(self, lat: float, lon: float) -> dict:
        key = (round(lat, 2), round(lon, 2))
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_SEC:
            return hit[1]
        imperial = self.config.units == "imperial"
        data = self._get(FORECAST_URL, {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,apparent_temperature,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,weather_code",
            "timezone": "auto", "forecast_days": 2,
            **({"temperature_unit": "fahrenheit"} if imperial else {}),
        })
        self._cache[key] = (time.monotonic(), data)
        return data

    # -- what it says -----------------------------------------------------

    def run(self, **params: object) -> str:
        when = str(params.get("when") or "").lower()
        need = str(params.get("need") or "").lower()
        # Answer the question that was asked. "Will it rain?" answered with a
        # temperature range is a worse reply than no reply -- it sounds like an
        # answer and is not one.
        about = "general"
        if "rain" in params or need == "umbrella":
            about = "rain"
        elif need in ("jacket", "coat"):
            about = "cold"
        if not when:
            when = "today" if need or about != "general" else "now"
        if when in ("right now", ""):
            when = "now"

        place = str(params.get("location") or "").strip()
        try:
            if place:
                found = self._geocode(place)
                if found is None:
                    # Never the nearest match: "Redding" for "Reading" is a
                    # different continent, said with the same confidence.
                    return f"I couldn't find a place called {place}."
                lat, lon, label = found
            else:
                lat, lon = self.config.latitude, self.config.longitude
                label = self.config.place_name
            data = self._forecast(lat, lon)
        except httpx.HTTPError as e:
            log.error("weather lookup failed: %s", e)
            return "Sorry, I couldn't reach the weather service."

        try:
            return self._describe(data, label, when, about)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            # Shape changed. Saying so beats reading out a fabricated zero.
            log.error("unexpected forecast shape: %s", e)
            return "The weather service gave me an answer I didn't understand."

    def _describe(self, data: dict, place: str, when: str,
                  about: str = "general") -> str:
        daily = data["daily"]
        day = 1 if when == "tomorrow" else 0
        high = round(float(daily["temperature_2m_max"][day]))
        low = round(float(daily["temperature_2m_min"][day]))
        rain = daily.get("precipitation_probability_max") or [None, None]
        rain_pct = rain[day]

        timeword = {"now": "today", "today": "today", "tomorrow": "tomorrow"}[when]

        if about == "rain":
            if not isinstance(rain_pct, (int, float)):
                return f"I don't have a rain forecast for {place}."
            pct = round(rain_pct)
            if pct < 10:
                return f"No rain expected in {place} {timeword}."
            if pct < RAIN_MENTION_PCT:
                return f"Probably not, only a {pct} percent chance in {place} {timeword}."
            return f"Yes, a {pct} percent chance of rain in {place} {timeword}."

        if about == "cold":
            # No judgment about what warrants a coat: that is a matter of
            # opinion and of who is asking. The number is not.
            return f"{timeword.capitalize()} in {place} it's {low} to {high}."

        if when == "now":
            cur = data["current"]
            temp = round(float(cur["temperature_2m"]))
            feels = round(float(cur.get("apparent_temperature", temp)))
            sentence = f"It's {temp} and {condition(cur['weather_code'])} in {place}"
            if abs(feels - temp) >= FEELS_LIKE_GAP:
                sentence += f", feels like {feels}"
            sentence += f", with a high of {high}."
        else:
            word = "Tomorrow" if when == "tomorrow" else "Today"
            sentence = (
                f"{word} in {place}: {condition(daily['weather_code'][day])}, "
                f"{low} to {high}."
            )

        if isinstance(rain_pct, (int, float)) and rain_pct >= RAIN_MENTION_PCT:
            sentence += f" {round(rain_pct)} percent chance of rain."
        return sentence


SKILL = WeatherSkill()
