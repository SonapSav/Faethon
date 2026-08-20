"""Weather, and the reason the regex path never captures a place name.

That is credit_skill's lesson at a larger order of magnitude. "OpenRouter" is
one coined word with a handful of spellings and it still needed a loose
pattern. A place name is an open set of proper nouns, many foreign, and Whisper
will hand back "Redding" for Reading -- both of which the geocoder resolves
happily, one in England and one in California. The failure is not an error; it
is the forecast for the wrong continent, spoken with total confidence.

So the daily phrasings go through the regex path with no location captured at
all, and named places go through the model, which has far better priors for
turning a mangled transcript into a real place.

No network here: the HTTP layer is stubbed.
"""

from __future__ import annotations

import httpx
import pytest

from faethon.skills.weather_skill import SKILL, WeatherSkill, condition

FORECAST = {
    "current": {"temperature_2m": 34.8, "apparent_temperature": 44.1, "weather_code": 0},
    "daily": {
        "temperature_2m_max": [38.2, 40.1],
        "temperature_2m_min": [32.4, 31.3],
        "precipitation_probability_max": [0, 80],
        "weather_code": [0, 3],
    },
}


@pytest.fixture
def rig():
    class Rig(WeatherSkill):
        forecast = FORECAST
        geocode: list | None = None
        error: Exception | None = None
        calls = 0

        def _get(self, url, params):
            Rig.calls += 1
            if self.error is not None:
                raise self.error
            if "geocoding" in url:
                return {"results": self.geocode} if self.geocode is not None else {}
            return self.forecast

    Rig.calls = 0
    s = Rig()
    s._config = type("C", (), {
        "latitude": 25.2, "longitude": 55.3,
        "place_name": "Dubai", "units": "metric",
    })()
    return s


def say(skill, text):
    m = skill.match(text)
    assert m is not None, f"no pattern matched {text!r}"
    return skill.run(**m)


# -- the phrasings -----------------------------------------------------------


@pytest.mark.parametrize("heard", [
    "what's the weather",
    "what is the weather like",
    "what's the weather tomorrow",
    "will it rain",
    "is it going to rain tomorrow",
    "do I need an umbrella",
    "do I need a jacket",
    "how hot is it",
    "how cold is it today",
    "what's the temperature",
    "what's the forecast tomorrow",
])
def test_the_daily_phrasings(rig, heard):
    assert rig.match(heard) is not None


@pytest.mark.parametrize("heard", [
    "what's the weather like in the housing market",
    "how cold is your reception to this idea",
    "what is the forecast for the economy",
    "will it rain on my parade at the weekend",
])
def test_phrasings_that_only_look_like_weather(rig, heard):
    assert rig.match(heard) is None, f"intercepted {heard!r}"


def test_the_regex_path_never_captures_a_location(rig):
    """The whole design. A captured place name is a forecast for somewhere
    else, said with the same confidence as the right one."""
    for heard in ["what's the weather", "what's the weather in Paris",
                  "will it rain in Reading tomorrow"]:
        m = rig.match(heard)
        if m is not None:
            assert "location" not in m, f"{heard!r} captured a location"


# -- answering the question that was asked -----------------------------------


def test_a_rain_question_is_answered_about_rain(rig):
    """"Will it rain?" answered with a temperature range is worse than no
    answer: it sounds like one and is not."""
    assert "No rain expected" in say(rig, "will it rain")
    assert "80 percent chance of rain" in say(rig, "will it rain tomorrow")


def test_an_umbrella_is_a_rain_question(rig):
    assert "No rain expected" in say(rig, "do I need an umbrella")
    assert "80 percent" in say(rig, "do I need an umbrella tomorrow")


def test_a_jacket_is_a_temperature_question(rig):
    """And it gives the numbers rather than a verdict -- what warrants a coat
    is a matter of opinion and of who is asking."""
    said = say(rig, "do I need a jacket")
    assert "32 to 38" in said


def test_the_general_answer_leads_with_now(rig):
    said = say(rig, "what's the weather")
    assert said.startswith("It's 35 and clear in Dubai")
    assert "high of 38" in said


def test_tomorrow_is_a_range_not_a_reading(rig):
    said = say(rig, "what's the weather tomorrow")
    assert "Tomorrow in Dubai" in said and "31 to 40" in said


# -- how it reads aloud ------------------------------------------------------


def test_temperatures_are_whole_numbers(rig):
    """"Thirty four point eight degrees" is not how anyone says it, and the
    forecast has nothing like that precision anyway."""
    for said in [say(rig, "what's the weather"), say(rig, "what's the weather tomorrow")]:
        for raw in ("34.8", "44.1", "38.2", "32.4", "40.1", "31.3"):
            assert raw not in said, f"{raw} was read out verbatim: {said}"


def test_it_mentions_how_it_feels_only_when_that_differs(rig):
    """In Dubai in August the gap is routinely ten degrees, and worth saying.
    A one-degree gap is noise."""
    assert "feels like 44" in say(rig, "what's the weather")
    rig.forecast = {**FORECAST, "current": {
        "temperature_2m": 20.0, "apparent_temperature": 20.5, "weather_code": 0}}
    rig._cache.clear()          # the cache is doing its job; this test isn't it
    assert "feels like" not in say(rig, "what's the weather")


def test_rain_is_left_out_when_it_is_not_worth_saying(rig):
    """Two facts, maybe three. TTS bills per character and the follow-up
    window means "and the wind?" is one sentence away."""
    assert "chance of rain" not in say(rig, "what's the weather")


# -- named places ------------------------------------------------------------


def test_a_named_place_is_geocoded(rig):
    rig.geocode = [{"latitude": 48.85, "longitude": 2.35, "name": "Paris"}]
    assert "in Paris" in rig.run(location="Paris")


def test_an_unknown_place_is_refused_not_approximated(rig):
    """The geocoder answers "Redding, California" for "Reading" perfectly
    happily. Never take the nearest match."""
    rig.geocode = []
    said = rig.run(location="Barcelonia")
    assert "couldn't find a place called Barcelonia" in said
    assert "degrees" not in said


# -- when it cannot answer ---------------------------------------------------


def test_an_unreachable_service_says_so(rig):
    rig.error = httpx.ConnectError("no route")
    assert "couldn't reach the weather service" in say(rig, "what's the weather")


def test_an_unexpected_shape_is_not_read_out_as_zero(rig):
    rig.forecast = {"daily": {}}
    said = say(rig, "what's the weather")
    assert "didn't understand" in said
    assert "0" not in said


def test_an_unknown_condition_code_is_not_invented():
    assert condition(0) == "clear"
    assert condition(999) == "hard to describe"
    assert condition(None) == "hard to describe"


# -- caching -----------------------------------------------------------------


def test_a_follow_up_conversation_does_not_refetch(rig):
    """Three calls for data that changes hourly, and the follow-up window
    makes exactly that conversation likely."""
    say(rig, "what's the weather")
    before = type(rig).calls
    say(rig, "will it rain")
    say(rig, "what's the weather tomorrow")
    assert type(rig).calls == before
