"""Air quality, dust and UV -- and why they are one skill, not two.

Registry.discover walks the package alphabetically, which puts air_skill ahead
of every other skill in match order. That makes its patterns the ones most able
to steal a phrase they were not meant to catch, so the collision test below is
not decoration.

The unprompted half is the interesting half. It has to warn once when dust
crosses, stay quiet while it stays high, and re-arm when it drops -- the same
say-once contract as status.Announcer, for the same reason: a warning repeated
every half hour is noise, and noise gets ignored.

No network here: the HTTP layer is stubbed.
"""

from __future__ import annotations

import httpx
import pytest

from faethon.skills.air_skill import SKILL, AirSkill, describe_air, describe_uv

# The live reading on the afternoon this was built: a real dust event.
DUSTY = {
    "current": {"pm10": 222.1, "pm2_5": 62.4, "dust": 310.0,
                "uv_index": 8.65, "european_aqi": 90},
}
CLEAN = {
    "current": {"pm10": 18.0, "pm2_5": 6.0, "dust": 4.0,
                "uv_index": 2.1, "european_aqi": 15},
}


@pytest.fixture
def rig():
    class Rig(AirSkill):
        payload = DUSTY
        error: Exception | None = None
        calls = 0

        def _fetch(self, lat, lon, use_cache=True):
            Rig.calls += 1
            if self.error is not None:
                raise self.error
            return self.payload

    rig = Rig()
    rig._air_config = type("C", (), {
        "dust_warn": 200.0, "uv_warn": 8.0, "check_every_minutes": 30.0})()
    rig._config = type("C", (), {
        "latitude": 24.433, "longitude": 54.6511, "place_name": "Abu Dhabi"})()
    Rig.calls = 0
    return rig


# -- the bands ------------------------------------------------------------

@pytest.mark.parametrize("aqi,word", [
    (0, "good"), (19.9, "good"), (20, "fair"), (55, "moderate"),
    (80, "very poor"), (90, "very poor"), (100, "extremely poor"), (300, "extremely poor"),
])
def test_air_bands(aqi, word):
    assert describe_air(aqi) == word


@pytest.mark.parametrize("uv,word", [
    (0, "low"), (2.9, "low"), (3, "moderate"), (7, "high"),
    (8.65, "very high"), (11, "extreme"), (14, "extreme"),
])
def test_uv_bands(uv, word):
    assert describe_uv(uv) == word


# -- the spoken answers ---------------------------------------------------

def test_names_dust_when_dust_is_the_cause(rig):
    """"Very poor" alone leaves you wondering what is in the air."""
    said = rig.run()
    assert said == "The air is very poor in Abu Dhabi, mostly dust."


def test_clean_air_does_not_blame_dust(rig):
    rig.payload = CLEAN
    assert rig.run() == "The air is good in Abu Dhabi."


def test_uv_carries_its_meaning_not_just_a_number(rig):
    """A bare "8.65" is not an answer anybody can act on."""
    assert rig.run(kind="uv") == "The UV index is 9 in Abu Dhabi, which is very high."


def test_dust_reading(rig):
    assert rig.run(kind="dust") == "Dust is 310 micrograms in Abu Dhabi."


def test_barely_any_dust_says_so_plainly(rig):
    rig.payload = CLEAN
    assert rig.run(kind="dust") == "Hardly any dust in Abu Dhabi right now."


def test_kind_inferred_from_the_regex_group(rig):
    """The patterns name their kind via a capture group, not a parameter."""
    assert "UV index" in rig.run(uv="sunscreen")
    assert "Dust is" in rig.run(dust="dusty")


def test_missing_field_does_not_invent_a_reading(rig):
    rig.payload = {"current": {"european_aqi": 90}}
    assert rig.run(kind="uv") == "I don't have a UV reading for Abu Dhabi."


def test_network_failure_is_admitted(rig):
    rig.error = httpx.ConnectError("no route")
    assert "couldn't reach" in rig.run()


def test_garbled_response_is_admitted(rig):
    rig.payload = {"nonsense": True}
    assert "didn't understand" in rig.run()


# -- speaking up unasked --------------------------------------------------

def test_warns_once_then_stays_quiet(rig):
    """Say-once, the same contract as status.Announcer."""
    first = rig.tick()
    assert first is not None and "dust" in first.lower()

    rig._last_check = 0            # pretend the interval elapsed
    second = rig.tick()
    assert second is None or "dust" not in second.lower()


def test_rearms_after_dropping_below(rig):
    assert rig.tick() is not None          # dust warning
    rig.payload = CLEAN
    rig._last_check = 0
    assert rig.tick() is None              # clean: nothing to say, and re-armed
    rig.payload = DUSTY
    rig._last_check = 0
    assert "dust" in (rig.tick() or "").lower()


def test_respects_the_check_interval(rig):
    rig.tick()
    before = type(rig).calls
    rig.tick()
    rig.tick()
    assert type(rig).calls == before, "refetched inside the interval"


def test_uv_warns_on_its_own_after_dust_clears(rig):
    rig.tick()                             # dust fires first
    rig._last_check = 0
    said = rig.tick()                      # uv is over 8.0 too
    assert said is not None and "UV" in said


def test_a_failed_check_is_silent_not_a_spoken_error(rig):
    """Nobody wants to be told at 3am that an air API timed out."""
    rig.error = httpx.ConnectError("no route")
    assert rig.tick() is None


def test_tick_warnings_are_informational_so_quiet_hours_hold_them():
    assert SKILL.announce_urgency == "informational"


# -- match order ----------------------------------------------------------

def test_does_not_steal_from_the_other_skills():
    """air_skill sorts first alphabetically, so it matches before anything."""
    from faethon.skills.registry import Registry

    r = Registry.discover()
    assert [s.name for s in r][0] == "get_air_quality", "assumption changed"

    for phrase, owner in [
        ("what is the weather", "get_weather"),
        ("how hot is it", "get_weather"),
        ("do I need an umbrella", "get_weather"),
        ("what is your temperature", "get_health"),
        ("how are you feeling", "get_health"),
        ("what time is it", "get_time"),
        ("what is my credit balance", "get_credit_balance"),
    ]:
        hit = r.match(phrase)
        assert hit is not None and hit[0].name == owner, f"{phrase} -> {hit}"


@pytest.mark.parametrize("phrase", [
    "what's the air quality", "how's the air", "is the air ok",
    "is it dusty", "how much dust", "what's the uv", "what's the u.v. index",
    "how strong is the sun", "do i need sunscreen", "is it hazy outside",
])
def test_the_daily_phrasings_skip_the_model(phrase):
    assert SKILL.match(phrase) is not None, phrase


@pytest.mark.parametrize("phrase", [
    "the air conditioning is broken", "i saw a dust storm on the news",
    "she has an air of confidence", "sunscreen is expensive",
])
def test_does_not_fire_on_passing_mentions(phrase):
    assert SKILL.match(phrase) is None, phrase
