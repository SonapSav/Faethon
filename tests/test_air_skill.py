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

import time

import httpx
import pytest

from faethon import state
from faethon.skills.air_skill import (
    SKILL, AirSkill, daily_peaks, day_name, describe_air, describe_uv)

# The live reading on the afternoon this was built: a real dust event.
DUSTY = {
    "current": {"pm10": 222.1, "pm2_5": 62.4, "dust": 310.0,
                "uv_index": 8.65, "european_aqi": 90},
}
CLEAN = {
    "current": {"pm10": 18.0, "pm2_5": 6.0, "dust": 4.0,
                "uv_index": 2.1, "european_aqi": 15},
}

# Dust over the line, UV under it. tick() says at most one thing per check, so
# with both over threshold a restart legitimately announces the UV it never got
# to -- true, but it makes "did the dust repeat?" ambiguous to assert.
DUSTY_ONLY = {"current": dict(DUSTY["current"], uv_index=5.0)}

# The measured decay of the real event: 321, 307, 219, 132 over four days.
DAYS = ["2026-08-21", "2026-08-22", "2026-08-23", "2026-08-24"]
FORECAST = dict(
    DUSTY,
    daily={"time": DAYS, "uv_index_max": [9.1, 9.25, 8.55, 9.05]},
    hourly={
        "time": [f"{d}T{h:02d}:00" for d in DAYS for h in (0, 12)],
        "dust": [160, 321, 150, 307, 110, 219, 60, 132],
        "european_aqi": [70, 95, 72, 99, 60, 94, 45, 80],
    },
)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    class Rig(AirSkill):
        payload = DUSTY
        error: Exception | None = None
        calls = 0

        def _get(self, url, params):
            Rig.calls += 1
            if self.error is not None:
                raise self.error
            return self.payload

    # Never the real /var/lib/faethon: these tests write warning flags.
    monkeypatch.setattr(state, "state_dir", lambda: tmp_path)

    rig = Rig()
    rig._air_config = type("C", (), {
        "dust_warn": 200.0, "uv_warn": 8.0, "check_every_minutes": 30.0,
        "stale_after_hours": 6.0})()
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


def test_a_tick_leaves_the_forecast_intact(rig):
    """The bug this file missed the first time.

    tick() rebuilt the cache from only the current block, so every forecast
    question afterwards answered "I don't have a forecast" -- and tick runs at
    startup and every half hour, so that was nearly always the cache a question
    found. Today kept working, which is what made it look like a forecast
    problem rather than a caching one.
    """
    rig.payload = FORECAST
    rig.tick()
    assert set(rig._cache[1]) >= {"current", "hourly", "daily"}, "cache truncated"
    assert "307" in rig.run(kind="dust", when="tomorrow")
    assert "ease off" in rig.run(when="trend")


def test_a_tick_primes_the_cache_rather_than_costing_a_second_call(rig):
    rig.payload = FORECAST
    rig.tick()
    before = type(rig).calls
    rig.run()
    rig.run(when="tomorrow")
    assert type(rig).calls == before, "refetched what tick had already fetched"


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


# -- the next few days ----------------------------------------------------

def test_daily_peaks_takes_the_peak_not_the_mean():
    """Whether it gets bad at all beats what it averaged overnight."""
    peaks = daily_peaks(FORECAST["hourly"])
    assert peaks["2026-08-21"]["dust"] == 321
    assert peaks["2026-08-24"]["dust"] == 132


def test_day_names_come_from_the_api_date_not_the_pi_clock():
    """The Pi has no RTC; a weekday is what it would get wrong after a boot."""
    assert day_name("2026-08-21", 0) == "today"
    assert day_name("2026-08-22", 1) == "tomorrow"
    assert day_name("2026-08-24", 3) == "Monday"
    assert day_name("not-a-date", 2) == "in 2 days"


def test_tomorrow_uses_the_forecast_not_the_current_reading(rig):
    rig.payload = FORECAST
    said = rig.run(kind="dust", when="tomorrow")
    assert "307" in said and "tomorrow" in said
    assert "310" not in said, "answered from the current reading"


def test_tomorrow_air_and_uv(rig):
    rig.payload = FORECAST
    assert rig.run(when="tomorrow") == "The air should be very poor in Abu Dhabi tomorrow."
    assert "9" in rig.run(kind="uv", when="tomorrow")


def test_uv_tomorrow_uses_the_native_daily_max(rig):
    """UV is the one field the API aggregates itself; do not re-derive it."""
    rig.payload = dict(FORECAST, daily={"time": DAYS, "uv_index_max": [1, 11.4, 3, 3]})
    assert "extreme" in rig.run(kind="uv", when="tomorrow")


def test_trend_gives_a_direction_and_a_day(rig):
    rig.payload = FORECAST
    said = rig.run(when="trend")
    assert "ease off" in said and "321" in said and "132" in said and "Monday" in said


def test_trend_notices_it_getting_worse(rig):
    rig.payload = dict(FORECAST, hourly=dict(
        FORECAST["hourly"], dust=[50, 60, 80, 100, 150, 200, 260, 300]))
    said = rig.run(when="trend")
    assert "worse" in said


def test_trend_says_so_when_nothing_changes(rig):
    rig.payload = dict(FORECAST, hourly=dict(
        FORECAST["hourly"], dust=[300, 310, 300, 305, 300, 308, 300, 302]))
    assert "Not much change" in rig.run(when="trend")


def test_trend_on_the_air_is_read_as_dust(rig):
    """"Is the air going to clear" is a dust question in this climate."""
    rig.payload = FORECAST
    assert rig.run(trend="air") == rig.run(trend="dust")


def test_missing_forecast_is_admitted_not_invented(rig):
    """The stub payloads carry no hourly block; that must not become a number."""
    rig.payload = DUSTY
    assert "don't have a forecast" in rig.run(when="tomorrow")
    assert "don't have enough forecast" in rig.run(when="trend")


def test_a_flat_zero_baseline_does_not_divide_by_zero(rig):
    rig.payload = dict(FORECAST, hourly=dict(
        FORECAST["hourly"], dust=[0, 0, 0, 0, 0, 0, 0, 0]))
    assert "Not much change" in rig.run(when="trend")


@pytest.mark.parametrize("phrase,expected", [
    ("is the dust going to clear", "trend"),
    ("will the dust clear", "trend"),
    ("is the air getting better", "trend"),
    ("is the dust dying down", "trend"),
    ("will it be dusty tomorrow", "tomorrow"),
    ("what's the air like tomorrow", "tomorrow"),
    ("what's the uv tomorrow", "tomorrow"),
])
def test_forecast_phrasings_skip_the_model(phrase, expected):
    params = SKILL.match(phrase)
    assert params is not None, phrase
    assert expected in params, f"{phrase} -> {params}"


def test_today_is_still_the_default(rig):
    rig.payload = FORECAST
    assert rig.run() == "The air is very poor in Abu Dhabi, mostly dust."


# -- surviving a restart ------------------------------------------------------
# The warning is say-once, but the flag lived only in memory, so every restart
# announced the same dust again. Faethon restarts on a voice command, which
# made this easy to trigger and irritating in exactly the conditions it fires.


def restarted(rig):
    """A fresh instance, as a restart produces, sharing the same state dir."""
    fresh = type(rig)()
    fresh._air_config = rig._air_config
    fresh._config = rig._config
    return fresh


def test_a_restart_does_not_repeat_the_warning(rig):
    rig.payload = DUSTY_ONLY
    assert "dust" in (rig.tick() or "").lower()

    again = restarted(rig)
    again.payload = DUSTY_ONLY
    assert again.tick() is None, "repeated the warning after a restart"


def test_a_long_gap_forgets_because_the_episode_may_have_ended(rig, monkeypatch):
    """An episode ends when the reading drops, which Faethon only sees while
    running. After a gap it cannot know that did not happen."""
    from faethon import clock

    monkeypatch.setattr(clock, "is_synced", lambda: True)
    rig.tick()

    seven_hours = time.time() + 7 * 3600
    monkeypatch.setattr(time, "time", lambda: seven_hours)
    again = restarted(rig)
    assert "dust" in (again.tick() or "").lower(), "stayed silent across a long gap"


def test_a_short_gap_stays_silent(rig, monkeypatch):
    from faethon import clock

    monkeypatch.setattr(clock, "is_synced", lambda: True)
    rig.payload = DUSTY_ONLY
    rig.tick()

    soon = time.time() + 600
    monkeypatch.setattr(time, "time", lambda: soon)
    again = restarted(rig)
    again.payload = DUSTY_ONLY
    assert again.tick() is None


def test_an_unsynced_clock_keeps_quiet_rather_than_repeating(rig, monkeypatch):
    """The Pi has no RTC. Wrong in the silent direction beats wrong in the
    direction the user just asked to be rid of."""
    from faethon import clock

    monkeypatch.setattr(clock, "is_synced", lambda: False)
    rig.payload = DUSTY_ONLY
    rig.tick()
    much_later = time.time() + 30 * 3600
    monkeypatch.setattr(time, "time", lambda: much_later)
    again = restarted(rig)
    again.payload = DUSTY_ONLY
    assert again.tick() is None


def test_clearing_persists_so_the_next_episode_is_announced(rig):
    rig.tick()                                   # warned
    rig.payload = CLEAN
    rig._last_check = 0
    assert rig.tick() is None                    # dropped back, flag cleared

    again = restarted(rig)
    again.payload = DUSTY
    assert "dust" in (again.tick() or "").lower(), "new episode went unannounced"


def test_the_stamp_refreshes_even_when_nothing_is_said(rig, monkeypatch):
    """Otherwise a quiet week of good air would look like a week of not
    looking, and the flags would expire for the wrong reason."""
    from faethon import clock

    monkeypatch.setattr(clock, "is_synced", lambda: True)
    rig.tick()
    saved = state.load("air_warnings", {})
    first = saved["at"]

    later = time.time() + 3600
    monkeypatch.setattr(time, "time", lambda: later)
    rig._last_check = 0
    rig.tick()
    assert state.load("air_warnings", {})["at"] > first


def test_a_corrupt_state_file_does_not_stop_it(rig, tmp_path):
    (tmp_path / "air_warnings.json").write_text("{ not json")
    assert "dust" in (rig.tick() or "").lower()


def test_a_stamp_from_the_future_is_not_trusted(rig, monkeypatch):
    """A clock corrected backwards would otherwise look freshly observed
    forever, silencing the warning permanently."""
    from faethon import clock

    monkeypatch.setattr(clock, "is_synced", lambda: True)
    rig.tick()
    backwards = time.time() - 48 * 3600
    monkeypatch.setattr(time, "time", lambda: backwards)
    assert "dust" in (restarted(rig).tick() or "").lower()


def test_a_failed_check_does_not_refresh_the_stamp(rig, monkeypatch):
    """A stamp is a claim to have seen a reading; a timeout saw nothing."""
    from faethon import clock

    monkeypatch.setattr(clock, "is_synced", lambda: True)
    rig.tick()
    before = state.load("air_warnings", {})["at"]

    rig.error = httpx.ConnectError("no route")
    monkeypatch.setattr(time, "time", lambda: before + 3600)
    rig._last_check = 0
    rig.tick()
    assert state.load("air_warnings", {})["at"] == before


def test_a_restart_delivers_the_other_warning_rather_than_repeating(rig):
    """With dust and UV both over the line, a check says only one thing.

    So a restart is not silent -- it announces the UV it never reached. That
    is the point: the flag suppresses what was already said, not everything.
    """
    first = rig.tick()
    assert "dust" in (first or "").lower()

    again = restarted(rig)
    second = again.tick()
    assert second is not None and "UV" in second
    assert "dust" not in second.lower()

    third = restarted(again)
    assert third.tick() is None, "nothing left to say, yet it said something"


def test_an_unknown_when_answers_now_rather_than_tomorrow(rig):
    """Found by asking in Greek: the model sent when="now", outside the enum.

    The dispatch tested for "today" and "trend" and let everything else fall
    into tomorrow, so a question about right now was answered with tomorrow's
    forecast -- and worded "should be", which reads as a forecast without
    admitting it answered a different question.
    """
    rig.payload = FORECAST
    now = rig.run(when="now")
    assert now == rig.run(), "when='now' did not answer about now"
    assert "tomorrow" not in now

    for odd in ("right now", "currently", "", "TODAY", "nonsense"):
        assert "tomorrow" not in rig.run(when=odd), f"when={odd!r} drifted to tomorrow"


def test_the_real_values_still_dispatch(rig):
    rig.payload = FORECAST
    assert "tomorrow" in rig.run(when="tomorrow")
    assert "ease off" in rig.run(when="trend")
