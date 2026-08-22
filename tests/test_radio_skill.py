"""Internet radio on the other Pi, selected by frequency rather than by name.

The name approach was measured and abandoned. With `stt.language` pinned to
"en", Whisper renders Greek station names as an unpredictable mix of
transliteration and translation -- Ρυθμός came back as "rithmos" but Δρόμος
came back as "the road" and Μελωδία as "melody". No string comparison recovers
that, and no single rule covers a set that is sometimes one and sometimes the
other. Frequencies survive both languages intact.

No network here: the HTTP layer is stubbed.
"""

from __future__ import annotations

import pytest

from faethon.skills.radio_skill import (
    RadioSkill, SKILL, frequency_of, parse_frequency, say_frequency)

STATIONS = [
    {"id": 2, "name": "Happy 104", "sort_order": 1},
    {"id": 4, "name": "Ρυθμός 94,9", "sort_order": 2},
    {"id": 5, "name": "Δρόμος 89,8", "sort_order": 3},
    {"id": 7, "name": "Μέντα 88", "sort_order": 4, "last_played": "2026-08-20T10:00:00"},
    {"id": 16, "name": "Red 96,3", "sort_order": 5, "last_played": "2026-08-21T10:00:00"},
    {"id": 20, "name": "Galaxy 92", "sort_order": 6},
]
PLAYING = {"playing": True, "station": STATIONS[4], "volume": 39,
           "now_playing": "Some Artist - Some Track"}


@pytest.fixture
def rig():
    class Rig(RadioSkill):
        status_body = dict(PLAYING)
        down = False
        calls: list = []

        def _call(self, method, path, **kw):
            Rig.calls.append((method, path, kw.get("json")))
            if self.down:
                self._down_until = 1e18
                return None
            if path == "/api/stations":
                return list(STATIONS)
            if path == "/api/player/status":
                return dict(self.status_body)
            if path == "/api/player/volume":
                # A faithful fake: the real server changes what status reports,
                # and the restore logic reads it back to decide whether the
                # duck actually landed. A stub that ignored this made the
                # restore look broken when it was behaving correctly.
                # self, not Rig: a test that assigns rig.status_body creates
                # an instance attribute, and writing the class one here would
                # be shadowed by it -- the update lands somewhere nothing
                # reads. Cost an hour twice; the stub misleads more quietly
                # than the code it stands in for.
                self.status_body = dict(self.status_body, volume=kw["json"]["level"])
            return {}

    Rig.calls = []
    Rig.status_body = dict(PLAYING)
    r = Rig()
    r._config = type("C", (), {"base_url": "http://x", "timeout_seconds": 4.0,
                               "volume_step": 10, "cache_seconds": 300.0,
                               "duck": True, "duck_to": 15})()
    return r


# -- the thing that made frequencies necessary --------------------------------

@pytest.mark.parametrize("heard,want", [
    ("94.9", 94.9),          # English STT
    ("94,9", 94.9),          # Greek STT, same station
    ("Play 94.9.", 94.9),
    ("VALLE TO 94,9", 94.9),  # the Greek verb mangles; the digits do not
    ("102.2", 102.2),
    ("88", 88.0),
])
def test_frequencies_survive_both_languages(heard, want):
    assert parse_frequency(heard) == want


@pytest.mark.parametrize("text", ["5", "2026", "play 8", "set a timer for 30", ""])
def test_numbers_that_are_not_frequencies_are_refused(text):
    """The FM band check is all that separates this from every other number."""
    assert parse_frequency(text) is None


@pytest.mark.parametrize("name,want", [
    ("Ρυθμός 94,9", 94.9), ("Μέντα 88", 88.0), ("Happy 104", 104.0),
    ("Hot FM 104,6", 104.6), ("Galaxy 92", 92.0), ("No Frequency Here", None),
])
def test_frequency_read_out_of_a_station_name(name, want):
    assert frequency_of(name) == want


def test_neighbouring_frequencies_do_not_collide(rig):
    """92, 92.3 and 92.9 are three different stations."""
    assert rig.by_frequency(92.0)["id"] == 20
    assert rig.by_frequency(92.3) is None
    assert rig.by_frequency(104.0)["id"] == 2


def test_spoken_back_the_way_it_was_asked():
    assert say_frequency(94.9) == "94.9"
    assert say_frequency(88.0) == "88"


# -- doing things -------------------------------------------------------------

def test_play_by_frequency(rig):
    assert rig.run(freq="94,9") == "Playing 94.9."
    assert ("POST", "/api/player/play/4", None) in type(rig).calls


def test_an_unknown_frequency_is_refused_not_approximated(rig):
    assert "don't have a station on 99.9" in rig.run(freq="99.9")


def test_a_number_outside_the_band_says_so_rather_than_mishearing(rig):
    """"I didn't catch that" would send someone repeating a number that can
    never work."""
    said = rig.run(freq="111.1")
    assert "111.1" in said and "isn't a frequency" in said


def test_stop_and_resume(rig):
    assert rig.run(stop="radio") == "Radio off."
    rig.status_body = {"playing": False, "station": STATIONS[4], "volume": 39}
    assert rig.run(resume="radio") == "Playing 96.3."


def test_resume_says_so_when_it_is_already_on(rig):
    assert "already on" in rig.run(resume="radio")


def test_next_wraps_around(rig):
    rig.status_body = dict(PLAYING, station=STATIONS[-1])   # last in sort order
    assert rig.run(step="next") == "Playing 104."           # back to the first


def test_previous_from_the_first_wraps_to_the_last(rig):
    rig.status_body = dict(PLAYING, station=STATIONS[0])
    assert rig.run(step="previous") == "Playing 92."


def test_volume_is_clamped(rig):
    assert rig.run(level=140) == "Radio volume is set to 100%."
    assert rig.run(level=-5) == "Radio volume is set to 0%, muted."


def test_volume_is_a_level_not_a_percentage(rig):
    """"Radio volume 5" means half, the same as it does for Faethon's own."""
    assert rig.run(level=5) == "Radio volume is set to 50%."
    assert type(rig).calls[-1] == ("POST", "/api/player/volume", {"level": 50})


def test_a_percentage_said_back_is_still_understood(rig):
    """Announcing "50%" teaches people to say "50" -- read as a level that
    would clamp to maximum, which is how volume_skill got this wrong first."""
    assert rig.run(level=50) == "Radio volume is set to 50%."
    assert rig.run(level=50, unit="%") == "Radio volume is set to 50%."
    assert rig.run(level=50, unit="percent") == "Radio volume is set to 50%."


def test_volume_nudges_by_one_level(rig):
    """From 39, which is level 4: up lands on 50, not 49."""
    assert rig.run(vol="up") == "Radio volume is set to 50%."
    assert rig.run(vol="down") == "Radio volume is set to 40%."


def test_a_nudge_lands_on_a_level_boundary(rig):
    """A radio left at 45 from somebody's phone should not stay between two
    steps -- it goes to 60, not 55."""
    rig.status_body = dict(PLAYING, volume=45)
    assert rig.run(vol="up") == "Radio volume is set to 60%."


def test_nudging_down_from_the_bottom_mutes_rather_than_wrapping(rig):
    rig.status_body = dict(PLAYING, volume=0)
    assert rig.run(vol="down") == "Radio volume is set to 0%, muted."


def test_nudging_up_from_the_top_stays_at_the_top(rig):
    rig.status_body = dict(PLAYING, volume=100)
    assert rig.run(vol="up") == "Radio volume is set to 100%."


def test_now_playing_includes_the_track(rig):
    assert rig.run() == "96.3 is playing Some Artist - Some Track."


def test_now_playing_when_off(rig):
    rig.status_body = {"playing": False}
    assert rig.run() == "The radio is off."


def test_a_name_still_works_through_the_model(rig):
    """The Latin-script twelve, and anything the model resolves for us."""
    assert rig.run(station="Happy") == "Playing 104."
    assert "don't have a station called" in rig.run(station="Nonexistent FM")


# -- when the other Pi is off -------------------------------------------------

def test_an_unreachable_host_is_admitted(rig):
    rig.down = True
    assert rig.run(freq="94.9") == "I can't reach the radio right now."


def test_it_stops_retrying_a_dead_host(rig):
    """A timeout per turn is a timeout inside a conversation."""
    rig.down = True
    rig.run(freq="94.9")
    assert rig.available is False
    before = len(type(rig).calls)
    rig.run(freq="94.9")
    assert len(type(rig).calls) == before, "retried a host known to be down"


def test_the_station_list_is_cached(rig):
    rig.stations()
    before = len(type(rig).calls)
    rig.stations()
    rig.stations()
    assert len(type(rig).calls) == before


# -- it sorts before set_volume and set_timer, so this is not decoration -------

RADIO_PHRASES = [
    "play 94.9", "play 94,9", "put on 102.2", "tune to 88", "switch to 96,3",
    "play the radio", "put the radio on", "turn on the radio",
    "stop the radio", "turn off the radio", "next station", "previous station",
    "turn the radio up", "turn up the radio", "turn the radio down",
    "radio volume 50", "set the radio volume to 30",
    "what's playing", "what station is this",
]


@pytest.mark.parametrize("phrase", RADIO_PHRASES)
def test_radio_phrases_reach_the_skill(phrase):
    assert SKILL.match(phrase) is not None, phrase


NOT_RADIO = [
    "turn it up", "turn it down", "volume up", "mute", "set the volume to 5",
    "what time is it", "set a timer for 5 minutes", "stop the timer",
    "restart yourself", "how much time is left on the timer",
    "what is the air quality", "what is my budget",
]


@pytest.mark.parametrize("phrase", NOT_RADIO)
def test_it_steals_nothing_from_the_skills_it_outranks(phrase):
    from faethon.skills.registry import Registry

    hit = Registry.discover().match(phrase)
    assert hit is not None, phrase
    assert hit[0].name != "control_radio", f"{phrase!r} was stolen by the radio"


def test_an_unreachable_host_is_not_reported_as_a_missing_station(rig):
    """The bug this file caught.

    With the list unfetchable, "I don't have a station on 94.9" is a confident
    wrong answer -- it sends someone looking for a station that is really
    there. Absence of the list is not absence of the station.
    """
    rig.down = True
    for said in (rig.run(freq="94.9"), rig.run(station="Happy")):
        assert said == "I can't reach the radio right now."
        assert "don't have" not in said


# -- listing what is on the dial ----------------------------------------------
# Asked for explicitly so the list can be checked after adding or removing a
# station on the other Pi. Which is exactly why it is never cached: a cached
# answer to "what stations do you have" is the one answer never worth giving.


def test_listing_is_fetched_fresh_every_time(rig):
    rig.stations()                                   # warm the cache
    before = len(type(rig).calls)
    rig.run(list="")
    rig.run(list="")
    fetches = [c for c in type(rig).calls[before:] if c[1] == "/api/stations"]
    assert len(fetches) == 2, "a cached station list defeats the point of asking"


def test_listing_reads_frequencies_ascending(rig):
    said = rig.run(list="")
    assert said == "I have 6 stations: 88, 89.8, 92, 94.9, 96.3, 104."


def test_listing_refreshes_the_cache_for_the_next_selection(rig):
    """Add a station, ask what there is, then play it -- without a restart."""
    rig.run(list="")
    assert rig.by_frequency(94.9) is not None


def test_a_long_list_summarises_rather_than_reciting(rig, monkeypatch):
    """Nineteen frequencies already take 29.6s to say. Twice that stops being
    an answer."""
    from faethon.skills import radio_skill

    many = [{"id": i, "name": f"Station {80 + i // 10},{i % 10}", "sort_order": i}
            for i in range(60)]
    monkeypatch.setattr(rig, "stations", lambda fresh=False: many)
    said = rig.run(list="")
    assert said.startswith("I have ")
    assert "more." in said
    assert said.count(",") <= radio_skill.MAX_SPOKEN + 2


def test_listing_admits_an_unreachable_host(rig):
    rig.down = True
    assert rig.run(list="") == "I can't reach the radio right now."


def test_stations_without_a_frequency_do_not_break_the_list(rig, monkeypatch):
    monkeypatch.setattr(rig, "stations",
                        lambda fresh=False: [{"id": 1, "name": "Some Station"}])
    assert "none of them list a frequency" in rig.run(list="")


@pytest.mark.parametrize("phrase", [
    "what stations do you have",
    "what stations are available",
    "what stations have you got",
    "which stations can i pick",
    "list the stations",
    "tell me the stations",
    "list all the radio stations",
    "what's the station list",
])
def test_listing_phrases_reach_the_list_not_now_playing(phrase):
    params = SKILL.match(phrase)
    assert params is not None, phrase
    assert "list" in params, f"{phrase!r} would answer with what's playing"


@pytest.mark.parametrize("phrase", ["what's playing", "what station is this"])
def test_now_playing_is_still_a_different_question(phrase):
    params = SKILL.match(phrase)
    assert params is not None and "list" not in params


# -- ducking ------------------------------------------------------------------
# Turning the radio down while Faethon holds the floor. Not for the microphone
# -- measured, the radio reaches it at about -88 dBFS -- but so a person can
# hear the reply over the music.


def _settle(rig):
    t = rig._duck_thread
    if t:
        t.join(timeout=5)


def test_ducking_lowers_and_restores(rig):
    with rig.ducked():
        _settle(rig)
        assert ("POST", "/api/player/volume", {"level": 15}) in type(rig).calls
    assert ("POST", "/api/player/volume", {"level": 39}) in type(rig).calls


def test_the_duck_never_blocks_the_turn(rig):
    """Measured blocking at a worst case of 4094ms -- four seconds of silence
    before the "go ahead" chime, with someone standing there waiting."""
    import time

    t0 = time.monotonic()
    rig._duck()
    blocked = time.monotonic() - t0
    _settle(rig)
    assert blocked < 0.05, f"duck blocked the turn for {blocked*1000:.0f}ms"


def test_the_duck_is_recorded_before_the_request(rig, tmp_path):
    """A POST that times out may still have arrived.

    That is exactly how the radio ended up stuck at 15 with nothing recorded
    to restore it from: the write happened after the call, so a timeout lost
    the only means of undoing it.
    """
    from faethon import state

    seen = {}

    def failing(method, path, **kw):
        if path == "/api/player/status":
            return dict(PLAYING)
        seen["state_at_post_time"] = state.load("radio_duck", None)
        return None                       # as if it timed out

    rig._call = failing
    rig._duck()
    _settle(rig)
    assert seen["state_at_post_time"] == {"was": 39, "set": 15}


def test_a_duck_that_never_landed_is_not_restored(rig):
    """If the volume is not what we set, the change did not happen -- putting
    the radio somewhere it never was would be worse than doing nothing."""
    from faethon import state

    state.save("radio_duck", {"was": 39, "set": 15})
    rig.status_body = dict(PLAYING, volume=39)      # never actually ducked
    before = len([c for c in type(rig).calls if c[1] == "/api/player/volume"])
    rig._unduck()
    after = len([c for c in type(rig).calls if c[1] == "/api/player/volume"])
    assert after == before, "restored a duck that never happened"
    assert state.load("radio_duck", None) is None


def test_a_volume_you_set_mid_conversation_wins(rig):
    """Your instruction is the more recent one."""
    from faethon import state

    state.save("radio_duck", {"was": 39, "set": 15})
    rig.status_body = dict(PLAYING, volume=55)      # user said "radio volume 55"
    rig._unduck()
    assert ("POST", "/api/player/volume", {"level": 39}) not in type(rig).calls


def test_nothing_is_ducked_when_the_radio_is_off(rig):
    rig.status_body = {"playing": False}
    rig._duck()
    _settle(rig)
    assert not [c for c in type(rig).calls if c[1] == "/api/player/volume"]


def test_a_radio_already_quieter_is_left_alone(rig):
    rig.status_body = dict(PLAYING, volume=10)
    rig._duck()
    _settle(rig)
    assert not [c for c in type(rig).calls if c[1] == "/api/player/volume"]


def test_ducking_can_be_switched_off(rig):
    rig._config.duck = False
    rig._duck()
    _settle(rig)
    assert not [c for c in type(rig).calls if c[1] == "/api/player/volume"]


def test_a_crash_mid_turn_is_repaired_at_startup(rig):
    """Otherwise the radio sits at 15 until somebody reaches for their phone,
    and nobody would connect a quiet radio to an assistant that died an hour
    ago."""
    from faethon import state

    state.save("radio_duck", {"was": 39, "set": 15})
    rig.status_body = dict(PLAYING, volume=15)
    rig.restore_after_crash()
    assert ("POST", "/api/player/volume", {"level": 39}) in type(rig).calls
    assert state.load("radio_duck", None) is None


def test_ducking_never_raises(rig):
    """A courtesy that can break a turn is worse than no courtesy."""
    def explode(*a, **kw):
        raise RuntimeError("radiohost fell over")

    rig._call = explode
    with rig.ducked():
        _settle(rig)
    # reaching here without an exception is the assertion


# -- changing the volume during a conversation --------------------------------
# Both of these were bugs, and they share a cause: the live reading during a
# turn is the *ducked* volume, not the listener's. Anything that acts on it
# gets the wrong number and cancels the duck by writing over it.


def test_setting_a_volume_while_ducked_does_not_cancel_the_duck(rig):
    """Otherwise the confirmation plays over a radio back at full volume --
    the one thing ducking exists to prevent, defeated by the one command that
    is actually about volume."""
    with rig.ducked():
        _settle(rig)
        assert rig.run(level=5) == "Radio volume is set to 50%."
        assert rig.status_body["volume"] == 15, "jumped to the new level mid-reply"
    assert rig.status_body["volume"] == 50, "did not land on the new level"


def test_nudging_while_ducked_steps_from_the_listeners_level(rig):
    """From a real 60, "up" is 70. Reading the ducked 15 instead made it 30
    and lost the setting entirely -- the worse of the two, because nothing
    about it looks wrong until you notice the radio is quiet."""
    rig.status_body = dict(PLAYING, volume=60)
    with rig.ducked():
        _settle(rig)
        assert rig.run(vol="up") == "Radio volume is set to 70%."
        assert rig.status_body["volume"] == 15
    assert rig.status_body["volume"] == 70


def test_nudging_down_while_ducked(rig):
    rig.status_body = dict(PLAYING, volume=60)
    with rig.ducked():
        _settle(rig)
        assert rig.run(vol="down") == "Radio volume is set to 50%."
    assert rig.status_body["volume"] == 50


def test_outside_a_conversation_the_volume_changes_immediately(rig):
    """No duck in force, so nothing is deferred."""
    assert rig.run(level=4) == "Radio volume is set to 40%."
    assert rig.status_body["volume"] == 40


def test_a_deferred_volume_survives_several_changes_in_one_turn(rig):
    """The last one said is the one that lands."""
    with rig.ducked():
        _settle(rig)
        rig.run(level=5)
        rig.run(level=8)
        rig.run(vol="down")
        assert rig.status_body["volume"] == 15
    assert rig.status_body["volume"] == 70
