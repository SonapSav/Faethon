"""How Faethon reports on itself, and when it speaks up unasked.

The half that earns its keep is tick(). A CPU temperature you have to ask for
is nearly useless -- you would ask a week after the wake word started failing.
Throttling degrades this assistant specifically: openWakeWord runs
continuously, a Pi 4 soft-throttles at 80C, and the symptom is missed or late
wake words.

Under-voltage is the one worth catching. A marginal supply causes intermittent
throttling that looks exactly like a software bug, and vcgencmd latches it
since boot -- so it can be reported long after the moment that caused it, which
is the only way anyone would notice.

One skill rather than five because Registry.match is first-match-wins in import
order: separate temperature, IP, SSID and status skills would compete over the
same questions and resolve by filename.

Nothing here reads the real machine.
"""

from __future__ import annotations

import pytest

from faethon.skills import health_skill
from faethon.skills.health_skill import (
    THROTTLED_EVER,
    UNDER_VOLTAGE_EVER,
    UNDER_VOLTAGE_NOW,
    HealthSkill,
    describe_temp,
    spoken_address,
)


@pytest.fixture
def rig(monkeypatch):
    state = {"temp": 48.0, "flags": 0x0,
             "link": ("wlan0", "192.168.0.61", "HomeNet-5G"), "clock": 0.0}
    monkeypatch.setattr(health_skill, "cpu_temp", lambda: state["temp"])
    monkeypatch.setattr(health_skill, "throttle_flags", lambda: state["flags"])
    monkeypatch.setattr(health_skill, "link", lambda: state["link"])
    monkeypatch.setattr(health_skill.time, "monotonic", lambda: state["clock"])
    monkeypatch.setattr(health_skill.socket, "gethostname", lambda: "faethon",
                        raising=False)

    class Rig(HealthSkill):
        pass

    s = Rig()
    s.state = state
    return s


def say(skill, text):
    m = skill.match(text)
    assert m is not None, f"no pattern matched {text!r}"
    return skill.run(**m)


# -- speaking the numbers ----------------------------------------------------


@pytest.mark.parametrize("celsius,word", [
    (40, "normal"), (59.9, "normal"), (65, "warm"), (75, "hot"),
    (81, "throttling"), (90, "throttling"),
])
def test_the_temperature_carries_its_meaning(celsius, word):
    """A bare number alarms anyone who doesn't know Pi thresholds and
    reassures anyone who does, which makes it the wrong output."""
    assert describe_temp(celsius) == word


@pytest.mark.parametrize("ip,spoken", [
    ("192.168.0.61", "1 9 2, 1 6 8, 0, 61"),
    ("10.0.0.5", "10, 0, 0, 5"),
    ("192.168.1.100", "1 9 2, 1 6 8, 1, 1 0 0"),
])
def test_an_address_is_said_so_it_can_be_written_down(ip, spoken):
    """Auditioned through the speaker: the raw string ran 5.15s and this 4.27s,
    and this one you can actually transcribe. Three-digit octets are spelled
    out because "one hundred" is ambiguous where "1 0 0" is not."""
    assert spoken_address(ip) == spoken


def test_a_malformed_address_is_left_alone():
    assert spoken_address("not-an-ip") == "not-an-ip"


# -- the questions -----------------------------------------------------------


def test_status_gives_temperature_and_link(rig):
    said = say(rig, "what's your status")
    assert "48 degrees" in said and "HomeNet-5G" in said


@pytest.mark.parametrize("heard", ["how hot are you", "what's your temperature",
                                   "are you overheating"])
def test_a_temperature_question_gets_only_that(rig, heard):
    said = say(rig, heard)
    assert "48 degrees" in said
    assert "wi-fi" not in said


def test_the_address_leads_with_the_hostname(rig):
    """Stable, half the length spoken, and the only one anyone can reliably
    write down by ear. The IP changes; the hostname does not."""
    said = say(rig, "what's your IP")
    assert said.index("faethon dot local") < said.index("1 9 2")


def test_ethernet_is_reported_as_ethernet(rig):
    rig.state["link"] = ("eth0", "192.168.0.61", "")
    assert "ethernet" in say(rig, "what network are you on")


def test_no_network_says_so(rig):
    rig.state["link"] = ("", "", "")
    assert "don't seem to be on a network" in say(rig, "what network are you on")


def test_an_unreadable_temperature_says_so(rig, monkeypatch):
    monkeypatch.setattr(health_skill, "cpu_temp", lambda: None)
    assert "can't read my temperature" in say(rig, "what's your temperature")


@pytest.mark.parametrize("heard", ["how are you", "how are you today",
                                   "are you sure", "what's the weather"])
def test_pleasantries_are_left_to_the_model(rig, heard):
    """"How are you" is a greeting, not a status request. Answering it with a
    CPU temperature would be odd."""
    assert rig.match(heard) is None


# -- speaking up unasked -----------------------------------------------------


def test_a_healthy_pi_says_nothing(rig):
    assert rig.tick() is None


def test_under_voltage_is_reported(rig):
    """The classic Pi failure: a marginal supply or a mediocre cable, causing
    intermittent trouble that looks like a software bug."""
    rig.state["flags"] = UNDER_VOLTAGE_EVER
    said = rig.tick() or ""
    assert "under-voltage" in said and "power supply" in said


def test_a_latched_flag_counts_as_much_as_a_live_one(rig):
    """vcgencmd remembers it since boot, which is the only way anyone would
    ever catch a supply that dips under load."""
    rig.state["flags"] = UNDER_VOLTAGE_NOW
    assert rig.tick() is not None


def test_throttling_is_reported(rig):
    rig.state["flags"] = THROTTLED_EVER
    said = rig.tick() or ""
    assert "throttling" in said


def test_it_says_each_thing_once(rig):
    """Repeating it every minute would turn information into nagging, which is
    the whole reason status.py exists."""
    rig.state["flags"] = UNDER_VOLTAGE_EVER
    assert rig.tick() is not None
    for _ in range(5):
        rig.state["clock"] += 61
        assert rig.tick() is None


def test_power_is_reported_before_heat(rig):
    """Both set is the usual case, since under-voltage causes throttling. The
    supply is the cause and the actionable one."""
    rig.state["flags"] = UNDER_VOLTAGE_EVER | THROTTLED_EVER
    assert "power supply" in (rig.tick() or "")


def test_it_does_not_fork_a_process_per_frame(rig):
    """tick() runs once per 80ms audio frame while idle, and vcgencmd is a
    subprocess."""
    calls = []
    import faethon.skills.health_skill as m

    original = rig.state["flags"]
    m.throttle_flags = lambda: (calls.append(1), original)[1]
    rig.tick()
    for _ in range(100):
        rig.state["clock"] += 1
        rig.tick()
    assert len(calls) <= 3, f"forked {len(calls)} times in 100 seconds"


def test_a_pi_without_vcgencmd_is_survived(rig, monkeypatch):
    monkeypatch.setattr(health_skill, "throttle_flags", lambda: None)
    assert rig.tick() is None
    assert "48 degrees" in say(rig, "what's your temperature")
