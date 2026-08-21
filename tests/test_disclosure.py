"""The ledger of what leaves this machine.

Counted at the HTTP layer because the turn log understates it by about half:
127 transcripts came back in a day against 74 logged turns, since one exchange
is three or four requests. A ledger of conversations is not a ledger of
disclosures.
"""

from __future__ import annotations

import json

import pytest

from faethon import disclosure
from faethon.disclosure import Ledger, summarise
from faethon.skills.disclosure_skill import SKILL, _minutes


@pytest.fixture
def ledger(tmp_path):
    return Ledger()


def test_kinds_are_derived_from_the_endpoint():
    assert disclosure.kind_for("/audio/transcriptions") == disclosure.VOICE
    assert disclosure.kind_for("/audio/speech") == disclosure.TEXT
    assert disclosure.kind_for("/chat/completions") == disclosure.TEXT
    assert disclosure.kind_for("/credits") == disclosure.ACCOUNT


def test_an_unknown_endpoint_is_assumed_to_carry_text():
    """Overstating a disclosure beats silently omitting a new endpoint."""
    assert disclosure.kind_for("/some/future/thing") == disclosure.TEXT


def test_records_survive_a_round_trip(ledger):
    ledger.record("openrouter.ai", "/audio/speech", disclosure.TEXT)
    rows = ledger.read()
    assert len(rows) == 1
    assert rows[0]["host"] == "openrouter.ai"
    assert rows[0]["kind"] == disclosure.TEXT
    assert rows[0]["asked"] is True


def test_background_calls_are_marked_unasked(ledger):
    ledger.asked = False
    ledger.record("air-quality-api.open-meteo.com", "/v1/air-quality",
                  disclosure.LOCATION)
    ledger.asked = True
    ledger.record("openrouter.ai", "/credits", disclosure.ACCOUNT)
    s = summarise(ledger.read())
    assert s["unasked"] == 1, "the tick that ran with nobody home was not flagged"


def test_withheld_is_recorded_because_it_leaves_no_other_trace(ledger):
    """A follow-up window that hears nothing makes no request at all, so a
    ledger of requests can never show what was declined."""
    ledger.withheld()
    s = summarise(ledger.read())
    assert s["calls"] == 0, "silence counted as something sent"
    assert s["withheld"] == 1


def test_a_corrupt_line_is_skipped_not_fatal(ledger, tmp_path):
    ledger.record("openrouter.ai", "/credits", disclosure.ACCOUNT)
    with (tmp_path / "disclosures.jsonl").open("a") as f:
        f.write("{ not json\n")
    ledger.record("openrouter.ai", "/credits", disclosure.ACCOUNT)
    assert len(ledger.read()) == 2


def test_a_disabled_ledger_writes_nothing(tmp_path):
    off = Ledger(enabled=False)
    off.record("openrouter.ai", "/audio/speech", disclosure.TEXT)
    assert off.read() == []


def test_it_never_raises(monkeypatch, ledger):
    """A ledger that can break a turn is worse than no ledger."""
    monkeypatch.setattr(disclosure.state, "state_dir",
                        lambda: (_ for _ in ()).throw(OSError("read-only")))
    ledger.record("openrouter.ai", "/audio/speech", disclosure.TEXT)   # must not raise


def test_no_payloads_are_stored(ledger):
    """A ledger able to recite your conversations would be storing them."""
    ledger.record("openrouter.ai", "/chat/completions", disclosure.TEXT)
    row = ledger.read()[0]
    assert set(row) == {"at", "host", "path", "kind", "asked"}


# -- what the client does -----------------------------------------------------

def test_the_client_records_every_attempt_not_every_success(tmp_path, monkeypatch):
    """A retried upload crossed the wire twice; a timeout still sent its body."""
    import httpx
    from faethon.providers.client import OpenRouterClient

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"text": "ok"})

    c = OpenRouterClient("key")
    c._client = httpx.Client(base_url="https://openrouter.ai/api/v1",
                             transport=httpx.MockTransport(handler))
    monkeypatch.setattr("faethon.providers.client.time.sleep", lambda *_: None)
    c.post_json("/audio/transcriptions", {})

    rows = disclosure.LEDGER.read()
    sent = [r for r in rows if r["path"].endswith("/audio/transcriptions")]
    assert len(sent) == 2, "a retried request was only counted once"
    assert all(r["kind"] == disclosure.VOICE for r in sent)


# -- the spoken answer --------------------------------------------------------

def test_durations_are_spoken_not_counted():
    assert _minutes(45) == "45 seconds"
    assert _minutes(150) == "2.5 minutes"
    assert _minutes(900) == "15 minutes"


def test_it_leads_with_audio_and_consent(tmp_path):
    for _ in range(3):
        disclosure.LEDGER.record("openrouter.ai", "/audio/transcriptions",
                                 disclosure.VOICE)
    disclosure.LEDGER.asked = False
    disclosure.LEDGER.record("air-quality-api.open-meteo.com", "/v1/air-quality",
                             disclosure.LOCATION)
    disclosure.LEDGER.asked = True
    disclosure.LEDGER.withheld()

    said = SKILL.run()
    assert "3 of them carrying audio from this room" in said
    assert "1 with nobody asking" in said
    assert "location went to the weather service once" in said
    assert "1 stretch of silence and sent nothing" in said


def test_an_empty_ledger_says_so_rather_than_claiming_nothing_happened(tmp_path):
    assert "record" in SKILL.run()


@pytest.mark.parametrize("phrase", [
    "what did you send to the cloud today",
    "what did you send",
    "what have you sent today",
    "what data did you send",
    "who did you talk to",
    "what leaves this house",
    "how much audio did you send",
    "are you recording me",
])
def test_the_question_reaches_the_skill(phrase):
    assert SKILL.match(phrase) is not None, phrase


@pytest.mark.parametrize("phrase", [
    "what did you say",
    "send a message to john",
    "what time is it",
])
def test_it_does_not_overreach(phrase):
    assert SKILL.match(phrase) is None, phrase


# -- it is read aloud, so the grammar is part of the answer --------------------

def _fake(monkeypatch, calls=0, voice=0, location=0, unasked=0, withheld=0):
    rows = ([{"kind": disclosure.VOICE, "host": "o", "asked": True}] * voice
            + [{"kind": disclosure.TEXT, "host": "o", "asked": True}] * calls
            + [{"kind": disclosure.LOCATION, "host": "m", "asked": False}] * location
            + [{"kind": disclosure.TEXT, "host": "o", "asked": False}] * unasked
            + [{"kind": "withheld", "host": "", "asked": True}] * withheld)
    monkeypatch.setattr(disclosure.LEDGER, "read", lambda since_seconds=None: list(rows))


def test_singulars_are_not_spoken_as_plurals(monkeypatch):
    """One request, one silence -- every count in the sentence is singular."""
    _fake(monkeypatch, calls=1, withheld=1)
    said = SKILL.run()
    assert "1 request went out" in said
    assert "requests" not in said
    assert "1 stretch of silence" in said
    assert "stretc " not in said, "the plural was built by slicing a string"
    assert "sent nothing" in said


def test_a_single_location_call_says_once(monkeypatch):
    _fake(monkeypatch, location=1)
    said = SKILL.run()
    assert "weather service once." in said
    assert "1 times" not in said


def test_plurals_are_spoken_as_plurals(monkeypatch):
    _fake(monkeypatch, calls=4, location=3, withheld=5)
    said = SKILL.run()
    assert "7 requests went out" in said
    assert "3 times" in said
    assert "5 stretches of silence" in said
    assert "none of them" in said
