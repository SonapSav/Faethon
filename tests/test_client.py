"""The shared HTTP client: retries, backoff, errors, cost.

Every leg of the pipeline goes through this, and none of it was covered. It is
also exactly the kind of code whose bugs are invisible: a retry loop that gives
up one attempt early, or a backoff that sleeps for an hour, produces a turn
that failed for no visible reason rather than a stack trace.

Driven through httpx's MockTransport, so the real retry loop runs against
scripted responses and no socket is opened. time.sleep is neutered -- the
delays are asserted on directly instead of waited out.
"""

from __future__ import annotations

import httpx
import pytest

from faethon.providers import client as client_mod
from faethon.providers.client import (
    BACKOFF_BASE,
    MAX_ATTEMPTS,
    OpenRouterClient,
    OpenRouterError,
)


@pytest.fixture
def rig(monkeypatch):
    """A client whose transport is scripted and whose sleeps are recorded."""
    slept: list[float] = []
    monkeypatch.setattr(client_mod.time, "sleep", slept.append)

    class Rig:
        responses: list = []
        requests: list[httpx.Request] = []

        def handler(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            nxt = self.responses.pop(0) if self.responses else httpx.Response(200, json={})
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        @property
        def slept(self) -> list[float]:
            return slept

        @property
        def attempts(self) -> int:
            return len(self.requests)

    r = Rig()
    r.requests = []
    r.responses = []
    c = OpenRouterClient("sk-or-test")
    c._client = httpx.Client(
        base_url=client_mod.BASE_URL, transport=httpx.MockTransport(r.handler)
    )
    r.client = c
    return r


# -- the happy path ----------------------------------------------------------


def test_a_json_post_returns_the_parsed_body(rig):
    rig.responses = [httpx.Response(200, json={"text": "hello"})]
    assert rig.client.post_json("/audio/transcriptions", {"a": 1}) == {"text": "hello"}
    assert rig.attempts == 1
    assert rig.slept == []


def test_get_and_post_use_the_right_method(rig):
    rig.responses = [httpx.Response(200, json={}), httpx.Response(200, json={})]
    rig.client.get_json("/credits")
    rig.client.post_json("/chat/completions", {})
    assert [r.method for r in rig.requests] == ["GET", "POST"]


def test_the_api_key_is_sent(rig):
    rig.responses = [httpx.Response(200, json={})]
    rig.client.post_json("/x", {})
    # The header lives on the client the fixture replaced, so assert on the
    # real one rather than the mock's.
    real = OpenRouterClient("sk-or-secret")
    assert real._client.headers["Authorization"] == "Bearer sk-or-secret"
    real.close()


def test_a_missing_key_fails_loudly_at_construction():
    """Better here than as a 401 on the first thing anyone says."""
    with pytest.raises(OpenRouterError, match="No OpenRouter API key"):
        OpenRouterClient("")


# -- retries -----------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_failures_are_retried(rig, status):
    rig.responses = [httpx.Response(status), httpx.Response(200, json={"ok": True})]
    assert rig.client.post_json("/x", {}) == {"ok": True}
    assert rig.attempts == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_failures_are_not_retried(rig, status):
    """Retrying a bad request just spends the user's time twice."""
    rig.responses = [httpx.Response(status, text="nope")]
    with pytest.raises(OpenRouterError):
        rig.client.post_json("/x", {})
    assert rig.attempts == 1


def test_a_connection_error_is_retried_then_reported(rig):
    rig.responses = [httpx.ConnectError("unreachable")] * MAX_ATTEMPTS
    with pytest.raises(OpenRouterError, match="failed after"):
        rig.client.post_json("/x", {})
    assert rig.attempts == MAX_ATTEMPTS


def test_it_gives_up_rather_than_retrying_forever(rig):
    rig.responses = [httpx.Response(503)] * (MAX_ATTEMPTS + 5)
    with pytest.raises(OpenRouterError):
        rig.client.post_json("/x", {})
    assert rig.attempts == MAX_ATTEMPTS


def test_backoff_grows_between_attempts(rig):
    """Hammering a rate limit at a fixed interval is how you stay limited."""
    rig.responses = [httpx.Response(429), httpx.Response(429), httpx.Response(200, json={})]
    rig.client.post_json("/x", {})
    assert rig.slept == [BACKOFF_BASE, BACKOFF_BASE * 2]


def test_a_retry_after_header_wins_over_the_backoff(rig):
    """The server knows better than our guess."""
    rig.responses = [
        httpx.Response(429, headers={"retry-after": "7"}),
        httpx.Response(200, json={}),
    ]
    rig.client.post_json("/x", {})
    assert rig.slept == [7.0]


def test_an_absurd_retry_after_is_capped(rig):
    """A spoken turn cannot wait five minutes for a header to be polite."""
    rig.responses = [
        httpx.Response(429, headers={"retry-after": "300"}),
        httpx.Response(200, json={}),
    ]
    rig.client.post_json("/x", {})
    assert rig.slept == [30.0]


def test_an_unparseable_retry_after_falls_back_to_the_backoff(rig):
    """It may be an HTTP-date, which this does not parse."""
    rig.responses = [
        httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        httpx.Response(200, json={}),
    ]
    rig.client.post_json("/x", {})
    assert rig.slept == [BACKOFF_BASE]


# -- what the error carries --------------------------------------------------


def test_an_http_failure_carries_its_status(rig):
    """status is what tells "out of credit" from "no network", and they need
    different clips and different fixes."""
    rig.responses = [httpx.Response(402, text="insufficient credits")]
    with pytest.raises(OpenRouterError) as caught:
        rig.client.post_json("/x", {})
    assert caught.value.status == 402


def test_a_connection_failure_carries_no_status(rig):
    rig.responses = [httpx.ConnectError("unreachable")] * MAX_ATTEMPTS
    with pytest.raises(OpenRouterError) as caught:
        rig.client.post_json("/x", {})
    assert caught.value.status is None


def test_the_error_names_the_endpoint(rig):
    rig.responses = [httpx.Response(404, text="missing")]
    with pytest.raises(OpenRouterError, match="/audio/speech"):
        rig.client.post_json("/audio/speech", {})


# -- cost --------------------------------------------------------------------


def test_cost_accumulates_across_calls(rig):
    rig.responses = [
        httpx.Response(200, json={"usage": {"cost": 0.25}}),
        httpx.Response(200, json={"usage": {"cost": 0.5}}),
    ]
    rig.client.post_json("/x", {})
    rig.client.post_json("/x", {})
    assert rig.client.spent == pytest.approx(0.75)


def test_a_response_without_usage_costs_nothing(rig):
    rig.responses = [httpx.Response(200, json={"text": "hi"})]
    rig.client.post_json("/x", {})
    assert rig.client.spent == 0.0


def test_streamed_usage_can_be_recorded_separately(rig):
    """Streamed replies carry usage in a trailing SSE chunk, which this client
    never sees as a whole JSON body."""
    rig.client.record_usage({"cost": 0.125})
    assert rig.client.spent == pytest.approx(0.125)


@pytest.mark.parametrize("usage", [None, {}, {"cost": None}, {"cost": "free"}, "nonsense"])
def test_malformed_usage_does_not_crash_a_turn(rig, usage):
    rig.client.record_usage(usage)
    assert rig.client.spent == 0.0
