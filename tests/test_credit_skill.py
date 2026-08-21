"""Reporting the OpenRouter credit balance.

Two things carry the weight. The balance is a subtraction -- OpenRouter returns
total_credits and total_usage, not what is left -- and the regex has to survive
Whisper's spelling. "OpenRouter" is a coined word, so it comes back as
OpenRooter, Open Router, Open Rooter and worse; a pattern matching only the
correct spelling fails by falling through to the LLM, which cannot see the
balance and will cheerfully invent one.

Offline: the HTTP call is stubbed.
"""

from __future__ import annotations

import pytest

from faethon.providers.client import OpenRouterError
from faethon.skills import credit_skill
from faethon.skills.credit_skill import SKILL, CreditSkill


@pytest.fixture
def skill(monkeypatch):
    s = CreditSkill()
    s._key = "sk-or-test"
    s._looked = True
    s.response = {"data": {"total_credits": 30, "total_usage": 28.228023831}}
    s.error: Exception | None = None

    class FakeClient:
        def __init__(self, key, timeout=None):
            s.timeout = timeout

        def __enter__(self): return self
        def __exit__(self, *a): return False

        def get_json(self, path):
            s.path = path
            if s.error is not None:
                raise s.error
            return s.response

    monkeypatch.setattr(credit_skill, "OpenRouterClient", FakeClient)
    return s


# -- the number --------------------------------------------------------------


def test_the_balance_is_what_is_left_not_what_was_granted(skill):
    """OpenRouter returns credits and usage; the balance is the difference."""
    assert skill.run() == "Your OpenRouter balance is 1.77 dollars."


def test_it_reads_the_credits_endpoint(skill):
    skill.run()
    assert skill.path == "/credits"


def test_two_decimals_always(skill):
    """Asked for explicitly, and "1.8" spoken aloud is not how money sounds."""
    for credits, usage, expected in [
        (30, 28.228023831, "1.77"),
        (10, 8.0, "2.00"),
        (5, 0.005, "5.00"),
        (2, 1.996, "0.00"),
    ]:
        skill.response = {"data": {"total_credits": credits, "total_usage": usage}}
        assert expected in skill.run(), f"{credits} - {usage}"


def test_it_says_dollars_rather_than_a_symbol(skill):
    """It goes straight to TTS without the LLM tidying it up."""
    said = skill.run()
    assert "dollars" in said
    assert "$" not in said


def test_running_out_is_called_out(skill):
    """The assistant stops working entirely at zero, so a bare "0.00" would
    understate it."""
    skill.response = {"data": {"total_credits": 30, "total_usage": 30}}
    assert "out of credit" in skill.run()


def test_an_overdrawn_account_does_not_report_a_negative(skill):
    skill.response = {"data": {"total_credits": 30, "total_usage": 31.5}}
    said = skill.run()
    assert "-" not in said
    assert "out of credit" in said


# -- when it cannot answer ---------------------------------------------------


def test_an_unreachable_api_says_so(skill):
    skill.error = OpenRouterError("connection refused")
    assert "couldn't reach OpenRouter" in skill.run()


def test_an_unexpected_response_shape_is_not_read_out_as_zero(skill):
    """If the endpoint changes, saying "0.00 dollars" would be a confident lie
    that reads as "you are out of credit"."""
    skill.response = {"data": {"balance": 1.77}}
    said = skill.run()
    assert "didn't understand" in said
    assert "0.00" not in said


def test_no_api_key_means_unavailable(monkeypatch):
    s = CreditSkill()
    s._key = None
    s._looked = True
    assert not s.available
    assert "don't have an OpenRouter key" in s.run()


def test_it_does_not_hang_a_spoken_turn(skill):
    """The client's 60s default would leave the user in silence."""
    skill.run()
    assert skill.timeout <= 15


# -- the phrasings -----------------------------------------------------------


@pytest.mark.parametrize("heard", [
    "What is my OpenRouter credit balance",
    "What's my OpenRooter credit balance",
    "Whats my Open Router credit balance",
    "what is my open rooter balance",
    "What's my Open Rooter credit balance?",
    "OpenRouter credits",
    "how much credit do I have left",
    "how much money is left",
    "what is my balance",
    "how much have I got left",
])
def test_phrasings_that_reach_it(heard):
    assert SKILL.match(heard) is not None, f"no pattern matched {heard!r}"


@pytest.mark.parametrize("heard", [
    "what is the weather",
    "balance the books",
    "what time is it",
    "turn the volume up",
])
def test_phrasings_that_should_not(heard):
    assert SKILL.match(heard) is None, f"unexpectedly matched {heard!r}"


# -- phrasings that must never reach the model --------------------------------
# Asked "what is my budget with OpenRouter", which no pattern matched, the model
# answered "You have 19 dollars and 34 cents left" without calling anything --
# against a real balance of $1.35 -- and repeated it twice when asked again.
# The direction is what makes it serious: overstating by 14x reads as
# reassurance at exactly the moment the account is nearly empty.


@pytest.mark.parametrize("phrase", [
    "what is my budget with openrouter",
    "what's my budget with open router",
    "what is my budget",
    "what is my openrouter budget",
    "how much is left on openrouter",
    "how much is remaining on open router",
    "how much have i got left with openrouter",
    "what is my openrouter balance",
    "what is my credit balance",
    "how much credit do i have",
    "how much money do i have left",
    "what is my balance",
])
def test_money_questions_never_reach_the_model(phrase):
    assert SKILL.match(phrase) is not None, phrase


@pytest.mark.parametrize("phrase", [
    "how much money is a raspberry pi",
    "what is the budget of the film",
    "money can not buy happiness",
    "how much does a pizza cost",
    "how much time is left on the timer",
    "tell me about budgeting",
])
def test_broad_money_words_do_not_overreach(phrase):
    """"budget" and "money" are common English; they must still be quiet."""
    assert SKILL.match(phrase) is None, phrase


def test_the_prompt_forbids_inventing_live_figures():
    """The regex covers the phrasings we thought of. This covers the rest."""
    from faethon.config import load_config

    prompt = load_config().llm.system_prompt.lower()
    assert "cannot know any live figure" in prompt
    assert "never state one from memory" in prompt
