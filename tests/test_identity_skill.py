"""What Faethon says it is.

Identity is the one category where the model's priors are guaranteed wrong --
Faethon postdates the training data, so the only source of truth is the prompt.
With the prompt saying "You are Rhasspy, a voice assistant" and nothing else,
the model described the Rhasspy project instead, and invented a lineage:

    Q: are you related to the Rhasspy project
    A: Yes, I am named after the Rhasspy project ... built by the same team.

There is no transcript to check it against, and the answer lands in the memory
buffer, so the model stays consistent with its own confabulation for the rest
of the conversation. Hence a deterministic skill rather than better prompting
alone.
"""

from __future__ import annotations

import pytest

from faethon.config import load_config
from faethon.skills.identity_skill import SKILL

FORBIDDEN = "rhasspy"


# -- the phrasings -----------------------------------------------------------


@pytest.mark.parametrize("heard", [
    "what are you",
    "What are you?",
    "who are you",
    "what is your name",
    "what's your name",
    "what are you called",
    "who made you",
    "who built you",
    "who created you",
    "who programmed you",
    "are you ChatGPT",
    "are you chat gpt",
    "are you Siri",
    "are you Alexa",
    "are you a robot",
    "are you an AI",
    "are you human",
])
def test_questions_about_itself(heard):
    assert SKILL.match(heard) is not None, f"no pattern matched {heard!r}"


@pytest.mark.parametrize("heard", [
    "what are you doing",
    "what are you doing today",
    "who made this cake",
    "are you sure",
    "are you awake",
    "what is the time",
])
def test_questions_that_only_look_similar(heard):
    """"What are you doing" answering with a hardware description is the same
    class of failure as "what is the time complexity"."""
    assert SKILL.match(heard) is None, f"intercepted {heard!r}"


def test_what_can_you_do_is_left_to_the_model():
    """That answer is dynamic -- the model reads it off the live tool schemas,
    so it stays correct as skills are added. A hardcoded list would rot on the
    next one."""
    assert SKILL.match("what can you do") is None
    assert SKILL.match("what are your skills") is None


@pytest.mark.parametrize("heard", [
    "why do I say hey rhasspy",
    "why do i say hey raspy to wake you",
    "why are you called rhasspy",
    "why is it called Rasppy",
    "what does rhasspy mean",
])
def test_questions_about_the_wake_phrase(heard):
    """The prompt cannot fix these -- the user's own question reintroduces the
    token, and the model answers "Rhasspy is the name of the software running
    on this Raspberry Pi", which is false."""
    assert SKILL.match(heard) is not None, f"no pattern matched {heard!r}"


def test_it_never_speaks_its_own_wake_phrase():
    """Measured: Faethon saying "hey rhasspy" scores 0.9984 on the wake model
    through the speaker and microphone -- above the 0.7 that wakes it, and far
    above the 0.1 barge-in listens at. It would interrupt itself mid-sentence,
    every time anyone asked why the wake word is what it is.
    """
    import re

    for kind in ("identity", "origin", "comparison", "wake"):
        said = SKILL.run(kind=kind)
        assert not re.search(r"rh?as+p{0,2}y", said, re.I), (
            f"the {kind} answer speaks the wake phrase: {said!r}"
        )


# -- what it says ------------------------------------------------------------


def test_it_says_what_it_is():
    said = SKILL.run()
    assert "Faethon" in said
    assert "Raspberry Pi" in said


def test_each_kind_answers_differently():
    said = {
        SKILL.run(),
        SKILL.run(**SKILL.match("who made you")),
        SKILL.run(**SKILL.match("are you ChatGPT")),
        SKILL.run(**SKILL.match("why do I say hey rhasspy")),
    }
    assert len(said) == 4


def test_origin_denies_a_company_rather_than_inventing_one():
    """The model's failure was inventing a team. Saying plainly that there
    isn't one is what leaves nothing to invent."""
    said = SKILL.run(**SKILL.match("who built you"))
    assert "no company" in said.lower()


def test_comparison_answers_positively():
    """Naming what it is, not listing what it isn't -- denial is what invited
    the confabulation."""
    said = SKILL.run(**SKILL.match("are you ChatGPT"))
    assert "Faethon" in said


def test_the_tool_path_can_ask_for_a_kind():
    """The model reaches this with an explicit kind rather than a regex group."""
    assert "no company" in SKILL.run(kind="origin").lower()
    assert SKILL.run(kind="comparison").startswith("No")
    assert "voice assistant" in SKILL.run(kind="identity")


def test_an_unknown_kind_falls_back_to_identity():
    assert "Faethon" in SKILL.run(kind="nonsense")


# -- the regression guards ---------------------------------------------------


@pytest.mark.parametrize("kind", ["identity", "origin", "comparison", "wake"])
def test_no_answer_names_the_other_project(kind):
    assert FORBIDDEN not in SKILL.run(kind=kind).lower()


def test_the_system_prompt_does_not_name_the_other_project():
    """The prompt is the backstop for phrasings the patterns miss, and naming
    the entity is what activates it.

    Measured: a variant that mentioned the wake phrase *in order to explain it
    away* made things worse -- "I am built on the Rhasspy project's wake-word
    detection and voice assistant framework". This is what stops a future
    prompt edit quietly reintroducing the token.
    """
    prompt = load_config().llm.system_prompt.lower()
    assert FORBIDDEN not in prompt
    assert "faethon" in prompt


def test_the_prompt_still_grounds_what_it_is():
    """Removing the old name is only half the fix; the prompt has to say what
    it is, or the model has nothing to answer from."""
    prompt = load_config().llm.system_prompt.lower()
    assert "raspberry pi" in prompt
