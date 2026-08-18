"""Router tests. No network: the LLM leg is stubbed."""

from __future__ import annotations

import pytest

from faethon.config import Config, load_config
from faethon.memory import Memory
from faethon.providers.llm import LLMReply, ToolCall
from faethon.router import Router
from faethon.skills.base import Skill
from faethon.skills.registry import Registry


class Echo(Skill):
    name = "echo"
    tag = "test"
    description = "echo something back"
    patterns = [r"\becho (?P<word>\w+)\b"]
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    def run(self, **params):
        return f"echo {params.get('word', '')}".strip()


class Exploding(Skill):
    name = "explode"
    tag = "test"
    description = "always raises"
    patterns = [r"\bexplode\b"]

    def run(self, **params):
        raise RuntimeError("boom")


class Strict(Skill):
    """A skill with an explicit signature rather than **params.

    Perfectly legal to write, and the case where a bad tool call from the model
    would otherwise raise TypeError straight out of the loop.
    """

    name = "strict"
    tag = "test"
    description = "needs exactly one named argument"
    patterns = [r"\bstrict\b"]
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}}

    def run(self, city: str) -> str:  # type: ignore[override]
        return f"city is {city}"


class Offline(Skill):
    name = "offline"
    tag = "test"
    description = "never available"
    patterns = [r"\boffline\b"]

    @property
    def available(self) -> bool:
        return False

    def run(self, **params):
        raise AssertionError("must not run")


@pytest.fixture
def config() -> Config:
    return load_config()


@pytest.fixture
def router(config, monkeypatch):
    """Router whose LLM leg returns whatever the test sets in `replies`."""
    registry = Registry([Echo(), Exploding(), Offline(), Strict()])
    r = Router(config, client=object(), registry=registry, memory=Memory(3))

    r.replies: list[LLMReply] = []
    calls: list[dict] = []
    r.calls = calls

    def fake_complete(client, messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return r.replies.pop(0) if r.replies else LLMReply(text="fallback")

    monkeypatch.setattr("faethon.router.llm_mod.complete", fake_complete)
    return r


def test_regex_path_never_calls_the_llm(router):
    assert router.handle("please echo banana") == "echo banana"
    assert router.calls == []


def test_unmatched_text_goes_to_the_llm(router):
    router.replies = [LLMReply(text="The sky scatters blue light.")]
    assert router.handle("why is the sky blue") == "The sky scatters blue light."
    assert len(router.calls) == 1


def test_llm_can_reach_a_skill_by_tool_call(router):
    # Phrasing the regex doesn't cover still lands on the skill.
    router.replies = [LLMReply(tool_calls=[ToolCall("echo", {"word": "pear"})])]
    assert router.handle("could you repeat the word pear for me") == "echo pear"


def test_unavailable_skill_is_reported_not_run(router):
    out = router.handle("go offline")
    assert "isn't available" in out
    assert router.calls == []      # must not fall through to the LLM


def test_unavailable_skills_are_hidden_from_the_llm(router):
    router.replies = [LLMReply(text="ok")]
    router.handle("something unmatched")
    tool_names = [t["function"]["name"] for t in router.calls[0]["tools"]]
    assert "offline" not in tool_names
    assert "echo" in tool_names


def test_skill_exception_becomes_speech(router):
    out = router.handle("explode now")
    assert "went wrong" in out.lower()


def test_hallucinated_tool_name_is_survivable(router):
    router.replies = [LLMReply(tool_calls=[ToolCall("no_such_skill", {})])]
    assert "don't have a skill" in router.handle("do the thing")


def test_bad_tool_arguments_do_not_crash(router):
    # Model invents an argument name a strict-signature skill won't accept.
    router.replies = [LLMReply(tool_calls=[ToolCall("strict", {"wrong_arg": 1})])]
    assert "couldn't run" in router.handle("do the thing").lower()


def test_skills_taking_kwargs_tolerate_extra_arguments(router):
    # The **params convention absorbs surplus keys instead of failing the turn.
    router.replies = [LLMReply(tool_calls=[ToolCall("echo", {"word": "plum", "extra": 1})])]
    assert router.handle("do the thing") == "echo plum"


def test_history_is_sent_on_the_next_llm_call(router):
    router.replies = [LLMReply(text="first"), LLMReply(text="second")]
    router.handle("question one")
    router.handle("question two")

    contents = [m["content"] for m in router.calls[1]["messages"]]
    assert "question one" in contents
    assert "first" in contents


def test_empty_transcript_is_ignored(router):
    assert router.handle("   ") == ""
    assert router.calls == []
