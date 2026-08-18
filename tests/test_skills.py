from __future__ import annotations

import pytest

from faethon.skills.base import Skill
from faethon.skills.registry import Registry
from faethon.skills.time_skill import SKILL as TIME_SKILL


class Dummy(Skill):
    name = "dummy"
    tag = "test"
    description = "test skill"
    patterns = [r"\bturn (?P<state>on|off) the (?P<thing>\w+)\b", r"\bping\b"]
    parameters = {
        "type": "object",
        "properties": {"state": {"type": "string"}, "thing": {"type": "string"}},
    }

    def run(self, **params):
        if not params:
            return "pong"
        return f"{params['thing']} {params['state']}"


class Unavailable(Dummy):
    name = "broken"
    patterns = [r"\bbroken\b"]

    @property
    def available(self) -> bool:
        return False


def test_named_groups_become_params():
    assert Dummy().match("please turn off the lamp") == {"state": "off", "thing": "lamp"}


def test_pattern_without_groups_matches_with_empty_params():
    # {} is a match; None is "no match". The distinction matters to the router.
    assert Dummy().match("ping") == {}


def test_no_match_returns_none():
    assert Dummy().match("what is the weather") is None


def test_registry_rejects_duplicate_names():
    r = Registry([Dummy()])
    with pytest.raises(ValueError, match="duplicate"):
        r.add(Dummy())


def test_registry_match_returns_skill_and_params():
    skill, params = Registry([Dummy()]).match("turn on the fan")
    assert skill.name == "dummy"
    assert params == {"state": "on", "thing": "fan"}


def test_tool_schemas_omit_unavailable_skills():
    r = Registry([Dummy(), Unavailable()])
    names = [s["function"]["name"] for s in r.tool_schemas()]
    assert names == ["dummy"]


def test_tool_schema_shape():
    schema = Dummy().tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dummy"
    assert "parameters" in schema["function"]


def test_discover_finds_the_time_skill():
    r = Registry.discover()
    assert r.get("get_time") is not None


@pytest.mark.parametrize(
    "phrase",
    [
        "what's the time",
        "what time is it",
        "whats the time",
        "tell me the date",
        "what is the date",
        "current time",
    ],
)
def test_time_skill_phrasings(phrase):
    assert TIME_SKILL.match(phrase) is not None


def test_time_skill_runs_with_no_params():
    # The regex path can produce {}, so run() must cope.
    assert TIME_SKILL.run().startswith("It's ")


def test_time_skill_hour_has_no_leading_zero():
    # "09:05" would be spoken as "oh nine oh five".
    out = TIME_SKILL.run(kind="time")
    assert not out.startswith("It's 0")
