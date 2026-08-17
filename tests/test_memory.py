from __future__ import annotations

from roxy.memory import Memory


def test_evicts_oldest_beyond_limit():
    m = Memory(max_turns=10)
    for i in range(15):
        m.add(f"q{i}", f"a{i}")
    assert len(m) == 10

    msgs = m.messages("sys", "now")
    contents = [x["content"] for x in msgs]
    assert "q4" not in contents      # evicted
    assert "q5" in contents          # oldest survivor
    assert "q14" in contents


def test_message_ordering_and_roles():
    m = Memory(max_turns=5)
    m.add("hello", "hi there")
    msgs = m.messages("be brief", "what time is it")

    assert [x["role"] for x in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == "be brief"
    assert msgs[-1]["content"] == "what time is it"


def test_incomplete_turns_are_not_stored():
    m = Memory()
    m.add("question", "")
    m.add("", "answer")
    assert len(m) == 0


def test_clear():
    m = Memory()
    m.add("a", "b")
    m.clear()
    assert len(m) == 0
