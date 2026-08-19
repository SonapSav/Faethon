from __future__ import annotations

from faethon.memory import Memory


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


# -- idle expiry -------------------------------------------------------------
# Without it, context has no end: ask about France in the morning and "how big
# is it?" in the afternoon still answers about France, which is wrong in a way
# nothing announces. The clock runs from the last turn, not from process start.


class Clock:
    """A hand-wound clock, so these don't sleep."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def aged(idle_timeout_sec=600.0, turns=1):
    clock = Clock()
    m = Memory(10, idle_timeout_sec=idle_timeout_sec, now=clock)
    for i in range(turns):
        m.add(f"question {i}", f"answer {i}")
    return m, clock


def test_it_forgets_after_the_idle_window():
    m, clock = aged()
    clock.advance(601)
    assert m.expire_if_idle() is True
    assert len(m) == 0


def test_it_keeps_the_conversation_inside_the_window():
    m, clock = aged()
    clock.advance(599)
    assert m.expire_if_idle() is False
    assert len(m) == 1


def test_the_clock_runs_from_the_last_turn_not_the_first():
    """Otherwise a long conversation expires while it is still going on."""
    m, clock = aged()
    clock.advance(550)
    m.add("still here", "still listening")
    clock.advance(550)                    # 1100s since the first turn
    assert m.expire_if_idle() is False, "expired mid-conversation"
    clock.advance(60)                     # 610s since the last one
    assert m.expire_if_idle() is True


def test_an_empty_buffer_expires_nothing():
    """The usual state while idle. It must not churn or log every frame."""
    m, clock = aged(turns=0)
    clock.advance(10_000)
    assert m.expire_if_idle() is False


def test_zero_disables_expiry():
    m, clock = aged(idle_timeout_sec=0)
    clock.advance(100_000)
    assert m.expire_if_idle() is False
    assert len(m) == 1


def test_clearing_by_voice_also_restarts_the_clock():
    """Otherwise the first turn of a fresh conversation inherits the old idle
    time and can expire almost immediately."""
    m, clock = aged()
    clock.advance(590)
    m.clear()
    m.add("brand new", "conversation")
    clock.advance(100)
    assert m.expire_if_idle() is False
    assert len(m) == 1


def test_expiry_leaves_no_history_in_the_next_prompt():
    m, clock = aged(turns=3)
    clock.advance(601)
    m.expire_if_idle()
    prompt = m.messages("SYS", "something new")
    assert [x["role"] for x in prompt] == ["system", "user"]
