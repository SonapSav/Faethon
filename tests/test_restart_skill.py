"""Restarting the assistant by voice.

Two failure modes matter more than the happy path, and neither is visible by
reading the code:

* Exiting inside run() would kill the process before the reply is spoken.
  Silence followed by a dead assistant is indistinguishable from a crash, and
  the user would have no idea whether the command was even heard.
* Being offered to the LLM as a tool would put a restart within reach of
  something that decides for itself when to use its tools. "This keeps
  freezing, what should I do?" is a reasonable thing to say to an assistant
  and an unreasonable thing to be restarted for.

Rebooting the device is deliberately not implemented: it would have needed a
polkit rule or a sudoers entry, and the privilege was declined rather than
granted. Restarting the service needs neither -- the unit is Restart=always,
so the process just exits.
"""

from __future__ import annotations

import pytest

from faethon.config import load_config
from faethon.memory import Memory
from faethon.router import Router
from faethon.skills.base import Skill
from faethon.skills.registry import Registry
from faethon.skills.restart_skill import SKILL as RESTART
from faethon.skills.time_skill import SKILL as TIME


@pytest.fixture
def router():
    return Router(
        load_config(), client=object(),
        registry=Registry([RESTART, TIME]), memory=Memory(10),
    )


# -- the two things that must not go wrong -----------------------------------


def test_the_reply_is_produced_without_exiting():
    """run() must return, not exit. The sentence has to be spoken first."""
    said = RESTART.run()
    assert "Restarting" in said


def test_the_exit_is_deferred_until_after_the_reply():
    with pytest.raises(SystemExit) as exit_info:
        RESTART.after_reply()
    assert exit_info.value.code == 0, "a non-zero exit would look like a crash"


def test_it_is_hidden_from_the_llm():
    """A helpful model with a restart tool is a restart waiting to happen."""
    from faethon.skills.identity_skill import SKILL as IDENTITY

    registry = Registry([RESTART, IDENTITY])
    offered = [t["function"]["name"] for t in registry.tool_schemas()]
    assert "restart_assistant" not in offered
    assert "describe_self" in offered, "ordinary skills should still be offered"


# -- the router's half of the deferral ---------------------------------------


def test_the_router_holds_the_action_until_asked(router):
    list(router.handle_streaming("restart yourself"))
    action = router.take_pending_action()
    assert action is not None
    with pytest.raises(SystemExit):
        action()


def test_the_action_is_handed_over_only_once(router):
    """Taken rather than read, so a second turn cannot re-trigger it."""
    list(router.handle_streaming("restart yourself"))
    assert router.take_pending_action() is not None
    assert router.take_pending_action() is None


def test_an_ordinary_skill_leaves_no_action(router):
    list(router.handle_streaming("what time is it"))
    assert router.take_pending_action() is None


def test_an_ordinary_skill_clears_a_stale_action(router):
    """An abandoned restart turn must not fire on someone else's question."""
    list(router.handle_streaming("restart yourself"))
    list(router.handle_streaming("what time is it"))
    assert router.take_pending_action() is None, "a stale restart survived"


def test_skills_declare_the_deferral_by_overriding(router):
    """The router checks for an overridden after_reply rather than a flag,
    which cannot fall out of step with the method it describes."""
    assert type(RESTART).after_reply is not Skill.after_reply
    assert type(TIME).after_reply is Skill.after_reply


# -- the wiring in the main loop ---------------------------------------------


def test_the_turn_runs_the_action_after_speaking(monkeypatch):
    """Ordering is the whole point: speak, then exit."""
    from faethon.__main__ import Faethon
    from faethon.speech import Spoken

    order = []
    f = Faethon.__new__(Faethon)
    f.config = load_config()
    f.client = type("C", (), {"spent": 0.0})()
    f._running = True
    f.router = Router(
        f.config, client=object(),
        registry=Registry([RESTART, TIME]), memory=Memory(10),
    )

    monkeypatch.setattr(Faethon, "_capture_utterance", lambda self, s, t=None: b"\x00" * 32000)
    monkeypatch.setattr(
        "faethon.__main__.stt_mod.transcribe",
        lambda *a, **kw: "restart yourself",
    )

    def fake_speak(self, read_frame, text):
        order.append("spoke")
        list(self.router.handle_streaming(text))
        return Spoken("Restarting now.")

    monkeypatch.setattr(Faethon, "_speak_reply", fake_speak)

    with pytest.raises(SystemExit):
        f._handle_turn(lambda: b"")
    assert order == ["spoke"], "the action ran before the reply was spoken"


# -- the phrasings -----------------------------------------------------------


@pytest.mark.parametrize("heard", [
    "restart yourself",
    "Restart yourself.",
    "restart the assistant",
    "restart the service",
    "restart Faethon",
    "restart Phaethon",
    "restart Rhasspy",
    "reboot yourself",
])
def test_phrasings_that_reach_it(heard):
    assert RESTART.match(heard) is not None, f"no pattern matched {heard!r}"


@pytest.mark.parametrize("heard", [
    "restart my router",
    "I had to restart my laptop this morning",
    "what time is it",
    "reboot the raspberry pi",     # deliberately not implemented
    "restart",
])
def test_phrasings_that_should_not(heard):
    assert RESTART.match(heard) is None, f"unexpectedly matched {heard!r}"
