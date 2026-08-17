"""Request-payload construction.

The reasoning guard is the important one here. DeepSeek V4 is a hybrid model
that decides per-request whether to think first, and thinking tokens are billed
against max_tokens. On a 100-token spoken-reply budget it can spend the whole
allowance reasoning and return empty content -- measured at 99/100 tokens on one
arithmetic question -- so Roxy says nothing at all. It's intermittent, so a
regression here would be easy to miss by hand.
"""

from __future__ import annotations

from roxy.providers.llm import _payload

BASE = dict(
    messages=[{"role": "user", "content": "hi"}],
    model="deepseek/deepseek-v4-flash",
    max_tokens=100,
    temperature=0.7,
    tools=None,
)


def test_reasoning_is_disabled_by_default():
    assert _payload(**BASE)["reasoning"] == {"enabled": False}


def test_reasoning_can_be_turned_on():
    # When on, we must send no reasoning key at all rather than
    # {"enabled": True} -- letting the model use its own default.
    assert "reasoning" not in _payload(**BASE, reasoning=True)


def test_disabling_uses_enabled_false_not_exclude():
    # "exclude": true and "max_tokens": 0 both still burn the whole budget
    # thinking and return empty content. They hide reasoning; they don't stop
    # it. Only "enabled": false does.
    reasoning = _payload(**BASE)["reasoning"]
    assert reasoning.get("exclude") is None
    assert reasoning.get("max_tokens") is None
    assert reasoning == {"enabled": False}


def test_tools_add_auto_tool_choice():
    tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    payload = _payload(**{**BASE, "tools": tools})
    assert payload["tool_choice"] == "auto"
    assert payload["tools"] == tools


def test_no_tool_choice_without_tools():
    assert "tool_choice" not in _payload(**BASE)


def test_core_parameters_are_passed_through():
    payload = _payload(**BASE)
    assert payload["model"] == "deepseek/deepseek-v4-flash"
    assert payload["max_tokens"] == 100
    assert payload["temperature"] == 0.7
