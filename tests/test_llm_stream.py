"""The streamed chat completion: parsing SSE into speech and tool calls.

Untested until now, and the most intricate code in the project. It has to cope
with deltas split anywhere, tool-call arguments arriving in fragments keyed by
index, OpenRouter's keepalive comments, usage in a trailing chunk with no
choices, and a stream that simply stops. Every one of those failures is quiet:
a dropped fragment produces a tool call with the wrong arguments, and a missed
finish_reason produces half a sentence spoken as though it were the whole
answer.

Driven through httpx's MockTransport with scripted SSE bodies. No network.
"""

from __future__ import annotations

import httpx
import pytest

from faethon.providers import client as client_mod
from faethon.providers.client import OpenRouterClient
from faethon.providers.llm import complete_streaming


def sse(*lines: str) -> bytes:
    return ("\n".join(lines) + "\n").encode()


def delta(content: str) -> str:
    return (
        'data: {"choices":[{"delta":{"content":%s},"finish_reason":null}]}'
        % _json_str(content)
    )


def _json_str(s: str) -> str:
    import json

    return json.dumps(s)


def stop(reason: str = "stop") -> str:
    return 'data: {"choices":[{"delta":{},"finish_reason":"%s"}]}' % reason


@pytest.fixture
def reply_from(monkeypatch):
    """Build a StreamingReply fed by a scripted SSE body."""
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)

    def build(body: bytes, status: int = 200):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, content=body)

        c = OpenRouterClient("sk-or-test")
        c._client = httpx.Client(
            base_url=client_mod.BASE_URL, transport=httpx.MockTransport(handler)
        )
        return c, complete_streaming(
            c, [{"role": "user", "content": "hi"}],
            model="m", max_tokens=100, temperature=0.7,
        )

    return build


# -- content -----------------------------------------------------------------


def test_content_deltas_are_yielded_in_order(reply_from):
    _, reply = reply_from(sse(delta("Hello"), delta(" there"), stop(), "data: [DONE]"))
    assert list(reply) == ["Hello", " there"]
    assert reply.text == "Hello there"
    assert reply.finish_reason == "stop"


def test_keepalive_comments_are_ignored(reply_from):
    """OpenRouter sends ': OPENROUTER PROCESSING' while a provider is slow.

    Treated as data, it would be parsed as a chunk and spoken aloud.
    """
    _, reply = reply_from(sse(
        ": OPENROUTER PROCESSING", "", delta("Fine"), ": OPENROUTER PROCESSING",
        delta(" thanks"), stop(), "data: [DONE]",
    ))
    assert list(reply) == ["Fine", " thanks"]


def test_an_unparseable_chunk_is_skipped_not_fatal(reply_from):
    """One bad line should cost one fragment, not the whole reply."""
    _, reply = reply_from(sse(
        delta("Good"), "data: {not json", delta(" enough"), stop(), "data: [DONE]",
    ))
    assert list(reply) == ["Good", " enough"]


def test_empty_deltas_are_not_yielded(reply_from):
    """A chunk carrying only a finish_reason would otherwise be spoken as ''."""
    _, reply = reply_from(sse(delta("Hi"), 'data: {"choices":[{"delta":{}}]}',
                              stop(), "data: [DONE]"))
    assert list(reply) == ["Hi"]


def test_done_ends_the_stream(reply_from):
    _, reply = reply_from(sse(delta("One"), "data: [DONE]", delta("Two")))
    assert list(reply) == ["One"]


# -- tool calls --------------------------------------------------------------


def test_tool_call_fragments_are_reassembled(reply_from):
    """Arguments arrive split across chunks. Concatenated in the wrong order,
    or dropped, the skill runs with the wrong parameters and says so
    confidently."""
    _, reply = reply_from(sse(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"name":"set_volume","arguments":"{\\"lev"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"arguments":"el\\": 7}"}}]}}]}',
        stop("tool_calls"), "data: [DONE]",
    ))
    assert list(reply) == [], "a tool call should yield no speech"
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "set_volume"
    assert reply.tool_calls[0].arguments == {"level": 7}


def test_two_tool_calls_are_kept_apart_by_index(reply_from):
    _, reply = reply_from(sse(
        'data: {"choices":[{"delta":{"tool_calls":['
        '{"index":0,"function":{"name":"get_time","arguments":"{}"}},'
        '{"index":1,"function":{"name":"set_volume","arguments":"{\\"level\\":3}"}}]}}]}',
        stop("tool_calls"), "data: [DONE]",
    ))
    list(reply)
    assert [c.name for c in reply.tool_calls] == ["get_time", "set_volume"]
    assert reply.tool_calls[1].arguments == {"level": 3}


def test_malformed_tool_arguments_do_not_crash_the_turn(reply_from):
    """A model that emits broken JSON should get the skill's defaults, not a
    traceback out of the main loop."""
    _, reply = reply_from(sse(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"name":"get_time","arguments":"{oh dear"}}]}}]}',
        stop("tool_calls"), "data: [DONE]",
    ))
    list(reply)
    assert reply.tool_calls[0].name == "get_time"
    assert reply.tool_calls[0].arguments == {}


def test_a_nameless_tool_call_is_dropped(reply_from):
    _, reply = reply_from(sse(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"arguments":"{}"}}]}}]}',
        stop("tool_calls"), "data: [DONE]",
    ))
    list(reply)
    assert reply.tool_calls == []


# -- how it ended ------------------------------------------------------------


def test_a_truncated_stream_is_noticed(reply_from, caplog):
    """No finish_reason means the connection died mid-sentence. Speaking the
    fragment as though it were the answer is the failure to avoid."""
    _, reply = reply_from(sse(delta("The first half")))
    assert list(reply) == ["The first half"]
    assert reply.finish_reason is None
    assert "truncated" in caplog.text.lower()


def test_hitting_the_token_cap_is_reported(reply_from, caplog):
    _, reply = reply_from(sse(delta("Going on a bit"), stop("length"), "data: [DONE]"))
    list(reply)
    assert reply.finish_reason == "length"
    assert "cut off" in caplog.text.lower()


def test_usage_from_the_trailing_chunk_is_billed(reply_from):
    """Usage arrives with no choices at all, after the content."""
    client, reply = reply_from(sse(
        delta("Hi"), stop(),
        'data: {"choices":[],"usage":{"cost":0.0009}}',
        "data: [DONE]",
    ))
    list(reply)
    assert client.spent == pytest.approx(0.0009)


def test_an_http_error_surfaces_before_any_audio(reply_from):
    from faethon.providers.client import OpenRouterError

    _, reply = reply_from(b"insufficient credits", status=402)
    with pytest.raises(OpenRouterError) as caught:
        list(reply)
    assert caught.value.status == 402
