"""LLM via OpenRouter's chat completions endpoint.

Used for two things: answering conversationally, and -- when a skill's regex
patterns didn't match -- deciding via tool-calling whether a skill should run
and with what parameters.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .client import OpenRouterClient

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    """Exactly one of `text` or `tool_calls` is meaningful per reply."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (json.JSONDecodeError, TypeError, ValueError):
            # A model that emits malformed arguments shouldn't crash the loop;
            # run the skill with no parameters and let it use its defaults.
            log.warning("tool %s: unparseable arguments %r", name, raw_args)
            args = {}
        calls.append(ToolCall(name=name, arguments=args))
    return calls


def _payload(
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    reasoning: bool = False,
    provider_sort: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    if not reasoning:
        # DeepSeek V4 is a hybrid: it decides per-request whether to think
        # first, and thinking tokens come out of max_tokens. On a short budget
        # it can spend the entire allowance reasoning and return empty content
        # -- Faethon then says nothing at all. It's intermittent, which makes it
        # worse: the same question can work and then not.
        #
        # Only "enabled": false actually stops it. "exclude": true and
        # "max_tokens": 0 merely hide the reasoning while still paying for it.
        payload["reasoning"] = {"enabled": False}

    if provider_sort:
        # OpenRouter serves this model from many providers and defaults to
        # cheapest-first, which is the wrong objective for speech: the reply
        # is late by however long the provider takes to produce its first
        # token, and the cheap ones are erratic. Measured over five trials
        # each, with this system prompt and tool schemas:
        #
        #   default      median ttft 2.76s, worst 5.22s
        #   latency      median ttft 1.50s, worst 2.94s
        #   throughput   median ttft 0.81s, worst 11.36s
        #
        # "throughput" wins on the median and loses badly on the tail, which
        # is the half that gets noticed -- a reply that is occasionally 11s
        # late is worse than one reliably at 1.5s.
        payload["provider"] = {"sort": provider_sort}
    return payload


class StreamingReply:
    """A chat completion consumed token by token.

    Iterate it for content deltas as they arrive. Once iteration finishes,
    `text` holds everything that was said and `tool_calls` holds any tools the
    model asked for.

    A response is in practice either content or tool calls, not both, so
    nothing is yielded when the model decides to call a tool -- the caller
    finds out by checking `tool_calls` after the (empty) iteration.
    """

    def __init__(self, client: OpenRouterClient, payload: dict[str, Any]) -> None:
        self._client = client
        self._payload = payload
        self.text = ""
        self.tool_calls: list[ToolCall] = []
        self.finished = False
        #: "stop", "length", "tool_calls", ... or None if the stream ended
        #: without one, which means it was cut short.
        self.finish_reason: str | None = None

    def __iter__(self) -> Iterator[str]:
        # Tool call fragments arrive split across chunks and keyed by index.
        pending: dict[int, dict[str, str]] = {}

        with self._client.post_stream("/chat/completions", self._payload) as response:
            for line in response.iter_lines():
                line = line.strip()
                # OpenRouter sends ": OPENROUTER PROCESSING" keepalive comments.
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    log.warning("unparseable SSE chunk: %r", data[:120])
                    continue

                # Usage arrives in a trailing chunk with no choices.
                self._client.record_usage(obj.get("usage"))

                choices = obj.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason"):
                    self.finish_reason = choices[0]["finish_reason"]
                delta = choices[0].get("delta") or {}

                content = delta.get("content")
                if content:
                    self.text += content
                    yield content

                for fragment in delta.get("tool_calls") or []:
                    slot = pending.setdefault(
                        fragment.get("index", 0), {"name": "", "arguments": ""}
                    )
                    fn = fragment.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

        self.tool_calls = _parse_tool_calls(
            {"tool_calls": [
                {"function": slot} for slot in pending.values() if slot["name"]
            ]}
        )
        self.finished = True

        if self.finish_reason is None:
            # The connection ended before the model said it was done. The text
            # so far is a fragment; say so rather than speaking half a sentence
            # as though it were the whole answer.
            log.warning(
                "llm stream ended with no finish_reason after %d chars -- truncated",
                len(self.text),
            )
        elif self.finish_reason == "length":
            log.warning(
                "llm hit the %s-token cap; reply is cut off",
                self._payload.get("max_tokens"),
            )

        log.info(
            "llm streamed (%s): %s",
            self.finish_reason,
            f"tool={[c.name for c in self.tool_calls]}"
            if self.tool_calls
            else repr(self.text),
        )


def complete_streaming(
    client: OpenRouterClient,
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    reasoning: bool = False,
    provider_sort: str = "",
) -> StreamingReply:
    payload = _payload(
        messages, model, max_tokens, temperature, tools, reasoning, provider_sort
    )
    payload["stream"] = True
    # Without this the streamed response carries no cost information at all.
    payload["stream_options"] = {"include_usage": True}
    return StreamingReply(client, payload)


def complete(
    client: OpenRouterClient,
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    reasoning: bool = False,
    provider_sort: str = "",
) -> LLMReply:
    data = client.post_json(
        "/chat/completions",
        _payload(
            messages, model, max_tokens, temperature, tools, reasoning, provider_sort
        ),
    )
    choices = data.get("choices") or []
    if not choices:
        log.warning("llm returned no choices: %s", str(data)[:300])
        return LLMReply()

    message = choices[0].get("message") or {}
    reply = LLMReply(
        text=(message.get("content") or "").strip(),
        tool_calls=_parse_tool_calls(message),
    )
    log.info(
        "llm: %s",
        f"tool={[c.name for c in reply.tool_calls]}" if reply.wants_tool else repr(reply.text),
    )
    return reply
