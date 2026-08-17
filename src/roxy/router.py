"""Turn a transcript into something to say.

This is the flowchart's "does the text include a skill command?" diamond.

Two paths, cheapest first:

  1. local_skill_match -- regex against the transcript. Free, instant, and
     offline. Handles the phrasings people actually repeat every day.
  2. llm_fallback -- the LLM gets the transcript plus every available skill as
     a tool. It either invokes one with extracted parameters, or just answers
     conversationally.

So an unanticipated phrasing still reaches the right skill; it just costs one
API call to get there.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from .config import Config
from .memory import Memory
from .providers import llm as llm_mod
from .providers.client import OpenRouterClient, OpenRouterError
from .skills.registry import Registry

log = logging.getLogger(__name__)


class Router:
    def __init__(
        self,
        config: Config,
        client: OpenRouterClient,
        registry: Registry,
        memory: Memory | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.registry = registry
        self.memory = memory or Memory(config.llm.history_turns)

    def handle(self, text: str) -> str:
        """Return what Roxy should say in response to `text`."""
        if not text.strip():
            return ""

        spoken = self._local_skill_match(text)
        if spoken is None:
            spoken = self._llm_fallback(text)

        self.memory.add(text, spoken)
        return spoken

    def handle_streaming(self, text: str) -> Iterator[str]:
        """Same routing as `handle`, but yields speakable chunks as they form.

        The skill path yields one chunk (a skill's answer is already short and
        arrives all at once). The LLM path yields sentences as the model
        produces them, so speech can start before generation finishes.

        Memory is only recorded once the caller has consumed the generator.
        """
        if not text.strip():
            return

        hit = self.registry.match(text)
        if hit is not None:
            skill, params = hit
            if not skill.available:
                log.info("skill %s matched but is unavailable", skill.name)
                spoken = skill.unavailable_reason
            else:
                log.info("regex -> %s(%s)", skill.name, params)
                spoken = self._run(skill.name, params)
            if spoken:
                yield spoken
            self.memory.add(text, spoken)
            return

        from .speech import sentence_chunks  # local: avoids a circular import

        messages = self.memory.messages(self.config.llm.system_prompt, text)
        reply = llm_mod.complete_streaming(
            self.client,
            messages,
            model=self.config.models.llm,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            tools=self.registry.tool_schemas() or None,
            reasoning=self.config.llm.reasoning,
        )

        said: list[str] = []
        try:
            for chunk in sentence_chunks(reply):
                said.append(chunk)
                yield chunk
        except OpenRouterError as e:
            log.error("llm stream failed: %s", e)
            if not said:
                spoken = "Sorry, I couldn't reach my brain just then."
                yield spoken
                self.memory.add(text, spoken)
                return

        # No content streamed: the model chose a tool instead.
        for call in reply.tool_calls:
            log.info("tool -> %s(%s)", call.name, call.arguments)
            result = self._run(call.name, call.arguments)
            if result:
                said.append(result)
                yield result

        spoken = " ".join(said)
        if not spoken:
            spoken = "Sorry, I didn't catch that."
            yield spoken
        self.memory.add(text, spoken)

    def _local_skill_match(self, text: str) -> str | None:
        """The regex path. None means "no skill matched, try the LLM"."""
        hit = self.registry.match(text)
        if hit is None:
            return None

        skill, params = hit
        if not skill.available:
            # Matched a known command whose skill can't run -- say so rather
            # than falling through to the LLM, which would invent an answer.
            log.info("skill %s matched but is unavailable", skill.name)
            return skill.unavailable_reason

        log.info("regex -> %s(%s)", skill.name, params)
        return self._run(skill.name, params)

    def _llm_fallback(self, text: str) -> str:
        messages = self.memory.messages(self.config.llm.system_prompt, text)
        try:
            reply = llm_mod.complete(
                self.client,
                messages,
                model=self.config.models.llm,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                tools=self.registry.tool_schemas() or None,
                reasoning=self.config.llm.reasoning,
            )
        except OpenRouterError as e:
            log.error("llm call failed: %s", e)
            return "Sorry, I couldn't reach my brain just then."

        if not reply.wants_tool:
            return reply.text or "Sorry, I didn't catch that."

        # The model picked a skill. Run them in order and speak the results.
        results = []
        for call in reply.tool_calls:
            log.info("tool -> %s(%s)", call.name, call.arguments)
            results.append(self._run(call.name, call.arguments))
        return " ".join(r for r in results if r)

    def _run(self, name: str, params: dict) -> str:
        skill = self.registry.get(name)
        if skill is None:
            # The model hallucinated a tool name, or a skill was removed.
            log.warning("no such skill: %s", name)
            return "I don't have a skill for that yet."
        if not skill.available:
            return skill.unavailable_reason
        try:
            return skill.run(**params)
        except TypeError as e:
            # Wrong or unexpected arguments -- a bad tool call, not a crash.
            log.warning("skill %s rejected params %s: %s", name, params, e)
            return f"I couldn't run {name.replace('_', ' ')} with those details."
        except Exception:
            log.exception("skill %s raised", name)
            return f"Something went wrong running {name.replace('_', ' ')}."
