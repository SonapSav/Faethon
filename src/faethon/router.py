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
        # `is not None`, not `or`: Memory defines __len__, so an empty one is
        # falsy and `or` would silently swap the caller's memory for a private
        # one. Faethon passes a fresh -- therefore empty -- Memory in, so that
        # is the normal case, not the edge case.
        self.memory = memory if memory is not None else Memory(
            config.llm.history_turns
        )
        #: Set for the duration of one turn when a skill wipes memory, so the
        #: exchange that asked for the wipe is not then recorded into it.
        self._skip_record = False

    def handle(self, text: str) -> str:
        """Return what Faethon should say in response to `text`."""
        if not text.strip():
            return ""
        self._skip_record = False

        spoken = self._local_skill_match(text)
        if spoken is None:
            spoken = self._llm_fallback(text)

        self._record(text, spoken)
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
        # Reset per turn rather than after use: a barge-in can abandon this
        # generator before the recording step, and a flag left set would
        # silently swallow the next turn instead.
        self._skip_record = False

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
            self._record(text, spoken)
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
            provider_sort=self.config.llm.provider_sort,
        )

        said: list[str] = []
        recorded = False

        def remember(spoken: str) -> None:
            """Record this exchange once, whichever way the generator ends."""
            nonlocal recorded
            if not recorded:
                recorded = True
                self._record(text, spoken)

        try:
            try:
                for chunk in sentence_chunks(reply):
                    said.append(chunk)
                    yield chunk
            except OpenRouterError as e:
                log.error("llm stream failed: %s", e)
                if not said:
                    spoken = "Sorry, I couldn't reach my brain just then."
                    yield spoken
                    remember(spoken)
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
            remember(spoken)
        finally:
            # Barge-in closes this generator part-way through. Record what was
            # said before the interruption: dropping it entirely would leave
            # the next turn's context claiming Faethon never answered, so
            # "what did you just say?" would draw a blank and asking again
            # would replay the whole reply from the top.
            remember(" ".join(said))

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
                provider_sort=self.config.llm.provider_sort,
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
            spoken = skill.run(**params)
        except TypeError as e:
            # Wrong or unexpected arguments -- a bad tool call, not a crash.
            log.warning("skill %s rejected params %s: %s", name, params, e)
            return f"I couldn't run {name.replace('_', ' ')} with those details."
        except Exception:
            log.exception("skill %s raised", name)
            return f"Something went wrong running {name.replace('_', ' ')}."

        if skill.clears_memory:
            # Now, not at the end of the turn: a barge-in can abandon this
            # exchange part-way, and "cleared, mostly" is not a useful state.
            self.memory.clear()
            self._skip_record = True
            log.info("memory cleared by %s", name)
        return spoken

    def _record(self, text: str, spoken: str) -> None:
        """Add the exchange to memory, unless a skill just wiped it.

        The wipe has to take the exchange that caused it as well, or the first
        entry in the empty buffer is the request to empty it.
        """
        if self._skip_record:
            return
        self.memory.add(text, spoken)
