"""The skill contract.

One file per skill, each exposing a module-level ``SKILL``. A skill declares
its metadata once and becomes reachable two ways:

  * regex -- ``patterns`` are tried against the transcript locally, costing
    nothing and adding no latency
  * tool-calling -- ``description`` and ``parameters`` are handed to the LLM,
    which can invoke the skill with extracted arguments when the phrasing
    wasn't anticipated

Writing a skill means filling in both, then dropping the file in this package.
The registry finds it; nothing else needs editing.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class Skill(ABC):
    #: Identifier used as the tool-calling function name. Keep it snake_case.
    name: ClassVar[str]
    #: Grouping tag: "home", "utility", "media", ...
    tag: ClassVar[str]
    #: Shown to the LLM to decide whether this skill applies.
    description: ClassVar[str]
    #: Regex sources tried against the transcript. Named groups become params.
    patterns: ClassVar[list[str]] = []
    #: JSON Schema for the tool-calling parameters.
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    @property
    def available(self) -> bool:
        """Whether the skill can run right now.

        Override for skills with a dependency that may be missing -- an unset
        API key, an unreachable hub. A skill that matches but is unavailable
        gets a spoken explanation rather than silence.
        """
        return True

    @property
    def unavailable_reason(self) -> str:
        return f"The {self.name.replace('_', ' ')} skill isn't available right now."

    def match(self, text: str) -> dict[str, str] | None:
        """Return extracted params if a pattern matches, else None.

        An empty dict is a valid match (skill takes no parameters), which is
        why this returns None rather than {} for "no match".
        """
        for pattern in self._compiled:
            m = pattern.search(text)
            if m:
                return {k: v for k, v in m.groupdict().items() if v is not None}
        return None

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def run(self, **params: Any) -> str:
        """Perform the skill. Returns what Roxy should say back.

        Must tolerate being called with no arguments: the regex path may match
        a pattern with no capture groups, and a model may omit optional ones.
        """
