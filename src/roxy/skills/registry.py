"""Skill auto-discovery.

Imports every module in this package and collects their ``SKILL`` objects, so
adding a skill is a matter of adding a file.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any

from .base import Skill

log = logging.getLogger(__name__)


class Registry:
    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills or []:
            self.add(skill)

    def add(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill

    @classmethod
    def discover(cls) -> Registry:
        import roxy.skills as pkg

        registry = cls()
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name in ("base", "registry") or info.name.startswith("_"):
                continue
            module = importlib.import_module(f"{pkg.__name__}.{info.name}")
            skill = getattr(module, "SKILL", None)
            if skill is None:
                log.warning("roxy.skills.%s defines no SKILL; ignoring", info.name)
                continue
            registry.add(skill)

        log.info(
            "%d skill(s): %s",
            len(registry),
            ", ".join(f"{s.name}[{s.tag}]" for s in registry),
        )
        return registry

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self):
        return iter(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def by_tag(self, tag: str) -> list[Skill]:
        return [s for s in self if s.tag == tag]

    def match(self, text: str) -> tuple[Skill, dict[str, str]] | None:
        """First skill whose regex matches. Order is registration order."""
        for skill in self:
            params = skill.match(text)
            if params is not None:
                return skill, params
        return None

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Tool definitions for the LLM. Unavailable skills are omitted so the
        model can't offer something that would only fail."""
        return [s.tool_schema() for s in self if s.available]
