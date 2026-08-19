"""How much OpenRouter credit is left.

    GET /credits -> {"data": {"total_credits": 30, "total_usage": 28.23}}

The balance is the difference; OpenRouter does not return it directly.

Most of the care here is in the regex. "OpenRouter" is a coined word, so
Whisper spells it however it sounds -- OpenRouter, Open Router, OpenRooter,
Open Rooter -- and a pattern matching only the correct spelling would work
until it didn't, silently falling through to the LLM which cannot see the
balance and would invent one. The pattern therefore accepts the vowel cluster
loosely rather than exactly.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..providers.client import OpenRouterClient, OpenRouterError
from .base import Skill

log = logging.getLogger(__name__)

#: Short: this runs inside a spoken turn, and the default 60s would leave the
#: user standing in silence wondering whether it heard them.
TIMEOUT_SEC = 10.0

#: However Whisper decides to spell it.
_OPENROUTER = r"open[\s.\-]*r[oeu]{1,3}ter"


class CreditSkill(Skill):
    name = "get_credit_balance"
    tag = "utility"
    description = (
        "Report how much OpenRouter API credit is left on the account, in "
        "dollars. Use when the user asks about their credit, balance, or how "
        "much they have left to spend."
    )

    patterns = [
        rf"\b{_OPENROUTER}\b.{{0,30}}\b(?:credit|credits|balance|funds)\b",
        rf"\b(?:credit|credits|balance|funds)\b.{{0,30}}\b{_OPENROUTER}\b",
        r"\bwhat(?:'s|s| is)? my (?:credit |account )?balance\b",
        r"\bhow much (?:credit|money) (?:do i have|is (?:there )?left|have i got)\b",
        r"\bhow much (?:do i have|have i got) left\b",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._key: str | None = None
        self._looked = False

    @property
    def api_key(self) -> str:
        if not self._looked:
            self._looked = True
            self._key = Settings().openrouter_api_key.get_secret_value() or None
        return self._key or ""

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def unavailable_reason(self) -> str:
        return "I don't have an OpenRouter key to check the balance with."

    def run(self, **params: object) -> str:
        if not self.available:
            return self.unavailable_reason

        try:
            with OpenRouterClient(self.api_key, timeout=TIMEOUT_SEC) as client:
                data = client.get_json("/credits").get("data") or {}
            granted = float(data["total_credits"])
            used = float(data["total_usage"])
        except OpenRouterError as e:
            log.error("credit lookup failed: %s", e)
            return "Sorry, I couldn't reach OpenRouter to check that."
        except (KeyError, TypeError, ValueError) as e:
            # The endpoint changed shape. Say so rather than reading out a zero.
            log.error("unexpected /credits response: %s", e)
            return "OpenRouter gave me an answer I didn't understand."

        balance = granted - used
        log.info("credit: %.4f granted, %.4f used, %.4f left", granted, used, balance)

        # Spelled out rather than "$", and always two decimals as asked. A bare
        # "1.8" would be read as one point eight, which is not how money sounds.
        if balance <= 0:
            return f"Your OpenRouter balance is {max(balance, 0):.2f} dollars. You're out of credit."
        return f"Your OpenRouter balance is {balance:.2f} dollars."


SKILL = CreditSkill()
