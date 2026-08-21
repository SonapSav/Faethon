"""Forget the conversation so far.

Faethon keeps the last ten exchanges in RAM so pronouns carry across turns
("what's the capital of France?" -> "how big is it?"). That is useful right up
until it isn't: a wrong answer that stays in context gets built on, and anything
said in front of a guest is still there for the next question.

The wipe deliberately includes the exchange that asked for it. Otherwise the
first thing in the fresh buffer is "clear the buffer" / "Memory is cleared",
which is both useless as context and confusing to ask about afterwards. The
router does the clearing -- skills hold no reference to memory -- via the
``clears_memory`` flag.
"""

from __future__ import annotations

from .base import Skill


class ClearMemorySkill(Skill):
    name = "clear_memory"
    tag = "utility"
    clears_memory = True
    description = (
        "Forget the conversation so far, discarding everything Faethon "
        "remembers of it. Use when the user wants to clear the memory or "
        "buffer, or start a fresh conversation."
    )

    patterns = [
        r"\bclear the buffer\b",
        r"\bclear (?:the |your |our )?memory\b",
        r"\bforget (?:the |our |this )?(?:conversation|everything)\b",
        r"\b(?:start a |begin a )?(?:new|fresh) conversation\b",
        # Clearing is destructive, so it must not depend on the model choosing
        # to call it -- a model that says "Memory is cleared" without calling
        # anything leaves someone believing a conversation was forgotten when
        # every word of it is still in the buffer.
        r"\bforget what we (?:said|talked about|discussed|were saying)\b",
        r"\b(?:wipe|erase|delete|reset|dump) (?:the |your |our )?memory\b",
        r"\bforget (?:it |that |all of it )?(?:all|everything)\b",
    ]

    def run(self, **params: object) -> str:
        # The clearing itself belongs to the router, which owns the memory.
        # Reaching this line at all means the skill ran, and a deque clear
        # cannot fail, so there is no failure case to report.
        return "Memory is cleared."


SKILL = ClearMemorySkill()
