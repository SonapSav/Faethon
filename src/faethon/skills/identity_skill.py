"""What Faethon is, who built it, and what it isn't.

Identity is the one category where the model's priors are guaranteed wrong:
Faethon postdates the training data, so any question about itself has an answer
that can only come from the prompt. When the prompt said "You are Rhasspy, a
voice assistant" and nothing else, the model did the only thing available and
described the Rhasspy project -- a real, well-documented open-source assistant
it does know about. Fluently, confidently, and wrongly:

    Q: are you related to the Rhasspy project
    A: Yes, I am named after the Rhasspy project ... built by the same team.

That is not a harmless slip. There is no transcript to scroll back through, so
whoever hears it believes it; and the answer lands in the ten-turn buffer and
is sent back as context, so the model stays consistent with its own
confabulation for the rest of the conversation.

The prompt now grounds the identity positively and never mentions the other
project -- measured to matter, see config.yaml. This skill is the other half:
these questions have one correct answer that never changes, so they should not
reach a model at all. Free, instant, offline, and unarguable.

Deliberately NOT handled here: "what can you do". That answer is dynamic --
the model reads it off the live tool schemas, so it updates itself as skills
are added. Hardcoding a capability list would rot on the next skill. Identity
is fixed; capabilities are not.
"""

from __future__ import annotations

from .base import Skill

#: The question has to end here, or "what are you doing" answers with a
#: description of the hardware. Same trap as "what is the time complexity".
_END = r"(?:\s+(?:exactly|then|anyway|really|please))*[^\w]*$"


class IdentitySkill(Skill):
    name = "describe_self"
    tag = "utility"
    description = (
        "Say what Faethon is, who built it, or how it differs from other "
        "assistants. Use for questions about Faethon itself -- what it is, "
        "its name, its origin, or whether it is some other product."
    )

    patterns = [
        rf"\bwhat are you{_END}",
        rf"\bwho are you{_END}",
        rf"\bwhat(?:'s|s| is) your name{_END}",
        rf"\bwhat are you called{_END}",
        rf"\bwho (?P<origin>made|built|created|designed|wrote|programmed) you\b",
        # The article is optional: "are you human" as well as "are you a robot".
        r"\bare you (?P<other>chat\s?gpt|gpt|siri|alexa|google|"
        r"(?:an? )?(?:ai|a\.i\.|robot|human|person|bot))\b",
        # Why the wake phrase is what it is. The user's own question puts the
        # token back in context, so the prompt cannot help -- measured, the
        # model answers "Rhasspy is the name of the software running on this
        # Raspberry Pi", which is false. Loose spelling, as ever: Whisper
        # writes a coined word however it sounds.
        r"\bwhy\b.{0,30}\bsay\b.{0,24}(?P<wake>rh?as+p{0,2}y)\b",
        r"\bwhy (?:are|is) (?:you|it|that) called\b.{0,24}(?P<wake>rh?as+p{0,2}y)\b",
        r"\bwhat does (?P<wake>rh?as+p{0,2}y) mean\b",
    ]

    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["identity", "origin", "comparison", "wake"],
                "description": (
                    "identity: what it is. origin: who built it. "
                    "comparison: whether it is some other assistant. "
                    "wake: why the wake phrase is what it is."
                ),
            }
        },
        "required": [],
    }

    def run(self, **params: object) -> str:
        kind = str(params.get("kind") or "").lower()
        if not kind:
            # The regex path signals intent by which group matched.
            if "origin" in params:
                kind = "origin"
            elif "wake" in params:
                kind = "wake"
            elif "other" in params:
                kind = "comparison"
            else:
                kind = "identity"

        if kind == "wake":
            # Deliberately does NOT repeat the phrase. Measured: Faethon
            # speaking its own wake word scores 0.9984 on the wake model
            # through the speaker and mic -- above the 0.7 that wakes it and
            # far above the 0.1 barge-in listens at. It would interrupt itself
            # mid-sentence, every time anyone asked.
            return (
                "That's just a trigger sound I listen for. It came with the "
                "wake word model I use, and it isn't my name."
            )
        if kind == "origin":
            # No company, no team, no product. Saying so plainly is what stops
            # the model inventing one.
            return (
                "I was built here at home, on a Raspberry Pi. "
                "There's no company behind me."
            )
        if kind == "comparison":
            # Answer positively and name what it is, rather than listing what
            # it isn't -- denial is what invited the confabulation in the
            # first place.
            return (
                "No, I'm Faethon. I run on a Raspberry Pi in this house, "
                "though I do use a cloud model to think."
            )
        return (
            "I'm Faethon, a voice assistant on a Raspberry Pi in this house. "
            "Only my listening is local. I use cloud services to think and speak."
        )


SKILL = IdentitySkill()
