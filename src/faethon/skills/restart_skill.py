"""Restart the assistant by voice.

Needs no privileges, which is the whole reason this is the only one of the two
restart skills that exists. The unit is `Restart=always` with `RestartSec=5`,
so the process simply exits and systemd brings it back about five seconds
later. Rebooting the Pi would have meant a polkit rule or a sudoers entry, and
was dropped rather than granted.

Two things this has to get right, both of which are invisible until they go
wrong:

* The exit happens in `after_reply`, not `run`. Killing the process inside
  `run()` means the reply is never spoken, and silence followed by a dead
  assistant looks exactly like a crash.
* `regex_only` keeps it out of the LLM's tool list. The model is helpful, and
  "this keeps freezing, what should I do?" should not put a restart within
  reach of something that decides for itself when to use its tools.

Memory is RAM-only, so restarting also forgets the conversation. That is the
existing behaviour of a restart, not something added here.
"""

from __future__ import annotations

import logging
import sys

from .base import Skill

log = logging.getLogger(__name__)


class RestartSkill(Skill):
    name = "restart_assistant"
    tag = "utility"
    regex_only = True
    description = (
        "Restart the Faethon assistant process. It comes back automatically "
        "after a few seconds."
    )

    patterns = [
        r"\brestart yourself\b",
        r"\brestart (?:the )?(?:assistant|service)\b",
        # However Whisper spells the two names it might hear.
        r"\brestart (?:fa?e|pha?e)thon\b",
        r"\brestart rh?as+py\b",
        r"\breboot yourself\b",
        r"\breset yourself\b",
        r"\breboot (?:the )?(?:assistant|service)\b",
        # Bare "restart" stays out on purpose, and a test pins it. Everything
        # else here was widened because a model that claims an action without
        # performing it is worse than a refusal -- but restarting inverts that.
        # An unwanted restart drops a conversation and costs ~20s of boot; a
        # model merely saying "restarting now" and not doing it is confusing
        # and harmless. So this is the one skill where the false positive is
        # the more expensive mistake, and it stays narrow.
    ]

    def run(self, **params: object) -> str:
        # Nothing happens here: the process has to survive long enough to say
        # this out loud. See after_reply.
        return "Restarting now. Back in a few seconds."

    def after_reply(self) -> None:
        """Exit, and let systemd start us again.

        SystemExit rather than os._exit: it unwinds the capture stream's
        context manager on the way out, so arecord is stopped properly instead
        of being orphaned holding the microphone.
        """
        log.info("restarting on request: exiting for systemd to bring us back")
        sys.exit(0)


SKILL = RestartSkill()
