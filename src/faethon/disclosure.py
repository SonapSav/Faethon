"""A ledger of everything that leaves this machine.

Counted at the HTTP layer, which is the only place that cannot be wrong. The
obvious source was the turn log, and it would have understated the truth by
about half: 127 transcripts came back in a day against 74 rows logged, because
a turn records a completed exchange while a single exchange is three or four
requests -- speech in, a completion, then two chunks of speech out. Counting
turns and calling it "what you sent" measures the conversation, not the
disclosure.

So this counts requests, and it lives where requests are made.

**What** left matters more than how many times. OpenRouter receives the sound
of the room; Open-Meteo receives your coordinates to about ten metres. Both are
"a call to a server" and they are not remotely the same disclosure, so every
record carries a kind rather than only a host. Thirty calls to a weather API
sounds like nothing until it is said as "your home location, thirty times".

Deliberately no payloads and no transcripts -- not the audio, not the text, not
the reply. A ledger that could recite what you said back to you would have to
be storing what you said, which defeats the thing it exists to reassure you
about. Host, path, kind, and whether a person asked for it. Nothing else.

Same shape as the turn log next to it: append-only JSONL in the state
directory, rotating on size, and it never raises. A ledger that can break a
turn is worse than no ledger.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterator

from . import state

log = logging.getLogger(__name__)

NAME = "disclosures.jsonl"

#: What a request hands over. The point of the whole file.
VOICE = "voice"        # audio recorded in this room
TEXT = "text"          # words you said, or words it is about to say
LOCATION = "location"  # where this house is
ACCOUNT = "account"    # billing metadata, nothing about you or the room

#: Endpoint -> what it discloses. Anything unlisted is assumed to carry text,
#: which is the cautious reading: better to overstate a disclosure than to
#: quietly leave a new endpoint out of the ledger entirely.
OPENROUTER_KINDS = {
    "/audio/transcriptions": VOICE,
    "/audio/speech": TEXT,
    "/chat/completions": TEXT,
    "/credits": ACCOUNT,
}


def kind_for(path: str) -> str:
    for suffix, kind in OPENROUTER_KINDS.items():
        if path.endswith(suffix):
            return kind
    return TEXT


class Ledger:
    """Appends one record per outbound request."""

    def __init__(self, enabled: bool = True, max_bytes: int = 5_000_000) -> None:
        self.enabled = enabled
        self.max_bytes = max_bytes
        #: Set False by the wake loop's background ticks, so the ledger can
        #: separate "you asked" from "it phoned out with nobody in the room".
        #: That second category is the one nobody anticipates.
        self.asked = True

    def record(self, host: str, path: str, kind: str, asked: bool | None = None) -> None:
        """Note one request. Never raises."""
        if not self.enabled:
            return
        try:
            file = state.state_dir() / NAME
            if self.max_bytes and file.exists() and file.stat().st_size >= self.max_bytes:
                file.replace(file.with_suffix(".jsonl.1"))
            record = {
                "at": round(time.time(), 3),
                "host": host,
                "path": path,
                "kind": kind,
                "asked": self.asked if asked is None else asked,
            }
            with file.open("a") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except (OSError, TypeError, ValueError) as e:
            log.warning("could not write disclosure ledger: %s", e)

    def withheld(self, reason: str = "no_speech") -> None:
        """Note a moment the microphone was live and nothing was sent.

        The most reassuring true fact available, and the only one that leaves
        no trace anywhere else: a follow-up window that hears nothing makes no
        request at all, so without recording it the ledger can only ever count
        what went out.
        """
        self.record("", reason, "withheld", asked=True)

    def read(self, since_seconds: float | None = None) -> list[dict[str, Any]]:
        """Records, newest last. Unreadable lines are skipped, not fatal."""
        out: list[dict[str, Any]] = []
        cutoff = time.time() - since_seconds if since_seconds else None
        for path in (state.state_dir() / NAME, ):
            try:
                with path.open() as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if cutoff is None or row.get("at", 0) >= cutoff:
                            out.append(row)
            except FileNotFoundError:
                continue
            except OSError as e:
                log.warning("could not read disclosure ledger: %s", e)
        return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold records into the shape both the spoken answer and the report need."""
    sent = [r for r in rows if r.get("kind") != "withheld"]
    return {
        "calls": len(sent),
        "withheld": len(rows) - len(sent),
        "unasked": sum(1 for r in sent if not r.get("asked", True)),
        "by_kind": _count(r.get("kind", "?") for r in sent),
        "by_host": _count(r.get("host", "?") for r in sent),
    }


def _count(values: Iterator[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


#: One ledger for the process, because the client and the skills that reach
#: the network directly must write to the same file.
LEDGER = Ledger()
