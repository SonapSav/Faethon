"""`faethon-misses` -- what people said that reached no skill.

Every threshold in this project that turned out right was measured, and the
patterns are the one part still set by guessing at how somebody might phrase
something. This closes that: it reads what people actually said and reports
which of it matched nothing, so patterns can be extended from evidence.

The failure it exists for is quiet. A phrasing the regex misses does not error
-- it goes to the model, which usually answers well enough that nobody
notices. "Radio volume at 7" fell through for a day that way, and turned up
only because somebody said the radio was ignoring them. In the journal it looks
like a `heard:` line with no `router ->` line after it, which is not something
anyone would spot by reading.

Reads journald and stores nothing. That is deliberate: the transcripts are
already there, and whether the turn log should keep its own copy is a decision
left open. Answering "which phrasing missed" by starting a second transcript
store would settle that question by accident.

    faethon-misses --days 7
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys

HEARD = re.compile(r"faethon: heard: (.+)$")
ROUTED = re.compile(r"faethon\.router: (?:regex|tool) -> (\w+)\(")
STREAMED = re.compile(r"llm streamed \((stop|tool_calls)\)")

#: Regex syntax that survives a naive word grab and looks like vocabulary.
#: `\b` leaves "bwhat", a split group leaves "thon". Escapes and group names
#: come out before any words go in.
_ESCAPES = re.compile(r"\\[a-zA-Z]|\(\?P?[<:=!][\w>]*|\(\?|[\[\](){}|?*+^$]")
#: Ordinary English that happens to appear in a pattern. Left in, these
#: produce hints that are actively wrong -- "about" comes from "forget what we
#: talked about" and pointed "tell me a joke about fatherhood" at clear_memory,
#: "yourself" from "restart yourself" pointed a question about self-description
#: at restart_assistant. A wrong hint is worse than none: it sends somebody to
#: the wrong skill to add a pattern that does not belong there.
_NOISE = {
    # capture-group names and scaffolding
    "action", "addr", "duration", "level", "unit", "freq", "when", "step",
    "kind", "list", "thon",
    # ordinary words, whatever pattern they came from
    "about", "yourself", "does", "called", "going", "been", "here", "there",
    "know", "much", "many", "some", "into", "with", "from", "over", "just",
    "like", "want", "need", "give", "make", "take", "than", "then", "them",
    "please", "tell", "your", "this", "that", "what", "have", "they", "will",
    "would", "could", "should", "still", "again", "back", "down", "last",
    "next", "before", "after", "right", "left", "long", "well", "good",
    "nice", "better", "worse", "open", "close", "full", "current", "time",
    "date", "year", "today", "tomorrow", "night", "outside", "room",
}


def vocabulary() -> dict[str, str]:
    """Distinctive words each skill's own patterns look for.

    Derived from the patterns rather than listed by hand, so it cannot go
    stale the moment somebody adds a phrasing.
    """
    from .skills.registry import Registry

    seen: dict[str, str] = {}
    for skill in Registry.discover():
        for pattern in skill.patterns:
            for word in re.findall(r"[a-z]{4,}", _ESCAPES.sub(" ", pattern).lower()):
                if word in _NOISE:
                    continue
                # A word two skills both look for says nothing about which one
                # was meant, so it is worse than useless as a hint.
                seen[word] = skill.name if seen.get(word, skill.name) == skill.name else ""
    return {w: s for w, s in seen.items() if s}


def read_journal(days: float) -> list[str]:
    out = subprocess.run(
        ["journalctl", "-u", "faethon", "--since", f"-{days * 24:.0f}h",
         "--no-pager", "-o", "cat"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        print("could not read the journal:", out.stderr.strip()[:200], file=sys.stderr)
        return []
    return out.stdout.splitlines()


def classify(lines: list[str]) -> list[tuple[str, str, str]]:
    """(what was heard, outcome, skill) per turn, in order.

    Outcomes: `routed` a pattern matched; `rescued` the model called a tool a
    pattern could have; `answered` the model replied itself; `nothing` no skill
    and no reply, which is a hard miss.
    """
    turns: list[tuple[str, str, str]] = []
    heard: str | None = None
    outcome = "nothing"
    skill = ""

    def close() -> None:
        if heard is not None:
            turns.append((heard, outcome, skill))

    for line in lines:
        m = HEARD.search(line)
        if m:
            close()
            heard, outcome, skill = m.group(1).strip(), "nothing", ""
            continue
        if heard is None:
            continue
        m = STREAMED.search(line)
        if m:
            outcome = "rescued" if m.group(1) == "tool_calls" else "answered"
            continue
        m = ROUTED.search(line)
        if m:
            # A tool call after the model already counts as rescued -- the
            # distinction that matters is whether a *pattern* could have.
            outcome = "rescued" if outcome == "rescued" else "routed"
            skill = m.group(1)
    close()
    return turns


def normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def matches_now(text: str) -> bool:
    """Whether today's patterns would catch this.

    The journal reaches back further than the last fix, so without this the
    report fills with phrasings already dealt with -- six of the first nine it
    found had been fixed hours earlier. A list of solved problems is a list
    nobody reads twice.
    """
    from .skills.registry import Registry

    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry.discover()
    return _REGISTRY.match(normalise(text)) is not None


_REGISTRY = None


def looks_like_a_command(text: str, vocab: dict[str, str]) -> str:
    """The skill whose vocabulary this phrase uses, if exactly one."""
    hits = {vocab[w] for w in normalise(text).split() if w in vocab}
    return hits.pop() if len(hits) == 1 else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--min", type=int, default=1,
                    help="only show phrasings heard at least this often")
    args = ap.parse_args()

    turns = classify(read_journal(args.days))
    if not turns:
        print("Nothing in the journal for that window.")
        return

    vocab = vocabulary()
    counts = collections.Counter(t[0] for t in turns)
    outcome_of = {t[0]: t[1] for t in turns}
    routed = sum(1 for t in turns if t[1] == "routed")

    print(f"{len(turns)} turns over {args.days:g} days -- "
          f"{routed} matched a pattern, {len(turns) - routed} did not\n")

    def section(title: str, rows: list[tuple[str, int, str]], note: str) -> None:
        if not rows:
            return
        print(f"{title}\n  {note}")
        for text, n, hint in sorted(rows, key=lambda r: (-r[1], r[0])):
            arrow = f"   -> {hint}?" if hint else ""
            print(f"  {n:3}x  {text[:58]!r}{arrow}")
        print()

    likely, rescued, chatter, fixed = [], [], [], 0
    for text, n in counts.items():
        if n < args.min:
            continue
        outcome = outcome_of[text]
        if outcome == "routed":
            continue
        if matches_now(text):
            fixed += n
            continue
        hint = looks_like_a_command(text, vocab)
        if outcome == "rescued":
            rescued.append((text, n, hint))
        elif hint:
            likely.append((text, n, hint))
        else:
            chatter.append((text, n, ""))

    if fixed:
        print(f"({fixed} turn(s) that missed at the time would match today -- "
              f"not listed)\n")

    section("likely missed commands", likely,
            "used a skill's own vocabulary, but nothing matched -- "
            "these are patterns worth adding")
    section("rescued by the model", rescued,
            "the model called a tool a pattern could have: right answer, "
            "one round trip slower and dearer")
    section("reached the model, no tool", chatter,
            "mostly ordinary conversation; skim for anything that reads "
            "like an instruction")


if __name__ == "__main__":  # pragma: no cover
    main()
