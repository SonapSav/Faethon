"""`faethon-sent` -- the full account of what left this machine.

The spoken answer is one sentence, because a sentence is what a voice
assistant gets. Everything worth knowing about a day's disclosures needs a
screen: which endpoints, how often, carrying what, and how much of it happened
with nobody in the room.

Grouped by what each request handed over rather than by host. A call to a
weather API and a call to a speech API are both "a request to a server", and
they are not the same thing to disclose.
"""

from __future__ import annotations

import argparse
import time

from . import disclosure

#: Plain descriptions. "voice" is accurate and "audio recorded in this room"
#: is what it means, which is the one that should be read.
KIND_MEANS = {
    disclosure.VOICE: "audio recorded in this room",
    disclosure.TEXT: "words you said, or words it said back",
    disclosure.LOCATION: "where this house is",
    disclosure.ACCOUNT: "billing metadata only",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="What Faethon sent to the cloud.")
    ap.add_argument("--days", type=float, default=1.0,
                    help="how far back to look (default: 1)")
    args = ap.parse_args()

    window = args.days * 86400
    rows = disclosure.LEDGER.read(since_seconds=window)
    if not rows:
        print("Nothing recorded. The ledger starts when the service does.")
        return

    s = disclosure.summarise(rows)
    span = max(r["at"] for r in rows) - min(r["at"] for r in rows)
    print(f"{s['calls']} requests left this machine over {span / 3600:.1f} hours\n")

    print("carrying what")
    for kind, n in s["by_kind"].items():
        print(f"  {kind:10} {n:5}   {KIND_MEANS.get(kind, ''):38}")

    print("\nto whom")
    for host, n in s["by_host"].items():
        print(f"  {host or '(nothing sent)':34} {n:5}   {100 * n / max(s['calls'], 1):4.1f}%")

    print("\nby endpoint")
    seen: dict[tuple[str, str], int] = {}
    for r in rows:
        if r.get("kind") == "withheld":
            continue
        key = (r.get("host", "?"), r.get("path", "?"))
        seen[key] = seen.get(key, 0) + 1
    for (host, path), n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  {host + path:58} {n:5}")

    # The two facts that are actually about consent rather than scale.
    print()
    if s["unasked"]:
        n = s["unasked"]
        # The article has to move with the number, or the plural reads
        # "a background checks that run".
        checks = ("a background check that runs" if n == 1
                  else "background checks that run")
        print(f"{n} of those went out with nobody asking -- {checks} "
              f"whether or not anyone is home.")
    if s["withheld"]:
        n = s["withheld"]
        print(f"{disclosure.times(n).capitalize()} the microphone was open and "
              f"nothing was sent: no speech was heard, so no audio left.")


if __name__ == "__main__":  # pragma: no cover
    main()
