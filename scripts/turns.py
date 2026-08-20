"""Summarise the turn log -- the point of keeping one.

    uv run python scripts/turns.py [--days N]

Reads /var/lib/faethon/turns.jsonl (or the local state dir), and prints what
the thresholds in config.yaml would want to know: where turns actually go, how
long each leg really takes including its tail, what it costs, and how long
conversations run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faethon import state          # noqa: E402
from faethon.turnlog import NAME   # noqa: E402


#: Below this, a monthly projection says more about when you ran the script
#: than about what Faethon costs.
MIN_SPAN_TO_PROJECT = 6 * 3600


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=0, help="only the last N days")
    args = ap.parse_args()

    path = state.state_dir() / NAME
    if not path.exists():
        raise SystemExit(f"no turn log at {path} yet")

    cutoff = time.time() - args.days * 86400 if args.days else 0
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("at", 0) >= cutoff:
            rows.append(row)

    if not rows:
        raise SystemExit("no turns in that window")

    span = max(r["at"] for r in rows) - min(r["at"] for r in rows)
    print(f"{len(rows)} turns over {span / 3600:.1f} hours\n")

    print("where they went")
    for route, n in Counter(r.get("route", "?") for r in rows).most_common():
        print(f"  {route:26} {n:5}  {100 * n / len(rows):4.0f}%")

    print("\nlatency, seconds")
    for leg in ("stt_s", "reply_s", "total_s"):
        v = [r[leg] for r in rows if leg in r]
        if v:
            print(f"  {leg[:-2]:8} median {percentile(v, .5):6.2f}   "
                  f"p90 {percentile(v, .9):6.2f}   max {max(v):6.2f}")

    total = sum(r.get("cost", 0.0) for r in rows)
    line = (f"\ncost  ${total:.4f} over {span / 3600:.1f} hours"
            f"   (${total / max(len(rows), 1):.5f} a turn")
    # Only project from a window long enough to mean anything. Extrapolating a
    # month from two minutes of testing produces a number that is both absurd
    # and alarming -- an early run of this printed "$292.21 a month".
    if span >= MIN_SPAN_TO_PROJECT:
        line += f", ${total / (span / 86400) * 30:.2f} a month at this rate"
    else:
        line += ", too short a window to project a month from"
    print(line + ")")

    wake = sum(1 for r in rows if r.get("wake"))
    print(f"\nconversations: {wake} started by a wake word, "
          f"{len(rows) - wake} follow-ups")
    if wake:
        print(f"  turns per conversation: {len(rows) / wake:.1f} average")
    held = [r.get("held", 0) for r in rows]
    print(f"  memory held: median {percentile(held, .5):.0f}, max {max(held)}")
    interrupted = sum(1 for r in rows if r.get("interrupted"))
    if interrupted:
        print(f"  interrupted: {interrupted}")


if __name__ == "__main__":
    main()
