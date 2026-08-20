"""Generate Faethon's two chimes.

    ack.wav   ascending  -- "go ahead, I'm listening"
    done.wav  descending -- "conversation over, wake me again"

Short sine blips with a raised-cosine envelope. The envelope matters: a bare
sine switched on and off clicks audibly.

The pair are deliberately the same two notes in opposite order. Rising and
falling is a distinction you hear without having to learn it, and it survives
being played quietly across a room, which a difference in timbre would not.

Run with:  uv run python scripts/make_chime.py
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

RATE = 24000
GAP = 0.02
AMPLITUDE = 0.28

A5, E6 = 880.0, 1318.5

CHIMES = {
    "ack.wav": [(A5, 0.075), (E6, 0.110)],
    # Quieter and a touch quicker than the ack: closing is an aside, not a
    # prompt, and it plays when nobody has spoken for five seconds.
    "done.wav": [(E6, 0.070), (A5, 0.100)],
}
# Three rising notes, so it is not mistakable for either of the two-note
# chimes: this one arrives when nobody asked for it and has to say "look up"
# rather than "go ahead" or "we're finished".
CHIMES["timer.wav"] = [(A5, 0.09), (E6, 0.09), (E6 * 1.5, 0.16)]
DONE_GAIN = 0.75


def tone(freq: float, dur: float) -> list[float]:
    n = int(RATE * dur)
    out = []
    for i in range(n):
        # Raised-cosine envelope over the whole blip: no click at either edge.
        env = 0.5 * (1 - math.cos(2 * math.pi * i / n))
        out.append(AMPLITUDE * env * math.sin(2 * math.pi * freq * i / RATE))
    return out


def main() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(exist_ok=True)

    for name, notes in CHIMES.items():
        gain = DONE_GAIN if name == "done.wav" else 1.0
        samples: list[float] = []
        for idx, (freq, dur) in enumerate(notes):
            if idx:
                samples += [0.0] * int(RATE * GAP)
            samples += [s * gain for s in tone(freq, dur)]

        pcm = b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples
        )

        out = assets / name
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        print(f"wrote {out} ({len(pcm)} bytes, {len(samples) / RATE:.2f}s)")


if __name__ == "__main__":
    main()
