"""Generate assets/ack.wav -- the sound Faethon makes when the wake word lands.

Two short ascending sine blips with a raised-cosine envelope. The envelope
matters: a bare sine switched on and off clicks audibly.

Run with:  uv run python scripts/make_chime.py
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

RATE = 24000
NOTES = [(880.0, 0.075), (1318.5, 0.110)]  # A5 then E6
GAP = 0.02
AMPLITUDE = 0.28


def tone(freq: float, dur: float) -> list[float]:
    n = int(RATE * dur)
    out = []
    for i in range(n):
        # Raised-cosine envelope over the whole blip: no click at either edge.
        env = 0.5 * (1 - math.cos(2 * math.pi * i / n))
        out.append(AMPLITUDE * env * math.sin(2 * math.pi * freq * i / RATE))
    return out


def main() -> None:
    samples: list[float] = []
    for idx, (freq, dur) in enumerate(NOTES):
        if idx:
            samples += [0.0] * int(RATE * GAP)
        samples += tone(freq, dur)

    pcm = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)

    out = Path(__file__).resolve().parents[1] / "assets" / "ack.wav"
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    print(f"wrote {out} ({len(pcm)} bytes, {len(samples) / RATE:.2f}s)")


if __name__ == "__main__":
    main()
