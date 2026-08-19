"""Synthesise the same sentence in several voices, for auditioning.

    uv run python scripts/make_voice_samples.py

Writes wav files to /tmp/faethon-voices. Play them with scripts/try-voices.sh.

To see every voice a provider offers, send it a bogus one -- the 400 lists
them all:

    curl -s https://openrouter.ai/api/v1/audio/speech \
      -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      -d '{"model":"deepgram/aura-2","input":"x","voice":"?"}'
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

from faethon.config import load_config
from faethon.providers import tts as tts_mod
from faethon.providers.client import OpenRouterClient, OpenRouterError

OUT = Path("/tmp/faethon-voices")

# A line with a name, a time, and a normal sentence -- enough to judge how a
# voice handles the things the assistant actually says.
TEXT = (
    "Hi, I'm Rhasspy. It's twenty past nine, and the sky looks blue "
    "because sunlight scatters off molecules in the air."
)

VOICES = [
    ("aura-2-thalia-en", "F", "Deepgram's default"),
    ("aura-2-asteria-en", "F", "warm, conversational"),
    ("aura-2-luna-en", "F", "younger, brighter"),
    ("aura-2-athena-en", "F", "calm, measured"),
    ("aura-2-orion-en", "M", "neutral, clear"),
    ("aura-2-apollo-en", "M", "confident"),
    ("aura-2-arcas-en", "M", "warm, natural"),
]


def main() -> int:
    cfg = load_config()
    model = cfg.models.tts
    if "aura-2" not in model:
        print(f"note: config uses {model}; these samples are Deepgram Aura-2 voices")
        model = "deepgram/aura-2"

    OUT.mkdir(exist_ok=True)
    client = OpenRouterClient(cfg.settings.openrouter_api_key.get_secret_value())

    print(f"{'file':<26}{'sex':<5}{'character':<24}{'audio':>8}")
    print("-" * 63)
    for voice, sex, desc in VOICES:
        rate = [cfg.tts.sample_rate]
        started = time.monotonic()
        try:
            pcm = b"".join(
                tts_mod.synthesize_stream(
                    client, TEXT, model=model, voice=voice,
                    on_rate=lambda r: rate.__setitem__(0, r),
                )
            )
        except OpenRouterError as e:
            print(f"{voice:<26}{sex:<5}FAILED: {str(e)[:40]}")
            continue

        seconds = len(pcm) / 2 / rate[0]
        path = OUT / f"{voice}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate[0])
            w.writeframes(pcm)

        # A sample far shorter than its peers means the provider truncated it.
        flag = "  <- suspiciously short" if seconds < 4 else ""
        print(f"{path.name:<26}{sex:<5}{desc:<24}{seconds:>7.2f}s{flag}")

    print(f"\nWrote to {OUT} in {time.monotonic() - started:.1f}s")
    print("Play them with: ./scripts/try-voices.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
