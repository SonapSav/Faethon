"""Render Faethon's fixed spoken lines to assets/*.wav.

Everything here is something Faethon has to be able to say when it cannot
reach OpenRouter -- which is most of the point. The failure clips exist
because every one of these states is otherwise silent: the mechanism for
speaking is the thing that has broken. Synthesising an apology for a network
outage over the network does not work.

The greeting is here for a different reason: a fixed sentence needs no round
trip on each boot, and it still plays when the wifi is not up yet.

Keep the assistant's name out of these. "Rhasspy" is most of the wake word --
an early greeting draft saying it measured 0.5036 on the wake model, against a
threshold of 0.7.

Re-run after editing PHRASES:  uv run python scripts/make_speech.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faethon.audio import playback          # noqa: E402
from faethon.config import ASSETS_DIR, load_config   # noqa: E402
from faethon.providers import tts as tts_mod         # noqa: E402
from faethon.providers.client import OpenRouterClient  # noqa: E402

PHRASES = {
    "greeting": "Hi, I am up and running! Say my name whenever you need me.",
    # Distinct clips for distinct fixes: one means check the router, the other
    # means top up the account. Hearing the wrong one sends you the wrong way.
    "no-network": "I can't reach the network right now.",
    "no-credit": "My OpenRouter credit has run out.",
    # Wording is coupled to credit.warn_below in config.yaml -- "half a dollar"
    # is only true because the threshold is 0.50. A test pins them together.
    "low-credit": "Your account is below half a dollar.",
    "no-mic": "I can't hear the microphone.",
    # Only played to someone who heard the failure. Announcing that the
    # microphone is back, to a person who never knew it had gone, is noise.
    "mic-back": "I can hear you again.",
    # Played by systemd, not by Faethon, when the service itself cannot run.
    "failed": "Something has gone wrong and I have stopped. Check the logs.",
}


def main() -> None:
    cfg = load_config()
    key = cfg.settings.openrouter_api_key.get_secret_value()
    if not key:
        raise SystemExit("No OPENROUTER_API_KEY -- needed once, to render these.")

    total = 0.0
    with OpenRouterClient(key) as client:
        for name, text in PHRASES.items():
            rate: dict[str, int] = {}
            pcm = b"".join(
                tts_mod.synthesize_stream(
                    client, text,
                    model=cfg.models.tts,
                    voice=cfg.tts.voice,
                    on_rate=lambda r: rate.setdefault("hz", r),
                )
            )
            hz = rate.get("hz", cfg.tts.sample_rate)
            out = ASSETS_DIR / f"{name}.wav"
            playback.write_wav(out, pcm, hz)
            print(f"  {out.name:14} {len(pcm) / 2 / hz:5.2f}s at {hz} Hz  \"{text}\"")
        total = client.spent
    print(f"\nrendered {len(PHRASES)} clips, cost ${total:.5f}")


if __name__ == "__main__":
    main()
