"""Generate assets/greeting.wav -- what Faethon says when the service starts.

Rendered once and committed, rather than synthesised at every startup. A fixed
sentence does not need a network round-trip on each boot, and baking it in
means the greeting still plays when OpenRouter is unreachable, out of credit,
or the wifi is not up yet -- which is exactly when knowing the service came
back is worth most.

Re-run after changing GREETING:  uv run python scripts/make_greeting.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faethon.audio import playback          # noqa: E402
from faethon.config import ASSETS_DIR, load_config   # noqa: E402
from faethon.providers import tts as tts_mod         # noqa: E402
from faethon.providers.client import OpenRouterClient  # noqa: E402

GREETING = "Hi, I am up and running! Say my name whenever you need me."


def main() -> None:
    cfg = load_config()
    key = cfg.settings.openrouter_api_key.get_secret_value()
    if not key:
        raise SystemExit("No OPENROUTER_API_KEY -- needed once, to render the file.")

    rate: dict[str, int] = {}
    with OpenRouterClient(key) as client:
        pcm = b"".join(
            tts_mod.synthesize_stream(
                client, GREETING,
                model=cfg.models.tts,
                voice=cfg.tts.voice,
                on_rate=lambda r: rate.setdefault("hz", r),
            )
        )
        cost = client.spent

    hz = rate.get("hz", cfg.tts.sample_rate)
    out = ASSETS_DIR / "greeting.wav"
    playback.write_wav(out, pcm, hz)
    print(f'wrote {out}')
    print(f'  "{GREETING}"')
    print(f"  {len(pcm)} bytes, {len(pcm) / 2 / hz:.2f}s at {hz} Hz, cost ${cost:.5f}")


if __name__ == "__main__":
    main()
