"""Text to speech via OpenRouter.

POST /audio/speech
  { "model": ..., "input": ..., "voice": ..., "response_format": "pcm" }
  -> raw audio bytes (NOT JSON)

We ask for pcm rather than mp3 so chunks can go straight to aplay as they
arrive -- no decoder in the middle, and Faethon starts talking before synthesis
has finished.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from .client import OpenRouterClient

log = logging.getLogger(__name__)

# Big enough to avoid syscall churn, small enough that time-to-first-sound
# stays under a frame or two of audio.
CHUNK_BYTES = 4096


def rate_from_content_type(content_type: str) -> int | None:
    """Pull the sample rate out of e.g. 'audio/pcm;rate=44100;channels=1'.

    TTS models disagree about this -- Kokoro and Aura-2 return 24 kHz, Fish
    Audio returns 44.1 kHz -- and playing at the wrong rate isn't subtle: 44.1k
    audio played as 24k runs at 0.54x speed. Taking it from the response beats
    hardcoding a number that silently rots when the model changes.
    """
    if "rate=" not in content_type:
        return None
    try:
        return int(content_type.split("rate=")[1].split(";")[0].strip())
    except (ValueError, IndexError):
        return None


def estimate_cost(text: str, cost_per_1k_chars: float) -> float:
    """What speaking `text` cost, near enough.

    An estimate because the speech endpoint returns raw audio and no usage
    body, so unlike every other leg there is nothing authoritative to read.
    /generation returns 404 for TTS ids, and /credits updates in batches too
    coarse to attribute a single call.

    Billed on the text sent rather than the audio produced: OpenRouter quotes
    fish-audio/s1 at 0.000015 per input token, and five calls totalling 345
    characters cost $0.001107 -- $0.0032 per thousand characters, which agrees
    with the quoted price at roughly four characters to the token.

    Leaving this uncounted understated a turn several times over, because
    speaking is the most expensive thing Faethon does.
    """
    return len(text) / 1000 * cost_per_1k_chars


def synthesize_stream(
    client: OpenRouterClient,
    text: str,
    *,
    model: str,
    voice: str | None,
    on_rate: Callable[[int], None] | None = None,
    cost_per_1k_chars: float = 0.0,
) -> Iterator[bytes]:
    """Yield raw s16le mono PCM chunks as they arrive.

    `on_rate` is called with the response's sample rate before any audio, so
    the caller can open the output device at the right rate.

    `voice` may be None, which is not the same as a default: a provider that
    accepts an empty voice may pick a different speaker per request, so Faethon
    sounds like a different person every time. Send one where the provider has
    one. An unknown voice is a 400, and the error lists the valid ones.
    """
    # Empty input makes the provider return 400, and whitespace-only burns a
    # request to get zero bytes back. Neither should reach the network.
    if not text.strip():
        return

    payload: dict[str, object] = {
        "model": model,
        "input": text,
        "response_format": "pcm",
    }
    if voice:
        payload["voice"] = voice

    if cost_per_1k_chars:
        client.record_usage({"cost": estimate_cost(text, cost_per_1k_chars)})

    with client.post_stream("/audio/speech", payload) as r:
        if on_rate is not None:
            rate = rate_from_content_type(r.headers.get("content-type", ""))
            if rate:
                on_rate(rate)
        for chunk in r.iter_bytes(CHUNK_BYTES):
            if chunk:
                yield chunk
