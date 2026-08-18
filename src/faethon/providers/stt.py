"""Speech to text via OpenRouter.

POST /audio/transcriptions
  { "model": ..., "input_audio": { "data": <base64>, "format": "wav" } }
  -> { "text": ..., "usage": {...} }

The endpoint takes bare base64, NOT a data: URI.
"""

from __future__ import annotations

import base64
import io
import logging
import wave

from .client import OpenRouterClient

log = logging.getLogger(__name__)


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw s16le mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe(
    client: OpenRouterClient,
    pcm: bytes,
    *,
    model: str,
    sample_rate: int,
    language: str = "en",
    temperature: float = 0.0,
) -> str:
    """Transcribe raw PCM. Returns the text, stripped; "" if nothing was heard.

    `language` is worth setting. Whisper otherwise detects the language per
    request from the audio alone, and on a short, accented, or noisy clip it
    can guess wrong -- then transcribes *into* that language, so an English
    question comes back as Greek or Spanish. Pinning it removes the failure
    mode outright. Pass "" to restore auto-detection for a multilingual house.
    """
    wav = pcm_to_wav(pcm, sample_rate)
    payload: dict[str, object] = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(wav).decode("ascii"),
            "format": "wav",
        },
        "temperature": temperature,
    }
    if language:
        payload["language"] = language
    data = client.post_json("/audio/transcriptions", payload)
    text = (data.get("text") or "").strip()
    log.info("stt (%.1fs audio): %r", len(pcm) / 2 / sample_rate, text)
    return text
