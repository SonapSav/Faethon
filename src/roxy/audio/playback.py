"""Audio playback via `aplay`.

Two modes:
  play_bytes  -- blocking, for short fully-buffered clips (the ack chime)
  stream      -- feed PCM in as it arrives, so TTS starts speaking before
                 synthesis has finished
"""

from __future__ import annotations

import logging
import subprocess
import wave
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


def _cmd(device: str, sample_rate: int) -> list[str]:
    return [
        "aplay",
        "-D", device,
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", "1",
        "-t", "raw",
        "-q",
        "-",
    ]


def play_bytes(pcm: bytes, device: str, sample_rate: int) -> None:
    """Play raw s16le mono PCM and wait for it to finish."""
    subprocess.run(_cmd(device, sample_rate), input=pcm, check=False)


def play_wav(path: Path, device: str) -> None:
    """Play a WAV file. Rate comes from the file header."""
    subprocess.run(["aplay", "-D", device, "-q", str(path)], check=False)


def play_wav_async(path: Path, device: str) -> subprocess.Popen:
    """Start playing a WAV without blocking -- used for the ack chime so we can
    begin recording the user's request immediately."""
    return subprocess.Popen(
        ["aplay", "-D", device, "-q", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@contextmanager
def stream(device: str, sample_rate: int):
    """Yield a write() that pipes PCM straight to the speaker as it arrives."""
    proc = subprocess.Popen(
        _cmd(device, sample_rate),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    def write(chunk: bytes) -> None:
        try:
            proc.stdin.write(chunk)
        except BrokenPipeError:
            log.warning("playback pipe closed early")

    try:
        yield write
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
