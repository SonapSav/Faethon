"""Audio playback via `aplay`.

Two modes:
  play_bytes  -- blocking, for short fully-buffered clips (the ack chime)
  stream      -- feed PCM in as it arrives, so TTS starts speaking before
                 synthesis has finished

A stream can also be aborted, which is how barge-in stops a reply mid-sentence.
That is a different operation from ending one: closing the pipe lets `aplay`
play out everything already buffered, which for a long reply means it keeps
talking for seconds after being told to stop. Measured on this Pi, from the
instant of the decision to actual silence:

    close stdin and wait   9112 ms
    terminate (SIGTERM)     234 ms
    kill (SIGKILL)          110 ms

SIGTERM is the one to want: near enough instant, and it lets aplay close the
PCM device rather than leaving ALSA to clean up after it.
"""

from __future__ import annotations

import logging
import subprocess
import threading
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


class Sink:
    """Somewhere to write PCM: call it to play, `abort()` to stop dead.

    Callable rather than a plain method so every existing `write(chunk)` call
    site keeps working.
    """

    def __init__(self, write, abort) -> None:
        self._write = write
        self._abort = abort

    def __call__(self, chunk: bytes) -> None:
        self._write(chunk)

    def abort(self) -> None:
        """Stop now, discarding whatever is already buffered.

        Safe to call from another thread, and safe to call twice. This is what
        barge-in uses; leaving the stream to close normally would play the rest
        of the reply out regardless.
        """
        self._abort()


@contextmanager
def stream(device: str, sample_rate: int):
    """Yield a `Sink` that pipes PCM straight to the speaker as it arrives."""
    proc = subprocess.Popen(
        _cmd(device, sample_rate),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    aborted = threading.Event()

    def write(chunk: bytes) -> None:
        if aborted.is_set():
            return
        try:
            proc.stdin.write(chunk)
        except (BrokenPipeError, ValueError):
            # ValueError: aborted between the check above and the write.
            if not aborted.is_set():
                log.warning("playback pipe closed early")

    def abort() -> None:
        if aborted.is_set():
            return
        # Set this first: the play thread may be blocked in write() under
        # aplay's backpressure, and terminating unblocks it with a broken pipe
        # that must not be logged as a fault.
        aborted.set()
        proc.terminate()

    try:
        yield Sink(write, abort)
    finally:
        if not aborted.is_set():
            try:
                proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        proc.wait()


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
