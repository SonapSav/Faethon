"""Microphone capture via an `arecord` subprocess.

We shell out to arecord rather than binding PortAudio because alsa-utils is
already on every Pi OS image, the device strings are the same ones you'd test
with by hand, and ALSA's "plug" layer does rate/channel conversion for us.

The stream yields fixed-size frames of signed 16-bit little-endian mono PCM.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)

# One read() while draining. Only bounds the syscall size; drain loops.
DRAIN_CHUNK = 1 << 16


class CaptureError(RuntimeError):
    pass


class Stream:
    """A live capture stream: call it for the next frame, drain() to discard.

    Callable rather than a plain method so every existing `read_frame()` call
    site keeps working.
    """

    def __init__(self, read, drain) -> None:
        self._read = read
        self._drain = drain

    def __call__(self) -> bytes:
        """Block until the next full frame is available."""
        return self._read()

    def drain(self) -> int:
        """Discard everything buffered so far; return the bytes dropped.

        arecord keeps running while Faethon talks, so by the time a reply ends
        the pipe holds a recording of that reply. Anything that listens next
        would hear Faethon and answer its own sentence. Call this after
        speaking, and never before -- the audio arriving right after a wake
        word is the user running straight on into their question.

        One call is enough however long the reply was. The pipe caps at 64 kB
        (2.04s at 16 kHz mono, measured), and once it is full ALSA drops the
        overflow rather than queueing it behind: after draining an 8s reply,
        capture reads at 1.00x real time immediately. So this does not need to
        loop until quiet, and must not -- it would never terminate on a live
        microphone.
        """
        return self._drain()


@contextmanager
def open_stream(device: str, sample_rate: int, frame_bytes: int):
    """Run arecord and yield a `Stream` returning exactly `frame_bytes` per call."""
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", "1",
        "-t", "raw",
        "-q",
    ]
    log.debug("starting capture: %s", " ".join(cmd))
    # bufsize=0 keeps stdout a raw unbuffered reader. Draining has to see the
    # pipe directly: bytes sitting in a BufferedReader are invisible to the
    # non-blocking read below, and would survive the drain.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
    )

    def read_frame() -> bytes:
        # .read(n) on a pipe can return short; loop until we have a full frame.
        buf = b""
        while len(buf) < frame_bytes:
            chunk = proc.stdout.read(frame_bytes - len(buf))
            if not chunk:
                err = (proc.stderr.read() or b"").decode(errors="replace").strip()
                raise CaptureError(f"capture stream ended{': ' + err if err else ''}")
            buf += chunk
        return buf

    def drain() -> int:
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        dropped = 0
        try:
            while True:
                try:
                    chunk = proc.stdout.read(DRAIN_CHUNK)
                except BlockingIOError:
                    break
                # None: nothing buffered right now. b"": the stream ended --
                # let the next read_frame() raise, rather than failing here.
                if not chunk:
                    break
                dropped += len(chunk)
        finally:
            os.set_blocking(fd, True)
        return dropped

    try:
        yield Stream(read_frame, drain)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def frames(device: str, sample_rate: int, frame_bytes: int) -> Iterator[bytes]:
    """Convenience iterator over capture frames. Runs until the caller stops."""
    with open_stream(device, sample_rate, frame_bytes) as read_frame:
        while True:
            yield read_frame()


def record_seconds(device: str, sample_rate: int, seconds: float) -> bytes:
    """Blocking fixed-length capture. Used by the audio check script."""
    frame_bytes = sample_rate * 2 // 10  # 100 ms
    want = int(sample_rate * seconds) * 2
    out = b""
    with open_stream(device, sample_rate, frame_bytes) as read_frame:
        while len(out) < want:
            out += read_frame()
    return out[:want]
