"""Microphone capture via an `arecord` subprocess.

We shell out to arecord rather than binding PortAudio because alsa-utils is
already on every Pi OS image, the device strings are the same ones you'd test
with by hand, and ALSA's "plug" layer does rate/channel conversion for us.

The stream yields fixed-size frames of signed 16-bit little-endian mono PCM.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    pass


@contextmanager
def open_stream(device: str, sample_rate: int, frame_bytes: int):
    """Run arecord and yield a callable returning exactly `frame_bytes` per call."""
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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

    try:
        yield read_frame
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
