"""Draining the microphone buffer.

arecord keeps recording while Faethon talks, so when a reply ends the pipe
holds a recording of that reply. Whatever listens next reads it back -- the
wake model chews through stale audio, and a follow-up window transcribes
Faethon's own sentence and answers it.

These drive `open_stream` against a real pipe with arecord replaced, so the
non-blocking read is genuinely exercised rather than mocked away.
"""

from __future__ import annotations

import os

import pytest

from faethon.audio import capture

RATE = 16000
FRAME_BYTES = 2560  # 80 ms


@pytest.fixture
def pipe(monkeypatch):
    """Replace arecord with a pipe we can push bytes into by hand."""
    read_fd, write_fd = os.pipe()
    reader = open(read_fd, "rb", buffering=0)
    writer = open(write_fd, "wb", buffering=0)

    class FakeProc:
        stdout = reader
        stderr = open(os.devnull, "rb")

        def terminate(self): pass
        def wait(self, timeout=None): return 0
        def kill(self): pass

    monkeypatch.setattr(capture.subprocess, "Popen", lambda *a, **kw: FakeProc())

    class Pipe:
        fd = read_fd

        def send(self, data: bytes) -> None:
            writer.write(data)

    try:
        yield Pipe()
    finally:
        if not writer.closed:
            writer.close()
        reader.close()


def test_drain_discards_what_was_buffered(pipe):
    with capture.open_stream("fake", RATE, FRAME_BYTES) as stream:
        pipe.send(b"\x11\x22" * 4000)          # Faethon talking over itself
        assert stream.drain() == 8000

        fresh = b"\x33\x44" * (FRAME_BYTES // 2)
        pipe.send(fresh)
        # The next frame is the new audio, not the stale reply.
        assert stream() == fresh


def test_drain_on_a_quiet_pipe_returns_immediately(pipe):
    """The load-bearing case: drain must never block.

    It runs on the main loop between a reply and the next listening window. If
    it waited for data, Faethon would hang until someone made a noise.
    """
    with capture.open_stream("fake", RATE, FRAME_BYTES) as stream:
        assert stream.drain() == 0


def test_reading_still_blocks_normally_after_a_drain(pipe):
    """Draining flips the fd non-blocking; it has to flip it back.

    Left non-blocking, read_frame would raise BlockingIOError at the first
    quiet moment instead of waiting for the microphone.
    """
    with capture.open_stream("fake", RATE, FRAME_BYTES) as stream:
        stream.drain()
        assert os.get_blocking(pipe.fd)

        pipe.send(b"\x00\x01" * (FRAME_BYTES // 2))
        assert len(stream()) == FRAME_BYTES


def test_drain_keeps_reading_past_the_first_chunk(pipe, monkeypatch):
    """A reply is far bigger than one read().

    Five seconds of 16kHz mono is 160kB. A drain that read once would leave
    most of Faethon's voice in the buffer, which is the whole failure this
    exists to prevent.
    """
    monkeypatch.setattr(capture, "DRAIN_CHUNK", 1024)
    with capture.open_stream("fake", RATE, FRAME_BYTES) as stream:
        pipe.send(b"\xab\xcd" * 5000)          # 10kB, ten chunks' worth
        assert stream.drain() == 10000
        assert stream.drain() == 0
