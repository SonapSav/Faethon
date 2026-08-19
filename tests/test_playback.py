"""Playback, and the difference between stopping and finishing.

Ending a stream closes the pipe and waits, which lets aplay play out
everything already buffered. Aborting kills it. Measured on this Pi from the
instant of the decision to actual silence:

    close stdin and wait   9112 ms
    terminate (SIGTERM)     234 ms

That gap is the whole of barge-in, and nothing in the code says so -- an
abort() rewritten to close stdin looks tidier and reads correctly. Until
these existed, the only test touching abort used a fake sink, so the real
terminate never ran in the suite at all.

No audio hardware: subprocess.Popen is replaced.
"""

from __future__ import annotations

import subprocess

import pytest

from faethon.audio import playback


class FakeProc:
    def __init__(self, *a, **kw):
        self.argv = a[0] if a else []
        self.events: list[str] = []
        self.written: list[bytes] = []
        self.stdin = self
        self.terminated = False
        self._closed = False

    # -- stdin ------------------------------------------------------------
    def write(self, chunk: bytes) -> None:
        if self.terminated:
            raise BrokenPipeError("aplay is gone")
        if self._closed:
            raise ValueError("I/O operation on closed file")
        self.written.append(chunk)

    def close(self) -> None:
        self._closed = True
        self.events.append("closed stdin")

    # -- process ----------------------------------------------------------
    def terminate(self) -> None:
        self.terminated = True
        self.events.append("terminated")

    def wait(self, timeout=None) -> int:
        self.events.append("waited")
        return 0


@pytest.fixture
def proc(monkeypatch):
    made: list[FakeProc] = []

    def fake_popen(*a, **kw):
        p = FakeProc(*a, **kw)
        made.append(p)
        return p

    monkeypatch.setattr(playback.subprocess, "Popen", fake_popen)
    return made


# -- the distinction that matters --------------------------------------------


def test_a_normal_stream_closes_the_pipe_and_waits(proc):
    """Finishing plays out what is buffered, which is what you want when the
    reply has actually ended."""
    with playback.stream("dev", 24000) as sink:
        sink(b"\x00\x01")
    assert proc[0].events == ["closed stdin", "waited"]
    assert not proc[0].terminated


def test_aborting_terminates_instead_of_draining(proc):
    """Closing the pipe here would keep talking for up to nine seconds after
    being told to stop."""
    with playback.stream("dev", 24000) as sink:
        sink(b"\x00\x01")
        sink.abort()
    assert "terminated" in proc[0].events
    assert "closed stdin" not in proc[0].events, "aborted stream was drained"


def test_abort_is_safe_to_call_twice(proc):
    """It arrives from the barge-in thread; the play loop aborts again on the
    way out to close a race."""
    with playback.stream("dev", 24000) as sink:
        sink.abort()
        sink.abort()
    assert proc[0].events.count("terminated") == 1


def test_writes_after_an_abort_are_dropped(proc):
    """Audio still queued behind the aborted sentence must not be played."""
    with playback.stream("dev", 24000) as sink:
        sink(b"\x01\x01")
        sink.abort()
        sink(b"\x02\x02")
        sink(b"\x03\x03")
    assert proc[0].written == [b"\x01\x01"]


def test_a_broken_pipe_after_abort_is_not_reported_as_a_fault(proc, caplog):
    """Terminating unblocks a play thread stuck in write(). That exception is
    the abort working, not a failure worth logging."""
    p_written = []

    with playback.stream("dev", 24000) as sink:
        sink.abort()
        proc[0].terminated = True
        sink(b"\x04\x04")
    assert "closed early" not in caplog.text


def test_a_broken_pipe_without_an_abort_is_reported(proc, caplog):
    """aplay dying on its own is a real fault and should say so."""
    import logging

    with caplog.at_level(logging.WARNING):
        with playback.stream("dev", 24000) as sink:
            proc[0].terminated = True     # died without us asking
            sink(b"\x05\x05")
    assert "closed early" in caplog.text


# -- the plumbing ------------------------------------------------------------


def test_the_stream_is_opened_at_the_rate_it_was_given(proc):
    with playback.stream("plughw:CARD=Test,DEV=0", 44100) as sink:
        sink(b"")
    argv = proc[0].argv
    assert "44100" in argv
    assert "plughw:CARD=Test,DEV=0" in argv
    assert argv[0] == "aplay"


def test_a_wav_round_trips(tmp_path):
    """write_wav is what renders the chimes and the status clips."""
    import wave

    pcm = bytes(range(256)) * 4
    out = tmp_path / "x.wav"
    playback.write_wav(out, pcm, 16000)
    with wave.open(str(out)) as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.readframes(w.getnframes()) == pcm
