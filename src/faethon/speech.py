"""Speak an LLM reply while it is still being generated.

Waiting for the whole reply before synthesising means the user hears nothing
for (LLM time + TTS time). Cutting the reply into sentences and synthesising
each as it lands turns that into (time to first sentence + TTS of that
sentence), with everything after it pipelined behind the audio already playing.

Two moving parts:

  sentence_chunks   splits a stream of token deltas at clause boundaries
  speak_streaming   synthesises chunks on a worker thread and writes them all
                    into one continuous aplay stream

One playback stream for the whole reply matters: a separate aplay per sentence
would put an audible gap and a device-open stutter between each one.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

from .audio import playback
from .config import Config
from .providers import tts as tts_mod
from .providers.client import OpenRouterClient, OpenRouterError

log = logging.getLogger(__name__)

# End of sentence: .!?… optionally followed by closing quotes/brackets, then
# whitespace or end of string. The lookahead is what stops "3.5" and "$1.20"
# from being treated as sentence ends.
_STRONG = re.compile(r"""[.!?…]["'”’)\]]*(?=\s|$)""")

# Fallback split points for a long sentence that never ends.
_WEAK = re.compile(r"""[,;:—–]["'”’)\]]*(?=\s)""")

# The first chunk can be short: getting *any* audio out fast is what makes
# Faethon feel responsive. Later chunks want to be longer so prosody holds
# together and we make fewer TTS calls.
FIRST_MIN_CHARS = 15
MIN_CHARS = 60
MAX_CHARS = 220

# How many sentences may be synthesised at once. Sequential synthesis cannot
# keep up with playback: each request costs ~2s of latency to return ~2s of
# audio, i.e. below real-time, so the speaker drains and ALSA underruns between
# sentences. Overlapping the requests is what makes the pipeline sustainable.
#
# 2 is deliberate: one sentence of lookahead is all that's needed to cover the
# gap, and piling more concurrent requests on the provider risks being
# throttled -- which would cost more than it saves.
LOOKAHEAD = 2

# Audio to accumulate before opening the device. Starting the instant the first
# bytes arrive means any hiccup upstream underruns the card, and recovery costs
# far more than this delay does.
PREBUFFER_MS = 700

# How often the play thread looks up from its queues to notice it has been
# stopped. Small next to the ~230ms it takes the speaker to fall silent anyway,
# so it costs nothing perceptible and saves blocking on a sentence that will
# never be spoken.
POLL_SEC = 0.05


def _find_cut(buf: str, min_chars: int, allow_weak: bool) -> int | None:
    """Index to split `buf` at, or None to keep accumulating."""
    for m in _STRONG.finditer(buf):
        if m.end() >= min_chars:
            return m.end()

    # A one-sentence reply has no strong boundary until the very end, which
    # would mean no audio until generation finished -- exactly the latency this
    # module exists to remove. So the first chunk may also break at a comma or
    # dash, which is a natural enough pause to synthesise on its own.
    if allow_weak:
        for m in _WEAK.finditer(buf):
            if m.end() >= min_chars:
                return m.end()

    if len(buf) < MAX_CHARS:
        return None

    # Overlong with no boundary at all -- break at a clause, then any space.
    for m in _WEAK.finditer(buf):
        if m.end() >= min_chars:
            return m.end()

    space = buf.rfind(" ", min_chars, MAX_CHARS)
    return space if space > 0 else None


def sentence_chunks(deltas: Iterable[str]) -> Iterator[str]:
    """Regroup token deltas into speakable chunks."""
    buf = ""
    first = True
    for delta in deltas:
        buf += delta
        while True:
            cut = _find_cut(
                buf,
                FIRST_MIN_CHARS if first else MIN_CHARS,
                allow_weak=first,
            )
            if cut is None:
                break
            chunk, buf = buf[:cut].strip(), buf[cut:]
            if chunk:
                first = False
                yield chunk
    if buf.strip():
        yield buf.strip()


class _Speaker:
    """Synthesises queued chunks into one continuous playback stream.

    Synthesis and playback run on separate threads:

        text -> [text queue] -> synth thread -> [pcm queue] -> play thread -> aplay

    They must not share a thread. `aplay` applies backpressure -- write()
    blocks once its buffer is full, which for a sentence of any length means
    blocking until most of that sentence has been *heard*. A single worker
    doing both would therefore not even request the next sentence until the
    current one finished playing, leaving a silent gap at every full stop the
    length of a whole TTS round-trip. Measured at 3-7s before this split.

    With them separated, sentence N+1 is synthesised while sentence N plays,
    and the audio is waiting in the queue by the time it's needed.
    """

    def __init__(self, client: OpenRouterClient, config: Config) -> None:
        self._client = client
        self._config = config
        # One PCM queue per sentence, held in speaking order. Synthesis fills
        # them out of order; playback drains them in order.
        self._order: queue.Queue[queue.Queue[bytes | None] | None] = queue.Queue()
        self._pool = ThreadPoolExecutor(
            max_workers=LOOKAHEAD, thread_name_prefix="tts"
        )
        self._error: BaseException | None = None
        self._first_audio: float | None = None
        self._started = time.monotonic()
        self._stopped = threading.Event()
        #: Set once the device is open, so another thread can abort playback.
        self._sink: playback.Sink | None = None
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        # Overwritten by the first response's Content-Type; config is only the
        # fallback for providers that don't declare a rate.
        self._rate = config.tts.sample_rate
        self._prebuffer_bytes = self._rate * 2 * PREBUFFER_MS // 1000

    def start(self) -> None:
        self._play_thread.start()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def error(self) -> BaseException | None:
        """Whatever stopped the audio, if anything did."""
        return self._error

    def stop(self) -> None:
        """Cut the reply off mid-sentence. Called from the barge-in thread.

        Aborting the sink is what silences the speaker; everything else here
        just stops work that would otherwise carry on being paid for and
        played -- sentences still being synthesised, and audio already queued
        behind the one being spoken.
        """
        if self._stopped.is_set():
            return
        self._stopped.set()
        if self._sink is not None:
            self._sink.abort()

    def send(self, chunk: str) -> None:
        if self._stopped.is_set():
            return
        # Claim this sentence's position in the running order *before*
        # submitting, so playback order never depends on which request finishes
        # first.
        pcm_q: queue.Queue[bytes | None] = queue.Queue()
        self._order.put(pcm_q)
        self._pool.submit(self._synth_one, chunk, pcm_q)

    def finish(self) -> float | None:
        """Wait for playback to drain. Returns time-to-first-audio."""
        self._order.put(None)
        self._play_thread.join()
        self._pool.shutdown(wait=True)
        if self._error is not None:
            log.error("tts pipeline failed: %s", self._error)
        return self._first_audio

    def _note_rate(self, rate: int) -> None:
        # Every sentence in a reply comes from the same model, so first wins.
        if rate != self._rate:
            log.debug("tts sample rate %d Hz (config said %d)", rate, self._rate)
            self._rate = rate

    def _synth_one(self, text: str, pcm_q: queue.Queue[bytes | None]) -> None:
        if self._stopped.is_set():
            pcm_q.put(None)
            return
        try:
            for pcm in tts_mod.synthesize_stream(
                self._client,
                text,
                model=self._config.models.tts,
                voice=self._config.tts.voice,
                on_rate=self._note_rate,
            ):
                if self._stopped.is_set():
                    # Stop pulling audio nobody will hear.
                    break
                pcm_q.put(pcm)
        except OpenRouterError as e:
            self._error = e
        except Exception as e:  # noqa: BLE001 - must not hang the main loop
            self._error = e
            log.exception("tts synthesis crashed")
        finally:
            # Always release the player, even on error, or it waits forever.
            pcm_q.put(None)

    def _play_loop(self) -> None:
        stream_cm = None
        write = None
        pending: list[bytes] = []
        pending_bytes = 0

        def open_device() -> None:
            nonlocal stream_cm, write
            # self._rate is set from the response header before any audio
            # arrives, so by here it reflects what we're actually about to play.
            stream_cm = playback.stream(
                self._config.audio.output_device, self._rate
            )
            write = stream_cm.__enter__()
            # Publish the sink before the first write: stop() can arrive from
            # the barge-in thread at any point once audio is audible.
            self._sink = write
            if self._stopped.is_set():
                # Stopped in the moment between the check in stop() and this
                # device opening. Nothing has been written yet, but the stream
                # still has to be aborted or the exit below will drain it.
                write.abort()
            self._first_audio = time.monotonic() - self._started

        try:
            while not self._stopped.is_set():
                # Poll rather than block: a barge-in must not have to wait for
                # a sentence that is still being synthesised to arrive first.
                try:
                    pcm_q = self._order.get(timeout=POLL_SEC)
                except queue.Empty:
                    continue
                if pcm_q is None:
                    break
                while not self._stopped.is_set():
                    try:
                        pcm = pcm_q.get(timeout=POLL_SEC)
                    except queue.Empty:
                        continue
                    if pcm is None:
                        break
                    if write is not None:
                        write(pcm)
                        continue
                    # Still filling the pre-roll.
                    pending.append(pcm)
                    pending_bytes += len(pcm)
                    if pending_bytes >= self._prebuffer_bytes:
                        open_device()
                        for buffered in pending:
                            write(buffered)
                        pending, pending_bytes = [], 0

            # Reply shorter than the pre-roll: play what we have.
            if pending and not self._stopped.is_set():
                open_device()
                for buffered in pending:
                    write(buffered)
        except Exception as e:  # noqa: BLE001
            self._error = e
            log.exception("playback crashed")
        finally:
            if stream_cm is not None:
                if self._stopped.is_set() and self._sink is not None:
                    # Belt and braces against the same race: exiting the
                    # context manager closes the pipe and waits, which plays
                    # out everything buffered -- 9.1s of it, measured. Abort
                    # is idempotent, so doing it twice costs nothing.
                    self._sink.abort()
                stream_cm.__exit__(None, None, None)


class Spoken(str):
    """What Faethon actually said, plus whether it was cut off part-way.

    A str subclass rather than a wrapper so every caller that compares,
    formats or truth-tests the reply keeps working untouched.
    """

    interrupted: bool = False
    #: True when the words were produced but never became audible, which is
    #: otherwise indistinguishable from Faethon ignoring you.
    failed: bool = False

    def __new__(
        cls, text: str, interrupted: bool = False, failed: bool = False
    ) -> "Spoken":
        self = super().__new__(cls, text)
        self.interrupted = interrupted
        self.failed = failed
        return self


def speak_streaming(
    client: OpenRouterClient,
    config: Config,
    chunks: Iterable[str],
    *,
    on_start: Callable[[_Speaker], None] | None = None,
) -> Spoken:
    """Speak chunks as they arrive. Returns everything that was said.

    `on_start` is handed the speaker as soon as one exists, giving a caller
    that is watching for barge-in something to stop.
    """
    speaker = _Speaker(client, config)
    speaker.start()
    if on_start is not None:
        on_start(speaker)

    spoken: list[str] = []
    try:
        for chunk in chunks:
            if speaker.stopped:
                # Barge-in. Stop pulling from the model as well as the
                # speaker: the rest of this reply is neither wanted nor free.
                break
            if not chunk.strip():
                continue
            spoken.append(chunk)
            speaker.send(chunk)
    finally:
        first = speaker.finish()
        close = getattr(chunks, "close", None)
        if close is not None:
            # Unwinds the router's generator, which closes the HTTP response
            # and records what was actually said. Without it an interrupted
            # turn leaves no trace in memory, and Faethon would repeat the
            # whole answer if asked again.
            close()

    if spoken and first is not None:
        log.info("first audio in %.2fs (%d chunk(s))", first, len(spoken))
    return Spoken(
        " ".join(spoken),
        interrupted=speaker.stopped,
        failed=speaker.error is not None,
    )
