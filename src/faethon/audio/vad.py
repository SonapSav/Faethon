"""End-of-utterance detection.

After the wake word fires we keep recording until the user stops talking.
webrtcvad decides speech vs. silence per 20 ms sub-frame.

Two distinct clocks, which is the crux of getting this right:

  start_timeout_ms   how long to wait for the user to BEGIN speaking
  silence_ms         how long to wait after they STOP before deciding they
                     finished

Conflating them makes Faethon give up on anyone who pauses to think. Worse, a
single 20 ms blip of room noise used to count as "they started", after which
the trailing-silence clock ran and killed the turn ~600 ms later. Speech onset
therefore requires `speech_onset_ms` of *sustained* voice, and a burst that
never amounts to a real utterance is treated as a false start -- we go back to
waiting rather than throwing the turn away.

webrtcvad only accepts 10/20/30 ms frames at 8/16/32/48 kHz, so the capture
frames get sliced into 20 ms pieces before being handed over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import webrtcvad

log = logging.getLogger(__name__)

VAD_FRAME_MS = 20

# Keep a little audio from before speech was confirmed, so the first phoneme
# isn't clipped -- onset is only detected once it's already underway.
PRE_ONSET_MS = 200


class Status(Enum):
    LISTENING = auto()   # still collecting
    DONE = auto()        # trailing silence reached; utterance complete
    TIMED_OUT = auto()   # hit max_ms mid-sentence
    NO_SPEECH = auto()   # never started talking within start_timeout_ms


@dataclass
class Result:
    status: Status
    pcm: bytes

    @property
    def usable(self) -> bool:
        return self.status in (Status.DONE, Status.TIMED_OUT) and bool(self.pcm)


class UtteranceRecorder:
    """Accumulates capture frames until the user stops speaking.

    Recording starts at the wake-word detection instant, so the wake word
    itself is never included.
    """

    def __init__(
        self,
        sample_rate: int,
        aggressiveness: int = 2,
        silence_ms: int = 600,
        min_ms: int = 400,
        max_ms: int = 15000,
        start_timeout_ms: int = 5000,
        speech_onset_ms: int = 120,
    ) -> None:
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"webrtcvad cannot handle {sample_rate} Hz")
        self.sample_rate = sample_rate
        self.silence_ms = silence_ms
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.start_timeout_ms = start_timeout_ms
        self.speech_onset_ms = speech_onset_ms
        self._vad = webrtcvad.Vad(aggressiveness)
        self._vad_frame_bytes = sample_rate * VAD_FRAME_MS // 1000 * 2
        self.reset()

    def reset(self, start_timeout_ms: int | None = None) -> None:
        """Clear state for a new recording.

        `start_timeout_ms` overrides the configured budget for this recording
        only -- a follow-up window after a reply is a different wait from the
        one after a wake word, but it is the same state machine.
        """
        self._start_timeout_ms = (
            self.start_timeout_ms if start_timeout_ms is None else start_timeout_ms
        )
        self._chunks: list[bytes] = []
        self._pending = b""       # leftover shorter than one VAD frame
        self._bytes = 0
        self._elapsed_ms = 0      # since the wake word, including waiting
        self._false_starts = 0
        self._begin_utterance()

    def _begin_utterance(self) -> None:
        """(Re)start tracking a candidate utterance, keeping the total clock."""
        self._silence_ms = 0
        self._speech_ms = 0
        self._run_speech_ms = 0   # consecutive voiced time, for onset
        self._heard_speech = False
        self._onset_bytes: int | None = None

    def _ms_to_bytes(self, ms: int) -> int:
        return self.sample_rate * ms // 1000 * 2

    def _classify(self, frame: bytes, frame_end_bytes: int) -> None:
        if self._vad.is_speech(frame, self.sample_rate):
            self._run_speech_ms += VAD_FRAME_MS
            self._silence_ms = 0
            if not self._heard_speech and self._run_speech_ms >= self.speech_onset_ms:
                # Confirmed: someone is actually talking.
                self._heard_speech = True
                self._onset_bytes = frame_end_bytes - self._ms_to_bytes(
                    self._run_speech_ms
                )
            if self._heard_speech:
                self._speech_ms += VAD_FRAME_MS
        else:
            self._run_speech_ms = 0
            if self._heard_speech:
                self._silence_ms += VAD_FRAME_MS

    def feed(self, frame: bytes) -> Status:
        """Add one capture frame. Returns the current status."""
        self._chunks.append(frame)
        self._bytes += len(frame)
        self._elapsed_ms += len(frame) // 2 * 1000 // self.sample_rate

        buf = self._pending + frame
        # Byte offset of the end of `buf` within the whole recording.
        buf_end = self._bytes
        offset = 0
        while offset + self._vad_frame_bytes <= len(buf):
            consumed_after = offset + self._vad_frame_bytes
            self._classify(
                buf[offset:consumed_after],
                buf_end - (len(buf) - consumed_after),
            )
            offset = consumed_after
        self._pending = buf[offset:]

        if self._heard_speech:
            if self._silence_ms >= self.silence_ms:
                if self._speech_ms >= self.min_ms:
                    return Status.DONE
                # Too brief to be a real request: a cough, a door, the tail of
                # the chime. Discard it and go back to waiting rather than
                # ending the turn.
                self._false_starts += 1
                log.debug(
                    "false start #%d (%dms of speech); still waiting",
                    self._false_starts, self._speech_ms,
                )
                self._chunks = []
                self._bytes = 0
                self._pending = b""
                self._begin_utterance()
                return Status.LISTENING
            if self._elapsed_ms >= self.max_ms:
                return Status.TIMED_OUT
        elif self._elapsed_ms >= self._start_timeout_ms:
            # Never started talking. A false wake, or they changed their mind.
            return Status.NO_SPEECH

        return Status.LISTENING

    def result(self, status: Status) -> Result:
        if not self._heard_speech:
            return Result(status=Status.NO_SPEECH, pcm=b"")

        pcm = b"".join(self._chunks)

        # Drop the wait before they began -- Whisper is billed by duration and
        # hallucinates on long silences. Keep a little pre-roll.
        if self._onset_bytes is not None:
            start = max(0, self._onset_bytes - self._ms_to_bytes(PRE_ONSET_MS))
            pcm = pcm[start:]

        if status is Status.DONE:
            trim = self._ms_to_bytes(self._silence_ms)
            pcm = pcm[:-trim] if 0 < trim < len(pcm) else pcm

        log.debug(
            "utterance %s: %.2fs speech, %.2fs kept, %.2fs since wake, %d false start(s)",
            status.name, self._speech_ms / 1000, len(pcm) / 2 / self.sample_rate,
            self._elapsed_ms / 1000, self._false_starts,
        )
        return Result(status=status, pcm=pcm)
