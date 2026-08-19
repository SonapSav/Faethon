"""Faethon's main loop.

    wake word -> chime -> record -> STT -> route -> TTS -+-> chime -> record ...
                                                         |
                                          silence -> chime -> back to wake word

Only the first turn of a conversation needs the wake word. After each reply the
mic reopens for `conversation.follow_up_ms`, so "and what about tomorrow?" works
without saying "hey rhasspy" again. Silence through that window plays the
closing chime and hands back to wake-word detection.

Half-duplex by design: Faethon cannot hear you while it is talking. Barge-in
would need echo cancellation, which is a bigger job than v1 warrants. arecord
does keep running throughout, so a reply ends with Faethon's own voice sitting
in the capture buffer -- see the drain in `_converse`, without which it hears
itself and answers its own sentence.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from .audio import capture, playback
from .audio.vad import Status, UtteranceRecorder
from .bargein import BargeInListener
from .config import ASSETS_DIR, Config, load_config
from .memory import Memory
from .providers import stt as stt_mod
from .providers import tts as tts_mod
from .providers.client import OpenRouterClient, OpenRouterError
from .router import Router
from .skills.registry import Registry
from .speech import Spoken, speak_streaming
from .wake import WakeWordDetector

log = logging.getLogger("faethon")

ACK_SOUND = ASSETS_DIR / "ack.wav"      # "go ahead"
CLOSE_SOUND = ASSETS_DIR / "done.wav"   # "we're finished; wake me again"


class Faethon:
    def __init__(self, config: Config, client: OpenRouterClient) -> None:
        self.config = config
        self.client = client
        self.registry = Registry.discover()
        self.memory = Memory(
            config.llm.history_turns,
            idle_timeout_sec=config.llm.history_idle_minutes * 60,
        )
        self.router = Router(config, client, self.registry, self.memory)
        self.detector = WakeWordDetector(
            config.wake.model,
            threshold=config.wake.threshold,
            refractory_sec=config.wake.refractory_sec,
        )
        self.recorder = UtteranceRecorder(
            config.audio.sample_rate,
            aggressiveness=config.utterance.vad_aggressiveness,
            silence_ms=config.utterance.silence_ms,
            min_ms=config.utterance.min_ms,
            max_ms=config.utterance.max_ms,
            start_timeout_ms=config.utterance.start_timeout_ms,
            speech_onset_ms=config.utterance.speech_onset_ms,
        )
        self._running = True

        if not config.tts.voice:
            # Leaving this empty is not the same as taking the provider's
            # default. Fish Audio picks a different speaker per request when no
            # voice is given -- measured 110-222 Hz across eight renders of one
            # sentence, versus 100-112 Hz with a voice set.
            log.warning(
                "no tts.voice set for %s -- some providers pick a random voice "
                "per request, so Faethon may sound like a different person each turn",
                config.models.tts,
            )

    def stop(self, *_: object) -> None:
        log.info("shutting down")
        self._running = False

    # -- pipeline stages -------------------------------------------------

    def _capture_utterance(self, read_frame, start_timeout_ms=None) -> bytes | None:
        """Record until the user stops talking. Returns None if nothing usable."""
        self.recorder.reset(start_timeout_ms)
        while self._running:
            status = self.recorder.feed(read_frame())
            if status is not Status.LISTENING:
                break
        else:
            return None

        result = self.recorder.result(status)
        if result.status is Status.TIMED_OUT:
            log.info("utterance hit the %dms cap", self.config.utterance.max_ms)
        if not result.usable:
            log.info("nothing to transcribe (%s)", result.status.name)
            return None
        return result.pcm

    def _speak(self, text: str) -> None:
        if not text:
            return
        log.info("saying: %s", text)
        try:
            with playback.stream(
                self.config.audio.output_device, self.config.tts.sample_rate
            ) as write:
                for chunk in tts_mod.synthesize_stream(
                    self.client,
                    text,
                    model=self.config.models.tts,
                    voice=self.config.tts.voice,
                ):
                    write(chunk)
        except OpenRouterError as e:
            # Nothing to say it with -- log loudly, stay alive.
            log.error("tts failed: %s", e)

    def _handle_turn(self, read_frame, start_timeout_ms=None) -> bool:
        """One exchange: record, transcribe, answer.

        Returns whether Faethon said anything. That is the signal `_converse`
        uses to decide whether to hold the conversation open -- if Faethon just
        spoke, the user may well reply, including when what it said was an
        apology for not understanding them.
        """
        pcm = self._capture_utterance(read_frame, start_timeout_ms)
        if pcm is None:
            return False

        # Clock starts when the user stops talking -- that's the silence they
        # actually perceive as lag.
        started = time.monotonic()
        audio_sec = len(pcm) / 2 / self.config.audio.sample_rate

        try:
            text = stt_mod.transcribe(
                self.client,
                pcm,
                model=self.config.models.stt,
                sample_rate=self.config.audio.sample_rate,
                language=self.config.stt.language,
                temperature=self.config.stt.temperature,
            )
        except OpenRouterError as e:
            log.error("stt failed: %s", e)
            self._speak("Sorry, I couldn't understand that.")
            return True
        t_stt = time.monotonic() - started

        if not text:
            log.info("empty transcript")
            return False

        log.info("heard: %s", text)

        # The reply is spoken sentence by sentence as the model generates it,
        # so the user hears the first words long before the last are decided.
        spoken = self._speak_reply(read_frame, text)

        total = time.monotonic() - started
        log.info(
            "turn: %.1fs audio | stt %.2fs | reply+speech %.2fs | total %.2fs%s | $%.5f",
            audio_sec, t_stt, total - t_stt, total,
            " | interrupted" if spoken.interrupted else "", self.client.spent,
        )
        if not spoken:
            log.info("nothing to say")
        # An interruption is a reason to keep listening even if nothing was
        # spoken before the cut: saying the wake word means "I want to talk".
        return bool(spoken) or spoken.interrupted

    def _speak_reply(self, read_frame, text: str) -> Spoken:
        """Speak the reply, watching for the wake word if barge-in is on."""
        chunks = self.router.handle_streaming(text)
        if not self.config.conversation.barge_in:
            return speak_streaming(self.client, self.config, chunks)

        # The speaker does not exist until speak_streaming builds one, but the
        # listener needs something to stop the moment it hears the wake word.
        # A one-element list is the handoff between the two threads.
        current: list = []

        def interrupt() -> None:
            if current:
                current[0].stop()

        listener = BargeInListener(
            read_frame, self.detector, interrupt,
            threshold=self.config.conversation.barge_in_threshold,
        )
        with listener:
            spoken = speak_streaming(
                self.client, self.config, chunks, on_start=current.append
            )
        if listener.error is not None:
            # The mic died while Faethon was talking. Raise it on the main
            # thread so the run loop's reconnect handles it as usual.
            raise listener.error
        return spoken

    def _converse(self, stream) -> None:
        """Run turns back to back until the user stops answering.

        The wake word opens the first turn; every turn after that is opened by
        the follow-up window alone.
        """
        # None means "use the configured wake-word budget" for the first turn.
        window_ms: int | None = None

        while self._running:
            playback.play_wav_async(ACK_SOUND, self.config.audio.output_device)
            replied = self._handle_turn(stream, window_ms)

            if replied:
                # Faethon has been talking over a live microphone. Drop what it
                # recorded of itself before listening, or the follow-up window
                # transcribes its own reply back to it.
                dropped = stream.drain()
                log.debug(
                    "dropped %.1fs of self-audio",
                    dropped / 2 / self.config.audio.sample_rate,
                )

            if not replied or not self.config.conversation.follow_up:
                # Only announce the close if a follow-up window was actually
                # open. A wake word that nobody followed with speech is a false
                # trigger, and answering it with a chime is just more noise.
                if window_ms is not None and not replied:
                    playback.play_wav(CLOSE_SOUND, self.config.audio.output_device)
                    log.info("no follow-up; conversation closed")
                return

            window_ms = self.config.conversation.follow_up_ms
            log.info("listening for a follow-up (%dms, no wake word needed)", window_ms)

    # -- main loop -------------------------------------------------------

    def run(self) -> None:
        log.info("Faethon is listening -- say the wake word")
        while self._running:
            try:
                self._listen_once()
            except capture.CaptureError as e:
                # The USB mic was unplugged, or the wireless link dropped.
                # Keep trying: it usually comes back.
                log.error("audio capture lost: %s -- retrying in 3s", e)
                time.sleep(3)

    def _listen_once(self) -> None:
        """Hold the mic open until a wake word fires, then hand off.

        The stream is reopened per turn so the speaker can have the audio
        device to itself while Faethon talks.
        """
        with capture.open_stream(
            self.config.audio.input_device,
            self.config.audio.sample_rate,
            self.config.audio.frame_bytes,
        ) as read_frame:
            while self._running:
                # Checked here rather than on a timer thread: this loop already
                # ticks once per 80ms frame while idle, so the buffer is wiped
                # at the deadline instead of merely being ignored by the next
                # turn -- which would leave it sitting in RAM.
                self.memory.expire_if_idle()
                if self.detector.process(read_frame()) is not None:
                    self._converse(read_frame)
                    # Flush the wake model: the tail of Faethon's own reply may
                    # still be in its feature buffer.
                    self.detector.reset()
                    log.info("listening again")


def main() -> int:
    parser = argparse.ArgumentParser(prog="faethon", description="Faethon voice assistant")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--config", help="path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # journald already timestamps; don't duplicate when running as a service.
    if not sys.stderr.isatty():
        logging.getLogger().handlers[0].setFormatter(
            logging.Formatter("%(levelname)-7s %(name)s: %(message)s")
        )

    config = load_config(Path(args.config) if args.config else None)

    try:
        client = OpenRouterClient(config.settings.openrouter_api_key.get_secret_value())
    except OpenRouterError as e:
        log.error("%s", e)
        return 1

    faethon = Faethon(config, client)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, faethon.stop)

    try:
        faethon.run()
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
