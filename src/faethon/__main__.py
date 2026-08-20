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
from .turnlog import TurnLog
from .status import (
    LOW_CREDIT,
    MIC_BACK,
    NO_MIC,
    NO_NETWORK,
    Announcer,
    CreditWatch,
    SilenceWatch,
    classify,
)
from .wake import WakeWordDetector

log = logging.getLogger("faethon")

ACK_SOUND = ASSETS_DIR / "ack.wav"      # "go ahead"
CLOSE_SOUND = ASSETS_DIR / "done.wav"   # "we're finished; wake me again"
# Pre-rendered rather than synthesised on each boot: a fixed sentence needs no
# network round-trip, and this way it still plays when OpenRouter is down, out
# of credit, or the wifi is not up -- when knowing the service came back is
# worth most. Regenerate with scripts/make_greeting.py.
GREETING_SOUND = ASSETS_DIR / "greeting.wav"
TIMER_SOUND = ASSETS_DIR / "timer.wav"   # "look up", not "go ahead"


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
        self.announcer = Announcer(config.audio.output_device)
        self.silence = SilenceWatch(config.audio.frame_ms)
        #: Consecutive failures to open the capture stream, so recovery can be
        #: reported. A USB microphone can take minutes to appear after a cold
        #: boot -- measured at 2m45s here -- and the retry loop handled that
        #: silently, leaving the journal ending on an error from long before
        #: everything started working.
        self._capture_failures = 0
        self.turn_log = TurnLog(
            config.turn_log.enabled, int(config.turn_log.max_mb * 1_000_000)
        )
        self.credit = CreditWatch(
            config.credit.warn_below,
            config.credit.check_every_minutes * 60,
            self._balance,
        )

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
                    cost_per_1k_chars=self.config.tts.cost_per_1k_chars,
                ):
                    write(chunk)
        except OpenRouterError as e:
            # Nothing to say it with -- log loudly, stay alive.
            log.error("tts failed: %s", e)

    def _handle_turn(self, read_frame, start_timeout_ms=None) -> bool:  # noqa: C901
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
        spent_before = self.client.spent
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
            # Not _speak(): that synthesises over the network which just
            # failed, so the apology was never heard. The clip is local.
            self.announcer.say(classify(e))
            return True
        t_stt = time.monotonic() - started

        if not text:
            log.info("empty transcript")
            return False

        log.info("heard: %s", text)

        # The reply is spoken sentence by sentence as the model generates it,
        # so the user hears the first words long before the last are decided.
        spoken = self._speak_reply(read_frame, text)

        # Anything a skill deferred until it had been heard -- currently only
        # the restart, which ends this process. Runs after the speaker is done
        # and before the follow-up window opens.
        action = self.router.take_pending_action()
        if action is not None:
            action()

        if spoken.failed:
            # The words exist but were never audible. Silence here reads as
            # Faethon having ignored the question.
            self.announcer.say(NO_NETWORK)
        else:
            self.announcer.recovered()

        total = time.monotonic() - started
        # This turn's cost, then the session's. Printing only the running total
        # here made every turn look more expensive than the last, which reads
        # exactly like context growing without bound.
        log.info(
            "turn: %.1fs audio | stt %.2fs | reply+speech %.2fs | total %.2fs%s "
            "| $%.5f this turn, $%.5f session (%d turn(s) held)",
            audio_sec, t_stt, total - t_stt, total,
            " | interrupted" if spoken.interrupted else "",
            self.client.spent - spent_before, self.client.spent, len(self.memory),
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

    def _over_budget(self, turns: int, elapsed: float) -> str | None:
        """Why this conversation should stop, or None to carry on.

        Both bounds exist because they fail differently. Turns bound the cost,
        since every one is a paid call. Seconds bound how long the microphone
        stays live without a deliberate trigger, which is the privacy claim the
        README makes for the follow-up window.
        """
        cfg = self.config.conversation
        if cfg.max_turns and turns >= cfg.max_turns:
            return f"{turns} turns without a wake word"
        if cfg.max_seconds and elapsed >= cfg.max_seconds:
            return f"{elapsed / 60:.1f} minutes without a wake word"
        return None

    def _converse(self, stream) -> None:
        """Run turns back to back until the user stops answering.

        The wake word opens the first turn; every turn after that is opened by
        the follow-up window alone.
        """
        # None means "use the configured wake-word budget" for the first turn.
        window_ms: int | None = None
        started = time.monotonic()
        turns = 0

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

            # Also between turns: the wake loop does not run during a
            # conversation, and one can now last five minutes. Worst case a
            # timer is one turn late rather than a whole conversation late.
            self._tick_skills(stream)

            turns += 1
            reason = self._over_budget(turns, time.monotonic() - started)
            if reason:
                # The chime rather than a sentence: a falling tone already
                # means "we're finished, wake me again", and explaining the
                # budget out loud would be noise at the one moment nobody
                # asked for a reply.
                log.info("conversation ended: %s", reason)
                playback.play_wav(CLOSE_SOUND, self.config.audio.output_device)
                return

            window_ms = self.config.conversation.follow_up_ms
            log.info("listening for a follow-up (%dms, no wake word needed)", window_ms)

    # -- main loop -------------------------------------------------------

    def _capture_ready(self) -> None:
        """The capture stream opened. Say so if anyone heard it fail.

        Recovery used to be entirely silent: the loop caught the error, slept,
        retried, and on success simply carried on. So the last thing in the
        journal was an error from minutes earlier, and the last thing heard in
        the room was "I can't hear the microphone" -- indistinguishable from
        having died, while it was in fact working.
        """
        # A fresh stream is a fresh silence clock; the old one was counting
        # frames from a device that has since gone away and come back.
        self.silence.reset()
        if not self._capture_failures:
            return
        log.info("microphone back after %d attempt(s)", self._capture_failures)
        self._capture_failures = 0
        if self.announcer.forget(NO_MIC):
            self.announcer.say(MIC_BACK)

    def _balance(self) -> float | None:
        """Remaining OpenRouter credit, or None if it can't be read.

        /credits reports what was granted and what has been spent, not what is
        left; the balance is the difference.
        """
        try:
            data = self.client.get_json("/credits").get("data") or {}
            return float(data["total_credits"]) - float(data["total_usage"])
        except (OpenRouterError, KeyError, TypeError, ValueError) as e:
            log.debug("credit check failed: %s", e)
            return None

    def _tick_skills(self, stream) -> None:
        """Give every skill a chance to speak without being asked.

        Cheap by contract -- called once per audio frame while idle -- and the
        skill returns text rather than speaking, so it needs no audio of its
        own. Anything raised here is logged and swallowed: a broken skill must
        not take down the loop that is listening for the wake word.
        """
        for skill in self.registry:
            try:
                said = skill.tick()
            except Exception:  # noqa: BLE001
                log.exception("tick failed in %s", skill.name)
                continue
            if said:
                self._announce(said)
                # Faethon has just talked over a live microphone, exactly as it
                # does during a turn -- and the turn path drains for this
                # reason. Without it the wake detector spends the next two
                # seconds chewing the announcement instead of hearing the room,
                # measured at 2.04s of backlog after a timer fires. Which is
                # precisely when someone says "cancel" or "set another".
                dropped = stream.drain()
                log.debug(
                    "dropped %.1fs of self-audio after announcing",
                    dropped / 2 / self.config.audio.sample_rate,
                )
                return

    def _announce(self, text: str) -> None:
        """Say something nobody asked for: chime first, then the words.

        The chime carries the attention on its own, so if speech fails -- no
        network, no credit -- the sound still lands and a timer is not silently
        lost.
        """
        log.info("announcing: %s", text)
        playback.play_wav(TIMER_SOUND, self.config.audio.output_device)
        self._speak(text)

    def _greet(self) -> None:
        """Say hello once, before the microphone is ever opened.

        The current wording scores 0.0001 on the wake model through the speaker
        and mic, so the ordering is not what makes it safe -- the words are. An
        earlier draft said "Hi, I am Rhasspy", which scored 0.5036: still under
        the 0.7 wake threshold, but five thousand times higher and well over
        the 0.1 barge-in listens at.

        So this stays first because it costs nothing and the margin belongs to
        the sentence rather than to the design. Reword the greeting to include
        the assistant's name and it comes straight back.
        """
        if not self.config.conversation.greet_on_start:
            return
        if not GREETING_SOUND.exists():
            log.warning(
                "no greeting at %s -- run scripts/make_greeting.py", GREETING_SOUND
            )
            return
        playback.play_wav(GREETING_SOUND, self.config.audio.output_device)

    def run(self) -> None:
        self._greet()
        log.info("Faethon is listening -- say the wake word")
        while self._running:
            try:
                self._listen_once()
            except capture.CaptureError as e:
                # The USB mic was unplugged, or the wireless link dropped.
                # Keep trying: it usually comes back.
                self._capture_failures += 1
                log.error("audio capture lost: %s -- retrying in 3s", e)
                self.announcer.say(NO_MIC)
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
            # Readiness is the first frame actually read, not the subprocess
            # starting. Popen succeeds immediately even when arecord is about
            # to exit with "Device or resource busy" -- announcing there made
            # a contended device flip between "I can hear you again" and "I
            # can't hear the microphone" every few seconds.
            opened = False
            while self._running:
                # Checked here rather than on a timer thread: this loop already
                # ticks once per 80ms frame while idle, so the buffer is wiped
                # at the deadline instead of merely being ignored by the next
                # turn -- which would leave it sitting in RAM.
                self.memory.expire_if_idle()
                self._tick_skills(read_frame)
                frame = read_frame()
                if not opened:
                    opened = True
                    self._capture_ready()
                if self.silence.feed(frame):
                    # Raises nothing and logs nothing on its own: a wireless
                    # mic with a flat transmitter hands over digital silence
                    # forever and looks perfectly healthy doing it.
                    log.error("microphone has been silent for minutes")
                    self.announcer.say(NO_MIC)
                if self.detector.process(frame) is not None:
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
