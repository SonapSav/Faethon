"""Smoke-test each provider in isolation.

    uv run faethon-probe stt [file.wav]     record 4s (or read a wav) and transcribe
    uv run faethon-probe llm "why is the sky blue"
    uv run faethon-probe tts "hello, I am Faethon"
    uv run faethon-probe chain                 record -> stt -> llm -> tts

Prints the running cost so you can see what a turn actually costs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import wave
from pathlib import Path

from .audio import capture, playback
from .config import load_config
from .providers import llm as llm_mod
from .providers import stt as stt_mod
from .providers import tts as tts_mod
from .providers.client import OpenRouterClient, OpenRouterError


def _read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need mono 16-bit; got "
                             f"{w.getnchannels()}ch {w.getsampwidth() * 8}-bit")
        return w.readframes(w.getnframes()), w.getframerate()


def _record(cfg, seconds: float) -> bytes:
    print(f"Recording {seconds:.0f}s -- speak now...", flush=True)
    pcm = capture.record_seconds(cfg.audio.input_device, cfg.audio.sample_rate, seconds)
    print("  done", flush=True)
    return pcm


def cmd_stt(cfg, client, args) -> None:
    if args.file:
        pcm, rate = _read_wav(Path(args.file))
    else:
        pcm, rate = _record(cfg, args.seconds), cfg.audio.sample_rate

    t0 = time.monotonic()
    text = stt_mod.transcribe(client, pcm, model=cfg.models.stt, sample_rate=rate,
                              language=cfg.stt.language,
                              temperature=cfg.stt.temperature)
    print(f"\n  [{time.monotonic() - t0:.2f}s] {text!r}")


def cmd_llm(cfg, client, args) -> None:
    messages = [
        {"role": "system", "content": cfg.llm.system_prompt},
        {"role": "user", "content": args.text},
    ]
    t0 = time.monotonic()
    reply = llm_mod.complete(
        client, messages,
        model=cfg.models.llm,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        provider_sort=cfg.llm.provider_sort,
    )
    print(f"\n  [{time.monotonic() - t0:.2f}s] {reply.text}")


def cmd_tts(cfg, client, args) -> None:
    """Streams to the speaker and reports time-to-first-audio, which is the
    number that decides whether Faethon feels responsive."""
    t0 = time.monotonic()
    first: float | None = None
    total = 0

    with playback.stream(cfg.audio.output_device, cfg.tts.sample_rate) as write:
        for chunk in tts_mod.synthesize_stream(
            client, args.text, model=cfg.models.tts, voice=args.voice or cfg.tts.voice
        ):
            if first is None:
                first = time.monotonic() - t0
            total += len(chunk)
            write(chunk)

    if first is None:
        print("  no audio returned")
        return
    secs = total / 2 / cfg.tts.sample_rate
    print(f"\n  first audio in {first:.2f}s; {total} bytes = {secs:.2f}s "
          f"at {cfg.tts.sample_rate} Hz (total {time.monotonic() - t0:.2f}s)")
    print("  If that sounded chipmunk-y or slow, tts.sample_rate is wrong.")


def cmd_say(cfg, client, args) -> None:
    """Ask the LLM and speak the reply as it streams.

    Compare against `faethon-probe llm` + `faethon-probe tts` run back to back: this
    should start talking far sooner.
    """
    from .providers import llm as llm_stream
    from .speech import sentence_chunks, speak_streaming

    messages = [
        {"role": "system", "content": cfg.llm.system_prompt},
        {"role": "user", "content": args.text},
    ]
    t0 = time.monotonic()
    first_token: list[float] = []

    reply = llm_stream.complete_streaming(
        client, messages,
        model=cfg.models.llm,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        provider_sort=cfg.llm.provider_sort,
    )

    def timed():
        for delta in reply:
            if not first_token:
                first_token.append(time.monotonic() - t0)
            yield delta

    spoken = speak_streaming(client, cfg, sentence_chunks(timed()))
    total = time.monotonic() - t0

    print(f"\n  said: {spoken}")
    if first_token:
        print(f"  first token in {first_token[0]:.2f}s")
    print(f"  total {total:.2f}s")


def cmd_bench(cfg, client, args) -> None:
    """Time-to-first-audio, streamed vs. buffered, on the same prompt."""
    from .providers import llm as llm_stream
    from .speech import sentence_chunks, speak_streaming

    messages = [
        {"role": "system", "content": cfg.llm.system_prompt},
        {"role": "user", "content": args.text},
    ]

    print("1/2  buffered: whole reply, then synthesise")
    t0 = time.monotonic()
    reply = llm_mod.complete(
        client, messages, model=cfg.models.llm,
        max_tokens=cfg.llm.max_tokens, temperature=cfg.llm.temperature,
        provider_sort=cfg.llm.provider_sort,
    )
    t_llm = time.monotonic() - t0
    first_buffered = None
    with playback.stream(cfg.audio.output_device, cfg.tts.sample_rate) as write:
        for chunk in tts_mod.synthesize_stream(
            client, reply.text, model=cfg.models.tts, voice=cfg.tts.voice
        ):
            if first_buffered is None:
                first_buffered = time.monotonic() - t0
            write(chunk)
    buffered_total = time.monotonic() - t0
    print(f"     llm {t_llm:.2f}s -> first audio {first_buffered:.2f}s "
          f"-> done {buffered_total:.2f}s")

    print("2/2  streamed: speak each sentence as it lands")
    t0 = time.monotonic()
    streaming = llm_stream.complete_streaming(
        client, messages, model=cfg.models.llm,
        max_tokens=cfg.llm.max_tokens, temperature=cfg.llm.temperature,
        provider_sort=cfg.llm.provider_sort,
    )
    speak_streaming(client, cfg, sentence_chunks(streaming))
    streamed_total = time.monotonic() - t0
    print(f"     done {streamed_total:.2f}s")

    if first_buffered:
        print(f"\n  time to first audio: {first_buffered:.2f}s buffered")
        print(f"  (streamed figure is logged above as 'first audio in ...')")


def cmd_chain(cfg, client, args) -> None:
    pcm = _record(cfg, args.seconds)

    t0 = time.monotonic()
    text = stt_mod.transcribe(client, pcm, model=cfg.models.stt,
                              sample_rate=cfg.audio.sample_rate,
                              language=cfg.stt.language,
                              temperature=cfg.stt.temperature)
    t_stt = time.monotonic() - t0
    print(f"  heard [{t_stt:.2f}s]: {text!r}")
    if not text:
        print("  nothing transcribed; stopping")
        return

    # Mirrors the real loop: reply and speech are pipelined, so they can't be
    # timed separately -- what matters is when the first word comes out.
    from .providers import llm as llm_stream
    from .speech import sentence_chunks, speak_streaming

    t1 = time.monotonic()
    streaming = llm_stream.complete_streaming(
        client,
        [{"role": "system", "content": cfg.llm.system_prompt},
         {"role": "user", "content": text}],
        model=cfg.models.llm,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        provider_sort=cfg.llm.provider_sort,
    )
    spoken = speak_streaming(client, cfg, sentence_chunks(streaming))
    t_reply = time.monotonic() - t1

    print(f"  said: {spoken}")
    print(f"\n  stt {t_stt:.2f}s + reply/speech {t_reply:.2f}s "
          f"= {t_stt + t_reply:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(prog="faethon-probe", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stt", help="transcribe a recording")
    p.add_argument("file", nargs="?", help="wav file; omit to record from the mic")
    p.add_argument("--seconds", type=float, default=4.0)
    p.set_defaults(fn=cmd_stt)

    p = sub.add_parser("llm", help="ask the model a question")
    p.add_argument("text")
    p.set_defaults(fn=cmd_llm)

    p = sub.add_parser("tts", help="speak some text")
    p.add_argument("text")
    p.add_argument("--voice", help="override the configured Kokoro voice")
    p.set_defaults(fn=cmd_tts)

    p = sub.add_parser("say", help="ask the LLM and speak the reply as it streams")
    p.add_argument("text")
    p.set_defaults(fn=cmd_say)

    p = sub.add_parser("bench", help="compare streamed vs buffered time-to-first-audio")
    p.add_argument("text", nargs="?", default="why is the sky blue")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("chain", help="record -> stt -> llm -> tts")
    p.add_argument("--seconds", type=float, default=4.0)
    p.set_defaults(fn=cmd_chain)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    try:
        with OpenRouterClient(cfg.settings.openrouter_api_key) as client:
            args.fn(cfg, client, args)
            print(f"  cost this run: ${client.spent:.6f}")
    except OpenRouterError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except capture.CaptureError as e:
        print(f"\nAUDIO ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
