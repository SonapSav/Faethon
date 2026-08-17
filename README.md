# Roxy

A voice assistant for the Raspberry Pi 4B. Cloud brain, edge ears: the Pi
listens locally for a wake word and nothing leaves the house until you say it.
Everything after that — transcription, thinking, speech — happens on
OpenRouter, so the Pi only ever runs one small model.

```
mic → wake word (local) → chime → record → STT → skill or LLM → TTS → speaker
                                                        ╰── streamed ──╯
```

The reply is spoken sentence by sentence as the model generates it, rather than
after. Waiting for the whole reply costs (generation + synthesis); streaming
costs (first clause + synthesis of that clause), with the rest pipelined behind
audio that's already playing. Measured on this Pi, that's about **4.5s → 2.7s**
to first word.

Synthesis and playback run on **separate threads**, which matters more than it
sounds. `aplay` applies backpressure, so a single worker doing both wouldn't
request the next sentence until the current one had finished playing — a silent
gap at every full stop the length of a whole TTS round-trip. Measured at
**+7.6s of dead air, now +1.7s** (and what's left is the initial synthesis,
not a gap between sentences).

## What it costs

Per spoken turn, roughly: Whisper Large v3 Turbo at $0.00000333/unit, DeepSeek
V4 Flash at $0.08/M input and $0.16/M output, Fish Audio S1 at about $1 per
thousand replies. Call it pennies a month for household use. `roxy-probe`
prints the actual cost of each call.

## Requirements

- Raspberry Pi 4B, 4GB or better (openWakeWord uses ~16% of one core)
- A USB microphone and a speaker
- An [OpenRouter](https://openrouter.ai/keys) API key with credit
- `alsa-utils` (ships with Pi OS)

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if you don't have uv
git clone <your-repo> roxy && cd roxy
uv sync                                            # installs Python 3.11 + deps
cp .env.example .env                               # then add your API key
```

`uv` fetches its own CPython 3.11 — the system Python is untouched. 3.11
specifically, because openWakeWord depends on `tflite-runtime`, which publishes
no wheel for 3.12 or later.

### Point it at your audio hardware

```bash
arecord -L    # capture devices
aplay -L      # playback devices
```

Put the device strings in `config.yaml`. **Use names, not card indices** —
indices shift when USB devices are re-plugged.

```yaml
audio:
  input_device: "plughw:CARD=Microphone,DEV=0"
  output_device: "plughw:CARD=Headphones,DEV=0"
```

Then confirm the hardware works before blaming anything above it:

```bash
./scripts/check-audio.sh
```

It records three seconds, reports the signal level, plays it back, and plays
the wake chime. If it says SILENT, the problem is the microphone, not Roxy.

### Check the cloud legs

```bash
uv run roxy-probe tts "hello, I am Roxy"
uv run roxy-probe llm "why is the sky blue"
uv run roxy-probe say "why is the sky blue"   # streamed: speaks as it thinks
uv run roxy-probe stt                          # records 4s, transcribes
uv run roxy-probe chain                        # the whole pipeline, timed
uv run roxy-probe bench                        # streamed vs buffered latency
```

Each prints what the call cost.

### Run it

```bash
uv run roxy            # foreground, Ctrl-C to stop
sudo ./scripts/install-service.sh    # or install as a systemd service
journalctl -u roxy -f
```

## Wake word

Ships with openWakeWord's stock **"hey jarvis"**. There is no pretrained
"Roxy" model — training a custom one is on the list, and swapping it in is a
one-line config change:

```yaml
wake:
  model: "models/hey_roxy.tflite"   # a path, or a pretrained name
  threshold: 0.5                     # raise if it false-triggers
```

Pretrained alternatives: `alexa`, `hey_mycroft`, `hey_rhasspy`.

## Writing a skill

Drop a file in `src/roxy/skills/`. The registry finds it — no imports to
update, no list to edit.

```python
from .base import Skill

class LightSkill(Skill):
    name = "set_light"
    tag = "home"
    description = "Turn a light on or off."      # the LLM reads this
    patterns = [r"\bturn (?P<state>on|off) the (?P<room>\w+) light\b"]
    parameters = {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": ["on", "off"]},
            "room": {"type": "string"},
        },
        "required": ["state", "room"],
    }

    def run(self, **params) -> str:
        ...
        return f"Turned the {params['room']} light {params['state']}."

SKILL = LightSkill()
```

Declaring it once makes it reachable two ways:

- **regex** — `patterns` are matched locally against the transcript. Free,
  instant, offline. This is the path for the phrasings you use every day.
- **tool-calling** — `description` and `parameters` go to the LLM, which
  invokes the skill with extracted arguments when the phrasing is one you
  didn't anticipate. Costs one API call.

Accept `**params` unless you want strict argument checking: the regex path can
match with no capture groups, and a model may omit optional arguments.

Override `available` for skills with a dependency that might be missing. An
unavailable skill is hidden from the LLM and, if its regex matches, explains
itself out loud rather than failing silently.

## Layout

| Path | What's there |
|---|---|
| `src/roxy/__main__.py` | the main loop |
| `src/roxy/wake.py` | openWakeWord wrapper |
| `src/roxy/audio/` | ALSA capture, playback, end-of-speech detection |
| `src/roxy/providers/` | OpenRouter STT, LLM, TTS |
| `src/roxy/speech.py` | sentence chunking and pipelined playback |
| `src/roxy/skills/` | skill contract, registry, and the skills themselves |
| `src/roxy/router.py` | regex-first, LLM-fallback routing |
| `config.yaml` | everything machine-specific |

Audio I/O shells out to `arecord`/`aplay` rather than binding PortAudio: it's
already on every Pi OS image, and the device strings are the same ones you test
with by hand.

## Tuning latency

Roxy logs a breakdown every turn, timed from the moment you stop speaking:

```
turn: 2.3s audio | stt 0.81s | reply+speech 2.66s | total 3.47s | $0.00004
```

The knobs, in the order they're worth reaching for:

| Symptom | Knob |
|---|---|
| Long pause before anything happens | `utterance.silence_ms` — dead air before Roxy even starts. 600ms is near the floor; below that it cuts people off mid-sentence. |
| Gives up while you're still thinking | `utterance.start_timeout_ms` — the budget for starting to speak, separate from `silence_ms`. |
| Gives up in a noisy room | `utterance.speech_onset_ms` — sustained voice needed before Roxy believes you've started. |
| `stt` line is slow | `models.stt` — `whisper-large-v3-turbo` is the fast default; plain `whisper-large-v3` is more accurate and slower. |
| Slow to start talking | `llm.max_tokens` — shorter replies finish sooner. Also try a different model; provider latency varies a lot more than the Pi does. |
| Choppy or oddly-paced speech | `MIN_CHARS` / `FIRST_MIN_CHARS` in `src/roxy/speech.py` — bigger chunks sound smoother but start later. |

Almost none of the delay is the Pi. Wake-word inference is 13ms per 80ms frame
(~16% of one core); the rest is network round-trips.

## Choosing a TTS model

The speaker consumes audio at exactly 1× real time. If synthesis is slower than
that, the card runs dry between sentences and **ALSA underruns** — recovery
costs far more than the gap itself, and a 4.4s reply stretched to 19s of wall
clock before this was understood. So the metric that matters is not price, it's
whether the model can deliver audio faster than it's spoken.

Benchmarked 2026-08-17, 3 trials each, one sentence:

| model | reliability | to first byte | total | rate | per 1000 replies |
|---|---|---|---|---|---|
| **fish-audio/s1** | 3/3 | **0.34s** | **1.17s** | 44100 | $1.05 |
| deepgram/aura-2 | 3/3 | 0.48s | 2.51s | 24000 | $2.10 |
| deepgram/flux-tts:free | 3/3 | 1.32s | 3.97s | 24000 | free |
| sesame/csm-1b | 3/3 | 5.71s | 6.43s | 24000 | $0.49 |
| hexgrad/kokoro-82m | 0/3 | — | — | — | $0.04 |

Fish S1 returns 3.95s of audio in 1.17s — 3.4× real time, with headroom to
spare. Kokoro is by far the cheapest and was the original choice, but it began
returning HTTP 200 and then hanging mid-body; it's worth retrying if you want
the saving.

Two gotchas the code now handles for you:

- **Sample rates differ.** Fish Audio returns 44.1kHz, Kokoro and Deepgram
  24kHz. Playing one at the other's rate isn't subtle — 44.1k played as 24k
  runs at 0.54× speed. The rate is read from the response `Content-Type`;
  `tts.sample_rate` in the config is only a fallback.
- **Voices differ.** Fish Audio rejects a `voice` field entirely (leave it
  `""`), Deepgram requires one. To see a provider's voices, send a bogus one —
  the 400 lists them all.

## A note on the LLM

`deepseek/deepseek-v4-flash` is a **hybrid reasoning model**: it decides
per-request whether to think before answering, and thinking tokens are billed
against `max_tokens`. On a 100-token spoken-reply budget that goes wrong —
measured at 99 of 100 tokens spent reasoning on one arithmetic question,
returning empty content, so Roxy says nothing at all. It's intermittent, which
makes it nastier: the same question can work and then not.

So `llm.reasoning: false` is the default, which sends `reasoning: {"enabled":
false}`. Two things that look like they'd work but don't — both still spend the
full budget thinking and merely *hide* the result:

```yaml
reasoning: {"exclude": true}      # WRONG - still burns the tokens
reasoning: {"max_tokens": 0}      # WRONG - still burns the tokens
```

Turn reasoning on only if you also raise `max_tokens` a long way, and accept
that a reply you have to wait several seconds for is a poor fit for speech.

It also **folds under pressure** unless told not to. Out of the box it would
answer "King Charles the Third" correctly, then on being told "that's wrong"
reply *"You're right — the United Kingdom doesn't have a king."* Confidently
wrong on the second try is worse than useless when there's no transcript to
scroll back through. The system prompt in `config.yaml` therefore tells it to
reconsider honestly and hold its ground when it was right — measured 4/4 held,
while still correcting itself when genuinely wrong (7 × 8).

## Known limits

- **Half-duplex.** Roxy can't hear you while it's talking; interrupting it
  would need echo cancellation.
- **Memory is 10 turns and RAM-only.** It forgets on restart, deliberately.
- **One wake word at a time.**

## Tests

```bash
uv run pytest
```

No network and no audio hardware required — the provider calls are stubbed.
