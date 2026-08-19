# Faethon

A voice assistant for the Raspberry Pi 4B. Cloud brain, edge ears: the Pi
listens locally for a wake word and nothing leaves the house until you say it.
Everything after that — transcription, thinking, speech — happens on
OpenRouter, so the Pi only ever runs one small model.

```
mic → wake word (local) → chime → record → STT → skill or LLM → TTS → speaker
                            ↑                           ╰── streamed ──╯     │
                            ╰───────────── follow-up window ─────────────────╯
```

Only the first thing you say needs the wake word. Faethon listens again for
five seconds after each reply, so "and what about tomorrow?" just works. Say
nothing and a falling chime closes the conversation. Say the wake word while
it's still talking and it stops.

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
thousand replies. Call it pennies a month for household use. `faethon-probe`
prints the actual cost of each call.

## Requirements

- Raspberry Pi 4B, 4GB or better (openWakeWord uses ~16% of one core)
- A USB microphone and a speaker
- An [OpenRouter](https://openrouter.ai/keys) API key with credit
- `alsa-utils` (ships with Pi OS)

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if you don't have uv
git clone <your-repo> faethon && cd faethon
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
the wake chime. If it says SILENT, the problem is the microphone, not Faethon.

### Check the cloud legs

```bash
uv run faethon-probe tts "hello, I am Faethon"
uv run faethon-probe llm "why is the sky blue"
uv run faethon-probe say "why is the sky blue"   # streamed: speaks as it thinks
uv run faethon-probe stt                          # records 4s, transcribes
uv run faethon-probe chain                        # the whole pipeline, timed
uv run faethon-probe bench                        # streamed vs buffered latency
```

Each prints what the call cost.

### Run it

```bash
uv run faethon            # foreground, Ctrl-C to stop
sudo ./scripts/install-service.sh    # or install as a systemd service
journalctl -u faethon -f
```

## Wake word

Wakes on **"Hey Rhasspy"** — openWakeWord's stock model, which measured 0.9999
on a real voice through the microphone here. The phrase is only a trigger; the
assistant calls itself Rhasspy in conversation, and the project is Faethon.

```yaml
wake:
  model: "hey_rhasspy"   # a pretrained name, or a path to a .onnx/.tflite
  threshold: 0.7         # raise if it false-triggers, lower if it misses you
```

Pretrained alternatives: `alexa`, `hey_mycroft`, `hey_jarvis`. Avoid `alexa` if
you own an Echo.

## Carrying on a conversation

After every reply the microphone reopens for a follow-up, so only the first
turn of a conversation needs the wake word.

```yaml
conversation:
  follow_up: true
  follow_up_ms: 5000
```

A rising chime means Faethon is listening; a falling one means the conversation
is over and the wake word is needed again. Nothing else distinguishes them, so
they're the same two notes in opposite order — audible across a room without
having to be learned.

`follow_up_ms` is worth thinking about as a privacy setting, not just a latency
one: it's the only period when the microphone is live without anyone having
deliberately triggered it, and it is spent in full, as silence, at the end of
every conversation. Below about 3s it clips people still deciding what to ask;
above about 8s the pause after a reply starts to feel like a hang. Set
`follow_up: false` for one question per wake word.

A follow-up turn is a turn like any other: it goes into the same 10-turn
memory, so pronouns carry across ("what's the capital of France?" → "how big is
it?").

### Interrupting a reply

Say **"Hey Rhasspy"** while Faethon is talking and it stops mid-sentence, then
listens. Useful when a one-line question gets a paragraph.

```yaml
conversation:
  barge_in: true
  barge_in_threshold: 0.1
```

Interrupting normally needs acoustic echo cancellation, because the microphone
hears the assistant far louder than it hears the room. Listening for a
*phrase* sidesteps that: a wake-word model isn't asking "is someone talking",
it's asking "was that 'hey rhasspy'", and Faethon never says its own wake word.
Playing a 30-second reply through the speaker and scoring the recorded
microphone gave a median of 0.0000 and a peak of 0.0014 — no false trigger,
against a threshold of 0.7.

**`barge_in_threshold` is much lower than `wake.threshold` and has to be.**
Faethon's voice doesn't trigger the detector, but it does mask yours. Playing a
reply and a real recording of the wake word through the speaker together, at
the equal loudness they measured at the mic:

| what the detector hears | score |
|---|---|
| wake word, quiet room | 0.9999 |
| wake word, over Faethon talking | **0.3681** |
| Faethon talking, no wake word | 0.0002 |

At the 0.7 used for waking, barge-in never fires at all. 0.1 sits 3.5× below
the masked phrase and 500× above the self-audio floor.

Detection falls off sharply once Faethon is louder than you are at the
microphone — measured against a file mix, the score went 0.998 at a quarter of
your volume, 0.925 at half, and 0.118 at equal. **If you have to shout, turn
the speaker down** rather than dropping the threshold further.

Stopping is not the same as finishing. Ending a reply closes the pipe and lets
`aplay` play out everything buffered, which measured **9.1 seconds** of talking
after the decision to stop; barge-in terminates it instead, at 234ms. It also
stops pulling from the model, so an interrupted reply stops costing tokens, and
records what was actually said to memory — otherwise "what did you just say?"
would draw a blank and asking again would replay the whole answer.

One limit: Faethon can only notice the interruption between sentences of the
*model's* output, so if generation stalls, the listening window opens a beat
late. The audio stops immediately either way.

Two things this needs that aren't obvious:

- **Faethon's own voice has to be flushed from the microphone.** `arecord` runs
  the whole time, including while Faethon talks, so a reply ends with a
  recording of that reply sitting in the capture buffer. Left there, the
  follow-up window transcribes it and Faethon answers its own sentence. The
  buffer is drained after speaking and never before — the audio arriving right
  after a wake word is you running straight on into your question.
- **The two waits are separate budgets.** `utterance.start_timeout_ms` is spent
  by someone who just said the wake word and is expected to speak;
  `conversation.follow_up_ms` is spent by someone who was merely spoken to, and
  usually on silence. Once you start talking, `silence_ms` takes over as
  normal — a short follow-up window won't cut off a long answer.

A custom-trained model lives at `models/hey_roxy.onnx` but is not used: it
scored 0.84-0.91 on synthesised voices and only 0.52-0.55 on a real one,
detecting roughly 1 utterance in 4. It appears to have been trained on
synthetic speech alone. If you train your own, **validate it against a
recording of yourself**, not against TTS — that mistake is what made a
threshold of 0.78 look reasonable when nothing could reach it.

## Volume

Say **"Hey Rhasspy, volume up"** or **"volume down"** to move one step. The
scale is 0 to 10, where 0 is muted and 10 is the mixer's maximum. Also
understands "louder", "quieter", "turn it up", "mute", "set the volume to 8",
and "what's the volume".

Percentages work as input too, and have to: Faethon announces "70%", so that's
what people say back. "Set the volume to 30%" and "volume 3" mean the same
thing, as do "to 7" and "to 70" — a number is read as a percentage when it
carries a unit or is simply too big to be a level. Any positive percentage
gives at least level 1, since rounding 3% down to silence would be a different
thing from what was asked.

Each change is spoken back as a percentage of the dial — one level is one
tenth, so level 7 announces **"Volume is set to 70%."** and level 0 says
**"Volume is set to 0%, muted."**, since 0% alone leaves it unclear whether the
speaker is silent or merely turned right down. The literal `%` is safe: Fish
Audio pronounces it, measured at 2.04s of audio against 2.09s for the word
spelled out and 1.72s with the sign removed.

The mapping is the part worth knowing about. ALSA's PCM control on a Pi is
scaled in **dB**, and the percentage `amixer` prints is a linear position in
that range rather than a loudness — so the obvious implementation, level 5
meaning 50%, is inaudible. Measured by playing a tone and recording it:

| `amixer` says | actual | loudness at the mic |
|---|---|---|
| 96% | 0 dB | 103 |
| 87% | −10 dB | 24 |
| 77% | −20 dB | 8 |
| 68% | −30 dB | 3 |
| 49% | −50 dB | 2 — silence |

Full volume already reads as 96%, and the entire useful range lives in the top
third of the percentage scale. So the levels are spaced evenly in dB instead,
which is roughly how loudness is perceived. Measured across the finished scale,
each step is a consistent **1.53–1.55×** in amplitude, so one "volume up" feels
like the same size step wherever you are on the dial.

`USABLE_RANGE_DB` in `src/faethon/skills/volume_skill.py` is the one number to
tune: 34 dB below maximum, the tone had reached the microphone's noise floor
and going lower changed nothing audible. Lower it if level 1 is still too loud
in a quiet room; raise it if the bottom of the scale is unusably quiet.

The mixer control is discovered at runtime rather than hardcoded, for the same
reason the rest of Faethon addresses ALSA by name: card indices move when USB
devices are re-plugged. If no playback control is found the skill reports
itself unavailable, so it's hidden from the LLM and explains itself out loud
rather than failing silently.

## Writing a skill

Drop a file in `src/faethon/skills/`. The registry finds it — no imports to
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
| `src/faethon/__main__.py` | the main loop |
| `src/faethon/wake.py` | openWakeWord wrapper |
| `src/faethon/audio/` | ALSA capture, playback, end-of-speech detection |
| `src/faethon/providers/` | OpenRouter STT, LLM, TTS |
| `src/faethon/speech.py` | sentence chunking and pipelined playback |
| `src/faethon/skills/` | skill contract, registry, and the skills themselves |
| `src/faethon/router.py` | regex-first, LLM-fallback routing |
| `config.yaml` | everything machine-specific |

Audio I/O shells out to `arecord`/`aplay` rather than binding PortAudio: it's
already on every Pi OS image, and the device strings are the same ones you test
with by hand.

## Tuning latency

Faethon logs a breakdown every turn, timed from the moment you stop speaking:

```
turn: 2.3s audio | stt 0.81s | reply+speech 2.66s | total 3.47s | $0.00004
```

The knobs, in the order they're worth reaching for:

| Symptom | Knob |
|---|---|
| Long pause before anything happens | `utterance.silence_ms` — dead air before Faethon even starts. 600ms is near the floor; below that it cuts people off mid-sentence. |
| Gives up while you're still thinking | `utterance.start_timeout_ms` — the budget for starting to speak, separate from `silence_ms`. |
| Closes the conversation too fast, or holds the mic open too long after a reply | `conversation.follow_up_ms`. |
| Gives up in a noisy room | `utterance.speech_onset_ms` — sustained voice needed before Faethon believes you've started. |
| `stt` line is slow | `models.stt` — `whisper-large-v3-turbo` is the fast default; plain `whisper-large-v3` is more accurate and slower. |
| Slow to start talking | `llm.provider_sort` first (see below), then `llm.max_tokens` — shorter replies finish sooner. Provider latency varies a lot more than the Pi does. |
| Choppy or oddly-paced speech | `MIN_CHARS` / `FIRST_MIN_CHARS` in `src/faethon/speech.py` — bigger chunks sound smoother but start later. |

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
- **Voices differ, and an empty voice is not a default.** Fish Audio S1 offers
  exactly one, `alloy`, and you have to ask for it: omitting the field does not
  fall back to a fixed speaker, it picks a different one per request. Measured
  over five renders of one sentence, pitch ranged 99–156 Hz unset against
  101–116 Hz with `alloy` — a different person each time you ask Faethon
  anything. Deepgram requires a voice outright. To see a provider's voices,
  send a bogus one; the 400 lists them.

## A note on the LLM

`deepseek/deepseek-v4-flash` is a **hybrid reasoning model**: it decides
per-request whether to think before answering, and thinking tokens are billed
against `max_tokens`. On a 100-token spoken-reply budget that goes wrong —
measured at 99 of 100 tokens spent reasoning on one arithmetic question,
returning empty content, so Faethon says nothing at all. It's intermittent, which
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

### Which provider serves it

OpenRouter serves this model from eighteen providers and, left alone, routes
cheapest-first. That optimises the wrong thing: a spoken reply is late by
however long the provider takes to produce its first token, and the cheap ones
are erratic — ten identical calls on the default routing all landed on
GMICloud and ranged 2.09s to 7.22s to first token.

```yaml
llm:
  provider_sort: "latency"   # "" | "price" | "latency" | "throughput"
```

Five trials each, with the real system prompt and tool schemas:

| `provider_sort` | median ttft | worst | who served it |
|---|---|---|---|
| `""` (cheapest-first) | 2.76s | 5.22s | GMICloud, StreamLake |
| **`"latency"`** | **1.50s** | **2.94s** | CoreWeave, GMICloud |
| `"throughput"` | 0.81s | 11.36s | Alibaba |

`"throughput"` has the best median and by far the worst tail, and the tail is
the half people notice — a reply that is occasionally eleven seconds late is
worse than one reliably at a second and a half.

It costs a little more, since routing on anything but price means not always
taking the cheapest of the eighteen (they span $0.068–$0.44 per M input against
GMICloud's $0.084). On a bill of pennies a month that is not the binding
constraint. Set it back to `""` for cheapest-first.

`faethon-probe llm` and `faethon-probe bench` send the same routing as the
assistant does, so their numbers stay comparable with what you hear.

It also **folds under pressure** unless told not to. Out of the box it would
answer "King Charles the Third" correctly, then on being told "that's wrong"
reply *"You're right — the United Kingdom doesn't have a king."* Confidently
wrong on the second try is worse than useless when there's no transcript to
scroll back through. The system prompt in `config.yaml` therefore tells it to
reconsider honestly and hold its ground when it was right — measured 4/4 held,
while still correcting itself when genuinely wrong (7 × 8).

## Known limits

- **Half-duplex except for the wake word.** Faethon can't understand you while
  it's talking — only spot its own wake phrase, which is what barge-in uses.
  Anything else you say during a reply is lost.
- **A follow-up window has no cap on how long a conversation can run.** Anything
  that keeps producing speech Whisper will transcribe — a television in the same
  room — can hold one open indefinitely, and every turn is a paid API call.
- **Memory is 10 turns and RAM-only.** It forgets on restart, deliberately.
- **One wake word at a time.**

## Tests

```bash
uv run pytest
```

No network and no audio hardware required — the provider calls are stubbed.
