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
audio that's already playing. Measured on this Pi 2026-08-17, that's about
**4.5s → 2.7s** to first word. Both halves of that comparison move with
provider latency; the ratio is the durable part, not the absolute figures.

Synthesis and playback run on **separate threads**, which matters more than it
sounds. `aplay` applies backpressure, so a single worker doing both wouldn't
request the next sentence until the current one had finished playing — a silent
gap at every full stop the length of a whole TTS round-trip. Measured at
**+7.6s of dead air, now +1.7s** (measured 2026-08-17; what's left is the
initial synthesis, not a gap between sentences).

## What it costs

Per spoken turn, roughly: Whisper Large v3 Turbo at $0.00000333/unit, DeepSeek
V4 Flash at $0.08/M input and $0.16/M output, Fish Audio S1 at $0.0032 per
thousand characters of input text. Call it pennies a month for household use.
`faethon-probe` prints the actual cost of each call.

**Speaking is the expensive part**, and it is the one leg with no usage to
read: `/audio/speech` returns raw audio, `/generation` 404s for TTS ids, and
`/credits` updates in batches too coarse to attribute a single call. So it is
estimated from the text sent, via `tts.cost_per_1k_chars`. Until that existed
the turn line omitted it entirely and reported a turn at a fraction of what it
actually cost.

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

`max_turns` and `max_seconds` bound how far a conversation can run without a
fresh wake word. That matters more than the cost: anything that keeps producing
transcribable speech — a television in the same room — re-opens the follow-up
window on every turn, so unbounded, the one period when the microphone is live
without a deliberate trigger becomes the steady state. Turns bound the spend,
seconds bound how long the mic stays open; either ends it, and the falling
chime says so.

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

## Clearing the conversation

Faethon keeps the last ten exchanges so pronouns carry across turns. Say
**"Hey Rhasspy, clear the buffer"** to drop them — also "clear the memory",
"forget the conversation", "start a new conversation" — and it answers
**"Memory is cleared."**

The wipe includes the exchange that asked for it, so the buffer really is
empty rather than holding "clear the buffer" as its first entry. That is the
whole subtlety: the router records every turn *after* the skill has run, so a
skill that clears memory has to suppress its own recording too. Skills declare
`clears_memory = True` and the router does both — skills hold no reference to
memory themselves.

It also forgets on its own after `llm.history_idle_minutes` of silence — 10 by
default. Without a bound, context never ends: ask about France in the morning
and "how big is it?" in the afternoon still answers about France, which is
wrong in a way nothing announces. The window was picked from 71 real
interactions here, whose gaps fall into two clumps with a trough between them —
53 gaps under a minute (one conversation), then almost nothing between 5 and 10
minutes, then a band of clearly separate sessions. 10 minutes sits in the
trough; 30 would have fired once in seventy.

The clock runs from the last turn, not the first, so a long conversation
doesn't expire while it's still going. The check happens in the wake-word loop,
which already ticks every 80 ms while idle — so the buffer is genuinely wiped
at the deadline rather than merely ignored by the next turn, which would leave
it sitting in RAM.

The buffer costs about 3–5 kB of RAM whatever you do with it. What it actually
costs is prompt size: ten turns adds roughly 250–700 tokens to every request,
which is latency on the slowest leg. Clearing it is a way to make Faethon
quicker as well as more private.

## Checking your credit

Say **"Hey Rhasspy, what's my OpenRouter credit balance"** and it answers
**"Your OpenRouter balance is 1.77 dollars."** — always two decimals, and the
word "dollars" rather than `$`, since a skill's reply goes straight to TTS with
no model in between to tidy it up.

The balance is a subtraction. `GET /credits` returns what was granted and what
has been spent, not what is left:

```json
{"data": {"total_credits": 30, "total_usage": 28.228023831}}
```

The regex is deliberately loose about the spelling. "OpenRouter" is a coined
word, so Whisper writes it however it sounds — OpenRouter, Open Router,
OpenRooter, Open Rooter — and a pattern matching only the correct spelling
fails in the worst way available: it falls through to the LLM, which cannot see
your balance and will invent a plausible one. `open[\s.\-]*r[oeu]{1,3}ter`
accepts the vowel cluster loosely instead.

## When it can't work

Every way Faethon breaks used to sound identical: silence. That isn't laziness
in the error handling, it's structural — the mechanism for speaking is the
thing that has broken. The code caught a failed transcription, which almost
always means the network is down, and called cloud TTS to apologise for it.

So the failure states have pre-rendered clips, played straight to the speaker
with no network, key, credit or model involved:

| you hear | what happened | what to do |
|---|---|---|
| "I can't reach the network right now." | request failed or retries exhausted | check the router |
| "My OpenRouter credit has run out." | HTTP 402 | top up |
| "I can't hear the microphone." | `CaptureError`, or minutes of digital silence | check the mic |
| "I can hear you again." | capture recovered, and the failure had been announced | nothing |
| "Your account is below half a dollar." | balance crossed `credit.warn_below` | top up |
| "Something has gone wrong and I have stopped." | the service gave up | `journalctl -u faethon` |

Credit and network are separate clips because the fixes are different, and
guessing wrong sends you to reboot a router that was working fine.

**The microphone one catches a failure that raises nothing.** A wireless mic
whose transmitter is off or flat still enumerates, still opens, and still hands
over frames — of digital silence, indefinitely, looking perfectly healthy. So
`SilenceWatch` notices when the stream has been *literally* all-zero for two
minutes. A live mic in a silent room never is: measured here, a quiet room
still peaks at 2–4 per frame.

Each status is announced once and then suppressed until something works again —
an outage lasts as long as it lasts, and repeating it every attempt turns
information into nagging.

The credit warning is the one that fires *before* anything breaks. At zero
every leg stops at once, so the failure is total silence — the useful moment is
earlier. It says only that the balance is low; ask **"what's my credit
balance"** for the figure, which the skill already answers live.

Its suppression is deliberately not the Announcer's. `recovered()` clears every
status and runs after each successful turn, so a credit warning routed through
it would be un-suppressed within seconds and repeat on every check. Instead it
warns once per crossing and re-arms only when the balance climbs back — which
only a top-up does, so it can't flap.

**Recovery is announced too**, which matters more than it sounds. A USB
microphone can take minutes to appear after a cold boot — measured at 2m45s
here — and the retry loop handled that correctly and *silently*. So the
journal ended on an error from long before everything started working, and the
last thing heard in the room was "I can't hear the microphone". Indistinguishable
from having died, while it was in fact fine.

Readiness is the first frame actually read, not the capture subprocess
starting: `Popen` succeeds immediately even when `arecord` is about to exit
with "Device or resource busy", and announcing there made a contended device
alternate between the two clips every few seconds. Recovery is only spoken if
the failure was — telling someone the microphone is back, when they never
heard it go, is noise.

The last row is systemd's, not Faethon's, via `OnFailure=faethon-failed.service`
— a `oneshot` that runs nothing but `aplay`, so it still works when a missing
key or a broken venv means no Faethon code can run at all. That also means the
unit has to be able to give up rather than retrying forever: a unit that never
fails can never announce that it failed, and this service spent its first half
hour crash-looping silently for exactly that reason.

The limit is **8 starts in 60 seconds**, tuned to separate the two cases by
*rate*, because systemd counts every start — including deliberate ones. A
crash loop restarts every `RestartSec=5`, so it trips this in about 40s. A
voice restart costs ~20s of speaking, exiting, waiting and greeting, so eight
of them need over two minutes and can't trip a 60-second window. An earlier
five-in-five-minutes setting put the unit into `start-limit-hit` after five
uses of "restart yourself" — healthy, restarted on purpose, declared dead.

Regenerate the clips with `uv run python scripts/make_speech.py`.

## Starting up

When the service starts, Faethon says **"Hi, I am up and running! Say my name
whenever you need me."** once — so a restart is audible from across the room,
and anyone in the house learns there's a wake word without being told.

It's pre-rendered to `assets/greeting.wav` rather than synthesised on each
boot. A fixed sentence needs no network round-trip, and this way it still
plays when OpenRouter is unreachable or out of credit — which is exactly when
knowing the service came back is worth most. Reword it in
`scripts/make_greeting.py` and re-run that; `greet_on_start: false` turns it
off.

**Keep the assistant's name out of the greeting.** The current wording scores
**0.0001** on the wake model through the speaker and mic. An earlier draft
opened "Hi, I am Rhasspy" and scored **0.5036** — still under the 0.7 wake
threshold, but five thousand times higher and well above the 0.1 that barge-in
listens at. Saying the name is most of saying the wake word.

The greeting also plays before the microphone is opened, which costs nothing
and means a future rewording can't wake Faethon up at every start. A test pins
the margin either way.

## Restarting it

Say **"Hey Rhasspy, restart yourself"** — also "restart the assistant",
"restart the service" — and it answers "Restarting now. Back in a few
seconds." then exits. `Restart=always` in the unit brings it back about five
seconds later, so this needs no privileges at all.

Two details that are invisible until they go wrong:

- **The exit happens after the reply, not during it.** Killing the process
  inside the skill means the sentence is never spoken, and silence followed by
  a dead assistant is indistinguishable from a crash. Skills defer that kind of
  work by overriding `after_reply`, which the main loop runs once the speaker
  has finished.
- **It is hidden from the LLM.** `regex_only = True` keeps it out of the tool
  list, so only the patterns can reach it. A model that decides for itself when
  to use its tools should not have a restart within reach of "this keeps
  freezing, what should I do?".

Restarting forgets the conversation, since memory is RAM-only. That is what a
restart has always done, not something this adds.

**Rebooting the Pi is deliberately not implemented.** It would need either a
polkit rule for `org.freedesktop.login1.reboot` or a sudoers entry plus
dropping `NoNewPrivileges` from the unit — a real privilege for a command one
mis-transcription away from taking the machine down. Use SSH.

## Asking it what it is

Say **"what are you"**, "who made you", "are you ChatGPT", or "why do I say hey
rhasspy" and you get a fixed, correct answer with no model involved.

This exists because identity is the one category where the model's priors are
guaranteed wrong — Faethon postdates the training data, so the only source of
truth is the prompt. The prompt used to open `You are Rhasspy, a voice
assistant`: four words, one of them a loaded proper noun with a real referent
and no grounding facts. Asked about itself, the model retrieved what it knows
about the Rhasspy project and described *that*, inventing a lineage on the way:

> **Q:** are you related to the Rhasspy project
> **A:** Yes, I am named after the Rhasspy project ... built by the same team.

There's no transcript to check that against, and the answer lands in the memory
buffer and is resent as context — so the model stays consistent with its own
confabulation for the rest of the conversation.

Three things fix it, and two of them are counterintuitive:

**Don't name the other project, even to deny it.** A prompt variant explaining
that "hey rhasspy" is only a trigger phrase measured *worse* — it produced "I
am built on the Rhasspy project's wake-word detection and voice assistant
framework". Naming the entity is what activates it. The prompt now grounds the
identity positively and never mentions it; a test asserts the token stays out.

**Faethon must never speak its own wake phrase.** "You wake me by saying hey
rhasspy" scores **0.9984** on the wake model through the speaker and mic —
above the 0.7 that wakes it and far above the 0.1 barge-in listens at. It would
interrupt itself mid-sentence every time anyone asked. So the answer explains
the wake word without quoting it, and a test enforces that across every reply.

**"What can you do" is deliberately left to the model.** That answer is
dynamic — read off the live tool schemas, so it stays correct as skills are
added. A hardcoded capability list would rot on the next one. Identity is
fixed; capabilities are not.

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
turn: 2.3s audio | stt 0.81s | reply+speech 2.66s | total 3.47s | $0.00004 this turn, $0.00021 session (4 turn(s) held)
```

The two figures are this turn and the process total; the count is how much
history is being resent. Cost per turn rises through a conversation because
every turn resends the whole history — measured at 181 tokens on the first
turn and 529 by the eleventh, roughly 3× — and then plateaus, because the
buffer starts evicting at `history_turns`. "Clear the buffer" puts it back to
the floor immediately.

That line is a good turn from 2026-08-17, not a promise. Both cloud legs vary
a great deal more than the Pi does: measured 2026-08-19, STT ran a median of
1.9s over eight identical calls with one at 19.4s, and the LLM leg ranged
0.63s to 13.38s before `provider_sort` was set. If a turn feels slow, compare
against the `turn:` lines either side of it rather than against this one —
the tail is provider behaviour, not something that regressed locally.

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
- **Memory is 10 turns and RAM-only.** It forgets on restart, after 10 minutes
  of silence, or when you say **"clear the buffer"**.
- **One wake word at a time.**

## Tests

```bash
uv run pytest
```

No network and no audio hardware required — the provider calls are stubbed.
