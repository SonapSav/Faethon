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
V4 Flash at $0.08/M input and $0.16/M output, MAI-Voice-2-Flash at $0.015 per
thousand characters of input text. In practice that is **about $2.24 a month**
of household use — measured over 17 hours of real turns, not guessed.
`faethon-probe` prints the actual cost of each call.

**Speaking is the expensive part** — around 83% of a turn — and it is the one
leg with no usage to read: `/audio/speech` returns raw audio, `/generation`
404s for TTS ids, and `/activity` needs a management key rather than an API
key. So it is estimated from the text sent, via `tts.cost_per_1k_chars`.

That constant was wrong for weeks. It sat at `0.0032`, a plausible-looking
figure from a single small sample, when the real rate is `0.015` — so every
cost Faethon reported understated the bill by **4.7×**, and `faethon-turns`
projected $0.76 a month against a true $2.24. Two credit readings taken either
side of a live session are what exposed it: $0.0144 actually spent against
$0.00432 reported.

The rate is now measured rather than quoted, against `/credits` with the
service stopped: 1124 characters billed $0.01686 and 450 characters billed
$0.00675, both exactly $0.000015 a character. A deliberately pause-heavy sample
spoke at 5.8 characters a second against the other's 8.7 and the per-character
figure did not move, which is what rules out billing on audio duration. End to
end the estimate now matches the bill to the cent.

**If you re-measure this, do not poll `/credits` for stability.** It lands in
batches up to two minutes late and sits perfectly still while the charge is
queued, so "wait until the number stops moving" returns early and reports a
fraction of the bill. That under-counting is what made duration look like the
better model. Drain for two minutes, fire the batch, then wait a fixed three to
four minutes. A test pins the shipped rate so it cannot drift back to a guess.

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

The scale lives in `faethon/levels.py` rather than in this skill, because [the
radio](#the-radio) on the other Pi uses the same one — somebody who has learned
that "volume 5" means half will say "radio volume 5" and mean it. Change it in
one place and both move.

Percentages work as input too, and have to: Faethon announces "70%", so that's
what people say back. "Set the volume to 30%" and "volume 3" mean the same
thing, as do "to 7" and "to 70" — a number is read as a percentage when it
carries a unit or is simply too big to be a level. Any positive percentage
gives at least level 1, since rounding 3% down to silence would be a different
thing from what was asked.

Each change is spoken back as a percentage of the dial — one level is one
tenth, so level 7 announces **"Volume is set to 70%."** and level 0 says
**"Volume is set to 0%, muted."**, since 0% alone leaves it unclear whether the
speaker is silent or merely turned right down. The literal `%` is safe:
MAI-Voice-2-Flash pronounces it, verified by round-tripping the clip back
through Whisper — `"Volume is set to 70%."` and `"Volume is set to 70
percent."` render to identical 2.29s of audio and transcribe identically. Worth
re-checking against any TTS model you switch to, since a model that silently
dropped the sign would announce `"Volume is set to 70"`.

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

## The radio

RadioHost runs on a second Pi at `radiohost.local:8000` — a plain HTTP/JSON API
on the LAN with no auth, the same trust model as the phone app that normally
drives it.

```
"Hey Rhasspy, play 94.9"        Playing 94.9.
"Βάλε το εκατόν δύο κόμμα δύο"  Playing 102.2.
"what's playing"                96.3 is playing Europe - The Final Countdown.
"next station"                  Playing 89.8.
"turn the radio up"             Radio volume is set to 50%.
"radio volume 5"                Radio volume is set to 50%.
"stop the radio"                Radio off.
"what stations do you have"     I have 19 stations: 88, 89.2, 89.8, 92, ...
```

The integration is four endpoints and a 94ms round trip. All the design sits in
one question: how does a spoken request become a station id.

**By frequency, because names do not survive.** With `stt.language` pinned to
`"en"`, Whisper renders the Greek station names as an unpredictable mix of
transliteration and translation:

| station | heard | |
|---|---|---|
| Ρυθμός | "rithmos" | transliterated |
| Μέντα | "Menta" | fine |
| Λάμψη | "lampsy" | mangled |
| **Δρόμος** | **"the road"** | **translated** |
| **Μελωδία** | **"melody"** | **translated** |

No string comparison recovers "the road" into Δρόμος, and no single rule covers
a set that is sometimes one and sometimes the other. Frequencies have none of
that problem — all nineteen stations carry one, they are unique, and digits are
language-neutral. "Play ninety four point nine" arrives as `94.9`; "Βάλε το
ενενήντα τέσσερα κόμμα εννιά" arrives as `94,9`. Both parse, and it is how
people refer to radio anyway.

An FM-band check is the only thing separating a frequency from every other
number in a sentence, so `2026` and "play 8" are refused. A bare "94 9" parses
as 94 rather than guessing at a decimal that would select a different station.

Names still reach the model as a fallback, which covers the Latin-script twelve
and can resolve "the road" back to Δρόμος with priors no matcher has. Same
split as the weather skill: the regex path never captures a name.

**Volume is the same 0–10 scale as the speaker**, shared through
`faethon/levels.py` rather than written twice, so "radio volume 5" means half
in both places. It inherits the trap too — announcing "50%" teaches people to
say "50" back, which read as a level would clamp to maximum.

Nudging works in levels, so "turn the radio up" from 45 lands on 60 rather than
55: after one nudge the radio sits on the scale Faethon speaks in instead of
between two of its steps.

Asked *during* a conversation, the change is deferred until the conversation
closes — see [ducking](#ducking) for why. Faethon confirms the new level
straight away; the radio arrives at it a moment later.

### Ducking

The radio drops to 15% while Faethon holds the floor, and goes back afterwards.
From the **wake word**, not from the reply — you are talking over the music
too, and a radio that dipped only while Faethon spoke would lurch up and down
through a conversation. Unprompted announcements duck as well, since a dust
warning is the one nobody is braced for.

**Not for the microphone.** That was the obvious worry — a radio in the same
room feeding the mic continuously and re-opening the follow-up window on every
turn, which is what the conversation cap exists to survive. Measured instead of
assumed: 45s of room audio scored a maximum of **0.0026** against a wake
threshold of 0.7, with the radio arriving at about **−88 dBFS** where speech
arrives at −20 to −30. The null result was checked by sweeping the radio's
volume (39/60/80 → −87.9/−78.1/−69.0 dBFS) to prove the measurement could see
it at all, so it means "far too quiet to matter" rather than "nothing was
playing". Move the speaker and that wants re-measuring; the same 45s recording
answers it.

Ducking exists for the **person in the room**, who has to hear the reply over
the music. A different question, which the acoustic measurement never asked.

Four things it has to get right, each of which was a bug before it was a rule:

- **A volume command must not cancel the duck.** Asking for "radio volume 5"
  mid-conversation used to move the radio immediately, so the confirmation
  played over it at full volume — ducking defeated by the one command that is
  actually about volume. The new level is now *deferred*: the radio stays down
  through the reply and lands on it when the conversation closes.
- **A nudge must step from your level, not the ducked one.** The live reading
  during a turn is 15, not the 60 you left it at, so "turn the radio up" read
  level 2 instead of level 6 and landed on 30. The setting was simply lost, and
  nothing about it looks wrong until you notice the radio is quiet. Anything
  that changes volume during a turn now goes through the duck.
- **A duck that never landed is not "restored".** The state is written *before*
  the request, because a POST that times out may still have arrived — that is
  exactly how the radio ended up stuck at 15 with nothing recorded to undo it.
  Writing first makes the restore self-correcting: if the volume is not what we
  think we set, the change never happened and the restore stands down. The same
  comparison covers somebody changing it on their phone mid-conversation, which
  is why one check does both jobs.
- **It never blocks a turn.** The duck runs in a thread. Measured blocking at a
  worst case of **4094ms** before that — four seconds of silence ahead of the
  "go ahead" chime, with somebody standing there waiting to speak. It is 0.7ms
  now.

A crash mid-turn is repaired at startup, or the radio would sit at 15 until
somebody reached for their phone — and nobody would connect a quiet radio to an
assistant that died an hour ago. Everything in the path swallows exceptions,
deliberately broadly: ducking is a courtesy, and a courtesy that can break a
turn is worse than no courtesy. Switch it off with `radio.duck`.

### The rest

`"what stations do you have"` is fetched **fresh every time**, never cached —
the list changes when stations are added on the other Pi, and finding out
whether it did is the whole reason for asking. It reads frequencies rather than
names, for the same two reasons selection does. It takes 29.6s to say all
nineteen, so the count comes first and barge-in cuts off the rest; past twenty
it summarises instead of reciting.

Every pattern needs an explicit radio marker, because this skill sorts *before*
`set_volume` and `set_timer` in the registry and a loose one would silently
take "turn it up" and "stop" from the skills that own them. Twelve phrases are
pinned in both directions.

An unreachable Pi is remembered for a minute rather than retried per turn, since
the timeout would land inside a conversation. And an unfetchable station list
reports as unreachable rather than "I don't have a station on 94.9" — absence
of the list is not absence of the station, and the wrong answer sends you
looking for something that is really there.

Radio calls appear in the disclosure ledger under a `lan` kind. They left this
machine, which is why they are counted, but they did not leave the house.

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

## When it may speak unprompted

Nine things can talk without being asked — timers, thermal and under-voltage
warnings, dust and UV crossings, and the status clips for a lost microphone, a
lost network and a low balance. Each decides on its own to say its piece once,
which reads fine one at a time and worse with every source added. So there's
one place that sees them all:

```yaml
announcements:
  quiet_start: "22:30"
  quiet_end: "07:30"
  min_gap_seconds: 30
```

The split that makes it simple isn't by message, it's by **whether you asked
for it**. A timer firing is *requested* — you set it, and an eight-hour timer
set in the evening should go off at three in the morning, because that's the
point of setting it. An under-voltage warning is *informational*: nobody asked,
the flag is latched, and it can wait until morning. Skills declare which they
are with `announce_urgency`.

A dust warning is *informational*, so the quiet hours hold it overnight. It is
also **say-once and persistent**: the flag survives a restart, or a voice
restart during a dust storm would re-announce the same dust every time. It
re-arms when the reading drops back below the threshold, and is forgotten after
a gap in observation long enough that the episode could have ended and a new
one begun unseen (`air.stale_after_hours`, default 6). A reboot stays silent; a
night with the Pi switched off does not.

**A held announcement is deferred, not dropped.** The budget is checked *before*
a skill is ticked, so a skill that's never ticked keeps its say-once state and
offers the same thing again once quiet hours end. That ordering is what makes
this need no queue and no re-delivery.

Anything said *during* a turn is exempt and never reaches the budget. If
transcription fails and Faethon says so, you're standing there waiting — that's
an answer, not an announcement.

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
`scripts/make_speech.py` and re-run that; `greet_on_start: false` turns it
off.

**Keep the assistant's name out of the greeting.** The current wording scores
**0.0001** on the wake model through the speaker and mic. An earlier draft
opened "Hi, I am Rhasspy" and scored **0.5036** — still under the 0.7 wake
threshold, but five thousand times higher and well above the 0.1 that barge-in
listens at. Saying the name is most of saying the wake word.

The greeting also plays before the microphone is opened, which costs nothing
and means a future rewording can't wake Faethon up at every start. A test pins
the margin either way.

## How it's doing

```
"Hey Rhasspy, what's your status"   I'm at 48 degrees, which is normal.
                                    I'm on wi-fi, connected to HomeNet-5G.
"How hot are you?"                  I'm at 48 degrees, which is normal.
"What's your IP?"                   You can reach me at faethon dot local,
                                    or at 1 9 2, 1 6 8, 0, 61.
```

One skill rather than five. `Registry.match` is first-match-wins in import
order, so separate temperature, IP, SSID and status skills would compete over
"how are you doing" and "what's your status", and which one won would depend on
filenames.

The temperature carries its meaning — "48 degrees, which is normal" — because a
bare number alarms anyone who doesn't know Pi thresholds and reassures anyone
who does. Same move `volume_skill` makes with dB.

The address leads with the **hostname**, because it's stable and the IP isn't,
and because it's the only one you can write down by ear. The IP form was
auditioned through the speaker: the raw string ran 5.15s, digits grouped by
octet 4.27s, and only the second is transcribable. Three-digit octets are
spelled out since "one hundred" is ambiguous where "1 0 0" isn't.

**The half that earns its keep is `tick()`.** A CPU temperature you have to ask
for is nearly useless — you'd ask a week after the wake word started failing.
Throttling degrades Faethon specifically: openWakeWord runs continuously, a Pi
4 soft-throttles at 80°C, and the symptom is missed or late wake words. So it
says so, once:

> "I've had under-voltage warnings, which usually means the power supply or the
> cable. It can make me unreliable."

Under-voltage is the one worth waiting for. A marginal supply or mediocre cable
causes intermittent trouble that looks exactly like a software bug, and
`vcgencmd` latches the flag since boot — so it can be reported long after the
moment that caused it, which is the only way anyone would ever catch it. Power
is reported before heat when both are set, since the supply is the cause.

Checked once a minute, not once per audio frame: `vcgencmd` forks a process.

## Weather

```
"Hey Rhasspy, what's the weather"       It's 35 and clear in Abu Dhabi, feels like 42, with a high of 43.
"Will it rain tomorrow?"                No rain expected in Abu Dhabi tomorrow.
"Do I need an umbrella?"                No rain expected in Abu Dhabi today.
"What's the weather in Paris?"          It's 24 and overcast in Paris, with a high of 25.
```

Open-Meteo — no key, no account. A keyless skill can't be half-configured, and
works on a fresh clone.

**The regex path never captures a place name**, and that's the whole design.
It's `credit_skill`'s lesson at a larger order of magnitude: "OpenRouter" is one
coined word with a handful of spellings and it still needed a loose pattern. A
place name is an open set of proper nouns, many foreign, and Whisper hands back
"Redding" for Reading — both of which the geocoder resolves happily, one in
England and one in California. That failure isn't an error, it's the forecast
for the wrong continent said with total confidence.

So the daily phrasings go through the regex path with no location captured at
all, and named places go through the model, which has far better priors for
turning a mangled transcript into a real place. An unknown place is refused
rather than approximated.

**It answers the question that was asked.** "Will it rain?" replied to with a
temperature range is worse than no reply — it sounds like an answer and isn't.
So a rain question gets a rain answer, an umbrella is a rain question, and a
jacket gets the numbers rather than a verdict, since what warrants a coat is a
matter of opinion and of who is asking.

Two or three facts, never more: TTS bills per character, and the follow-up
window means "and tomorrow?" is one sentence away. How it *feels* is mentioned
only when that differs by four degrees or more — here in August the gap is
routinely seven. Forecasts are cached for ten minutes, because a conversation
about the weather otherwise fetches the same hourly data three times.

## Air quality, dust and UV

```
"Hey Rhasspy, what's the air quality"   The air is very poor in Abu Dhabi, mostly dust.
"Is it dusty?"                          Dust is 310 micrograms in Abu Dhabi.
"What's the UV index?"                  The UV index is 9 in Abu Dhabi, which is very high.
"Do I need sunscreen?"                  (same — routed by capture group)
"Will it be dusty tomorrow?"            Dust should peak around 307 micrograms tomorrow in Abu Dhabi.
"Is the dust going to clear?"           It should ease off. Dust is around 321 micrograms today,
                                        down to about 132 micrograms by Monday.
```

Open-Meteo again, keyless, at the same coordinates as the weather so the two
cannot drift apart.

**Dust is why this exists.** It is the weather that changes your day here and is
completely invisible to a temperature forecast — on the afternoon this was
built the weather skill said *"It's 45 and clear in Abu Dhabi"* while dust sat
at 310 µg/m³ and the European index read very poor. Both sentences were true;
only one was useful.

**One skill, not three.** Air, dust and UV share an endpoint, a cache, a
location and a config section, and *"is it safe to go outside"* belongs to all
of them. Splitting them would duplicate that to gain nothing and put three
skills in competition for the same phrases — which `Registry.match` settles by
import order rather than by fit. That ordering matters more than usual here:
discovery walks the package alphabetically, so `air_skill` matches ahead of
every other skill. Its patterns are `_END`-anchored for that reason, and a test
asserts the weather, health, time and credit phrases still reach their owners.

**Readings carry their meaning**, the same move `volume_skill` makes with dB and
`health_skill` with thermals — "very poor, mostly dust" rather than "PM10 is
222". Dust is named as the cause only when it is most of the PM10, so the
attribution is earned. A number nobody can place is not an answer.

**The forecast half is lopsided, because the API is.** UV has a real daily
aggregate (`uv_index_max`) that the model produces itself, used as-is. Dust and
the air index are hourly only — there is no `european_aqi_max`, and asking for
one is a 400 — so their daily figures are aggregated here from 24 hourly
values, taking the *peak* rather than the mean: the question behind "will it be
dusty tomorrow" is whether it gets bad at any point, not what it averaged while
you were asleep.

**Three days, though the API serves seven.** The decay measured while writing
this — 321, 307, 219, 132, 112 — is real signal through about day three and
thins after. The tool description says so explicitly, so the model declines
rather than inventing a Thursday.

Days are named from the API's own date strings rather than the local clock. The
Pi has no RTC, and a weekday is exactly what it would get wrong in the first
seconds after a cold boot.

Current, hourly and daily ride in **one 3.8 kB request**, so the forecast costs
no extra round trip and the 15-minute cache serves all of it.

## The turn log

One line per turn in `/var/lib/faethon/turns.jsonl`, so the next threshold can
be measured rather than guessed.

```bash
uv run faethon-turns --days 7
```

```
142 turns over 71.3 hours

where they went
  llm                           83    58%
  regex:get_time                21    15%
  ...

latency, seconds
  stt      median   1.87   p90   4.10   max  19.41
  ...
```

Every number in this project that turned out right was measured — the wake
threshold by scoring a real voice, the idle window by plotting 71 real gaps,
the barge-in threshold by a mixing experiment. Each of those datasets had to be
manufactured by hand for the occasion. The ones that *weren't* measured show
it: the conversation cap is 20 turns and 5 minutes because three conversations
happened to be in the journal when it was written.

**Metadata only — routes, latencies, costs, text lengths, never transcripts.**
journald already keeps the words, and whether it should is a live decision;
recording them a second time here would answer it by accident.

Each row records the `tts_rate` it was costed at, and the report **recosts rows
written before that rate was measured** — exactly, from their `said_chars`,
rather than flagging them. Without that, 43 rows priced at the old $0.0032
dragged the projection to $0.76 a month against a true $2.24, which is precisely
the number someone uses to judge how long their balance lasts. It says how many
rows it touched rather than quietly presenting a corrected total as if it had
always been right. A row carrying its own rate is left alone even when that
differs from today's — a genuine price change is history, not an error.

It rotates one generation back at `max_mb`, because an SD card is the part of a
Pi that wears out. And it never raises: a log that can break a turn is worse
than no log.

State lives in `/var/lib/faethon`, and both the service and a foreground `uv
run faethon` use it. systemd sets `STATE_DIRECTORY` when it starts the service;
nothing sets it otherwise, so anything run by hand used to keep its own state
inside the checkout — meaning a timer set in the foreground was invisible to
the service, and reading the log meant setting the variable by hand.

## What it sent to the cloud

```
"Hey Rhasspy, what did you send to the cloud today?"

  1 request went out today, 1 with nobody asking.
  Your location went to the weather service once.
```

```bash
uv run faethon-sent --days 1
```

A cloud assistant is a microphone in a private house that talks to companies.
The honest thing is to be able to say what left, so this counts every outbound
request.

**Counted at the HTTP layer, because the turn log would understate it by about
half.** A turn is a completed exchange; a single exchange is three or four
requests — speech in, a completion, then two chunks of speech out. One measured
day had **127 transcripts against 74 logged turns**. Counting turns and calling
it "what you sent" measures the conversation, not the disclosure. So the
counter lives in `OpenRouterClient` and in the two skills that reach the
network without it, which is the only place that cannot disagree with reality.

**Per attempt, not per success.** A retried upload crossed the wire twice, and
a request that timed out still sent everything it was carrying. A ledger of
what arrived safely would understate what left.

**Records carry what was disclosed, not just where it went.** OpenRouter
receives the sound of the room; Open-Meteo receives this house's coordinates to
about ten metres. Both are "a request to a server" and they are not the same
thing to hand over — "thirty calls to a weather API" sounds like nothing until
it is said as *your home location, thirty times*.

| kind | what it means |
|---|---|
| `voice` | audio recorded in this room |
| `text` | words you said, or words it said back |
| `location` | where this house is |
| `account` | billing metadata only |

An unlisted endpoint counts as `text`. Overstating a disclosure beats silently
leaving a new endpoint out of the ledger.

**Two counters exist for the things nobody anticipates.**

Background ticks are flagged **unasked** — the dust check phones out every half
hour whether or not anyone is home, and that is the category people never think
of: the assistant talking to the internet about an empty house.

And a microphone that opened and sent nothing is recorded as **withheld**,
because it leaves no other trace. A follow-up window that hears no speech makes
no `/audio/transcriptions` call at all, so a ledger of requests could only ever
show what went out and never what was declined. It is the most reassuring true
fact available and it was invisible.

**No payloads and no transcripts** — not the audio, not the text, not the
reply. Host, path, kind, and whether a person asked. A ledger able to recite
your conversations back would have to be keeping your conversations, which
defeats the thing it exists to reassure you about. That is also why it does not
read journald, which *does* hold every transcript: answering a privacy question
with the most privacy-invasive source on the machine is the wrong trade.

The spoken answer leads with audio and consent rather than scale, because the
real question behind it is whether the thing in the corner is listening when
nobody asked it to. On one measured day **38 of 74 turns** were follow-up
windows — the microphone live, audio leaving, without a wake word. That is the
5-second window working exactly as designed, and it is still the fact most
worth surfacing.

## Timers

```
"Hey Rhasspy, set a timer for ten minutes"
"Hey Rhasspy, set a pasta timer for eight minutes"
"How long left on the pasta timer?"
"Cancel the pasta timer"
```

Several at once, each optionally named. When one comes due, a C major scale
falls a full octave — eight notes, the last held — and Faethon says which timer
it was. The first thing it does without being asked.

`done.wav` also falls, so length is what separates them: two notes over 0.19s
against eight over 0.94s. The discriminator is the run, not the direction.

**Relative only — "in ten minutes", never "at seven."** That's the clock, not
laziness. The Pi has no battery-backed RTC (`RTC time: n/a`), so the wall clock
is wrong for the first couple of minutes after a cold boot and then *steps*
when NTP corrects it. Measured here: stepping the clock 60 seconds moved
`time.time()` by 62.3s and `time.monotonic()` by 2.3s — the real elapsed. So a
running timer counts on the monotonic clock, which cannot jump.

**They survive a restart**, which needs the other clock, because monotonic
resets on reboot and means nothing across one. Each timer carries a wall-clock
deadline on disk and a monotonic deadline in memory, each used for what it's
good at. Verified: a 3-minute timer set, the service restarted, and it came
back reading 2:38 — it counted the time it was away.

Restoring waits for `/run/systemd/timesync/synchronized`, so a timer is never
restored against a clock that hasn't been corrected yet. Three outcomes by how
stale the deadline is:

| deadline | what happens |
|---|---|
| still ahead | resumes normally |
| just missed (under 5 min) | fires at once, saying it's late |
| long past — Pi was off overnight | dropped and logged, not announced |

State lives in `/var/lib/faethon`, declared as `StateDirectory=faethon` in the
unit: systemd creates it, owns it to the service user, and adds it to the
sandbox's writable paths, so neither `ProtectHome` nor `ProtectSystem` has to
be relaxed and nothing is written inside the checkout. Writes are atomic —
a crash mid-write would otherwise leave JSON that fails to parse, turning a
lost timer into a permanently broken one.

An announcement drains the microphone afterwards, the same way a spoken reply
does. Faethon talks over a live mic either way, and without the drain the wake
detector spends the next two seconds chewing its own voice instead of hearing
the room — measured at **2.04s of backlog** after a timer fires, which is
exactly when someone says "cancel" or "set another". With the drain it reads at
0.96× real time, i.e. live.

Timers are also the reason skills have a `tick()` hook: called while idle and
between turns, returning a sentence to speak or nothing. Returning text rather
than speaking keeps skills free of any audio dependency, exactly as `run()`
does. Worst case a timer is one turn late, since the wake loop doesn't run
during a conversation.

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

## Knowing what day it is

The model has no idea. Ungrounded it doesn't say so — it retrieves a date from
training data and answers from that:

> **Q:** what is the date today
> **A:** Today is Friday, March 14, 2025.

Nineteen months out, and "how long until Christmas" was computed off it: 318
days, when the answer was 127. Same shape as the identity confabulation.

So every request carries the current date and time in the system prompt, built
per request rather than at startup — a service up for a week would otherwise
still be insisting it's Monday. About twenty tokens, fractions of a cent a
month.

**It is omitted entirely until the clock can be trusted.** This Pi has no
battery-backed RTC, so for the first couple of minutes after a cold boot the
wall clock is whatever was restored from disk. A wrong date is worse than no
date, because it's exactly as confident as a right one.

Two things fell out of adding it, both worth knowing before you write a skill:

**A tool result ends the turn.** Skill output is yielded straight to speech and
never fed back to the model for a second pass. That's right for skills that
*do* something — "Volume is set to 60%" needs no rephrasing — but it means a
model reaching for a tool as an intermediate *step* produces the tool's output
instead of an answer. Asked "how long until Christmas", it called `get_time`
and the reply was "It's Thursday, August 20."

`get_time` is therefore `regex_only` now: with the date already in the prompt
the model has no need of it, and the regex path still answers instantly and for
nothing. The timer skill hit the same trap — "how long" reads as a timer query
— and its description now says explicitly that it's only for Faethon's own
timers.

**Reasoning is off, so the model thinks in the visible reply.** Its first
correct answer was four hundred characters of month-by-month arithmetic, read
aloud. The grounding line ends with "give only the answer, never the
arithmetic", which brought it to eighty-two.

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
| `src/faethon/levels.py` | the 0-10 volume scale, shared by speaker and radio |
| `src/faethon/turnlog.py` | one metadata line per turn |
| `src/faethon/disclosure.py` | one line per outbound request |
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
clock before this was understood.

That rules models out, but it no longer decides between them. Every current
candidate generates several times faster than the speech plays, so synthesis
never becomes the bottleneck and total generation time is a red herring. What
you actually wait through is **time to first audio**, because Faethon streams
and starts speaking on the first chunk.

Re-benchmarked 2026-08-21 on a real announcement, six calls each:

| model | to first audio | worst of 6 | rate | voices | per 1M chars |
|---|---|---|---|---|---|
| **microsoft/mai-voice-2-flash** | **0.35s** | 0.88s | 24000 | Azure catalogue | $15 |
| fish-audio/s1 | 0.43s | **0.53s** | 44100 | 1 | $15 |
| x-ai/grok-voice-tts-1.0 | 0.50s | 1.38s | 24000 | 5 | $15 |
| fish-audio/s2.1-pro-free | 0.50s | 0.86s | 44100 | n/a | free |
| deepgram/flux-tts:free | 0.51s | 1.62s | 24000 | 36 | free |
| qwen/qwen-audio-3.0-tts-flash | — | — | — | undiscoverable | $15 |
| sesame/csm-1b | 5.71s | — | 24000 | several | $7 |
| hexgrad/kokoro-82m | hangs | — | 24000 | 54 | $0.62 |

**MAI-Voice-2-Flash is the current choice**, in `en-US-AndrewNeural`. It is
quicker than Fish S1 in the middle and looser in the tail, so this is not a
clean latency win — it is a win on voice choice at no latency cost. S1 remains
the most consistent model measured, and switching back is two lines of config.

It also returned byte-identical output across all six calls, which nothing else
did. Deterministic synthesis means no speaker drift between renders.

**Ignore the first call after an idle gap.** It runs slow on every model here —
1.55s once on MAI, settling to 0.35s by the third call. Benchmark a model warm
or you will reject a good one. The MAI figures above are measured through
`synthesize_stream`, the path Faethon actually speaks through; the rest are
bare HTTP from the same session. Where both were run they agreed closely, 0.32s
against 0.35s median, so the ranking holds — but they are not all from the same
harness.

**Voice pace matters more than the numbers above.** The same announcement runs
6.0s in `en-US-AndrewNeural` against 8.1s in `en-AU-NatashaNeural` — 35% off
the wait on every reply, dwarfing the 0.1–0.2s differences in first-audio time.
It costs nothing either way, since billing is per character rather than per
second.

**Qwen is unusable**, not slow. It rejects a request with no voice, refuses to
enumerate its voices, and twelve documented DashScope names were all rejected.
A model whose voices you cannot name cannot be shipped.

**The free tier is a real lever if cost bites.** Speech is ~83% of spend, so a
free model takes the bill from ~$2.24/month to ~$0.38. The catches: Deepgram
Flux has the worst tail of anything measured (1.62s), and Fish S2.1 Pro Free
takes no voice parameter and carries no stated availability guarantee.

Three gotchas the code handles for you:

- **Sample rates differ.** Fish Audio returns 44.1kHz, most others 24kHz.
  Playing one at the other's rate isn't subtle — 44.1k played as 24k runs at
  0.54× speed. The rate is read from the response `Content-Type`;
  `tts.sample_rate` in the config is only a fallback.
- **An empty voice is not a default.** Several providers, MAI among them,
  reject a request with no voice outright. Those that accept one pick a
  different speaker per request: measured over five renders of one sentence on
  Fish S1, pitch ranged 99–156 Hz unset against 101–116 Hz with `alloy` — a
  different person each time you ask Faethon anything.
- **To see a provider's voices, send a bogus one.** The 400 sometimes
  enumerates them — Deepgram does, Fish and MAI do not. MAI takes Azure neural
  names; `en-US-AndrewNeural`, `en-US-AvaNeural`, `en-US-EmmaNeural`,
  `en-US-BrianNeural`, `en-GB-SoniaNeural`, `en-GB-RyanNeural` and
  `en-AU-NatashaNeural` are all confirmed working.

**Changing the voice means re-rendering `assets/*.wav`**, or Faethon greets you
in one voice and apologises for a network outage in another. Run
`uv run python scripts/make_speech.py`, then re-score every clip against the
wake model — a clip that crosses the threshold would make Faethon wake itself,
and the failure clips play precisely when it cannot reach the network to
recover. The current set peaks at 0.0004 against a threshold of 0.7.

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
- **A conversation is capped at 20 turns or 5 minutes** without a fresh wake
  word, after which the falling chime plays and it returns to wake-word
  detection. Deliberately loose: it bounds a runaway — a television in the same
  room re-opening the follow-up window on every turn — rather than trimming
  ordinary use, whose longest recorded conversation was five follow-ups.
- **Memory is 10 turns and RAM-only.** It forgets on restart, after 10 minutes
  of silence, or when you say **"clear the buffer"**.
- **One wake word at a time.**

## Tests

```bash
uv run pytest
```

No network and no audio hardware required — the provider calls are stubbed.

**And no writing to the real state directory.** An autouse fixture in
`tests/conftest.py` points `state.state_dir()` at a temporary path for every
test. That is not tidiness. The disclosure ledger writes on each outbound
request, the fake clients here make requests to a path called `x`, and
`state_dir()` resolves to `/var/lib/faethon` whenever it exists — so a full
test run silently appended forty rows to the live privacy ledger, the one file
whose entire value is being an accurate record of what actually happened. The
timers and the turn log had the same exposure and nobody had noticed, because
their rows look plausible. The ledger's did not.
