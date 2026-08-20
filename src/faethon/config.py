"""Configuration: config.yaml for settings, .env for secrets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ASSETS_DIR = PROJECT_ROOT / "assets"


class AudioConfig(BaseModel):
    input_device: str
    output_device: str
    sample_rate: int = 16000
    frame_ms: int = 80

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2  # s16le


class WakeConfig(BaseModel):
    model: str = "hey_jarvis_v0.1"
    threshold: float = 0.5
    refractory_sec: float = 2.0


class UtteranceConfig(BaseModel):
    vad_aggressiveness: int = Field(2, ge=0, le=3)
    silence_ms: int = 600
    min_ms: int = 400
    max_ms: int = 15000
    #: How long to wait for the user to start speaking at all. Separate from
    #: silence_ms, which only applies once they have started.
    start_timeout_ms: int = 5000
    #: Sustained voice needed before "they started talking" is believed. Stops
    #: a single noise blip from starting the end-of-speech clock.
    speech_onset_ms: int = 120


class ConversationConfig(BaseModel):
    """Whether a reply leaves the mic open for a follow-up.

    Without this every sentence needs the wake word again, which makes a
    back-and-forth exhausting: "hey rhasspy, what's the weather" ... "hey
    rhasspy, and tomorrow?".
    """

    #: Listen again after each reply, so only the first turn needs the wake word.
    follow_up: bool = True
    #: How long to wait for that follow-up before closing the conversation.
    #: Deliberately not utterance.start_timeout_ms: that budget is spent by
    #: someone who has just deliberately said the wake word and is expected to
    #: speak, while this one is usually spent on silence, and every millisecond
    #: of it is a millisecond the mic stays live after Faethon stops talking.
    follow_up_ms: int = 5000
    #: Let the wake word cut a reply off mid-sentence. Costs one wake-model
    #: inference per 80ms frame while Faethon is speaking (~16% of one core),
    #: and nothing at all when it is quiet.
    barge_in: bool = True
    #: A conversation cannot run past these without a fresh wake word. Not a
    #: latency tuning: the follow-up window is the only period when the
    #: microphone is live without anyone having deliberately triggered it, and
    #: anything that keeps producing transcribable speech -- a television in
    #: the same room -- re-opens it on every turn. Left unbounded, the
    #: exception becomes the steady state.
    #:
    #: Deliberately loose. This is a bound on runaway, not a trim on normal
    #: use: the longest real conversation recorded here was five follow-ups,
    #: so twenty turns and five minutes cannot touch one. 0 disables either.
    max_turns: int = Field(20, ge=0)
    max_seconds: float = Field(300.0, ge=0.0)
    #: Wake score needed to interrupt, which has to be far lower than the
    #: quiet-room threshold: Faethon's own voice is masking yours. Measured
    #: through the speaker and mic at equal loudness, the wake word scored
    #: 0.368 while Faethon talking over it scored 0.0002.
    barge_in_threshold: float = Field(0.1, gt=0.0, le=1.0)
    #: Say hello once when the service starts, so a restart is audible.
    greet_on_start: bool = True


class WeatherConfig(BaseModel):
    """Where "the weather" means, when nobody says a place.

    Machine-specific, so it lives here. Find your coordinates by right-clicking
    on any map; four decimal places is far more precision than a forecast has.
    """

    latitude: float = Field(25.2048, ge=-90.0, le=90.0)
    longitude: float = Field(55.2708, ge=-180.0, le=180.0)
    #: What Faethon calls it out loud.
    place_name: str = "Dubai"
    #: "metric" for Celsius, "imperial" for Fahrenheit.
    units: Literal["metric", "imperial"] = "metric"


class TurnLogConfig(BaseModel):
    """A line per turn, so future thresholds can be measured.

    Metadata only -- routes, latencies, costs, text lengths. Never transcripts:
    journald already keeps those and whether it should is a live decision, so
    recording them again here would answer it by accident.
    """

    enabled: bool = True
    #: Rotate one generation back at this size. An SD card is the part of a Pi
    #: that wears out, and roughly 200 bytes a turn means years before this
    #: matters -- but unbounded is unbounded.
    max_mb: float = Field(5.0, ge=0.0)


class CreditConfig(BaseModel):
    """Warning before the account runs dry.

    At zero every leg stops -- no transcription, no thinking, no speech -- so
    the failure is total silence, and the only thing still able to report it is
    a pre-rendered clip.
    """

    #: USD. 0 disables the warning.
    #:
    #: COUPLED TO THE AUDIO: the clip says "below half a dollar", so changing
    #: this means re-rendering it. scripts/make_speech.py holds the wording and
    #: a test pins the two together.
    warn_below: float = Field(0.50, ge=0.0)
    #: How often to look, at most. Checked after a turn rather than in the wake
    #: loop: a /credits call takes 250-450ms, which in that loop is three or
    #: four frames of deafness and a real chance of missing a wake word.
    check_every_minutes: float = Field(60.0, gt=0.0)


class ModelsConfig(BaseModel):
    stt: str
    llm: str
    tts: str


class STTConfig(BaseModel):
    #: ISO-639-1 code. Empty means let Whisper auto-detect, which is what
    #: produces the occasional English-transcribed-as-Greek surprise.
    language: str = "en"
    #: 0 makes decoding deterministic; higher values let it wander.
    temperature: float = 0.0


class LLMConfig(BaseModel):
    max_tokens: int = 100
    temperature: float = 0.7
    history_turns: int = 10
    #: Minutes of silence after which the conversation is forgotten. 0 keeps it
    #: until a restart or "clear the buffer".
    history_idle_minutes: float = Field(10.0, ge=0.0)
    system_prompt: str
    #: Let the model think before answering. Off by default: reasoning tokens
    #: come out of max_tokens, and on a spoken-reply budget they can consume
    #: the lot and leave nothing to say. Only turn on with a large max_tokens.
    reasoning: bool = False
    #: How OpenRouter should choose between the providers serving the model.
    #: Empty leaves its default, which is cheapest-first and erratic. A typo
    #: here would be forwarded and silently ignored, hence the closed set.
    provider_sort: Literal["", "price", "latency", "throughput"] = ""


class TTSConfig(BaseModel):
    #: Empty is not the same as a default: a provider that accepts no voice may
    #: pick a different speaker per request. Faethon warns at startup if unset.
    voice: str = ""
    #: Fallback only -- the real rate comes from the response Content-Type.
    sample_rate: int = 24000
    #: USD per 1000 characters of input text, used to estimate what speaking
    #: cost. The speech endpoint returns raw audio with no usage body, so
    #: nothing else can account for it -- and it is the largest part of a turn.
    #: 0 disables the estimate rather than reporting a wrong one.
    cost_per_1k_chars: float = Field(0.0032, ge=0.0)


class Settings(BaseSettings):
    """Secrets from the environment / .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: SecretStr, not str: pydantic prints the whole Config in an assertion
    #: diff or a validation error, and a plain string puts the key in every
    #: failing test's output and every traceback anyone pastes anywhere.
    #: Read it with .get_secret_value().
    openrouter_api_key: SecretStr = SecretStr("")


class Config(BaseModel):
    audio: AudioConfig
    wake: WakeConfig
    utterance: UtteranceConfig
    conversation: ConversationConfig = ConversationConfig()
    credit: CreditConfig = CreditConfig()
    turn_log: TurnLogConfig = TurnLogConfig()
    weather: WeatherConfig = WeatherConfig()
    models: ModelsConfig
    stt: STTConfig = STTConfig()
    llm: LLMConfig
    tts: TTSConfig
    settings: Settings


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"No config file at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return Config(**raw, settings=Settings())
