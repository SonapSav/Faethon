#!/usr/bin/env bash
# Audition TTS voices through Roxy's actual speaker.
#
#   ./scripts/try-voices.sh            play the samples in /tmp/roxy-voices
#   ./scripts/try-voices.sh regen      re-synthesise them first
#
# Pick one, then set it in config.yaml:
#     tts:
#       voice: "aura-2-orion-en"
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
UV="${UV:-$HOME/.local/bin/uv}"
DIR=/tmp/roxy-voices

DEV=$($UV run python -c "from roxy.config import load_config; print(load_config().audio.output_device)")

if [ "${1:-}" = "regen" ] || [ ! -d "$DIR" ]; then
  echo "Synthesising samples..."
  $UV run python scripts/make_voice_samples.py || exit 1
fi

shopt -s nullglob
files=("$DIR"/*.wav)
if [ ${#files[@]} -eq 0 ]; then
  echo "No samples in $DIR -- run: $0 regen" >&2
  exit 1
fi

echo "Playing ${#files[@]} voices through $DEV"
echo
for f in "${files[@]}"; do
  name=$(basename "$f" .wav)
  printf "  %-24s " "$name"
  aplay -D "$DEV" -q "$f" && echo "ok" || echo "PLAYBACK FAILED"
  sleep 0.4
done

echo
echo "Set your pick in config.yaml under tts.voice"
