#!/usr/bin/env bash
# Record a wake-word sample with a live level meter, then report the level.
#
#   ./scripts/mic-test.sh          record 10s
#   ./scripts/mic-test.sh 5        record 5s
#
# Run this yourself rather than having an assistant run it: the countdown has
# to appear in your terminal, in real time, for you to speak on cue.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SECS="${1:-10}"
DEV=$(grep -oP 'input_device:\s*"\K[^"]+' config.yaml)
OUT=/tmp/mic-test.wav

echo "device: $DEV"
echo
for i in 3 2 1; do printf "\r  starting in %d... " "$i"; sleep 1; done
printf "\r                        \r"
echo ">>> SPEAK NOW — say \"Hey Roxy\" a few times <<<"
echo

# -V mono draws a live meter while recording, so you get feedback as you talk.
arecord -D "$DEV" -f S16_LE -r 16000 -c 1 -V mono -d "$SECS" "$OUT"

echo
echo "=== result ==="
python3 - "$OUT" <<'PY'
import sys, wave, struct, math
w=wave.open(sys.argv[1]); n=w.getnframes()
s=struct.unpack("<%dh"%n, w.readframes(n))
peak=max(abs(x) for x in s); rms=math.sqrt(sum(x*x for x in s)/len(s))
db=lambda v: 20*math.log10(max(v,1)/32768)
print(f"  peak {peak} ({db(peak):.1f} dBFS)   RMS {rms:.0f} ({db(rms):.1f} dBFS)")
if peak < 60:      print("  >> DEAD LINK — transmitter not sending anything")
elif peak < 800:   print("  >> very low — either you were silent, or the link is weak")
elif peak > 32000: print("  >> CLIPPING — reduce transmitter gain")
else:              print(f"  >> good level")
print(f"  saved {sys.argv[1]}")
PY
