#!/usr/bin/env bash
# Roxy audio smoke test: prove the mic hears you and the speaker works,
# before blaming anything further up the stack.
#
#   ./scripts/check-audio.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
UV="${UV:-$HOME/.local/bin/uv}"

echo "=== Capture devices ==="
arecord -l || echo "  (none -- is the mic plugged in?)"
echo
echo "=== Playback devices ==="
aplay -l || echo "  (none)"
echo

DEV_IN=$($UV run python -c "from roxy.config import load_config; print(load_config().audio.input_device)")
DEV_OUT=$($UV run python -c "from roxy.config import load_config; print(load_config().audio.output_device)")
echo "Configured input : $DEV_IN"
echo "Configured output: $DEV_OUT"
echo

echo "=== Recording 3 seconds -- SAY SOMETHING NOW ==="
$UV run python - "$DEV_IN" <<'PY'
import sys, math, struct
from pathlib import Path
from roxy.config import load_config
from roxy.audio import capture, playback

cfg = load_config()
rate = cfg.audio.sample_rate
try:
    pcm = capture.record_seconds(sys.argv[1], rate, 3.0)
except capture.CaptureError as e:
    print(f"  CAPTURE FAILED: {e}")
    sys.exit(1)

s = struct.unpack("<%dh" % (len(pcm) // 2), pcm)
rms = math.sqrt(sum(x * x for x in s) / len(s))
peak = max(abs(x) for x in s)
dbfs = 20 * math.log10(peak / 32768) if peak else float("-inf")
print(f"  samples={len(s)}  RMS={rms:.1f}  peak={peak}  ({dbfs:.1f} dBFS)")

if peak < 500:
    print("  >> SILENT. The mic enumerates but no audio is arriving.")
    print("     Wireless mic? Check the transmitter is powered on and paired.")
    print("     Wired mic? Try another USB port, and check `arecord -L`.")
elif peak > 32000:
    print("  >> CLIPPING. Back away from the mic or lower its gain.")
else:
    print("  >> Signal looks good.")

out = Path("/tmp/roxy-audiocheck.wav")
playback.write_wav(out, pcm, rate)
print(f"  saved {out}")
PY
echo

echo "=== Playing it back ==="
aplay -D "$DEV_OUT" -q /tmp/roxy-audiocheck.wav && echo "  Did you hear yourself?" \
  || echo "  PLAYBACK FAILED -- check the speaker is connected to $DEV_OUT"
echo

echo "=== Playing the wake chime ==="
[ -f assets/ack.wav ] || $UV run python scripts/make_chime.py
aplay -D "$DEV_OUT" -q assets/ack.wav && echo "  Two ascending beeps?" || echo "  FAILED"
