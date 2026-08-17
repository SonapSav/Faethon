#!/usr/bin/env bash
# Install (or reinstall) Roxy as a systemd service.
#
#   sudo ./scripts/install-service.sh
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_SRC="$PROJECT_DIR/systemd/roxy.service"
UNIT_DST="/etc/systemd/system/roxy.service"

# The unit hardcodes paths for the machine it was written on. Rewrite them for
# wherever the project actually lives, and for whoever owns it.
OWNER="$(stat -c '%U' "$PROJECT_DIR")"
UV_BIN="$(sudo -u "$OWNER" bash -lc 'command -v uv' 2>/dev/null || echo "/home/$OWNER/.local/bin/uv")"

if [ ! -x "$UV_BIN" ]; then
  echo "uv not found for user $OWNER (looked at $UV_BIN)" >&2
  echo "Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "project : $PROJECT_DIR"
echo "user    : $OWNER"
echo "uv      : $UV_BIN"

sed -e "s|^User=.*|User=$OWNER|" \
    -e "s|^Group=.*|Group=$OWNER|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" \
    -e "s|^ExecStart=.*|ExecStart=$UV_BIN run --project $PROJECT_DIR roxy|" \
    -e "s|^EnvironmentFile=.*|EnvironmentFile=-$PROJECT_DIR/.env|" \
    -e "s|^ReadWritePaths=.*|ReadWritePaths=$PROJECT_DIR|" \
    "$UNIT_SRC" > "$UNIT_DST"

if ! id -nG "$OWNER" | grep -qw audio; then
  echo "WARNING: $OWNER is not in the 'audio' group; ALSA access may fail."
  echo "  fix with: sudo usermod -aG audio $OWNER"
fi

systemctl daemon-reload
systemctl enable roxy.service
systemctl restart roxy.service
sleep 2

systemctl --no-pager status roxy.service || true
echo
echo "Follow the log with:  journalctl -u roxy -f"
