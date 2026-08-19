#!/usr/bin/env bash
# Install (or reinstall) Faethon as a systemd service.
#
#   sudo ./scripts/install-service.sh
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_SRC="$PROJECT_DIR/systemd/faethon.service"
UNIT_DST="/etc/systemd/system/faethon.service"

# The unit hardcodes paths for the machine it was written on. Rewrite them for
# wherever the project actually lives, and for whoever owns it.
OWNER="$(stat -c '%U' "$PROJECT_DIR")"
UV_BIN="$(sudo -u "$OWNER" bash -lc 'command -v uv' 2>/dev/null || echo "/home/$OWNER/.local/bin/uv")"

# uv's cache sits outside the project, so ProtectHome hides it unless the unit
# names it. Create it if this is a fresh machine -- systemd refuses to start a
# unit whose ReadWritePaths doesn't exist.
UV_CACHE="$(sudo -u "$OWNER" bash -lc 'echo "${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"')"

if [ ! -x "$UV_BIN" ]; then
  echo "uv not found for user $OWNER (looked at $UV_BIN)" >&2
  echo "Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "project : $PROJECT_DIR"
echo "user    : $OWNER"
echo "uv      : $UV_BIN"
echo "uv cache: $UV_CACHE"

sudo -u "$OWNER" mkdir -p "$UV_CACHE"

sed -e "s|^User=.*|User=$OWNER|" \
    -e "s|^Group=.*|Group=$OWNER|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" \
    -e "s|^ExecStart=.*|ExecStart=$UV_BIN run --project $PROJECT_DIR faethon|" \
    -e "s|^EnvironmentFile=.*|EnvironmentFile=-$PROJECT_DIR/.env|" \
    -e "s|^ReadWritePaths=.*|ReadWritePaths=$PROJECT_DIR $UV_CACHE|" \
    "$UNIT_SRC" > "$UNIT_DST"

if ! id -nG "$OWNER" | grep -qw audio; then
  echo "WARNING: $OWNER is not in the 'audio' group; ALSA access may fail."
  echo "  fix with: sudo usermod -aG audio $OWNER"
fi

systemctl daemon-reload
systemctl enable faethon.service
systemctl restart faethon.service
sleep 2

systemctl --no-pager status faethon.service || true
echo
echo "Follow the log with:  journalctl -u faethon -f"
