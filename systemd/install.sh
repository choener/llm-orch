#!/usr/bin/env bash
# Install llm-orch as a hardened systemd service on a plain Linux host
# (Ubuntu/Debian droplet, and similar).  Counterpart to the NixOS module.
#
# Idempotent: safe to rerun, e.g. after rebuilding the binary.  An apikeys
# file is only installed when a source file is given; an existing
# /etc/llm-orch/apikeys.txt is never touched by a rerun without --apikeys.
#
# The script installs:
#   /usr/local/bin/llm-orch                     (the binary)
#   /etc/llm-orch/config.yaml                    (mode 0644)
#   /etc/llm-orch/apikeys.txt                    (mode 0400, llm-orch:llm-orch)
#   /etc/systemd/system/llm-orch.service        (the unit from this directory)
#
# It creates the llm-orch system user on first run.  /var/lib/llm-orch (the
# state directory) is created by systemd when the service first starts.
#
# Usage:
#   sudo systemd/install.sh [options]
#
# Options:
#   --binary PATH   compiled binary to install (default: target/release/llm-orch)
#   --config PATH   config file to install (default: config.yaml)
#   --apikeys PATH  apikeys file to install (default: apikeys.txt, skipped if
#                   the file does not exist in the working directory)
#   --no-start      install files but do not (re)start the service
#   -h, --help      show this help and exit
set -euo pipefail

BINDIR="/usr/local/bin"
SYSCONFDIR="/etc/llm-orch"
UNIT_DST="/etc/systemd/system/llm-orch.service"
SERVICE_USER="llm-orch"
SERVICE_GROUP="llm-orch"

BINARY="target/release/llm-orch"
CONFIG="config.yaml"
APIKEYS="apikeys.txt"
START=yes

usage() {
    sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary)  BINARY="${2:?--binary needs a path}"; shift 2 ;;
        --config)  CONFIG="${2:?--config needs a path}"; shift 2 ;;
        --apikeys) APIKEYS="${2:?--apikeys needs a path}"; shift 2 ;;
        --no-start) START=no; shift ;;
        -h|--help) usage 0 ;;
        *) echo "error: unknown option: $1" >&2; usage 1 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "error: run as root (sudo $0)" >&2
    exit 1
fi

for f in "$BINARY" "$CONFIG"; do
    if [ ! -f "$f" ]; then
        echo "error: $f not found (build with 'cargo build --release', see --binary/--config)" >&2
        exit 1
    fi
done
if [ ! -x "$BINARY" ]; then
    echo "error: $BINARY is not executable" >&2
    exit 1
fi

# ── System user ───────────────────────────────────────────────────────
if ! getent passwd "$SERVICE_USER" >/dev/null; then
    NOLOGIN="$(command -v nologin || echo /usr/sbin/nologin)"
    useradd --system --user-group \
        --home-dir /var/lib/llm-orch --no-create-home \
        --shell "$NOLOGIN" "$SERVICE_USER"
    echo "created system user $SERVICE_USER"
fi

# ── Files ─────────────────────────────────────────────────────────────
install -D -m 0755 "$BINARY" "$BINDIR/llm-orch"

install -D -m 0644 "$CONFIG" "$SYSCONFDIR/config.yaml"

# A missing key file is legal at runtime (fail-closed, all requests denied)
# but a hard error under --check-config, so track whether one is present.
APIKEYS_INSTALLED=no
if [ -f "$APIKEYS" ]; then
    install -D -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0400 \
        "$APIKEYS" "$SYSCONFDIR/apikeys.txt"
    APIKEYS_INSTALLED=yes
elif [ ! -f "$SYSCONFDIR/apikeys.txt" ]; then
    echo "note: no apikeys file given — the service will start fail-closed" >&2
    echo "note: (every request denied) until $SYSCONFDIR/apikeys.txt exists" >&2
fi

UNIT_SRC="$(cd "$(dirname "$0")" && pwd)/llm-orch.service"
install -m 0644 "$UNIT_SRC" "$UNIT_DST"
systemctl daemon-reload

# ── Validate before touching the running service ───────────────────────
# A bad install over a running daemon is safe by design (hot-reload rejects
# the new config and keeps the old one), but do not restart into a config
# that cannot even load.
if [ "$APIKEYS_INSTALLED" = yes ] || [ -f "$SYSCONFDIR/apikeys.txt" ]; then
    CHECK_CMD=("$BINDIR/llm-orch" --check-config "$SYSCONFDIR/config.yaml" \
        --apikeyfile "$SYSCONFDIR/apikeys.txt")
else
    CHECK_CMD=("$BINDIR/llm-orch" --check-config "$SYSCONFDIR/config.yaml")
fi
if ! "${CHECK_CMD[@]}"; then
    echo "error: config check failed — not (re)starting the service" >&2
    exit 1
fi

# ── Enable and (re)start ──────────────────────────────────────────────
systemctl enable llm-orch >/dev/null
if [ "$START" = yes ]; then
    if systemctl is-active --quiet llm-orch; then
        systemctl restart llm-orch
        echo "service restarted"
    else
        systemctl start llm-orch
        echo "service started"
    fi
fi

echo "done. follow with: journalctl -u llm-orch -f"
