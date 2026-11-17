#!/bin/sh
# Installs or removes the launchd job that runs scripts/tick.py every 2.5 minutes.
#
#   sh scripts/launchd.sh install   # write the job and start it (it ticks at once)
#   sh scripts/launchd.sh remove    # stop it and delete the job
#
# The Mac must stay awake for ticks to run. Output goes to data_local/tick.log.
set -eu

LABEL=com.oddsengine.tick
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
UV="$(command -v uv)"
DOMAIN="gui/$(id -u)"

case "${1:-}" in
install)
    mkdir -p "$REPO/data_local" "$(dirname "$PLIST")"
    cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$UV</string><string>run</string><string>python</string><string>scripts/tick.py</string></array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>StartInterval</key><integer>150</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$REPO/data_local/tick.log</string>
  <key>StandardErrorPath</key><string>$REPO/data_local/tick.log</string>
</dict>
</plist>
EOF
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "started $LABEL: a tick every 2.5 minutes, log in data_local/tick.log"
    ;;
remove)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed $LABEL"
    ;;
*)
    echo "usage: $0 install|remove" >&2
    exit 2
    ;;
esac
