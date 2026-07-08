#!/usr/bin/env bash
# Install the v21 cadence as a CONTINUOUS (back-to-back) macOS LaunchAgent.
# Runs run_cadence.sh over and over so the Mac is never idle. This UNLOADS the
# 4h calendar agent first so the two never run at the same time.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CAL_PLIST="com.chronos.v21.cadence.plist"       # the old every-4h agent
CONT_PLIST="com.chronos.v21.continuous.plist"   # this back-to-back agent
DEST_CAL="$HOME/Library/LaunchAgents/$CAL_PLIST"
DEST_CONT="$HOME/Library/LaunchAgents/$CONT_PLIST"

chmod +x "$HERE/run_cadence.sh"
mkdir -p "$HOME/Library/LaunchAgents" "$HERE/logs"

# 1) stop the every-4h agent if it's loaded (avoid double-running)
launchctl unload "$DEST_CAL" 2>/dev/null || true

# 2) (re)install the continuous agent
cp "$HERE/$CONT_PLIST" "$DEST_CONT"
launchctl unload "$DEST_CONT" 2>/dev/null || true
launchctl load "$DEST_CONT"

echo "Continuous cadence loaded — runs back-to-back with a ${THROTTLE:-60}s floor between passes."
echo "  Watch it roll:   tail -f $HERE/logs/cron_*.log"
echo "  Is it running:   launchctl list | grep chronos"
echo "  PAUSE it:        launchctl unload $DEST_CONT"
echo "  Back to 4h:      launchctl unload $DEST_CONT && bash $HERE/install_launchd.sh"
