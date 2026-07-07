#!/usr/bin/env bash
# Install the v21 cadence as a macOS LaunchAgent (runs every 4h, natively).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST="$HERE/com.chronos.v21.cadence.plist"
DEST="$HOME/Library/LaunchAgents/com.chronos.v21.cadence.plist"

chmod +x "$HERE/run_cadence.sh"
mkdir -p "$HOME/Library/LaunchAgents" "$HERE/logs"
cp "$PLIST" "$DEST"

# reload if already installed
launchctl unload "$DEST" 2>/dev/null || true
launchctl load "$DEST"
echo "Loaded. Scheduled runs: 00/04/08/12/16/20 local time."
echo "  Run one now:   launchctl start com.chronos.v21.cadence"
echo "  Check status:  launchctl list | grep chronos"
echo "  Stop/remove:   launchctl unload $DEST"
echo "  Live log:      tail -f $HERE/logs/cron_*.log"
