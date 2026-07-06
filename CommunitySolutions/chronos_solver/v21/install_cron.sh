#!/usr/bin/env bash
# Install the v21 4-hour cadence as a local cron job (REQUIREMENTS.md R6.2).
# Runs an escalating budget by hour-of-day so unsolved levels get deeper passes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"                       # set PYTHON=/path/to/.venv312/bin/python
# Every 4 hours; cheap passes most of the day, one deep pass at 02:00.
LINE_A="0 2 * * *   cd $HERE && $PY cadence_runner.py --bfs-timeout 1800 >> logs/cron.log 2>&1"
LINE_B="0 6,10,14,18,22 * * *   cd $HERE && $PY cadence_runner.py --bfs-timeout 600 >> logs/cron.log 2>&1"
( crontab -l 2>/dev/null | grep -v "v21/cadence_runner.py" ; echo "# chronos v21 cadence"; echo "$LINE_A"; echo "$LINE_B" ) | crontab -
echo "Installed. Current crontab:"; crontab -l | grep -A2 "chronos v21"
