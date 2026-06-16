#!/bin/bash
# v19 ExIt cycle — run every 15 min via cron.
#   1. solve more levels (BFS ladder, resumes corpus)
#   2. re-harvest transitions from the fuller corpus
#   3. re-train the world model on more data  -> better dynamics prior
# v17-style improvement logs (CAMPAIGN_LOG.md, WM_LOG.md) track every cycle.
# Lock-guarded so cycles never overlap. Scales automatically to however many
# games are discoverable (drop the 200-game testbed in environment_files or set
# V19_EXTRA_GAMES_DIR and it gets picked up).
set -u
# cron runs with a minimal PATH that lacks Homebrew — without this, `timeout`
# (brew coreutils, /opt/homebrew/bin) is "command not found" and every solve/
# train/attempt step silently skips. Put the real PATH back.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
DIR="/Users/shreyas/gitrepos/OpenSource/kaggle/arc3/CommunitySolutions/chronos_solver/v19/src"
VENV="/Users/shreyas/gitrepos/OpenSource/kaggle/arc3/.venv312/bin/activate"
LOCK="/tmp/v19_exit_cycle.lock"
LOG="$DIR/exit_cycle.log"

# skip if a previous cycle is still alive (PID-checked, stale-safe)
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "$(date '+%F %T') [skip] previous cycle still running" >> "$LOG"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

cd "$DIR" || exit 1
# shellcheck disable=SC1090
source "$VENV" 2>/dev/null
{
  echo "===== ExIt cycle START $(date '+%F %T') ====="
  # BREADTH focus: shorter cap = more games' first level attempted per cycle.
  # --shuffle front-loads fresh (no-solved-level) games, so compute spreads to
  # new games (where the score lives) rather than going deep on solved ones.
  echo "--- step 1: solve for BREADTH (cap 120s/level, ~9 min, fresh games first) ---"
  timeout 540 python solve_all.py --bfs-timeout 120 --shuffle
  echo "--- step 2: harvest transitions ---"
  python harvest_wm.py
  echo "--- step 3: retrain world model (20 epochs, ~5 min budget) ---"
  # --net-mult 4 continues the RTX-trained mult=4 lineage (warm-start + held-out
  # gate). Without it the default mult=1 run is blocked by train_wm's guard, so it
  # can never clobber the superior RTX wm_weights.pt.
  timeout 300 python train_wm_v19.py --epochs 20 --net-mult 4
  # DEPRIORITISED (breadth > one hard level): a light WM-attempt only, so it
  # never steals budget from breadth solving. Scale back up later if desired.
  echo "--- step 4: light WM-imagination attempt (deprioritised) ---"
  timeout 90 python wm_attempt.py --games 3 --budget 100
  echo "===== ExIt cycle DONE  $(date '+%F %T') ====="
} >> "$LOG" 2>&1
