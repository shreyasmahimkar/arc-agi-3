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
DIR="/Users/shreyas/gitrepos/OpenSource/kaggle/arc3/CommunitySolutions/chronos_solver/v19"
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
  echo "--- step 1: solve more (cap 180s/level, ~9 min budget, shuffled across all games) ---"
  timeout 540 python solve_all.py --bfs-timeout 180 --shuffle
  echo "--- step 2: harvest transitions ---"
  python harvest_wm.py
  echo "--- step 3: retrain world model (20 epochs, ~5 min budget) ---"
  timeout 300 python train_wm_v19.py --epochs 20
  echo "--- step 4: WM-imagination attempts on the frontier (crack what BFS can't) ---"
  timeout 240 python wm_attempt.py --games 8 --budget 200
  echo "===== ExIt cycle DONE  $(date '+%F %T') ====="
} >> "$LOG" 2>&1
