#!/usr/bin/env bash
# Wrapper that launchd (or cron) calls every 4h on the Mac. Self-contained:
# sets env, finds Ollama + the venv, runs one cadence pass, logs everything.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- config (edit these if your paths/model differ) ---------------------------
PY="${PY:-$HERE/../../../.venv312/bin/python}"
export V21_LLM_BACKEND="${V21_LLM_BACKEND:-ollama}"
export V21_OLLAMA_MODEL="${V21_OLLAMA_MODEL:-qwen2.5-coder:3b}"  # 3b coexists with heavy BFS (7b OOM'd -> Ollama 500)
BUDGET="${BUDGET:-1200}"                      # seconds/level; raised 600->1200 for deeper walls (ls20 L4 timed out at 600; L5-L6/ft09 L2-L5 need more BFS depth)
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"   # so `ollama` is found under launchd

# --- wall-cracking features (defaults ON; override in the environment) ---------
export V21_BLITZ="${V21_BLITZ:-1}"            # Stage-0 cheap-win probe (blitz.py)
export V21_EVOLVE_PROBE="${V21_EVOLVE_PROBE:-1}"   # let evolve actually PROMOTE (live rollout)
export V21_RUNTIME_CODER="${V21_RUNTIME_CODER:-1}" # on-the-fly WM writer ON (Qwen writes code for BFS/blitz-blocked walls)
export V21_BRAIN_PERCEPTION="${V21_BRAIN_PERCEPTION:-1}" # B1: perception connected-component click targets (one per blob) for vc33 same-colour walls

mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/cron_${STAMP}.log"

echo "[$STAMP] starting cadence (budget=${BUDGET}s, model=$V21_OLLAMA_MODEL)" | tee -a "$LOG"

# start Ollama if installed and not already serving (ignore failure -> mock fallback)
if command -v ollama >/dev/null 2>&1; then
  curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 || (ollama serve >>"$LOG" 2>&1 &)
  sleep 3
fi

# keep the Mac awake for the duration of the run, then run the cadence
caffeinate -i "$PY" cadence_runner.py --bfs-timeout "$BUDGET" --evolve --allow-network >>"$LOG" 2>&1
RC=$?
echo "[$(date -u +%Y%m%dT%H%M%SZ)] cadence exit=$RC" | tee -a "$LOG"
# keep only the last 30 cron logs
ls -1t logs/cron_*.log 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true
exit $RC
