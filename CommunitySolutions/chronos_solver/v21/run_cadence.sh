#!/usr/bin/env bash
# Wrapper that launchd (or cron) calls every 4h on the Mac. Self-contained:
# sets env, finds Ollama + the venv, runs one cadence pass, logs everything.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- config (edit these if your paths/model differ) ---------------------------
PY="${PY:-$HERE/../../../.venv312/bin/python}"
export V21_LLM_BACKEND="${V21_LLM_BACKEND:-ollama}"
export V21_OLLAMA_MODEL="${V21_OLLAMA_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}"  # q4 7b (~4.7GB) fits M1 alongside BFS; deadline watchdog guards any swap/hang. Fallback to 3b below if absent.
V21_OLLAMA_FALLBACK="${V21_OLLAMA_FALLBACK:-qwen2.5-coder:3b}"  # used if the primary model isn't pulled
BUDGET="${BUDGET:-600}"                       # seconds/level. Lowered 1200->600: BFS provably won't crack ls20 L5/L6 (117k states explored, 30k unique, still timed out at 1200s) — deeper BFS is wasted compute. Faster passes = more frequent CODE-branch signal for the coder; the walls now need learned/world-model methods (see BACKLOG R7 TRM), not raw search depth.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"   # so `ollama` is found under launchd

# --- wall-cracking features (defaults ON; override in the environment) ---------
export V21_BLITZ="${V21_BLITZ:-1}"            # Stage-0 cheap-win probe (blitz.py)
export V21_EVOLVE_PROBE="${V21_EVOLVE_PROBE:-1}"   # let evolve actually PROMOTE (live rollout)
export V21_BRAIN_PLANNER="${V21_BRAIN_PLANNER:-1}"  # Stage-3.4 Go-Explore/macro-BFS: collapses ls20 L5-L6 corridors that plain BFS can't reach in budget (macro REACH, not depth). Pure white-box search, no model needed; verify+shortest-gated. Runs before runtime_coder for UNSOLVED walls.
export V21_BLACKBOARD="${V21_BLACKBOARD:-1}"        # Epic C0 shared scratchpad: teachers WRITE verified wins as fragments+action_effects, students READ seeds+toddler action_order. Guarded (degrades to no-op on any failure); corpus untouched; verify+shortest-gated.
export V21_GOEXPLORE="${V21_GOEXPLORE:-1}"          # Stage-3.45 Epic C1 cell-archive Go-Explore: dedups on a COARSE downsampled-frame cell so ls20 L5-L6 corridors merge into a small return-to archive (vs macro_bfs's ~19k-state frontier); steered by the blackboard toddler order + primed by its fragments. verify+shortest-gated.
export V21_TODDLER="${V21_TODDLER:-1}"              # Epic C3 intuitive toddler: blends the corpus IntuitionPrior with the blackboard's ONLINE action_effects (frame-aware) behind order_actions, steering Go-Explore's action order. Degrades to bb.action_order/canonical on any failure; corpus untouched.
export V21_TODDLER_NET="${V21_TODDLER_NET:-1}"      # Epic C3/R11 NEURAL toddler: harvest (frame,action->changed/won) samples on walls, TRAIN a StochasticGoose-style frame-change CNN on the Mac GPU (MPS) each run, and use it to order Go-Explore's actions once trained. Degrades to the symbolic toddler if torch/data absent; corpus untouched.

# ---- Opus teacher (R13): API key sourced from an UNTRACKED secrets file --------
# Key lives in the git-IGNORED v21/.env (ANTHROPIC_API_KEY=sk-...) or ~/.chronos_secrets.
# Sourced here so it reaches the cadence env but never enters git or the logs.
set -a
[ -f "$HERE/.env" ] && . "$HERE/.env"
[ -f "$HOME/.chronos_secrets" ] && . "$HOME/.chronos_secrets"
set +a
export V21_OPUS_TEACHER="${V21_OPUS_TEACHER:-1}"    # Stage-3.6: on a wall all local stages failed, ask cloud Opus to read the WHITE-BOX source and construct the plan; verify+shortest-gated + exploit-refused. No-ops (skips) if ANTHROPIC_API_KEY is unset.
export V21_OPUS_ROUNDS="${V21_OPUS_ROUNDS:-2}"      # R7 teach-with-feedback: EXECUTE each proposed plan on a fork, feed the failure report (how far it got) back to Opus for up to N rounds instead of discarding a near-miss (this-run signal: ls20 L5 got a 19-action plan that failed verify). 1 = old single-shot.
export V21_OPUS_WM="${V21_OPUS_WM:-1}"              # Stage-3.7: ask Opus to WRITE an executable WorldModel .py from the white-box source (persisted to brain/wm/<gid>/model.py); exec+plan+verify on the engine. The B2 world-model spine. Needs ANTHROPIC_API_KEY.
export V21_OPUS_ARCH="${V21_OPUS_ARCH:-1}"          # Opus-as-ML-engineer: each run Opus DESIGNS an improved PyTorch net for the neural toddler; champion/challenger on held-out accuracy, ADOPT (brain/toddler/<gid>_arch.py) only if it beats the current net. Needs ANTHROPIC_API_KEY + torch + >=200 samples.
export V21_WORKSPACE_COUNTEREX="${V21_WORKSPACE_COUNTEREX:-1}"  # R7(a): persist each FAILED Opus-teacher wall plan as a blackboard dead_end and feed it back next run as a 'do NOT repeat' constraint (a fresh cadence otherwise re-proposes the same near-miss — run 073852Z: ls20 L5 rounds 1&2 both stalled at levels_completed=5). Env-gated, degrades to no-op; corpus untouched.
export V21_TEACHER_GROUND="${V21_TEACHER_GROUND:-1}"  # R8/B1: hand the Opus teacher the level-START valid ACTION6 click targets (perception component centroids) in its FIRST-round prompt so its clicks land on real objects, not dead coordinates (run 152556Z: vc33 L4 round 1 first action was a no-op — clicked empty space). Pure prompt-grounding; verify+shortest+exploit-gated; degrades to no-op.
export V21_TEACHER_GROUND2="${V21_TEACHER_GROUND2:-1}"  # R14 FULL grounding: hand Opus the REAL level-start frame as a symbolic scene digest (objects/centroids) PLUS a per-action effect table (each action pressed once -> changed / levels_completed), so it plans over the actual board instead of blind over the source (run 192513Z: ls20 L5 plans changed 86-90 cells but never crossed the goal — Opus never saw the board or the win state). Pure fork; verify+shortest+exploit-gated; degrades to the old prompt on any failure.

# ---- Phase 2: 274-game generalization corpus (CONTINGENT on the 3 games) --------
# Safe to leave ON: it is a no-op until ls20+ft09+vc33 are 100% solved, then it
# harvests toddler samples across the wide corpus (V21_PHASE2_MAX games/run) so the
# intuitive prior + world models generalize to unseen games. Never solves/commits.
export V21_PHASE2="${V21_PHASE2:-1}"
export V21_PHASE2_MAX="${V21_PHASE2_MAX:-40}"
export V21_WORLD_MODEL="${V21_WORLD_MODEL:-1}"      # Epic C2 persistent executable world model: on UNSOLVED walls, capture live one-step transitions -> build+MDL-refactor+save brain/wm/<gid>/model.json; next run load+verify it reproduces fresh transitions (is_trusted = cross-run reuse). Guarded; writes only brain/wm runtime state (gitignored); corpus untouched.
export V21_RUNTIME_CODER="${V21_RUNTIME_CODER:-1}" # on-the-fly WM writer ON (Qwen writes code for BFS/blitz-blocked walls)
export V21_BRAIN_PERCEPTION="${V21_BRAIN_PERCEPTION:-1}" # B1: perception connected-component click targets (one per blob) for vc33 same-colour walls

mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/cron_${STAMP}.log"

echo "[$STAMP] starting cadence (budget=${BUDGET}s, model=$V21_OLLAMA_MODEL)" | tee -a "$LOG"

# --- run heartbeat (ops observability) ----------------------------------------
# One-line epoch stamps the health-check reads directly, so it no longer has to
# parse cron_*.log filenames + convert sandbox-local mtimes to UTC to judge liveness.
# `.last_start` newer than `.last_end` by more than a real pass (~90m) => a run is
# hung or its process died mid-pass; `.last_start` itself older than the launchd
# interval => launchd is not ticking. Best-effort; never fails the run.
printf '%s %s\n' "$(date -u +%s)" "$STAMP" > logs/.last_start 2>/dev/null || true

# --- stale-run self-heal (ops) ------------------------------------------------
# A prior cadence that HUNG inside a stage (e.g. run 164123Z's RUNTIME_CODER/evolve
# tail) keeps its process alive, so its fcntl.flock on logs/.cadence.lock is never
# released and the lock FILE lingers. Every later launchd tick then either blocks on
# the flock or the loop's memory prescribes a MANUAL `pkill -f cadence_runner && rm
# -f logs/.cadence.lock && launchctl start ...` (see ITERATION_LOG 164123Z). That has
# now idled the runner for multi-hour windows twice. Auto-recover instead: reap any
# cadence_runner.py older than a ceiling that no legitimate full pass ever reaches
# (~20 levels x 600s + toddler train/evolve is <~90m), then clear the stale lock file
# so the fresh flock below starts clean. A LEGIT in-flight run (etimes < ceiling) is
# left untouched and its live flock still correctly rejects this double-start.
STALE_SECS="${V21_STALE_SECS:-10800}"   # 3h; >> a real pass, << an overnight idle gap
for _pid in $(pgrep -f "cadence_runner.py" 2>/dev/null); do
  [ "$_pid" = "$$" ] && continue
  _et="$(ps -o etimes= -p "$_pid" 2>/dev/null | tr -d ' ')"
  case "$_et" in ''|*[!0-9]*) continue;; esac        # skip if age unreadable
  if [ "$_et" -gt "$STALE_SECS" ]; then
    echo "[preflight] reaping hung cadence pid=$_pid (etimes=${_et}s > ${STALE_SECS}s)" | tee -a "$LOG"
    kill -9 "$_pid" 2>/dev/null || true
  fi
done
if ! pgrep -f "cadence_runner.py" >/dev/null 2>&1; then
  # no live cadence holds the flock -> a lingering lock file is stale; clear it.
  [ -e logs/.cadence.lock ] && echo "[preflight] clearing stale logs/.cadence.lock" | tee -a "$LOG"
  rm -f logs/.cadence.lock 2>/dev/null || true
fi

# start Ollama if installed and not already serving (ignore failure -> mock fallback)
if command -v ollama >/dev/null 2>&1; then
  curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 || (ollama serve >>"$LOG" 2>&1 &)
  sleep 3
  # Preflight: use the primary model only if it's actually pulled; else fall back
  # to the smaller model. Prevents the 404/500 (model-not-found) failure mode.
  if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$V21_OLLAMA_MODEL"; then
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$V21_OLLAMA_FALLBACK"; then
      echo "[preflight] '$V21_OLLAMA_MODEL' not pulled -> falling back to '$V21_OLLAMA_FALLBACK'" | tee -a "$LOG"
      export V21_OLLAMA_MODEL="$V21_OLLAMA_FALLBACK"
    else
      echo "[preflight] WARNING: neither '$V21_OLLAMA_MODEL' nor '$V21_OLLAMA_FALLBACK' pulled; runtime_coder will hit safety nets" | tee -a "$LOG"
    fi
  fi
  echo "[preflight] using model=$V21_OLLAMA_MODEL" | tee -a "$LOG"
fi

# keep the Mac awake for the duration of the run, then run the cadence
caffeinate -i "$PY" cadence_runner.py --bfs-timeout "$BUDGET" --evolve --allow-network >>"$LOG" 2>&1
RC=$?
printf '%s %s exit=%s\n' "$(date -u +%s)" "$(date -u +%Y%m%dT%H%M%SZ)" "$RC" > logs/.last_end 2>/dev/null || true
echo "[$(date -u +%Y%m%dT%H%M%SZ)] cadence exit=$RC" | tee -a "$LOG"
# keep only the last 30 cron logs
ls -1t logs/cron_*.log 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true
exit $RC
