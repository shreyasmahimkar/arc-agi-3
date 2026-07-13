#!/usr/bin/env bash
# ONE-COMMAND Mac-side recovery for the v21 cadence loop.
#
# WHY: when the launchd runner dies (SIGKILL/OOM or the Mac sleeps) AND the sandbox
# health-check cycles keep committing offline-verified work, the loop wedges in a
# recurring, multi-step manual state that has stalled it ~10 cycles (Jul 9-12):
#   (a) a hung `cadence_runner.py` + a stranded `logs/.cadence.lock`;
#   (b) 2-3 stranded `.git/*.lock` files (index.lock / HEAD.lock / packed-refs.lock)
#       that the SANDBOX cannot remove (EPERM on the mounted .git), so every cycle's
#       commit lands on the SIDE ref `refs/heads/v21-auto-sync` instead of `main`;
#   (c) `main` therefore sits N commits behind `v21-auto-sync` and un-pushed.
# The fix used to be a fiddly hand-typed sequence buried in commit messages. This
# script collapses it into: `bash resume_loop.sh`  (run ONCE on the Mac after a
# restart / wake — the Mac, unlike the sandbox, CAN remove its own stale locks).
#
# SAFETY: only removes a git lock when it is BOTH (i) older than STALE_SECS and
# (ii) has no live git process behind it; fast-forwards `main` ONLY when it is a
# strict ancestor of the side ref (never a merge-commit, never `reset --hard`, never
# touches user work); refuses and reports if the tree is dirty or the refs diverged.
# `DRY_RUN=1 bash resume_loop.sh` prints every action without executing it.
#
# Sourcing contract (for test_resume_loop.sh): `RESUME_LOOP_LIB=1 . resume_loop.sh`
# defines the functions WITHOUT running main().
set -uo pipefail

SIDE_REF="${RESUME_SIDE_REF:-refs/heads/v21-auto-sync}"
MAIN_REF="${RESUME_MAIN_REF:-refs/heads/main}"
STALE_SECS="${RESUME_STALE_SECS:-600}"   # a git lock younger than this may be a LIVE op
DRY_RUN="${DRY_RUN:-0}"

_say()  { echo "[resume] $*"; }
_run()  { if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] $*"; else eval "$*"; fi; }
_git_running() { pgrep -fl '[g]it (commit|add|rebase|merge|push|pull|fetch)' >/dev/null 2>&1; }

# Portable "seconds since file mtime" (Linux `stat -c`, macOS/BSD `stat -f`).
_age_secs() {
  local f="$1" m now; now=$(date +%s)
  m=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null) || { echo 999999; return; }
  echo $(( now - m ))
}

# Remove only genuinely-stale git locks (old enough AND no live git). Safe on the Mac.
clear_stale_git_locks() {
  local gd cleared=0 f age
  gd="$(git rev-parse --git-dir 2>/dev/null)" || { _say "not a git repo"; return 1; }
  if _git_running; then _say "a live git op is running — leaving locks alone"; return 0; fi
  shopt -s nullglob
  for f in "$gd"/index.lock "$gd"/HEAD.lock "$gd"/packed-refs.lock "$gd"/refs/heads/*.lock; do
    [ -e "$f" ] || continue
    age=$(_age_secs "$f")
    if [ "$age" -ge "$STALE_SECS" ]; then
      _run "rm -f '$f'" && { _say "cleared stale lock ${f##*/} (age ${age}s)"; cleared=1; }
    else
      _say "lock ${f##*/} is only ${age}s old (< ${STALE_SECS}s) — NOT clearing (may be live)"
    fi
  done
  shopt -u nullglob
  [ "$cleared" = 1 ] || _say "no stale git locks to clear"
  return 0
}

# Fast-forward MAIN_REF up to SIDE_REF, but ONLY when it is a safe pure fast-forward.
# Echos a status token as the last word: FF / ALREADY / NOSIDE / DIRTY / DIVERGED / FAIL.
reconcile_main() {
  git rev-parse --verify -q "$SIDE_REF" >/dev/null 2>&1 || { _say "no ${SIDE_REF} — nothing to reconcile NOSIDE"; return 0; }
  local side main behind
  side="$(git rev-parse "$SIDE_REF")"
  main="$(git rev-parse "$MAIN_REF" 2>/dev/null || echo '')"
  if [ "$side" = "$main" ]; then _say "${MAIN_REF} already at side ref ALREADY"; return 0; fi
  if ! git merge-base --is-ancestor "$MAIN_REF" "$SIDE_REF" 2>/dev/null; then
    _say "${MAIN_REF} is NOT an ancestor of ${SIDE_REF} — refusing (manual reconcile) DIVERGED"; return 1
  fi
  behind="$(git rev-list --count "${MAIN_REF}..${SIDE_REF}" 2>/dev/null || echo '?')"
  # If main is the checked-out branch, use a real ff-only merge so index+worktree move
  # with the ref; otherwise just advance the ref. Never touch a dirty worktree.
  if [ "$(git symbolic-ref -q HEAD || true)" = "$MAIN_REF" ]; then
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
      _say "worktree dirty — commit/stash then re-run; will not ff a dirty tree DIRTY"; return 1
    fi
    _run "git merge --ff-only '$SIDE_REF'" && { _say "fast-forwarded ${MAIN_REF} +${behind} to ${side:0:12} FF"; return 0; }
    _say "ff-only merge failed FAIL"; return 1
  else
    _run "git update-ref '$MAIN_REF' '$side'" && { _say "advanced ${MAIN_REF} +${behind} to ${side:0:12} FF"; return 0; }
    _say "update-ref failed FAIL"; return 1
  fi
}

main() {
  cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || { _say "cannot cd to repo root"; exit 1; }
  local V21="CommunitySolutions/chronos_solver/v21"

  _say "1/5 reaping any hung cadence_runner"
  _run "pkill -f cadence_runner.py" || true

  _say "2/5 clearing stranded runtime + git locks"
  _run "rm -f '$V21/logs/.cadence.lock'" || true
  clear_stale_git_locks || true

  _say "3/5 reconciling ${MAIN_REF} with ${SIDE_REF}"
  reconcile_main || _say "reconcile skipped/failed — inspect manually before push"

  _say "4/5 pushing ${MAIN_REF}"
  _run "git push origin ${MAIN_REF##refs/heads/}" || _say "push failed — re-run once network is up"

  _say "5/5 restarting the launchd cadence"
  if command -v launchctl >/dev/null 2>&1; then
    _run "launchctl start com.chronos.v21.cadence"
  else
    _say "launchctl not present (non-Mac) — start the runner however this host schedules it"
  fi
  _say "done — next launchd tick should write a fresh cron_*.log + logs/.last_start"
}

# Only run main() when executed directly, not when sourced by the test harness.
if [ "${RESUME_LOOP_LIB:-0}" != 1 ]; then
  main "$@"
fi
