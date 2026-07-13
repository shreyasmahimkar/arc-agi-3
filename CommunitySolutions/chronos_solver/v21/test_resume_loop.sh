#!/usr/bin/env bash
# Offline test for resume_loop.sh — exercises the fast-forward reconcile + stale-lock
# logic in a throwaway repo (no launchd/network needed). Run: bash test_resume_loop.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "  ok: $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL: $1"; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
git init -q; git config user.email t@t; git config user.name t
git commit -q --allow-empty -m base
BASE="$(git rev-parse HEAD)"

# Load resume_loop.sh as a library (defines functions, does NOT run main()).
RESUME_LOOP_LIB=1 . "$HERE/resume_loop.sh"

# --- 1. clean fast-forward: main is a strict ancestor of the side ref ---------
git commit -q --allow-empty -m c1
git commit -q --allow-empty -m c2
git update-ref refs/heads/v21-auto-sync HEAD
git update-ref refs/heads/main "$BASE"
git symbolic-ref HEAD refs/heads/main
git reset -q --hard "$BASE"          # main worktree at base, side ref 2 ahead
out="$(reconcile_main)"; echo "$out" | grep -q ' FF$' && ok "ff advances main" || bad "ff advances main ($out)"
[ "$(git rev-parse refs/heads/main)" = "$(git rev-parse refs/heads/v21-auto-sync)" ] \
  && ok "main == side after ff" || bad "main not fast-forwarded"

# --- 2. idempotent: second run is a no-op ALREADY -----------------------------
out="$(reconcile_main)"; echo "$out" | grep -q ' ALREADY$' && ok "already-synced no-op" || bad "not idempotent ($out)"

# --- 3. diverged: side ref is NOT a descendant of main -> refuse --------------
git checkout -q -b tmp "$BASE"; git commit -q --allow-empty -m divergent
git update-ref refs/heads/v21-auto-sync HEAD
git symbolic-ref HEAD refs/heads/main; git reset -q --hard refs/heads/main
before="$(git rev-parse refs/heads/main)"
out="$(reconcile_main)" || true; echo "$out" | grep -q ' DIVERGED$' && ok "diverged refused" || bad "diverged not refused ($out)"
[ "$(git rev-parse refs/heads/main)" = "$before" ] && ok "main untouched on diverge" || bad "main moved on diverge"

# --- 4. dirty worktree -> refuse to fast-forward ------------------------------
git update-ref refs/heads/v21-auto-sync "$BASE"   # reset side ref to a clean ancestor case
git commit -q --allow-empty -m c3
git update-ref refs/heads/v21-auto-sync HEAD
git reset -q --hard "$BASE"
echo dirty > dirty_file            # untracked change makes the tree dirty
out="$(reconcile_main)" || true; echo "$out" | grep -q ' DIRTY$' && ok "dirty tree refused" || bad "dirty not refused ($out)"
rm -f dirty_file

# --- 5. stale-lock clearing: old lock removed, fresh lock kept ----------------
GD="$(git rev-parse --git-dir)"
: > "$GD/index.lock"; touch -d '2000-01-01' "$GD/index.lock" 2>/dev/null || touch -t 200001010000 "$GD/index.lock"
: > "$GD/HEAD.lock"   # fresh (now) -> must be kept
RESUME_STALE_SECS=600 clear_stale_git_locks >/dev/null 2>&1
[ ! -e "$GD/index.lock" ] && ok "stale index.lock cleared" || bad "stale index.lock survived"
[ -e "$GD/HEAD.lock" ] && ok "fresh HEAD.lock kept" || bad "fresh HEAD.lock wrongly removed"
rm -f "$GD/HEAD.lock"

# --- 6. DRY_RUN never mutates refs --------------------------------------------
git update-ref refs/heads/v21-auto-sync "$BASE"
git commit -q --allow-empty -m c4; git update-ref refs/heads/v21-auto-sync HEAD; git reset -q --hard "$BASE"
before="$(git rev-parse refs/heads/main)"
DRY_RUN=1 reconcile_main >/dev/null 2>&1 || true
[ "$(git rev-parse refs/heads/main)" = "$before" ] && ok "DRY_RUN leaves refs unchanged" || bad "DRY_RUN mutated refs"

# --- 7. no side ref -> graceful NOSIDE ----------------------------------------
git update-ref -d refs/heads/v21-auto-sync
out="$(reconcile_main)"; echo "$out" | grep -q ' NOSIDE$' && ok "no side ref graceful" || bad "no side ref not handled ($out)"

echo "test_resume_loop: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
