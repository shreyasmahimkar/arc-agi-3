#!/usr/bin/env bash
# Offline test for git_safe_commit.sh. Builds a throwaway git repo (never touches
# the real one), then exercises: syntax, the normal commit+detached-push path, the
# plumbing path (private GIT_INDEX_FILE, no .git/index — forced via
# V21_FORCE_ALT_INDEX=1), and the side-ref fallback used when the branch ref is
# locked (forced via V21_FORCE_SIDE_REF=1). No network.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/git_safe_commit.sh"
PASS=0; FAIL=0
ok()  { echo "PASS: $1"; PASS=$((PASS+1)); }
bad() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

# 1) syntax
if bash -n "$SCRIPT"; then ok "syntax (bash -n)"; else bad "syntax (bash -n)"; fi

# scratch repo with the v21 subtree layout the script expects
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
git init -q
git config user.email t@t; git config user.name t; git config commit.gpgsign false
V21="CommunitySolutions/chronos_solver/v21"
mkdir -p "$V21"
cp "$SCRIPT" "$V21/git_safe_commit.sh"
echo "seed" > "$V21/file.txt"
git add -A && git commit -qm "seed"
BASE="$(git rev-parse HEAD)"
git init -q --bare "$TMP/remote.git"
git remote add origin "$TMP/remote.git"
git push -q origin HEAD 2>/dev/null || true

# 2) normal path: change a v21 file, commit; assert a NEW commit landed with it
echo "change-normal" > "$V21/file.txt"
bash "$V21/git_safe_commit.sh" "test: normal path" >/dev/null 2>&1; wait 2>/dev/null || true
H1="$(git rev-parse HEAD)"
[ "$H1" != "$BASE" ] && ok "normal path created a new commit" || bad "normal path created a new commit"
git show --stat HEAD | grep -q "file.txt" && ok "normal commit includes the v21 change" || bad "normal commit includes the v21 change"

# 3) plumbing path (forced): lands a commit on the branch WITHOUT using .git/index
echo "change-plumb" > "$V21/file.txt"
: > .git/index.lock   # prove the plumbing path doesn't depend on removing it
V21_FORCE_ALT_INDEX=1 V21_NO_PUSH=1 bash "$V21/git_safe_commit.sh" "test: plumbing" >/dev/null 2>&1; wait 2>/dev/null || true
rm -f .git/index.lock
H2="$(git rev-parse HEAD)"
[ "$H2" != "$H1" ] && ok "plumbing path advanced the branch" || bad "plumbing path advanced the branch"
git show HEAD:"$V21/file.txt" 2>/dev/null | grep -q "change-plumb" && ok "plumbing commit captured working-tree v21 change" || bad "plumbing commit captured working-tree v21 change"
[ "$(git rev-parse HEAD^)" = "$H1" ] && ok "plumbing commit parent = prior HEAD (clean chain)" || bad "plumbing commit parent = prior HEAD (clean chain)"

# 4) side-ref fallback (forced): must land on refs/heads/v21-auto-sync, NOT move HEAD
echo "change-side" > "$V21/file.txt"
HB="$(git rev-parse HEAD)"
V21_FORCE_SIDE_REF=1 V21_NO_PUSH=1 bash "$V21/git_safe_commit.sh" "test: side ref" >/dev/null 2>&1; wait 2>/dev/null || true
[ "$(git rev-parse HEAD)" = "$HB" ] && ok "side-ref fallback left HEAD untouched" || bad "side-ref fallback left HEAD untouched"
if git rev-parse --verify -q refs/heads/v21-auto-sync >/dev/null; then ok "side-ref v21-auto-sync created"; else bad "side-ref v21-auto-sync created"; fi
git show refs/heads/v21-auto-sync:"$V21/file.txt" 2>/dev/null | grep -q "change-side" && ok "side-ref commit captured the v21 change" || bad "side-ref commit captured the v21 change"

# 5) no-op: working tree matches HEAD -> no new commit, no crash
git checkout -q HEAD -- "$V21/file.txt"   # sync working tree to HEAD first
NOOP_BEFORE="$(git rev-parse HEAD)"
V21_FORCE_ALT_INDEX=1 V21_NO_PUSH=1 bash "$V21/git_safe_commit.sh" "test: noop" >/dev/null 2>&1; wait 2>/dev/null || true
[ "$(git rev-parse HEAD)" = "$NOOP_BEFORE" ] && ok "plumbing no-op leaves HEAD unchanged" || bad "plumbing no-op leaves HEAD unchanged"

# 6) detached-push contract: prints the backgrounding line (non-blocking push)
echo "p" > "$V21/file.txt"
if bash "$V21/git_safe_commit.sh" "test: push bg" 2>&1 | grep -q 'push backgrounded'; then
  ok "push runs detached (prints 'push backgrounded')"
else
  bad "push runs detached (prints 'push backgrounded')"
fi
wait 2>/dev/null || true

echo "----"
echo "git_safe_commit: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
