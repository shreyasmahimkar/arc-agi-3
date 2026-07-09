#!/usr/bin/env bash
# Safe commit that co-exists with the autonomous coder + launchd cadence (which
# also commit to this repo). Waits for any LIVE git op to finish, then clears a
# genuinely-stale lock (no git running) before committing. Never clobbers a live
# commit. Usage:  ./git_safe_commit.sh "my message"
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
MSG="${1:-v21: manual sync}"

_git_running() { pgrep -fl '[g]it (commit|add|rebase|merge|push|pull)' >/dev/null 2>&1; }

# wait up to ~6 min for a live git op (e.g. the coder's pre-commit benchmark) to finish
for _ in $(seq 1 72); do
  { [ -e .git/index.lock ] || [ -e .git/HEAD.lock ]; } && _git_running && { sleep 5; continue; }
  break
done

# a lock with NO git process behind it is stale -> safe to remove
if { [ -e .git/index.lock ] || [ -e .git/HEAD.lock ]; } && ! _git_running; then
  echo "clearing stale git lock(s)"; rm -f .git/index.lock .git/HEAD.lock .git/*.lock
fi

git add CommunitySolutions/chronos_solver/v21
git commit -m "$MSG" && echo "committed" || echo "(nothing to commit or commit blocked)"
git push 2>/dev/null && echo "pushed" || echo "(push skipped/failed — run manually if needed)"
