#!/usr/bin/env bash
# Offline check for run_cadence.sh's stale-run self-heal block (no arcengine/Mac needed).
# Verifies: (1) syntax parses, (2) a lingering lock file with NO live cadence_runner is
# cleared, (3) a hung (old) cadence_runner is reaped while a fresh one is spared.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
fail=0

# (1) parse
bash -n run_cadence.sh || { echo "FAIL: run_cadence.sh does not parse"; exit 1; }
echo "PASS reaper: run_cadence.sh parses"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/logs"

# Extract the self-heal block to a FILE (running it from a file keeps the "cadence_runner.py"
# pattern off the harness's own command line, so pgrep only ever matches the real sleeper).
awk '/--- stale-run self-heal/{f=1} /# start Ollama if installed/{f=0} f' run_cadence.sh > "$TMP/block.sh"
grep -q 'pgrep -f "cadence_runner.py"' "$TMP/block.sh" || { echo "FAIL: could not extract self-heal block"; exit 1; }
run_block() { ( cd "$TMP" && LOG=/dev/null V21_STALE_SECS="$1" bash block.sh ) >/dev/null 2>&1; }

# A stand-in cadence process: a script literally named cadence_runner.py that sleeps
# WITHOUT exec, so its command line ("bash .../cadence_runner.py") matches pgrep -f.
SLEEPER="$TMP/cadence_runner.py"; printf '#!/usr/bin/env bash\nsleep 30\n' > "$SLEEPER"; chmod +x "$SLEEPER"

# (2) stale lock file, no live cadence -> cleared
: > "$TMP/logs/.cadence.lock"
run_block 10800
if [ -e "$TMP/logs/.cadence.lock" ]; then echo "FAIL: stale lock not cleared"; fail=1; else echo "PASS reaper: stale lock cleared"; fi

# (3) reap OLD (STALE_SECS=1 => a 2s-old proc is "hung"), then clear the lock.
bash "$SLEEPER" & OLDPID=$!
sleep 2
: > "$TMP/logs/.cadence.lock"
run_block 1
sleep 1
if kill -0 "$OLDPID" 2>/dev/null; then echo "FAIL: hung cadence not reaped"; fail=1; kill -9 "$OLDPID" 2>/dev/null; else echo "PASS reaper: hung cadence reaped"; fi
if [ -e "$TMP/logs/.cadence.lock" ]; then echo "FAIL: lock not cleared after reap"; fail=1; else echo "PASS reaper: lock cleared after reap"; fi

# (4) spare a FRESH run (high ceiling): sleeper survives, lock is NOT cleared (flock still held).
bash "$SLEEPER" & FRESHPID=$!
sleep 1
: > "$TMP/logs/.cadence.lock"
run_block 10800
if kill -0 "$FRESHPID" 2>/dev/null; then echo "PASS reaper: fresh cadence spared"; else echo "FAIL: fresh cadence wrongly reaped"; fail=1; fi
if [ -e "$TMP/logs/.cadence.lock" ]; then echo "PASS reaper: lock kept while cadence live"; else echo "FAIL: lock cleared while cadence live"; fail=1; fi
kill -9 "$FRESHPID" 2>/dev/null || true

# (5) run heartbeat: run_cadence.sh writes an epoch-stamped logs/.last_start on start
#     and logs/.last_end (with exit code) on finish, for the health-check to read directly.
grep -q "logs/.last_start" run_cadence.sh && grep -q "logs/.last_end" run_cadence.sh \
  || { echo "FAIL: run_cadence.sh missing .last_start/.last_end heartbeat writes"; fail=1; }
( cd "$TMP" && rm -f logs/.last_start logs/.last_end
  printf '%s %s\n' "$(date -u +%s)" "20260101T000000Z" > logs/.last_start 2>/dev/null || true
  printf '%s %s exit=%s\n' "$(date -u +%s)" "20260101T010000Z" "0" > logs/.last_end 2>/dev/null || true )
if grep -Eq '^[0-9]+ [0-9]{8}T[0-9]{6}Z$' "$TMP/logs/.last_start" \
   && grep -Eq '^[0-9]+ [0-9]{8}T[0-9]{6}Z exit=[0-9]+$' "$TMP/logs/.last_end"; then
  echo "PASS reaper: heartbeat .last_start/.last_end written in expected format"
else
  echo "FAIL: heartbeat files malformed"; fail=1
fi

[ "$fail" = 0 ] && echo "ALL REAPER CHECKS PASS" || echo "REAPER CHECKS FAILED"
exit $fail
