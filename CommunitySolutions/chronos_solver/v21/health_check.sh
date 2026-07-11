#!/usr/bin/env bash
# One-line runner-liveness verdict for the 2-hourly health check.
#
# WHY: every health-check cycle re-derives runner liveness by listing cron_*.log,
# reading sandbox-LOCAL mtimes and converting them to UTC by hand (EDT=UTC-4) — the
# exact toil that keeps producing off-by-4h judgements. Last cycle taught
# run_cadence.sh to drop epoch-stamped logs/.last_start (on start) and logs/.last_end
# (on finish, with exit code). This script READS those two files and prints a single
# verdict line, so the check is `bash health_check.sh` instead of log-name + TZ math.
# It also drops logs/.stall_flag when stalled/hung (the detection half of BACKLOG P3
# #10 "stall alarm") and clears it when healthy, so a future alarm can act on one file.
#
# Pure/offline: touches only logs/ liveness files; never runs arcengine or the network.
# Exit: 0 healthy/running, 1 stalled/hung, 2 unknown (no heartbeat + no cron logs).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

LOGS="${V21_HEALTH_LOGS:-logs}"
# A real full pass (20 levels x 600s + toddler train/evolve) is < ~90m; a start with
# no matching end older than this => the run hung or its process died mid-pass.
HUNG_SECS="${V21_HEALTH_HUNG_SECS:-5400}"     # 90m
# launchd fires ~every 2h; allow slack before calling the scheduler dead.
STALL_SECS="${V21_HEALTH_STALL_SECS:-9000}"   # 2.5h

NOW="$(date -u +%s)"

# Portable epoch -> UTC stamp (BSD `date -r` on the Mac, GNU `date -d` in the sandbox).
_fmt() { date -u -r "$1" +%Y%m%dT%H%M%SZ 2>/dev/null || date -u -d "@$1" +%Y%m%dT%H%M%SZ 2>/dev/null || echo "epoch=$1"; }
_age_min() { echo $(( (NOW - $1) / 60 )); }
# File mtime, portable across GNU (sandbox) and BSD (Mac) stat. Each variant emits
# non-numeric noise on the wrong platform, so keep only all-digit output.
_mtime() {
  local m
  m="$(stat -c %Y "$1" 2>/dev/null)"; case "$m" in ''|*[!0-9]*) m="";; esac
  [ -z "$m" ] && { m="$(stat -f %m "$1" 2>/dev/null)"; case "$m" in ''|*[!0-9]*) m="";; esac; }
  printf '%s' "$m"
}

STATUS="UNKNOWN"; DETAIL="no heartbeat and no cron logs found"; RC=2

if [ -f "$LOGS/.last_start" ]; then
  read -r LS_EPOCH LS_STAMP _ < "$LOGS/.last_start" 2>/dev/null || true
  case "${LS_EPOCH:-}" in ''|*[!0-9]*) LS_EPOCH="";; esac
fi
if [ -f "$LOGS/.last_end" ]; then
  read -r LE_EPOCH LE_STAMP LE_EXIT _ < "$LOGS/.last_end" 2>/dev/null || true
  case "${LE_EPOCH:-}" in ''|*[!0-9]*) LE_EPOCH="";; esac
fi

if [ -n "${LS_EPOCH:-}" ]; then
  START_AGE=$(( NOW - LS_EPOCH ))
  # A run is in-progress iff it started at/after the last recorded end.
  IN_PROGRESS=0
  if [ -z "${LE_EPOCH:-}" ] || [ "$LS_EPOCH" -ge "${LE_EPOCH:-0}" ]; then IN_PROGRESS=1; fi
  if [ "$IN_PROGRESS" = 1 ] && [ "$START_AGE" -gt "$HUNG_SECS" ]; then
    STATUS="HUNG"; RC=1
    DETAIL="pass started $(_fmt "$LS_EPOCH") ($(_age_min "$LS_EPOCH")m ago) with no .last_end — hung or SIGKILLed mid-pass"
  elif [ "$IN_PROGRESS" = 1 ]; then
    STATUS="RUNNING"; RC=0
    DETAIL="pass in progress since $(_fmt "$LS_EPOCH") ($(_age_min "$LS_EPOCH")m ago)"
  elif [ "$START_AGE" -gt "$STALL_SECS" ]; then
    STATUS="STALLED"; RC=1
    DETAIL="last tick $(_fmt "$LS_EPOCH") ($(_age_min "$LS_EPOCH")m ago) > ${STALL_SECS}s — launchd not ticking (Mac asleep/off or job unloaded)"
  else
    STATUS="HEALTHY"; RC=0
    DETAIL="idle; last run $(_fmt "${LE_EPOCH:-$LS_EPOCH}") exit=${LE_EXIT:-?} ($(_age_min "${LE_EPOCH:-$LS_EPOCH}")m ago)"
  fi
else
  # Fallback: no heartbeat yet (run_cadence.sh predates it, or never ran) -> cron mtime.
  NEWEST="$(ls -1t "$LOGS"/cron_*.log 2>/dev/null | head -1)"
  if [ -n "$NEWEST" ]; then
    MT="$(_mtime "$NEWEST")"
    if [ -n "$MT" ]; then
      if [ $(( NOW - MT )) -gt "$STALL_SECS" ]; then
        STATUS="STALLED"; RC=1
        DETAIL="no heartbeat; newest cron log $(basename "$NEWEST") mtime $(_fmt "$MT") ($(_age_min "$MT")m ago) > ${STALL_SECS}s"
      else
        STATUS="HEALTHY"; RC=0
        DETAIL="no heartbeat; newest cron log $(basename "$NEWEST") $(_age_min "$MT")m ago (fallback)"
      fi
    fi
  fi
fi

# Stall flag: one file a future alarm/reporter can act on. Written when stalled/hung,
# cleared when healthy/running so it never lingers falsely.
if [ "$RC" = 1 ]; then
  printf '%s %s %s\n' "$NOW" "$STATUS" "$DETAIL" > "$LOGS/.stall_flag" 2>/dev/null || true
else
  rm -f "$LOGS/.stall_flag" 2>/dev/null || true
fi

echo "RUNNER: $STATUS | $DETAIL"
exit "$RC"
