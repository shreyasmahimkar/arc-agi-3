#!/usr/bin/env python3
"""v19 campaign orchestrator — escalating-cap solving with v17-style improvement
logging.

Runs the solving campaign in escalating passes (cheap caps first, up to a 5-HOUR
cap for the stubborn levels). Every pass is resumable (solved levels are skipped),
and after each pass we record the IMPROVEMENT (exactly which new levels were
solved) to CAMPAIGN_LOG.md — so each iteration provably moves the needle, the way
v17's ITERATIONS.md did.

Caps (seconds): 180 → 900 → 3600 → 18000 (5 h). Decide-as-you-go: the 5 h pass
only runs if earlier passes were still making progress (else it's wasted).

Usage:
    python run_campaign.py                      # full escalation
    python run_campaign.py --caps 180,900       # custom
    python run_campaign.py --max-cap 18000      # ceiling
"""
import argparse, glob, json, os, subprocess, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.join(HERE, "solutions")
LOG = os.path.join(HERE, "CAMPAIGN_LOG.md")
DEFAULT_CAPS = [180, 900, 3600, 18000]   # 3 min → 15 min → 1 h → 5 h


def corpus_state():
    """{game: {level:int -> action_count}}."""
    out = {}
    for f in glob.glob(os.path.join(SOL, "*.json")):
        g = os.path.basename(f)[:-5]
        try:
            out[g] = {int(k): len(v) for k, v in json.load(open(f)).items()}
        except Exception:
            pass
    return out


def totals(state):
    lv = sum(len(v) for v in state.values())
    games = sum(1 for v in state.values() if v)
    return games, lv


def diff_new(before, after):
    """List of 'game Lk' newly present in after."""
    new = []
    for g, lvls in after.items():
        b = before.get(g, {})
        for k in sorted(lvls):
            if k not in b:
                new.append(f"{g} L{k}")
    return new


def frontier_line(state):
    parts = []
    for g in sorted(state):
        lv = sorted(state[g])
        if lv:
            parts.append(f"{g}:L{lv[0]}-L{lv[-1]}({len(lv)})")
    return " | ".join(parts)


def log_pass(pass_i, cap, before, after, secs):
    games, lv = totals(after)
    new = diff_new(before, after)
    first = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        if first:
            f.write("# v19 solving campaign — improvement log (v17-style)\n\n")
            f.write("Each row is one escalating pass. **NEW** = levels solved that\n")
            f.write("were not solved before — the honest per-iteration improvement.\n\n")
            f.write("| pass | cap(s) | games | total_levels | NEW this pass | mins |\n")
            f.write("|---|---|---|---|---|---|\n")
        newstr = (f"+{len(new)} (" + ", ".join(new[:8]) +
                  ("…" if len(new) > 8 else "") + ")") if new else "+0"
        f.write(f"| {pass_i} | {cap} | {games} | {lv} | {newstr} | {secs/60:.1f} |\n")
        if pass_i == "final" or True:
            pass
    # rewrite the live frontier snapshot section at the tail each time
    snap = (f"\n<!-- frontier@{datetime.now():%H:%M:%S} -->\n"
            f"**Frontier ({games} games, {lv} levels):** {frontier_line(after)}\n")
    with open(LOG, "a") as f:
        f.write(snap)
    return games, lv, new


def run_pass(cap):
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, "solve_all.py"),
                        "--bfs-timeout", str(cap)],
                       cwd=HERE)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default=None, help="comma list of caps (s)")
    ap.add_argument("--max-cap", type=int, default=18000)
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")] if args.caps else \
        [c for c in DEFAULT_CAPS if c <= args.max_cap]

    print(f"[orchestrator] escalating caps {caps} (max {args.max_cap}s = {args.max_cap/3600:.1f}h)")
    for i, cap in enumerate(caps, 1):
        before = corpus_state()
        print(f"[orchestrator] === pass {i}: cap {cap}s ({cap/60:.0f} min) ===")
        secs = run_pass(cap)
        after = corpus_state()
        games, lv, new = log_pass(i, cap, before, after, secs)
        print(f"[orchestrator] pass {i} done: {games} games, {lv} levels, "
              f"+{len(new)} NEW ({secs/60:.1f} min). Logged -> CAMPAIGN_LOG.md")
        # decide-as-you-go: if a pass added nothing AND we've already spent real
        # time, escalating further (e.g. to 5 h) is unlikely to pay off — stop.
        if i >= 2 and not new and cap >= 3600:
            print("[orchestrator] no improvement at >=1h cap — stopping escalation.")
            break
    print("[orchestrator] campaign complete.")


if __name__ == "__main__":
    main()
