#!/usr/bin/env python3
"""Benchmark gate — the genuine (no-cache) solve depth for the tracked games must
NOT regress. This is the pre-commit guardrail: TDD-style, we lock the current
behaviour as the baseline and every commit must still meet it.

  python benchmark.py            # CHECK current depth >= baseline (exit 1 on regression)
  python benchmark.py --update   # RE-MEASURE true deepest level and record the baseline
  python benchmark.py --game ar25  # check/measure a single game

Genuine solving only: importing test_ls20 sets V19_STORE_SOLUTIONS=0 and
V19_CACHE_FALLBACK=0, so a level only counts if the agent searched and solved it
itself (never a cached answer). Baselines live in benchmark.json.

"Shall not move" = shall not REGRESS. A level cleared above the baseline is an
improvement — it prints a note to run --update to lock the new floor in.
"""
import json, os, sys, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_ls20 import run_episode          # import sets honest mode (cache OFF)

GAMES = ["ar25", "ls20"]
BASELINE = os.path.join(HERE, "benchmark.json")
BFS_TIMEOUT = 150
MAX_STEPS = 4000


def measure(game, target):
    best, steps, secs = run_episode(game, target=target, max_steps=MAX_STEPS,
                                    bfs_timeout=BFS_TIMEOUT, death_cap=8, verbose=False)
    return best, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="record current depth as the baseline")
    ap.add_argument("--game", help="restrict to one game")
    args = ap.parse_args()
    games = [args.game] if args.game else GAMES
    base = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}

    if args.update:
        for g in games:
            print(f"[measure] {g}: finding true deepest genuine level…")
            d, secs = measure(g, target=99)
            base[g] = d
            print(f"  {g}: {d} levels ({secs:.0f}s)")
        json.dump(base, open(BASELINE, "w"), indent=2)
        print("baselines recorded ->", {g: base[g] for g in games})
        return 0

    if not base:
        print("no benchmark.json yet — run: python benchmark.py --update")
        sys.exit(2)
    print("=== BENCHMARK GATE (genuine, cache OFF) — depth must not regress ===")
    t0 = time.time(); fails = []
    for g in games:
        floor = base.get(g)
        if floor is None:
            print(f"  {g}: no baseline (run --update) — skipping"); continue
        print(f"[check] {g}: must clear >= {floor} levels…")
        got, secs = measure(g, target=floor)        # early-stops once the floor is reached
        if got >= floor:
            extra = "  (>= floor; run --update if it now goes deeper)" if got > floor else ""
            print(f"  {g}: {got}/{floor} OK ({secs:.0f}s){extra}")
        else:
            print(f"  {g}: {got} < baseline {floor}  *** REGRESSION *** ({secs:.0f}s)")
            fails.append((g, got, floor))
    print(f"--- total {time.time()-t0:.0f}s ---")
    if fails:
        print("BENCHMARK REGRESSED — commit blocked:")
        for g, got, floor in fails:
            print(f"  {g}: {got} < {floor}")
        sys.exit(1)
    print("BENCHMARK held ✅ — all games meet their baseline.")
    sys.exit(0)


if __name__ == "__main__":
    main()
